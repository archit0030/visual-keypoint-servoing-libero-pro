"""Train one K-Net keypoint detector (run once per camera).

Training data are HDF5 shards rendered from the 45 training demonstrations
with the objects re-placed at random (see README). Each shard holds
  img_a, img_w   (N, 128, 128, 3) uint8   agentview / wrist images
  uv             (N, 3, 2, 2)  pixel keypoints [bowl_1, bowl_2, plate] x camera x (u, v)
  vis            (N, 3, 2)     keypoint visible in that camera

Loss = focal (heatmap) + L1 (sub-pixel offset at the peaks) + 0.1 BCE (visibility).
Augmentation: brightness/contrast +-20%, hue +-0.02, pixel noise 0.02, shifts of +-4 px.

  python train_detector.py --camera agentview --train data/renders/train/*.h5 \
      --dev data/renders/dev/*.h5 --out checkpoints/train_agentview
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from vks.detector import KNet, OUT, SIGMA, STRIDE, decode_peaks, focal_loss

CLS = torch.tensor([0, 0, 1])   # label slots bowl_1, bowl_2 -> class bowl; plate -> class plate
CFG = dict(lr=1e-3, weight_decay=1e-4, batch=128, epochs=30, seed=0, vis_weight=0.1, shift_px=4,
           brightness=0.2, contrast=0.2, hue=0.02, noise=0.02)
YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def load_shards(paths, camera):
    key, j = ('img_a', 0) if camera == 'agentview' else ('img_w', 1)
    imgs, uv, vis = [], [], []
    for p in paths:
        with h5py.File(p, 'r') as f:
            imgs.append(f[key][()])
            uv.append(f['uv'][()][:, :, j])
            vis.append(f['vis'][()][:, :, j])
    return np.concatenate(imgs), np.concatenate(uv).astype(np.float32), np.concatenate(vis)


def make_targets(pts, valid, size=OUT, sigma=SIGMA):
    """Batched version of detector.encode_targets on the GPU."""
    b, _ = valid.shape
    dev = pts.device
    x, y = pts[..., 0] / STRIDE, pts[..., 1] / STRIDE
    col = torch.clamp(torch.floor(x), max=size - 1)
    row = torch.clamp(torch.floor(y), max=size - 1)
    rr = torch.arange(size, device=dev, dtype=pts.dtype).view(1, 1, size, 1)
    cc = torch.arange(size, device=dev, dtype=pts.dtype).view(1, 1, 1, size)
    g = torch.exp(-((rr - row[..., None, None]) ** 2 + (cc - col[..., None, None]) ** 2) / (2 * sigma ** 2))
    g = g * valid[..., None, None]
    heat = torch.zeros(b, 2, size, size, device=dev, dtype=pts.dtype)
    for c in (0, 1):
        heat[:, c] = g[:, CLS.to(dev) == c].amax(1)
    off = torch.zeros(b, 2, size, size, device=dev, dtype=pts.dtype)
    mask = torch.zeros(b, size, size, device=dev, dtype=torch.bool)
    bi, ki = torch.nonzero(valid, as_tuple=True)
    r, c = row[bi, ki].long(), col[bi, ki].long()
    off[bi, 0, r, c] = (x - col)[bi, ki]
    off[bi, 1, r, c] = (y - row)[bi, ki]
    mask[bi, r, c] = True
    return heat, off, mask


def shift(img, pts, valid, dx, dy, size=128):
    """Integer translation of images and labels; labels pushed out of the image become invalid."""
    out = torch.zeros_like(img)
    for i in range(img.shape[0]):
        x, y = int(dx[i]), int(dy[i])
        src = img[i, :, max(0, -y):size - max(0, y), max(0, -x):size - max(0, x)]
        out[i, :, max(0, y):max(0, y) + src.shape[1], max(0, x):max(0, x) + src.shape[2]] = src
    p = pts + torch.stack([dx, dy], -1).to(pts)[:, None, :]
    inside = (p[..., 0] >= 0) & (p[..., 0] <= size) & (p[..., 1] >= 0) & (p[..., 1] <= size)
    return out, p, valid & inside


def photometric(x, gen, cfg=CFG):
    b = x.shape[0]
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(b, 1, 1, 1, device=x.device, generator=gen)
    x = x * u(1 - cfg['brightness'], 1 + cfg['brightness'])
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * u(1 - cfg['contrast'], 1 + cfg['contrast']) + mean
    ang = 2 * np.pi * u(-cfg['hue'], cfg['hue']).view(b)
    yiq = torch.einsum('ij,bjhw->bihw', YIQ.to(x.device), x)
    cs, sn = torch.cos(ang).view(b, 1, 1), torch.sin(ang).view(b, 1, 1)
    i, q = yiq[:, 1] * cs - yiq[:, 2] * sn, yiq[:, 1] * sn + yiq[:, 2] * cs
    x = torch.einsum('ij,bjhw->bihw', torch.linalg.inv(YIQ).to(x.device), torch.stack([yiq[:, 0], i, q], 1))
    x = x + cfg['noise'] * torch.randn(x.shape, device=x.device, generator=gen)
    return x.clamp(0, 1)


@torch.no_grad()
def pixel_error(net, imgs, uv, vis, device, batch=256, match_px=10.0):
    """Median / p95 pixel error of the nearest same-class peak, and the miss rate (no peak within 10 px)."""
    net.eval()
    errs = []
    for s in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[s:s + batch]).to(device).permute(0, 3, 1, 2).float() / 255
        y = net(x)
        heat = torch.sigmoid(y['heat'].float()).cpu().numpy()
        off = y['offset'].float().cpu().numpy()
        for i in range(len(heat)):
            peaks = decode_peaks(heat[i], off[i])
            for slot, cls in ((0, 0), (2, 1)):
                if not vis[s + i, slot]:
                    continue
                d = [np.hypot(u - uv[s + i, slot, 0], v - uv[s + i, slot, 1]) for u, v, _ in peaks[cls]]
                errs.append(min(d) if d and min(d) <= match_px else np.inf)
    net.train()
    e = np.array(errs)
    ok = e[np.isfinite(e)]
    return dict(n=len(e), median_px=float(np.median(ok)) if len(ok) else None,
                p95_px=float(np.percentile(ok, 95)) if len(ok) else None, miss_rate=float(np.isinf(e).mean()))


def train(camera, train_paths, dev_paths, out, epochs, device, cfg=CFG):
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg['seed'])
    gen = torch.Generator(device=device).manual_seed(cfg['seed'])
    cpu_gen = torch.Generator().manual_seed(cfg['seed'])
    imgs, uv, vis = load_shards(train_paths, camera)
    dev = load_shards(dev_paths, camera) if dev_paths else None
    print(f'{camera}: {len(imgs)} training images')

    net = KNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    batch = min(cfg['batch'], len(imgs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * (len(imgs) // batch)))
    imgs_t, pts_t, vis_t = torch.from_numpy(imgs), torch.from_numpy(uv), torch.from_numpy(vis)
    best = None
    for ep in range(1, epochs + 1):
        tic, losses = time.time(), []
        perm = torch.randperm(len(imgs), generator=cpu_gen)
        for s in range(0, len(imgs) - batch + 1, batch):
            idx = perm[s:s + batch]
            x = imgs_t[idx].to(device).permute(0, 3, 1, 2).float() / 255
            p = torch.nan_to_num(pts_t[idx].to(device), nan=-1000.0)
            v = vis_t[idx].to(device)
            dx = torch.randint(-cfg['shift_px'], cfg['shift_px'] + 1, (len(idx),), generator=cpu_gen)
            dy = torch.randint(-cfg['shift_px'], cfg['shift_px'] + 1, (len(idx),), generator=cpu_gen)
            x, p, v = shift(x, p, v, dx, dy)
            x = photometric(x, gen, cfg)
            heat_t, off_t, mask = make_targets(p, v)
            vis_target = torch.stack([v[:, :2].any(1), v[:, 2]], 1).float()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device != 'cpu'):
                y = net(x)
            l_heat = focal_loss(y['heat'], heat_t)
            m = mask[:, None].expand_as(off_t)
            l_off = F.l1_loss(y['offset'].float()[m], off_t[m], reduction='sum') / mask.sum().clamp(min=1)
            l_vis = F.binary_cross_entropy_with_logits(y['vis'].float(), vis_target)
            loss = l_heat + l_off + cfg['vis_weight'] * l_vis
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        torch.save(net.state_dict(), out / f'epoch_{ep:02d}.pt')
        row = dict(epoch=ep, loss=float(np.mean(losses)), seconds=round(time.time() - tic, 1))
        if dev is not None:
            row['dev'] = pixel_error(net, *dev, device)
            score = row['dev']['p95_px'] if row['dev']['p95_px'] is not None else float('inf')
            if best is None or score < best[1]:
                best = (ep, score)
        print(row, flush=True)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
    if best:
        print(f'best dev epoch: {best[0]} (p95 {best[1]:.2f} px)')
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', required=True, choices=['agentview', 'wrist'])
    ap.add_argument('--train', nargs='+', required=True, type=Path)
    ap.add_argument('--dev', nargs='*', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    cam = 'agentview' if args.camera == 'agentview' else 'robot0_eye_in_hand'
    train(cam, args.train, args.dev, args.out, args.epochs, args.device)
