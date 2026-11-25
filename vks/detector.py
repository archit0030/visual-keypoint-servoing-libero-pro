"""K-Net: a small CenterNet-style keypoint detector (one per camera).

Input is a 128x128 RGB image plus two CoordConv channels. Output is at stride 2
(64x64): a heatmap per class (0 = black bowl, 1 = plate), a shared 2-channel
sub-pixel offset and a per-class visibility logit. About 1.2 M parameters.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CLASSES = ('bowl', 'plate')
OUT, STRIDE, SIGMA = 64, 2, 1.5
MAX_PER_CLASS = (3, 1)   # at most 3 bowls and 1 plate per image
CONF = 0.5


def encode_targets(uv, size=OUT, sigma=SIGMA):
    """{class: [(u, v), ...]} pixel labels -> (heatmap, offsets, peak mask)."""
    heat = np.zeros((len(CLASSES), size, size), np.float32)
    off = np.zeros((2, size, size), np.float32)
    mask = np.zeros((size, size), bool)
    rr, cc = np.mgrid[0:size, 0:size]
    for k, pts in uv.items():
        for u, v in pts:
            x, y = u / STRIDE, v / STRIDE
            col, row = min(int(np.floor(x)), size - 1), min(int(np.floor(y)), size - 1)
            heat[k] = np.maximum(heat[k], np.exp(-((rr - row) ** 2 + (cc - col) ** 2) / (2 * sigma ** 2)))
            off[:, row, col] = x - col, y - row
            mask[row, col] = True
    return heat, off, mask


def _local_max(h):
    p = np.pad(h, 1, constant_values=-np.inf)
    shifts = [p[1 + dr:1 + dr + h.shape[0], 1 + dc:1 + dc + h.shape[1]] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    return h >= np.max(shifts, axis=0)


def decode_peaks(heat, off, conf=CONF, max_per_class=MAX_PER_CLASS):
    """3x3 NMS on the heatmap -> per class [(u, v, confidence)], most confident first."""
    out = []
    for k, cap in enumerate(max_per_class):
        h = heat[k]
        rows, cols = np.nonzero(_local_max(h) & (h >= conf))
        order = np.argsort(-h[rows, cols], kind='stable')[:cap]
        out.append([(float(STRIDE * (c + off[0, r, c])), float(STRIDE * (r + off[1, r, c])), float(h[r, c]))
                    for r, c in zip(rows[order], cols[order])])
    return out


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.proj = None
        if stride != 1 or cin != cout:
            self.proj = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.proj is None else self.proj(x)))


def _up(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class KNet(nn.Module):
    """ResNet-style encoder to stride 8, then two upsampling blocks with skips back to stride 2."""

    def __init__(self, width=(32, 64, 128)):
        super().__init__()
        a, b, c = width
        self.stem = nn.Sequential(nn.Conv2d(5, a, 7, 2, 3, bias=False), nn.BatchNorm2d(a), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(Block(a, a), Block(a, a))
        self.s2 = nn.Sequential(Block(a, b, 2), Block(b, b))
        self.s3 = nn.Sequential(Block(b, c, 2), Block(c, c), Block(c, c))
        self.up1, self.up2 = _up(c, b), _up(b, b)
        self.skip1 = nn.Conv2d(a, b, 1)
        self.heat = nn.Conv2d(b, len(CLASSES), 1)
        self.offset = nn.Conv2d(b, 2, 1)
        self.vis = nn.Linear(c, len(CLASSES))
        nn.init.constant_(self.heat.bias, -2.19)   # prior p = 0.1, as in CenterNet
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing='ij'))
        self.register_buffer('coords', coords[None], persistent=False)

    def forward(self, x):
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1).to(x.dtype)], 1)
        f1 = self.s1(self.stem(x))     # 64x64
        f2 = self.s2(f1)               # 32x32
        f3 = self.s3(f2)               # 16x16
        y = self.up1(F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=False)) + f2
        y = self.up2(F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)) + self.skip1(f1)
        return dict(heat=self.heat(y), offset=self.offset(y), vis=self.vis(f3.mean((2, 3))))


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss, normalised by the number of peaks."""
    p = torch.sigmoid(logits.float()).clamp(1e-6, 1 - 1e-6)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


class Detector:
    """Runs one K-Net per camera; returns {camera: {'bowl': [(u, v, c)], 'plate': [...]}}."""

    def __init__(self, net_agentview, net_wrist, device='cuda', conf=CONF):
        self.nets = {'agentview': net_agentview.to(device).eval(), 'robot0_eye_in_hand': net_wrist.to(device).eval()}
        self.device, self.conf = device, conf

    @torch.no_grad()
    def __call__(self, img_agentview, img_wrist):
        out = {}
        for cam, img in (('agentview', img_agentview), ('robot0_eye_in_hand', img_wrist)):
            x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
            y = self.nets[cam](x)
            heat = torch.sigmoid(y['heat'][0]).float().cpu().numpy()
            off = y['offset'][0].float().cpu().numpy()
            out[cam] = dict(zip(CLASSES, decode_peaks(heat, off, self.conf)))
        return out


def load_net(path, device='cpu'):
    net = KNet()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net.to(device)
