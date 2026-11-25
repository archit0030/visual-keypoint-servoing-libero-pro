"""Run the VKS policy on the test layouts.

Families (data/layouts.json):
  Standard  original LIBERO-Spatial layouts (init states 0-29)
  P2        LIBERO-PRO position swap: plate and cookie box exchange places (init 0-29)
  F-obj     source bowl moved 6-15 cm
  F-dest    plate moved 6-18 cm
  F-both    bowl and plate both moved (6-18 cm each)

Each episode: reset to the layout, 10 settle steps with the gripper open, then
up to 220 policy steps. Success is env.check_success().

  python evaluate.py --family P2 --n 5
  python evaluate.py --ids standard_i00 fdest_j173_c57_r177_a205 --video
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from vks.policy import VKSPolicy

ROOT = Path(__file__).resolve().parent
FAMILIES = ('Standard', 'P2', 'F-obj', 'F-dest', 'F-both')
SETTLE_STEPS, MAX_STEPS = 10, 220
SETTLE_ACTION = np.array([0., 0., 0., 0., 0., 0., -1.])


def load_layouts():
    meta = json.loads((ROOT / 'data' / 'layouts.json').read_text())['layouts']
    with np.load(ROOT / 'data' / 'layout_states.npz') as f:
        states = {k: f[k].copy() for k in f.files}
    return [dict(m, state=states[m['id']]) for m in meta]


def make_env(scene):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(ROOT / 'assets' / f'{scene}.bddl'),
                             camera_heights=128, camera_widths=128)
    env.env.set_xml_processor(None)
    return env


class VideoWriter:
    """Saves the two camera images the policy sees (upscaled to 256 px) with a caption bar."""

    def __init__(self, path, title):
        self.path, self.title, self.frames = path, title, []

    def add(self, obs, text):
        from PIL import Image, ImageDraw
        views = [Image.fromarray(obs[k][::-1]).resize((256, 256), Image.BILINEAR)
                 for k in ('agentview_image', 'robot0_eye_in_hand_image')]
        img = Image.new('RGB', (512, 296), (24, 28, 36))
        img.paste(views[0], (0, 40))
        img.paste(views[1], (256, 40))
        d = ImageDraw.Draw(img)
        d.text((10, 6), self.title, fill=(240, 240, 240))
        d.text((10, 22), text, fill=(170, 220, 170) if 'SUCCESS' in text else (200, 200, 200))
        self.frames.append(np.asarray(img))

    def close(self):
        import imageio
        frames = self.frames + [self.frames[-1]] * 20
        with imageio.get_writer(self.path, fps=20, codec='libx264', macro_block_size=8,
                                ffmpeg_params=['-crf', '28', '-pix_fmt', 'yuv420p']) as w:
            for f in frames:
                w.append_data(f)


def run_episode(env, policy, layout, video=None):
    env.seed(0)
    env.reset()
    obs = env.set_init_state(layout['state'])
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(SETTLE_ACTION)
    policy.reset()
    success_step, stages = None, []
    for t in range(MAX_STEPS):
        action, info = policy.step(obs)
        stages.append(info['stage'])
        if video:
            video.add(obs, f'step {t:3d}  stage {info["stage"]}')
        obs, _, _, _ = env.step(action)
        if env.check_success():
            success_step = t
            break
    if video:
        video.add(obs, f'step {t:3d}  ' + ('SUCCESS: bowl on plate' if success_step is not None else 'not completed'))
        video.close()
    bowl, plate = np.asarray(obs['akita_black_bowl_1_pos']), np.asarray(obs['plate_1_pos'])
    return dict(success=success_step is not None, success_step=success_step, steps=t + 1,
                final_stage=stages[-1], final_bowl_plate_xy_mm=float(1e3 * np.linalg.norm(bowl[:2] - plate[:2])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    ap.add_argument('--ids', nargs='+', help='run only these layout ids')
    ap.add_argument('--n', type=int, help='first n layouts of each family')
    ap.add_argument('--out', type=Path, default=ROOT / 'runs')
    ap.add_argument('--video', action='store_true', help='save an mp4 per episode in <out>/videos')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    layouts = load_layouts()
    if args.ids:
        layouts = [l for l in layouts if l['id'] in args.ids]
    else:
        layouts = [l for l in layouts if l['family'] in args.family]
        if args.n:
            layouts = [l for f in args.family for l in [x for x in layouts if x['family'] == f][:args.n]]

    args.out.mkdir(parents=True, exist_ok=True)
    if args.video:
        (args.out / 'videos').mkdir(exist_ok=True)
    policy = VKSPolicy.from_config(device=args.device)
    envs, counts = {}, {}
    with (args.out / 'episodes.jsonl').open('w') as log:
        for lay in layouts:
            if lay['scene'] not in envs:
                envs[lay['scene']] = make_env(lay['scene'])
            env = envs[lay['scene']]
            video = None
            if args.video:
                video = VideoWriter(args.out / 'videos' / f"{lay['id']}.mp4", f"VKS | {lay['family']} | {lay['id']}")
            tic = time.time()
            res = run_episode(env, policy, lay, video)
            row = dict(id=lay['id'], family=lay['family'], **res, seconds=round(time.time() - tic, 1))
            log.write(json.dumps(row) + '\n')
            log.flush()
            k = counts.setdefault(lay['family'], [0, 0])
            k[0] += res['success']
            k[1] += 1
            print(f"{lay['id']:32s} success={res['success']!s:5s} step={res['success_step']}", flush=True)
    for env in envs.values():
        env.close()

    summary = {f: dict(successes=s, episodes=n, success_rate_pct=round(100 * s / n, 1)) for f, (s, n) in counts.items()}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
