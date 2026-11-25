import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_layouts():
    from evaluate import load_layouts
    layouts = load_layouts()
    counts = {}
    for l in layouts:
        counts[l['family']] = counts.get(l['family'], 0) + 1
    assert counts == {'Standard': 30, 'P2': 30, 'F-obj': 53, 'F-dest': 69, 'F-both': 32}
    assert all(l['state'].ndim == 1 for l in layouts)


def test_policy_runs_on_a_fake_observation():
    from vks.policy import VKSPolicy
    policy = VKSPolicy.from_config(device='cpu')
    rng = np.random.default_rng(0)
    obs = dict(agentview_image=rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
               robot0_eye_in_hand_image=rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
               robot0_eef_pos=np.array([-0.2, 0.0, 1.1]), robot0_eef_quat=np.array([1.0, 0.0, 0.0, 0.0]),
               robot0_gripper_qpos=np.array([0.04, -0.04]))
    for _ in range(3):
        action, info = policy.step(obs)
        assert action.shape == (7,) and np.all(np.abs(action) <= 1.0)
