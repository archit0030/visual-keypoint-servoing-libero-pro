import json
from pathlib import Path

import numpy as np

from vks.controller import ControllerConfig, Gains, PhaseParams, PhaseTracker, StageController, tracking_action

CFG = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'vks.json').read_text())


def make():
    s = CFG['settings']
    gains = Gains(s['kp'], s['kr'], s['speed_cap_m'], PhaseParams(**s['phase']))
    return StageController(ControllerConfig.from_json(CFG['controller']), gains), gains


def test_tracking_is_speed_limited():
    a = tracking_action([1.0, 0, 0], [0, 0, 0], kp=1.0, speed_cap_m=0.04)
    assert np.allclose(a, [0.8, 0, 0])
    assert np.all(np.abs(tracking_action([5, 5, 5], [0, 0, 0], 100.0, 10.0)) <= 1.0)


def test_phase_tracker_detects_grasp_and_release():
    p = PhaseParams(w_empty_m=0.001)
    tr = PhaseTracker(p)
    assert [tr.update(1.0, 0.03) for _ in range(3)][-1] == 1      # closed on an object
    assert [tr.update(-1.0, 0.03) for _ in range(3)][-1] == 0     # opened again


def test_controller_goes_above_bowl_first():
    ctrl, _ = make()
    q = np.array([0.0, 1.0, 0.0, 0.0])
    bowl = np.array([-0.07, 0.0, 0.90])
    action, info = ctrl(np.array([0.0, 0.0, 1.2]), q, 0, -1.0, 0.08, {'bowl_1': bowl, 'plate': None})
    assert info['stage'] == 'A'
    assert action[6] == -1.0                       # gripper open
    assert action[0] < 0                           # moving towards the bowl in x
    assert np.all(np.abs(action[:6]) <= 1.0)


def test_controller_waits_without_detections():
    ctrl, _ = make()
    action, info = ctrl(np.zeros(3), np.array([1.0, 0, 0, 0]), 0, -1.0, 0.08, {})
    assert info['stage'] == 'A' and np.all(action[:6] == 0)
