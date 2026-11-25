"""The full VKS policy: camera images + proprioception in, OSC_POSE action out.

Only these observation keys are used (no object poses from the simulator):
agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat,
robot0_gripper_qpos.
"""
import json
from pathlib import Path
import numpy as np

from .controller import ControllerConfig, Gains, PhaseParams, PhaseTracker, StageController
from .detector import Detector, load_net
from .geometry import Calibration, mat_to_quat_wxyz, quat_xyzw_to_mat
from .perception import Perception, SelectorParams

ROOT = Path(__file__).resolve().parents[1]


class VKSPolicy:
    def __init__(self, detector, perception, controller_cfg, gains, hand_to_site):
        self.detector, self.perception = detector, perception
        self.cfg, self.gains = controller_cfg, gains
        self.hand_to_site = np.asarray(hand_to_site, float)
        self.open_thr = gains.phase.w_empty_m + gains.phase.margin_m
        self.reset()

    @classmethod
    def from_config(cls, path=ROOT / 'configs' / 'vks.json', device='cuda'):
        import torch
        torch.backends.cudnn.benchmark = False
        cfg = json.loads(Path(path).read_text())
        calib = Calibration.from_json(cfg['calibration'])
        variances = {(role, cam, int(ph)): float(v) for role, per_cam in cfg['variances'].items()
                     for cam, per_ph in per_cam.items() for ph, v in per_ph.items()}
        perception = Perception(calib, variances, SelectorParams(**cfg['selector']))
        detector = Detector(load_net(ROOT / cfg['weights']['agentview'], device),
                            load_net(ROOT / cfg['weights']['wrist'], device), device=device)
        s = cfg['settings']
        gains = Gains(kp=float(s['kp']), kr=float(s['kr']), speed_cap_m=float(s['speed_cap_m']),
                      phase=PhaseParams(**s['phase']))
        return cls(detector, perception, ControllerConfig.from_json(cfg['controller']), gains, calib.hand_to_site)

    def reset(self):
        self.controller = StageController(self.cfg, self.gains)
        self.tracker = PhaseTracker(self.gains.phase)
        self.perception.reset()
        self.t, self.cmd, self.phase = 0, -1.0, 0

    def step(self, obs):
        qg = np.asarray(obs['robot0_gripper_qpos'], float)
        width = float(qg[0] - qg[1])
        if self.t > 0:
            self.phase = self.tracker.update(self.cmd, width)
        eef = np.asarray(obs['robot0_eef_pos'], float)
        q_hand = np.asarray(obs['robot0_eef_quat'], float)               # xyzw, hand body
        q_site = mat_to_quat_wxyz(quat_xyzw_to_mat(q_hand) @ self.hand_to_site)

        peaks = self.detector(obs['agentview_image'], obs['robot0_eye_in_hand_image'])
        gripper_open = bool(self.cmd < 0 or (self.phase == 0 and width <= self.open_thr))
        det = self.perception.update(peaks, eef, q_hand, self.phase, gripper_open)
        action, info = self.controller(eef, q_site, self.phase, self.cmd, width, det)

        self.cmd = float(action[6])
        self.t += 1
        return np.asarray(action, float), dict(info, phase=self.phase)
