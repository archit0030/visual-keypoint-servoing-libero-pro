"""Object-relative stage controller.

Stages:  A  go above the bowl          Hb  hover and look, lock bowl estimate
         B  descend to grasp pose      C   close gripper
         D  lift                       E   move above the plate
         Hp hover and look, lock plate F   descend to release pose
         G  open gripper               R   retreat upwards

Grasp / release poses are offsets from the current bowl / plate estimate
(averaged over the demonstrations); the clearance heights h_g, h_lift, h_r are
demonstration medians. Motion between stages is a straight-line P-controller.
"""
from collections import deque
from dataclasses import dataclass
import mujoco
import numpy as np

POS_SCALE_M = 0.05   # metres per unit of OSC translation action
ROT_SCALE_RAD = 0.5  # radians per unit of OSC rotation action
GRIP = dict(A=-1.0, Hb=-1.0, B=-1.0, C=1.0, D=1.0, E=1.0, Hp=1.0, F=1.0, G=-1.0, R=-1.0)
HOLDING = ('D', 'E', 'Hp', 'F')
REACH = ('A', 'Hb', 'B')
ROLES = ('bowl_1', 'plate')


# ---------------------------------------------------------------- low-level control laws

def _limit(a):
    peak = np.max(np.abs(a))
    return a / peak if peak > 1.0 else a


def tracking_action(target, eef, kp, speed_cap_m):
    target, eef = np.asarray(target, float), np.asarray(eef, float)
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(eef))):
        return np.zeros(3)
    v = kp * (target - eef)
    norm = np.linalg.norm(v)
    if norm > speed_cap_m:
        v = v * (speed_cap_m / norm)
    return _limit(v / POS_SCALE_M)


def rotation_action(q_target, q_current, kr):
    """Axis-angle of R_target R_current^T (world frame), scaled by kr."""
    qt, qc = np.asarray(q_target, float), np.asarray(q_current, float)
    if not (np.all(np.isfinite(qt)) and np.all(np.isfinite(qc))):
        return np.zeros(3)
    qt, qc = qt / np.linalg.norm(qt), qc / np.linalg.norm(qc)
    inv, delta, rotvec = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_negQuat(inv, qc)
    mujoco.mju_mulQuat(delta, qt, inv)
    if delta[0] < 0:
        delta = -delta
    mujoco.mju_quat2Vel(rotvec, delta, 1.0)
    return _limit(rotvec * (kr / ROT_SCALE_RAD))


@dataclass(frozen=True)
class PhaseParams:
    w_empty_m: float      # finger width when the gripper closes on nothing
    n_closed: int = 3
    margin_m: float = 0.002
    eps_m: float = 0.001
    stable_steps: int = 1
    h_in: int = 1
    h_out: int = 3


class PhaseTracker:
    """0 = reach, 1 = transport (holding the bowl), from gripper command and finger width only."""

    def __init__(self, params):
        self.p = params
        self.reset()

    def reset(self):
        self.cmds = deque(maxlen=self.p.n_closed)
        self.widths = deque(maxlen=self.p.stable_steps + 1)
        self.phase, self.count = 0, 0

    def update(self, last_cmd, width):
        p = self.p
        self.cmds.append(last_cmd)
        self.widths.append(width)
        thr = p.w_empty_m + p.margin_m
        if self.phase == 0:
            closed = len(self.cmds) == p.n_closed and all(c > 0 for c in self.cmds)
            stable = len(self.widths) == p.stable_steps + 1 and bool(np.all(np.abs(np.diff(self.widths)) < p.eps_m))
            self.count = self.count + 1 if closed and width > thr and stable else 0
            limit = p.h_in
        else:
            self.count = self.count + 1 if last_cmd < 0 or width <= thr else 0
            limit = p.h_out
        if self.count >= limit:
            self.phase, self.count = 1 - self.phase, 0
        return self.phase


class Estimator:
    """Median filter over recent detections, plus latching (freeze the estimate while hovering)."""

    def __init__(self, filter_n=5, gate_m=0.04, release_frames=3):
        self.filter_n, self.gate_m, self.release_frames = int(filter_n), gate_m, int(release_frames)
        self.buf = {r: deque(maxlen=self.filter_n) for r in ROLES}
        self.last = {r: None for r in ROLES}
        self.locks = {r: None for r in ROLES}
        self.moved = {r: 0 for r in ROLES}

    def update(self, detections):
        for r in ROLES:
            d = detections.get(r)
            self.last[r] = None if d is None else np.asarray(d, float)
            if d is not None:
                self.buf[r].append(self.last[r])

    def raw(self, role):
        return self.last[role]

    def filtered(self, role):
        b = self.buf[role]
        if not b:
            return None
        return b[-1].copy() if self.filter_n == 1 else np.median(np.array(b), axis=0)

    def value(self, role):
        return self.locks[role] if self.locks[role] is not None else self.filtered(role)

    def latch(self, role):
        self.locks[role], self.moved[role] = self.filtered(role), 0

    def unlatch(self, *roles):
        for r in roles or ROLES:
            self.locks[r], self.moved[r] = None, 0

    def check_release(self, role):
        """Unlock if the detection stays more than gate_m away from the lock for release_frames frames."""
        if self.locks[role] is None:
            return False
        d = self.last[role]
        gone = d is not None and np.linalg.norm(d - self.locks[role]) > self.gate_m
        self.moved[role] = self.moved[role] + 1 if gone else 0
        if self.moved[role] >= self.release_frames:
            self.unlatch(role)
            return True
        return False


# ---------------------------------------------------------------- stage machine

@dataclass(frozen=True)
class ControllerConfig:
    g: np.ndarray          # grasp offset from the bowl keypoint
    q_g: np.ndarray        # grasp orientation (wxyz)
    r: np.ndarray          # release offset from the plate keypoint
    q_r: np.ndarray        # release orientation
    h_g: float             # approach height above the grasp pose
    h_lift: float          # lift height after grasping
    h_r: float             # approach height above the release pose
    eps_via: float = 0.02
    eps_grasp: float = 0.010
    eps_release: float = 0.015
    jump_m: float = 0.04
    empty_steps: int = 3
    hover: bool = True
    hover_max_steps: int = 6
    hover_agree_n: int = 3
    hover_agree_m: float = 0.003
    via_exit: bool = True
    via_xy_m: float = 0.02
    via_clear_m: float = 0.06
    via_progress_m: float = 0.002
    via_progress_steps: int = 10
    retreat: bool = True
    open_steps: int = 5
    retreat_m: float = 0.08
    tsr: bool = True       # transport-stall release
    tsr_steps: int = 60
    tsr_xy_m: float = 0.03
    filter_n: int = 5
    latch_gate_m: float = 0.04
    latch_release_frames: int = 3

    @classmethod
    def from_json(cls, d):
        return cls(**{k: np.asarray(v, float) if isinstance(v, list) else v for k, v in d.items()})


@dataclass(frozen=True)
class Gains:
    kp: float
    kr: float
    speed_cap_m: float
    phase: PhaseParams


class StageController:
    """One instance per episode: (eef, quat, phase, last cmd, width, detections) -> (action, info)."""

    def __init__(self, cfg, gains):
        self.c, self.s = cfg, gains
        self.thr = gains.phase.w_empty_m + gains.phase.margin_m
        self.est = Estimator(cfg.filter_n, cfg.latch_gate_m, cfg.latch_release_frames)
        self.stage, self.prev_kb, self.entry, self.empty = 'A', None, None, 0
        self.via_hist, self.hover_buf, self.hover_steps, self.hold = [], [], 0, None
        self.opened, self.retreat_target = 0, None
        self.t, self.d_entry = -1, None

    def _via_ok(self, eef, via, keypose):
        """Reached the via point above a keypose (or stopped making progress while above it)."""
        c = self.c
        err = float(np.linalg.norm(eef - via))
        if not c.via_exit:
            return err < c.eps_via
        self.via_hist.append(err)
        k = c.via_progress_steps
        stalled = len(self.via_hist) > k and self.via_hist[-1 - k] - err < c.via_progress_m
        return (np.linalg.norm(eef[:2] - via[:2]) <= c.via_xy_m and eef[2] >= keypose[2] + c.via_clear_m
                and (err <= c.eps_via or stalled))

    def _hover_done(self, role):
        """Done hovering once the last few detections agree within a few mm (or after hover_max_steps)."""
        c = self.c
        self.hover_buf.append(self.est.raw(role))
        last = self.hover_buf[-c.hover_agree_n:]
        agree = len(last) == c.hover_agree_n and all(x is not None for x in last) and \
            max(np.linalg.norm(a - b) for a in last for b in last) <= c.hover_agree_m
        if agree or self.hover_steps >= c.hover_max_steps:
            return True
        self.hover_steps += 1
        return False

    def _enter_hover(self, eef, role):
        self.hover_buf, self.hover_steps, self.hold = [self.est.raw(role)], 1, eef.copy()

    def __call__(self, eef, q, phase, cmd, width, detections):
        c, est = self.c, self.est
        eef = np.asarray(eef, float)
        self.t += 1
        est.update(detections)
        kb_f = est.filtered('bowl_1')
        jump = self.prev_kb is not None and kb_f is not None and np.linalg.norm(kb_f - self.prev_kb) > c.jump_m
        self.prev_kb = kb_f
        kb, kp = est.value('bowl_1'), est.value('plate')
        grasp = None if kb is None else kb + c.g
        release = None if kp is None else kp + c.r
        near = lambda p, eps: p is not None and np.linalg.norm(eef - p) < eps

        # at most one stage transition per step; interrupts first
        st = self.stage
        if st in HOLDING and phase == 0:                   # dropped the bowl -> start over
            st = 'A'
            est.unlatch()
        elif st in REACH and jump:                         # bowl estimate jumped -> start over
            st = 'A'
            est.unlatch('bowl_1')
        elif st == 'B' and est.check_release('bowl_1'):
            st = 'A'
        elif (c.tsr and st in ('E', 'Hp', 'F') and self.d_entry is not None and self.t - self.d_entry >= c.tsr_steps
              and release is not None and np.linalg.norm(eef[:2] - release[:2]) <= c.tsr_xy_m
              and eef[2] <= release[2] + c.h_r):          # stuck close above the plate -> release
            st, self.opened = 'G', 1
        elif st == 'A' and grasp is not None and self._via_ok(eef, grasp + [0.0, 0.0, c.h_g], grasp):
            if c.hover:
                st = 'Hb'
                self._enter_hover(eef, 'bowl_1')
            else:
                st = 'B'
        elif st == 'Hb' and self._hover_done('bowl_1'):
            est.latch('bowl_1')
            st = 'B'
        elif st == 'B' and near(grasp, c.eps_grasp):
            st, self.empty = 'C', 0
        elif st == 'C':
            self.empty = self.empty + 1 if phase == 0 and cmd > 0 and width <= self.thr else 0
            if phase == 1:
                st, self.entry = 'D', eef + [0.0, 0.0, c.h_lift]
            elif self.empty >= c.empty_steps:              # closed on nothing
                st = 'A'
                est.unlatch('bowl_1')
        elif st == 'D' and near(self.entry, c.eps_via):
            st = 'E'
        elif st == 'E' and release is not None and self._via_ok(eef, release + [0.0, 0.0, c.h_r], release):
            if c.hover:
                st = 'Hp'
                self._enter_hover(eef, 'plate')
            else:
                st = 'F'
        elif st == 'Hp' and self._hover_done('plate'):
            est.latch('plate')
            st = 'F'
        elif st == 'F' and near(release, c.eps_release):
            st, self.opened = 'G', 1
        elif st == 'G' and c.retreat:
            if self.opened >= c.open_steps:
                st, self.retreat_target = 'R', eef + [0.0, 0.0, c.retreat_m]
            else:
                self.opened += 1

        if st != self.stage:
            self.via_hist = []
            if st == 'D':
                self.d_entry = self.t
        self.stage = st

        kb, kp = est.value('bowl_1'), est.value('plate')
        grasp = None if kb is None else kb + c.g
        release = None if kp is None else kp + c.r
        target, q_ref = {
            'A': (None if grasp is None else grasp + [0.0, 0.0, c.h_g], c.q_g), 'Hb': (self.hold, c.q_g),
            'B': (grasp, c.q_g), 'C': (grasp, c.q_g), 'D': (self.entry, c.q_r),
            'E': (None if release is None else release + [0.0, 0.0, c.h_r], c.q_r), 'Hp': (self.hold, c.q_r),
            'F': (release, c.q_r), 'G': (None, None), 'R': (self.retreat_target, c.q_r)}[st]
        if target is None:
            action = np.r_[np.zeros(6), GRIP[st]]
        else:
            action = np.r_[tracking_action(target, eef, self.s.kp, self.s.speed_cap_m),
                           rotation_action(q_ref, q, self.s.kr), GRIP[st]]
        return action, dict(stage=st, bowl=kb, plate=kp)
