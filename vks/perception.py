"""Detections -> one 3D estimate of the bowl and the plate per frame.

1. back-project every detected peak onto its known plane (bowl / plate height)
2. pick the source bowl (the one nearest the table-centre region) and track it
3. per camera, keep the detection closest to the current track (4 cm gate)
4. fuse the two cameras with inverse-variance weights
"""
from dataclasses import dataclass
import numpy as np
from .geometry import TABLE_CENTER_REGION, distance_to_region

CAMERAS = ('agentview', 'robot0_eye_in_hand')
ROLES = ('bowl_1', 'plate')
CLASS_OF = {'bowl_1': 'bowl', 'plate': 'plate'}
GATE_M, CONF = 0.04, 0.5
REGION_CENTER = np.array([np.mean(TABLE_CENTER_REGION[0]), np.mean(TABLE_CENTER_REGION[1])])


@dataclass(frozen=True)
class SelectorParams:
    radius_m: float = 0.23           # bowls further than this from the table centre are not candidates
    gate_m: float = 0.04
    coast_frames: int = 10
    release_frames: int = 3
    conf_threshold: float = 0.5
    pregrasp_radius_m: float = 0.06  # lock the target once the gripper is this close


class BowlSelector:
    """Chooses which black bowl is the source bowl and keeps tracking the same one.

    The target is locked once the gripper is near it, and never switches while
    the gripper is closing or carrying the bowl.
    """

    def __init__(self, params=SelectorParams()):
        self.p = params
        self.reset()

    def reset(self):
        self.target, self.latched, self.misses, self.moved = None, False, 0, 0

    def _closest_to_centre(self, eligible):
        return min(eligible, key=lambda xy: (distance_to_region(xy), float(np.linalg.norm(xy - REGION_CENTER))))

    def update(self, detections, eef_xy, gripper_open, phase):
        p = self.p
        eef_xy = np.asarray(eef_xy, float)[:2]
        every = [np.asarray(xy, float)[:2] for xy, _ in detections]
        eligible = [xy for xy, (_, c) in zip(every, detections)
                    if c >= p.conf_threshold and distance_to_region(xy) <= p.radius_m]
        locked = phase == 1 or not gripper_open
        dist = lambda a, b: float(np.linalg.norm(a - b))

        if self.target is None:
            if not eligible:
                return None
            self.target = self._closest_to_centre(eligible).copy()
        elif self.latched:
            gated = [xy for xy in every if dist(xy, self.target) <= p.gate_m]
            if gated:
                self.target = min(gated, key=lambda xy: dist(xy, self.target)).copy()
            if locked:
                self.moved = 0
            else:
                # the locked bowl seems gone: eligible bowls exist but none is inside the gate
                gone = (bool(eligible) and min(dist(xy, self.target) for xy in eligible) > p.gate_m
                        and all(dist(xy, self.target) > p.gate_m for xy in every))
                self.moved = self.moved + 1 if gone else 0
                if self.moved >= p.release_frames:
                    self.latched, self.moved, self.misses = False, 0, p.coast_frames
                    return self.target.copy()
            return self.target.copy()
        elif not locked:
            gated = [xy for xy in eligible if dist(xy, self.target) <= p.gate_m]
            if gated:
                self.target = min(gated, key=lambda xy: dist(xy, self.target)).copy()
                self.misses = 0
            else:
                self.misses += 1
                if eligible and self.misses > p.coast_frames:
                    self.target = self._closest_to_centre(eligible).copy()
                    self.misses = 0

        if dist(eef_xy, self.target) < p.pregrasp_radius_m:
            self.latched, self.moved = True, 0
        return self.target.copy()


def fuse(estimates, variances):
    """Inverse-variance weighted mean of {camera: point}."""
    if not estimates:
        return None
    w = {c: 1.0 / variances[c] for c in estimates}
    return sum(w[c] * np.asarray(e, float) for c, e in estimates.items()) / sum(w.values())


def gate(candidates, track, gate_m=GATE_M, conf=CONF):
    """Nearest confident candidate to the track (within gate_m); the most confident one if there is no track."""
    ok = [(p, c) for p, c in candidates if c >= conf]
    if track is None:
        return max(ok, key=lambda x: x[1])[0] if ok else None
    near = [(float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(track)[:2])), p) for p, _ in ok]
    near = [x for x in near if x[0] <= gate_m]
    return min(near, key=lambda x: x[0])[1] if near else None


class Perception:
    def __init__(self, calib, variances, selector_params):
        """variances: {(role, camera, phase): m^2}, measured on held-out renders."""
        self.calib, self.var, self.sp = calib, variances, selector_params
        self.reset()

    def reset(self):
        self.selector = BowlSelector(self.sp)
        self.plate_track = None

    def update(self, peaks, eef_pos, eef_quat_xyzw, phase, gripper_open):
        points = {r: {} for r in ROLES}
        for cam in CAMERAS:
            for role in ROLES:
                pts = []
                for u, v, c in peaks.get(cam, {}).get(CLASS_OF[role], []):
                    p = self.calib.backproject(cam, (u, v), role, eef_pos, eef_quat_xyzw)
                    if p is not None:
                        pts.append((p, c))
                points[role][cam] = pts

        bowls = [(p[:2], c) for cam in CAMERAS for p, c in points['bowl_1'][cam]]
        target = self.selector.update(bowls, np.asarray(eef_pos)[:2], gripper_open, phase)
        track = {'bowl_1': None if target is None else np.r_[target, self.calib.planes['bowl_1']],
                 'plate': self.plate_track}

        out = {}
        for role in ROLES:
            if role == 'bowl_1' and track['bowl_1'] is None:
                out[role] = None
                continue
            chosen = {}
            for cam in CAMERAS:
                g = gate(points[role][cam], track[role])
                if g is not None:
                    chosen[cam] = g
            out[role] = fuse(chosen, {cam: self.var[(role, cam, int(phase))] for cam in CAMERAS})
        if out['plate'] is not None:
            self.plate_track = out['plate']
        return out
