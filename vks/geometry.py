"""Camera geometry: pixel -> 3D point on a known plane, for both cameras.

The agentview camera is fixed. The wrist camera pose is computed from the
robot's end-effector observation (robot0_eef_pos / robot0_eef_quat) and a fixed
hand -> camera transform, so no simulator access is needed at run time.
"""
from dataclasses import dataclass
import numpy as np

SIZE = 128
TABLE_CENTER_REGION = ((-0.10, -0.05), (-0.01, 0.01))   # where the source bowl starts (x range, y range)


def quat_xyzw_to_mat(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_wxyz(r):
    r = np.asarray(r, float).reshape(3, 3)
    tr = np.trace(r)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = np.empty(4)
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def distance_to_region(xy):
    (x0, x1), (y0, y1) = TABLE_CENTER_REGION
    dx = max(x0 - xy[0], 0.0, xy[0] - x1)
    dy = max(y0 - xy[1], 0.0, xy[1] - y1)
    return float(np.hypot(dx, dy))


def backproject(uv, cam_pos, cam_xmat, fovy, plane_z):
    """Pixel (u, v) in image coordinates -> world point on the plane z = plane_z."""
    u, v = float(uv[0]), SIZE - float(uv[1])
    focal = 0.5 * SIZE / np.tan(np.deg2rad(fovy) / 2)
    ray = np.asarray(cam_xmat, float).reshape(3, 3) @ np.array([(u - SIZE / 2) / focal, -(v - SIZE / 2) / focal, -1.0])
    if abs(ray[2]) < 1e-12:
        raise ValueError('ray parallel to plane')
    scale = (plane_z - cam_pos[2]) / ray[2]
    if scale <= 0:
        raise ValueError('plane behind camera')
    return np.asarray(cam_pos, float) + scale * ray


@dataclass
class Calibration:
    agent_pos: np.ndarray
    agent_xmat: np.ndarray
    agent_fovy: float
    wrist_fovy: float
    wrist_tf: dict            # wrist camera pose in the end-effector frame {'pos', 'rot'}
    hand_to_site: np.ndarray  # rotation from the hand body to the grip site
    planes: dict              # role -> z of the keypoint plane (bowl rim / plate rim height)

    def camera(self, cam, eef_pos, eef_quat_xyzw):
        if cam == 'agentview':
            return self.agent_pos, self.agent_xmat, self.agent_fovy
        r = quat_xyzw_to_mat(eef_quat_xyzw)
        pos = np.asarray(eef_pos, float) + r @ self.wrist_tf['pos']
        return pos, (r @ self.wrist_tf['rot']).ravel(), self.wrist_fovy

    def backproject(self, cam, uv, role, eef_pos, eef_quat_xyzw):
        pos, xmat, fovy = self.camera(cam, eef_pos, eef_quat_xyzw)
        try:
            return backproject(uv, pos, xmat, fovy, self.planes[role])
        except ValueError:
            return None

    @classmethod
    def from_json(cls, d):
        return cls(agent_pos=np.array(d['agent_pos']), agent_xmat=np.array(d['agent_xmat']),
                   agent_fovy=d['agent_fovy'], wrist_fovy=d['wrist_fovy'],
                   wrist_tf=dict(pos=np.array(d['wrist_tf']['pos']), rot=np.array(d['wrist_tf']['rot'])),
                   hand_to_site=np.array(d['hand_to_site']), planes=dict(d['planes']))
