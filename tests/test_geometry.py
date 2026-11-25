import numpy as np

from vks.geometry import backproject, mat_to_quat_wxyz, quat_xyzw_to_mat


def look_down(height=1.0):
    # camera at (0, 0, height) looking straight down (-z)
    return np.array([0.0, 0.0, height]), np.eye(3).ravel()


def test_centre_pixel_hits_point_below_camera():
    pos, xmat = look_down()
    p = backproject((64, 64), pos, xmat, 45.0, plane_z=0.2)
    assert np.allclose(p, [0.0, 0.0, 0.2])


def test_offset_pixel_matches_pinhole_model():
    pos, xmat = look_down()
    fovy = 60.0
    f = 64 / np.tan(np.deg2rad(fovy) / 2)
    p = backproject((64 + 10, 64), pos, xmat, fovy, plane_z=0.0)
    assert np.isclose(p[0], 10 / f * 1.0) and np.isclose(p[1], 0.0)


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = quat_xyzw_to_mat(q)
        w = mat_to_quat_wxyz(r)
        back = quat_xyzw_to_mat(np.r_[w[1:], w[0]])
        assert np.allclose(r, back)
