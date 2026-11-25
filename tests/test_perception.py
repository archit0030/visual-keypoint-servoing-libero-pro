import numpy as np

from vks.perception import BowlSelector, fuse, gate


def test_fuse_weights_by_inverse_variance():
    est = {'a': np.zeros(3), 'b': np.ones(3)}
    out = fuse(est, {'a': 1.0, 'b': 3.0})
    assert np.allclose(out, 0.25)


def test_gate_picks_nearest_inside_radius():
    track = np.array([0.0, 0.0, 0.0])
    cands = [(np.array([0.03, 0, 0]), 0.9), (np.array([0.01, 0, 0]), 0.6), (np.array([0.2, 0, 0]), 0.99)]
    assert np.allclose(gate(cands, track), [0.01, 0, 0])
    assert gate([(np.array([0.1, 0, 0]), 0.9)], track) is None


def test_selector_prefers_bowl_near_table_centre_and_keeps_it():
    sel = BowlSelector()
    centre_bowl, other_bowl = np.array([-0.075, 0.0]), np.array([0.05, 0.15])
    t = sel.update([(other_bowl, 0.9), (centre_bowl, 0.9)], eef_xy=[0.3, 0.3], gripper_open=True, phase=0)
    assert np.allclose(t, centre_bowl)
    # the other bowl alone does not steal the target while it is being tracked
    t = sel.update([(other_bowl, 0.9)], eef_xy=[0.3, 0.3], gripper_open=True, phase=0)
    assert np.allclose(t, centre_bowl)
