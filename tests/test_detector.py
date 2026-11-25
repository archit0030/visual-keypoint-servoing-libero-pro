import numpy as np
import torch

from vks.detector import KNet, decode_peaks, encode_targets


def test_parameter_count_and_shapes():
    net = KNet().eval()
    assert sum(p.numel() for p in net.parameters()) == 1_185_190
    with torch.no_grad():
        y = net(torch.rand(2, 3, 128, 128))
    assert y['heat'].shape == (2, 2, 64, 64)
    assert y['offset'].shape == (2, 2, 64, 64)
    assert y['vis'].shape == (2, 2)


def test_encode_then_decode_recovers_keypoints():
    labels = {0: [(30.3, 70.9), (90.0, 20.5)], 1: [(64.7, 64.2)]}
    heat, off, _ = encode_targets(labels)
    peaks = decode_peaks(heat, off)
    bowls = sorted((round(u, 3), round(v, 3)) for u, v, _ in peaks[0])
    assert bowls == sorted(labels[0])
    (u, v, c), = peaks[1]
    assert np.allclose((u, v), labels[1][0]) and c == 1.0


def test_low_confidence_is_ignored():
    heat = np.zeros((2, 64, 64), np.float32)
    heat[0, 10, 10] = 0.4
    assert decode_peaks(heat, np.zeros((2, 64, 64)))[0] == []
