"""bench/ F_A determinism and the full-resolution guard (PR #1 Bugbot review)."""
import numpy as np
import pytest
import torch

from bench.metrics import backproject, f1_3d
from bench.refmodel import _check_full_res


def _scene(h=200, w=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = 1.0 + 4.0 * torch.rand(h, w, generator=g)
    K = torch.tensor([[250.0, 0, w / 2], [0, 250.0, h / 2], [0, 0, 1]])
    return depth, K


def test_f1_3d_subsample_is_deterministic_and_model_independent():
    depth, K = _scene()
    gt = backproject(depth, K)
    mask = torch.ones_like(depth, dtype=torch.bool)
    noisy = backproject(depth * 1.03, K)
    a = f1_3d(gt, noisy, mask, max_depth=10.0, max_points=5000)
    b = f1_3d(gt, noisy, mask, max_depth=10.0, max_points=5000)
    assert a == b                                              # unseeded randperm made these differ
    # a perfect prediction scores the same subsampled as on every pixel (the AUC ceiling of this
    # protocol), i.e. the subset is taken from the GT mask and pairs GT/pred pixels one to one
    full = f1_3d(gt, gt.clone(), mask, max_depth=10.0, max_points=10 ** 9)
    assert f1_3d(gt, gt.clone(), mask, max_depth=10.0, max_points=5000) == pytest.approx(full)
    c = f1_3d(gt, noisy, mask, max_depth=10.0, max_points=5000, seed=1)
    assert np.isfinite(c)


def test_f1_3d_without_subsampling_ignores_seed():
    depth, K = _scene(40, 50)
    gt, pr = backproject(depth, K), backproject(depth * 1.05, K)
    mask = torch.ones_like(depth, dtype=torch.bool)
    assert f1_3d(gt, pr, mask, 10.0, seed=0) == f1_3d(gt, pr, mask, 10.0, seed=7)


def test_full_res_guard():
    _check_full_res((4, 6), depth=np.zeros((4, 6)), rays=np.zeros((3, 4, 6)), conf=None)
    with pytest.raises(ValueError, match="rays"):
        _check_full_res((4, 6), depth=np.zeros((4, 6)), rays=np.zeros((3, 2, 3)))
