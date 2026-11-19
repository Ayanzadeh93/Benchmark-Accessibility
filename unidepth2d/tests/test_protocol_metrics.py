"""Unit tests for benchmark/protocol.py: the metric definitions every result table rests on."""
import numpy as np
import pytest

from benchmark import protocol


def _scene(seed=0, h=48, w=64):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.5, 8.0, size=(h, w))
    valid = np.ones((h, w), bool)
    return gt, valid, rng


def test_perfect_prediction_scores_perfectly():
    gt, valid, _ = _scene()
    r = protocol.per_image_depth_metrics(gt, gt.copy(), valid)
    assert r["d1"] == 1.0 and r["d1_ssi"] == 1.0
    assert r["arel"] == pytest.approx(0, abs=1e-9)
    assert r["rmse"] == pytest.approx(0, abs=1e-9)


def test_global_scale_error_hits_raw_but_not_aligned_metrics():
    """The project's central diagnosis: a pure scale error ruins delta1, not delta1_ssi."""
    gt, valid, _ = _scene(1)
    r = protocol.per_image_depth_metrics(gt, gt * 1.5, valid)
    assert r["d1"] == 0.0                       # ratio 1.5 > 1.25 everywhere
    assert r["d1_ssi"] == pytest.approx(1.0)    # scale+shift alignment undoes it
    assert r["d1_si"] == pytest.approx(1.0)
    assert r["arel"] == pytest.approx(0.5, abs=1e-9)
    assert r["silog"] == pytest.approx(0, abs=1e-6)   # SILog ignores a global scale


def test_delta1_threshold_boundary():
    gt = np.full((4, 4), 2.0)
    valid = np.ones_like(gt, bool)
    inside = protocol.per_image_depth_metrics(gt, gt * 1.2499, valid)
    outside = protocol.per_image_depth_metrics(gt, gt * 1.2501, valid)
    assert inside["d1"] == 1.0 and outside["d1"] == 0.0
    assert outside["d2"] == 1.0


def test_invalid_pixels_are_ignored():
    gt, valid, _ = _scene(2)
    pred = gt.copy()
    valid[:10] = False
    pred[:10] = 1e3                              # garbage where GT is invalid
    r = protocol.per_image_depth_metrics(gt, pred, valid)
    assert r["d1"] == 1.0 and r["arel"] == pytest.approx(0, abs=1e-9)


def test_max_depth_masks_ground_truth_only():
    gt = np.array([[1.0, 2.0, 20.0, 30.0]])
    valid = np.ones_like(gt, bool)
    pred = np.array([[1.0, 2.0, 1.0, 1.0]])     # wrong only beyond the cap
    r = protocol.per_image_depth_metrics(gt, pred, valid, max_depth=10.0)
    assert r["d1"] == 1.0


def test_no_valid_pixels_returns_none():
    gt, valid, _ = _scene(3)
    assert protocol.per_image_depth_metrics(gt, gt, np.zeros_like(valid)) is None


def test_nonpositive_predictions_are_counted_not_crashing():
    gt, valid, _ = _scene(4)
    pred = gt.copy()
    pred[0, :5] = -1.0
    r = protocol.per_image_depth_metrics(gt, pred, valid)
    assert r["n_nonpos_pred"] == 5.0
    assert all(np.isfinite(v) for v in r.values())


def test_aggregate_is_unweighted_mean_and_skips_nan():
    rows = [{"d1": 1.0, "x": float("nan")}, {"d1": 0.0, "x": 2.0}]
    a = protocol.aggregate(rows)
    assert a["d1"] == 0.5 and a["x"] == 2.0
    assert protocol.aggregate([]) == {}


def test_ssi_align_recovers_affine_transform():
    rng = np.random.default_rng(5)
    g = rng.uniform(1, 5, 500)
    aligned = protocol.ssi_align(g, (g - 0.3) / 2.0)
    assert np.allclose(aligned, g, atol=1e-5)


def test_boundary_f1_perfect_scale_invariant_and_blur_sensitive():
    gt = np.ones((40, 40))
    gt[:, 20:] = 3.0                             # one sharp occluding edge
    valid = np.ones_like(gt, bool)
    assert protocol.boundary_f1_si(gt, gt.copy(), valid) == pytest.approx(1.0)
    assert protocol.boundary_f1_si(gt, gt * 2.7, valid) == pytest.approx(1.0)   # scale-invariant
    blurred = gt.copy()
    for c in range(12, 28):                      # linear ramp instead of a step
        blurred[:, c] = 1.0 + 2.0 * (c - 12) / 15.0
    f_blur = protocol.boundary_f1_si(gt, blurred, valid)
    assert f_blur is None or f_blur < 0.9
    assert protocol.boundary_f1_si(gt, np.ones_like(gt), valid) is None        # no predicted edge
