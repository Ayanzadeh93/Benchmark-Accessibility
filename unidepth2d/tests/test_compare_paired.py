"""Unit tests for benchmark/compare_paired.py (paired bootstrap on per-image metrics)."""
import numpy as np
import pytest

from benchmark.compare_paired import compare_rows, paired_bootstrap


def test_identical_inputs_give_zero_difference_and_no_significance():
    a = np.random.default_rng(0).uniform(0, 1, 100)
    r = paired_bootstrap(a, a.copy(), n_boot=2000)
    assert r["diff"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0
    assert r["significant"] is False


def test_clear_shift_is_detected_with_correct_sign():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.5, 0.7, 100)
    b = a + 0.05 + rng.normal(0, 0.005, 100)
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["significant"] is True
    assert r["ci_lo"] > 0 and r["diff"] == pytest.approx(0.05, abs=0.005)


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.5, 0.7, 100)
    b = a + rng.normal(0, 0.05, 100)
    r = paired_bootstrap(a, b, n_boot=4000)
    assert r["ci_lo"] < 0 < r["ci_hi"]
    assert r["significant"] is False


def test_pairing_matters():
    """A small consistent gain is detectable paired even when image-to-image spread is large."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0.2, 0.9, 100)        # large between-image spread
    b = a + 0.01                          # tiny but perfectly consistent gain
    assert paired_bootstrap(a, b, n_boot=2000)["significant"] is True


def test_nan_pairs_are_dropped():
    a = np.array([0.5, np.nan, 0.7, 0.6])
    b = np.array([0.6, 0.9, np.nan, 0.7])
    r = paired_bootstrap(a, b, n_boot=500)
    assert r["n"] == 2 and r["diff"] == pytest.approx(0.1)


def test_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        paired_bootstrap([np.nan], [np.nan])


def test_deterministic_given_seed():
    rng = np.random.default_rng(4)
    a, b = rng.uniform(0, 1, 50), rng.uniform(0, 1, 50)
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_compare_rows_direction_for_lower_is_better_metrics():
    rows_a = [{"d1": 0.6, "arel": 0.20} for _ in range(60)]
    rows_b = [{"d1": 0.7, "arel": 0.15} for _ in range(60)]
    res = compare_rows(rows_a, rows_b, ["d1", "arel", "missing"], n_boot=500)
    assert set(res) == {"d1", "arel"}
    assert res["d1"]["verdict"] == "B better" and res["arel"]["verdict"] == "B better"
    assert res["arel"]["lower_is_better"] is True
    assert res["arel"]["frac_images_b_better"] == 1.0
    worse = compare_rows(rows_b, rows_a, ["arel"], n_boot=500)
    assert worse["arel"]["verdict"] == "A better"


def test_compare_rows_rejects_different_image_counts():
    with pytest.raises(ValueError):
        compare_rows([{"d1": 1.0}], [{"d1": 1.0}, {"d1": 0.5}], ["d1"])
