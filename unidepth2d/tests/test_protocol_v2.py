"""Tests for benchmark/protocol_v2.py. Each test shows how a V2 definition differs from the
legacy one it sits next to (benchmark/protocol.py, compare_paired.py, scale_gap.py).

Env: depth or depth_cmp (numpy + scipy only). CPU-only, offline, synthetic data.
"""
import math

import numpy as np
import pytest

from benchmark import protocol, protocol_v2 as p2
from benchmark.compare_paired import compare_rows, paired_bootstrap


def _step(h=40, w=40, near=1.0, far=3.0, col=20):
    gt = np.full((h, w), near)
    gt[:, col:] = far
    return gt, np.ones((h, w), bool)


# ── boundary F1 ──────────────────────────────────────────────────────────────

def test_bf_matches_legacy_when_no_threshold_is_degenerate():
    gt, valid = _step()                              # ratio 3.0: an edge at every t
    for pred in (gt.copy(), gt * 2.7):
        assert p2.boundary_f1_si_v2(gt, pred, valid) == pytest.approx(1.0)
        assert protocol.boundary_f1_si(gt, pred, valid) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noisy = gt * rng.uniform(0.97, 1.03, gt.shape)
    assert p2.boundary_f1_si_v2(gt, noisy, valid) == pytest.approx(
        protocol.boundary_f1_si(gt, noisy, valid))  # noise edges fire at t=5 only... both agree


def test_bf_edge_that_fires_only_at_5pct_scores_point2_not_1():
    """Audit case: the legacy metric skips thresholds the prediction never reaches."""
    gt, valid = _step(far=3.0)
    weak = gt.copy()
    weak[:, 20:] = 1.07                              # predicted step of 7 %: an edge only at t=5
    assert protocol.boundary_f1_si(gt, weak, valid) == pytest.approx(1.0)      # legacy
    det = p2.boundary_f1_si_v2(gt, weak, valid, return_detail=True)
    assert det["bf1"] == pytest.approx(0.2)                                  # 1 of 5 thresholds
    assert det["n_t_zero_scored"] == 4


def test_bf_wrong_place_and_missing_edges_score_zero_instead_of_none():
    gt, valid = _step(col=10)
    wrong, _ = _step(col=30)                         # strong edge in the wrong place
    flat = np.ones_like(gt)                          # no predicted edge at all
    assert protocol.boundary_f1_si(gt, wrong, valid) is None
    assert protocol.boundary_f1_si(gt, flat, valid) is None
    assert p2.boundary_f1_si_v2(gt, wrong, valid) == 0.0
    assert p2.boundary_f1_si_v2(gt, flat, valid) == 0.0


def test_bf_blur_never_scores_higher_than_legacy():
    gt, valid = _step()
    blurred = gt.copy()
    for c in range(12, 28):
        blurred[:, c] = 1.0 + 2.0 * (c - 12) / 15.0
    legacy = protocol.boundary_f1_si(gt, blurred, valid)
    v2 = p2.boundary_f1_si_v2(gt, blurred, valid)
    assert v2 is not None and v2 < 0.9
    assert legacy is None or v2 <= legacy + 1e-12


def test_bf_image_without_gt_boundary_is_non_contributing_and_counted():
    flat_gt = np.full((30, 30), 2.0)
    valid = np.ones_like(flat_gt, bool)
    pred_with_edge, _ = _step(30, 30)
    assert p2.boundary_f1_si_v2(flat_gt, pred_with_edge, valid) is None
    gt, v = _step()
    vals = [p2.boundary_f1_si_v2(gt, gt, v), p2.boundary_f1_si_v2(gt, np.ones_like(gt), v),
            p2.boundary_f1_si_v2(flat_gt, flat_gt, valid)]
    agg = p2.aggregate_boundary_f1(vals)
    assert agg == {"bf1": 0.5, "n_images": 3, "n_contributing": 2, "n_no_gt_boundary": 1}
    # legacy dataset mean (run_bench) drops the None image AND the missed-edge image
    legacy = [protocol.boundary_f1_si(gt, gt, v), protocol.boundary_f1_si(gt, np.ones_like(gt), v)]
    assert np.mean([x for x in legacy if x is not None]) == 1.0


def test_bf_both_empty_policy():
    gt, valid = _step(far=1.12)                      # GT edge only at t=5,10
    pred = gt.copy()
    zero = p2.boundary_f1_si_v2(gt, pred, valid, both_empty="zero", return_detail=True)
    excl = p2.boundary_f1_si_v2(gt, pred, valid, both_empty="exclude", return_detail=True)
    assert zero["n_t_both_empty"] == 3 and zero["bf1"] == pytest.approx(0.4)
    assert excl["n_t"] == 2 and excl["bf1"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        p2.boundary_f1_si_v2(gt, pred, valid, both_empty="one")


def test_bf_invalid_pixels_and_no_pairs():
    gt, valid = _step()
    assert p2.boundary_f1_si_v2(gt, gt, np.zeros_like(valid)) is None
    v = valid.copy()
    v[:, 19:21] = False                              # mask the edge itself: no GT boundary pair
    assert p2.boundary_f1_si_v2(gt, gt, v) is None


def _box(h=40, w=40, inside=1.0, outside=3.0):
    d = np.full((h, w), outside)
    d[10:30, 10:30] = inside                         # near square: edges in all four directions
    return d


def test_depthpro_reference_on_box_and_threshold_weighting():
    gt = _box()
    assert p2.si_boundary_f1_depthpro(gt, gt) == pytest.approx(1.0)
    weak = _box(outside=1.12)                        # predicted step of 12 %
    ts = np.linspace(1.05, 1.25, 10)
    expected = ts[ts < 1.12].sum() / ts.sum()        # 4 of 10 thresholds, weighted by t
    assert p2.si_boundary_f1_depthpro(weak, gt) == pytest.approx(expected)       # 0.3768
    valid = np.ones_like(gt, bool)
    assert p2.boundary_f1_si_v2(gt, weak, valid) == pytest.approx(0.4)          # 2 of 5
    assert protocol.boundary_f1_si(gt, weak, valid) == pytest.approx(1.0)       # legacy


def test_depthpro_averages_directions_where_inhouse_pools_them():
    """A single straight step has GT edges in only one of DepthPro's four directional maps:
    DepthPro scores a perfect prediction 0.25, the pooled in-house metrics score 1.0."""
    gt, valid = _step()
    assert p2.si_boundary_f1_depthpro(gt, gt) == pytest.approx(0.25)
    assert p2.boundary_f1_si_v2(gt, gt, valid) == pytest.approx(1.0)
    assert protocol.boundary_f1_si(gt, gt, valid) == pytest.approx(1.0)


def test_depthpro_valid_mask_extension():
    gt = _box()
    gt_holes = gt.copy()
    gt_holes[0:5, :] = 0.0                           # invalid GT rows (sensor holes)
    v = gt_holes > 0
    # unmasked DepthPro sees the holes as huge inverse-depth edges the prediction lacks
    assert p2.si_boundary_f1_depthpro(gt, gt_holes) < 0.99
    assert p2.si_boundary_f1_depthpro(gt, gt_holes, valid=v) == pytest.approx(1.0)
    assert p2.si_boundary_f1_depthpro(gt, gt, valid=np.zeros_like(v)) == 0.0


# ── bias / b_std / dataset scale ────────────────────────────────────────────

def test_dataset_scale_separates_global_from_per_image_scale_error():
    rng = np.random.default_rng(1)
    gts = [rng.uniform(0.5, 8.0, (20, 20)) for _ in range(6)]
    valid = np.ones((20, 20), bool)
    # (a) the same 1.5x over-estimate on every image: one dataset scale fixes it
    preds = [g * 1.5 for g in gts]
    rs = [p2._valid_log_ratio(g, p, valid) for g, p in zip(gts, preds)]
    res = p2.d1_dataset_scale(rs)
    assert res["d1_raw"] == 0.0 and res["d1_dataset_scale"] == pytest.approx(1.0)
    b = [p2.per_image_log_shift(g, p, valid) for g, p in zip(gts, preds)]
    s = p2.dataset_scale_summary(b)
    assert s["b_mean"] == pytest.approx(-math.log(1.5)) and s["b_std"] == pytest.approx(0, abs=1e-12)
    # (b) per-image scales 0.6x .. 1.6x: per-image alignment (d1_si) fixes it, one scale cannot
    scales = np.linspace(0.6, 1.6, 6)
    preds = [g * k for g, k in zip(gts, scales)]
    rs = [p2._valid_log_ratio(g, p, valid) for g, p in zip(gts, preds)]
    res = p2.d1_dataset_scale(rs)
    d1_si = np.mean([protocol.per_image_depth_metrics(g, p, valid)["d1_si"]
                     for g, p in zip(gts, preds)])
    assert d1_si == pytest.approx(1.0) and res["d1_dataset_scale"] < 0.75
    s = p2.dataset_scale_summary([p2.per_image_log_shift(g, p, valid) for g, p in zip(gts, preds)])
    assert s["b_std"] == pytest.approx(np.std(-np.log(scales)))


def test_dataset_scale_full_pixels_vs_scale_gap_subsample_rule():
    """scale_gap.py applies the global shift to a 2,000-pixel subsample; V2 uses all pixels."""
    rng = np.random.default_rng(2)
    g = rng.uniform(0.5, 8.0, 40000)
    p = g * np.exp(rng.normal(0.0, 0.2, g.size))
    r = np.log(g) - np.log(p)
    full = p2.d1_dataset_scale([r])["d1_dataset_scale"]
    sel = np.random.default_rng(0).choice(g.size, 2000, replace=False)
    sub = p2.d1_dataset_scale([r[sel]], b=float(r.mean()))["d1_dataset_scale"]
    assert abs(full - sub) < 0.05 and full != sub


def test_log_ratio_store_cap_is_deterministic():
    st = p2.LogRatioStore(max_pixels=100)
    g = np.linspace(1, 5, 400).reshape(20, 20)
    st.add("a", g, g * 2, np.ones_like(g, bool))
    st.add("b", g, g * 2, np.ones_like(g, bool))
    assert st.n_full == [400, 400] and all(r.size == 100 for r in st.r)
    assert np.array_equal(st.r[0], st.r[1])
    assert st.add("c", g, g, np.zeros_like(g, bool)) is None


# ── pairing by id ────────────────────────────────────────────────────────────

def test_pair_by_id_catches_what_order_pairing_misses():
    a = [{"id": f"img{i}", "d1": float(i)} for i in range(5) if i != 3]
    b = [{"id": f"img{i}", "d1": float(i)} for i in range(5) if i != 1]
    res = compare_rows(a, b, ["d1"], n_boot=200)       # legacy: same count, silently mispaired
    assert res["d1"]["diff"] == pytest.approx(0.5)     # identical models look different
    with pytest.raises(ValueError, match="id sets differ"):
        p2.pair_rows_by_id(a, b)
    ids, ra, rb, info = p2.pair_rows_by_id(a, b, allow_intersection=True)
    assert ids == ["img0", "img2", "img4"] and info["only_a"] == ["img1"]
    assert [r["d1"] for r in ra] == [r["d1"] for r in rb]


def test_pair_by_id_rejects_duplicates_and_missing_ids():
    with pytest.raises(ValueError, match="duplicate"):
        p2.pair_rows_by_id([{"id": 1}, {"id": 1}], [{"id": 1}])
    with pytest.raises(ValueError, match="no 'id'"):
        p2.pair_rows_by_id([{"d1": 1.0}], [{"id": 1}])


# ── clustered bootstrap ──────────────────────────────────────────────────────

def test_clustered_bootstrap_delegates_without_clusters():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 50), rng.normal(0.2, 1, 50)
    ref = paired_bootstrap(a, b, n_boot=500, seed=1)
    got = p2.paired_bootstrap_clustered(a, b, None, n_boot=500, seed=1)
    assert got["ci_lo"] == ref["ci_lo"] and got["ci_hi"] == ref["ci_hi"]
    assert got["resampling"] == "image"


def test_clustered_bootstrap_widens_ci_for_near_duplicates():
    rng = np.random.default_rng(4)
    base_d = rng.normal(0.1, 1.0, 20)
    d = np.repeat(base_d, 8)                         # every scene appears 8 times
    clusters = np.repeat(np.arange(20), 8)
    a = np.zeros_like(d)
    naive = p2.paired_bootstrap_clustered(a, d, None, n_boot=2000, seed=0)
    clus = p2.paired_bootstrap_clustered(a, d, clusters, n_boot=2000, seed=0)
    assert clus["n_clusters"] == 20 and clus["diff"] == pytest.approx(naive["diff"])
    assert (clus["ci_hi"] - clus["ci_lo"]) > 2.0 * (naive["ci_hi"] - naive["ci_lo"])
    with pytest.raises(ValueError):
        p2.paired_bootstrap_clustered(a, d, clusters[:-1])
