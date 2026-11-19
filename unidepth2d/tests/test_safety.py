"""Tests for benchmark/safety.py (SF1-SF6, I1-I4/I6, sparse anchors, config, CLI guards).

Env: depth or depth_cmp (numpy, opencv-python, pyyaml). CPU-only, offline, synthetic data only.
"""
import json
import math
import os
import shutil

import numpy as np
import pytest

from benchmark import safety as sf
from benchmark.ocmr_conformal import _fnr_controlled_alert


@pytest.fixture
def cfg():
    return sf.default_config()


# ── config and labels ────────────────────────────────────────────────────────

def test_default_thresholds_are_the_plan_values(cfg):
    assert cfg["owner_confirmed"] is False
    assert cfg["sf1"]["near_m"] == 2.0 and cfg["sf1"]["over_factor"] == 1.25
    assert cfg["sf1"]["max_rate"] == 0.10 and cfg["sf1"]["max_int8_rise_pts"] == 0.5
    assert cfg["sf2"]["bands_m"] == [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]
    assert cfg["sf2"]["max_worse_than_ref_pts"] == 1.0
    s3 = cfg["sf3"]
    assert s3["roi_x"] == [0.25, 0.75] and s3["roi_y"] == [0.35, 1.0]
    assert s3["thresholds_m"] == [1.0, 2.0] and s3["target_miss"] == 0.05
    assert s3["max_test_miss"] == 0.06 and s3["max_false_alarm"] == 0.30
    assert cfg["sf4"] == {"max_miss_rise_pts": 1.0, "max_false_alarm_rise_pts": 3.0}
    assert cfg["sf5"]["max_d1_drop_pts"] == 8.0 and cfg["sf5"]["jpeg_quality"] == 40
    assert cfg["sf5"]["roll_deg"] == 15.0
    assert cfg["sf6"]["target_fps"] == 5.0 and cfg["sf6"]["max_p95_abs_change"] == 0.05
    i = cfg["int8"]
    assert i["i1_min_frac"] == 0.99 and i["i2_max_abs_dlog"] == pytest.approx(math.log(1.10))
    assert i["i2_near_m"] == 2.0 and i["i3_max_abs_mean_shift"] == 0.01
    assert (i["i4_max_d1_drop_pts"], i["i4_max_d1_drop_ub95_pts"]) == (1.0, 1.5)
    assert (i["i4_max_arel_rise"], i["i4_max_bf_rel_drop"]) == (0.005, 0.10)
    assert cfg["anchor"]["n_points"] == 150


def test_config_file_and_unknown_keys(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("sf1:\n  max_rate: 0.05\n", encoding="utf-8")
    c = sf.load_config(str(p))
    assert c["sf1"]["max_rate"] == 0.05 and c["sf1"]["near_m"] == 2.0
    j = tmp_path / "s.json"
    j.write_text(json.dumps({"sf3": {"max_false_alarm": 0.2}}), encoding="utf-8")
    assert sf.load_config(str(j))["sf3"]["max_false_alarm"] == 0.2
    with pytest.raises(ValueError, match="unknown safety config key 'sf1.max_rat'"):
        sf.load_config(overrides={"sf1": {"max_rat": 0.05}})
    with pytest.raises(ValueError, match="must be a mapping"):
        sf.load_config(overrides={"sf1": 3})
    assert sf.config_sha256(sf.default_config()) == sf.config_sha256(sf.default_config())
    assert sf.config_sha256(c) != sf.config_sha256(sf.default_config())


def test_research_only_until_owner_confirms_with_a_note(cfg):
    assert sf.claim_label(cfg) == sf.RESEARCH_ONLY_LABEL
    flip = sf.load_config(overrides={"owner_confirmed": True})
    assert sf.claim_label(flip) == sf.RESEARCH_ONLY_LABEL          # a bare flag is not enough
    ok = sf.load_config(overrides={"owner_confirmed": True,
                                   "owner_confirmation": "owner, 2027-01-05, SF1-SF6 as listed"})
    assert sf.claim_label(ok).startswith("owner-confirmed")
    g = sf.make_gate("SF1", "m", 0.01, 0.1, "<=", cfg)
    assert g["label"] == sf.RESEARCH_ONLY_LABEL and g["status"] == "pass"


def test_make_gate_status_rules(cfg):
    assert sf.make_gate("X", "m", 0.2, 0.1, "<=", cfg)["status"] == "fail"
    assert sf.make_gate("X", "m", 0.995, 0.99, ">=", cfg)["status"] == "pass"
    assert sf.make_gate("X", "m", -0.02, 0.01, "abs<=", cfg)["status"] == "fail"
    assert sf.make_gate("X", "m", None, 0.1, "<=", cfg)["status"] == "not_evaluated"
    assert sf.make_gate("X", "m", float("nan"), 0.1, "<=", cfg)["status"] == "not_evaluated"
    assert sf.make_gate("X", "m", float("inf"), 0.1, "<=", cfg)["status"] == "fail"
    assert sf.make_gate("X", "m", 0.3, None, "<=", cfg)["status"] == "report_only"
    with pytest.raises(ValueError):
        sf.make_gate("X", "m", 0.3, 0.1, "<", cfg)


# ── SF1 / SF2 / near-field bias ──────────────────────────────────────────────

def _two_depth(h=40, w=50, near=1.0, far=3.0):
    gt = np.full((h, w), far)
    gt[:, : w // 2] = near
    return gt, np.ones((h, w), bool)


def test_sf1_synthetic_over_estimation(cfg):
    gt, valid = _two_depth()
    near_idx = np.flatnonzero(gt.ravel() < 2.0)                    # 1,000 near pixels
    pred = gt.copy()
    pred.flat[near_idx[:200]] *= 1.3                               # 20 % over by > 1.25x
    s = sf.summarize_near_field([sf.near_field_counts(gt, pred, valid, cfg)], cfg)
    assert s["sf1"]["rate"] == pytest.approx(0.2) and s["sf1"]["n_near_px"] == 1000
    assert sf.gate_sf1(s, cfg)[0]["status"] == "fail"
    pred2 = gt.copy()
    pred2.flat[near_idx[:200]] *= 1.2                              # within 1.25x: not dangerous
    pred2[:, 25:] *= 2.0                                           # far errors never count for SF1
    s2 = sf.summarize_near_field([sf.near_field_counts(gt, pred2, valid, cfg)], cfg)
    assert s2["sf1"]["rate"] == 0.0 and sf.gate_sf1(s2, cfg)[0]["status"] == "pass"
    # int8 rise vs fp32: 20 % - 0 % = 20 points > 0.5
    g = sf.gate_sf1(s, cfg, fp32_summary=s2, tag="[int8]")
    assert g[1]["metric"].startswith("int8_rise_pts") and g[1]["value"] == pytest.approx(20.0)
    assert g[1]["status"] == "fail"


def test_nonfinite_predictions_count_as_far(cfg):
    gt, valid = _two_depth()
    pred = gt.copy()
    pred[0, 0] = np.nan                                            # a near pixel
    pred[1, 0] = np.inf
    c = sf.near_field_counts(gt, pred, valid, cfg)
    assert c["n_nonfinite_pred"] == 2 and c["sf1_n_over"] == 2
    assert c["band_1-2_miss"] == 2 and c["below_2_miss"] == 2
    assert c["nb_n"] == 998                                        # bias uses finite pixels only


def test_sf2_band_and_below_miss_rates(cfg):
    gt = np.concatenate([np.full(100, 0.5), np.full(100, 1.5), np.full(100, 3.0)]).reshape(15, 20)
    pred = gt.copy()
    pred.flat[:10] = 1.0               # band 0-1: predicted at the band edge -> missed (>= hi)
    pred.flat[100:150] = 2.5           # band 1-2: half missed
    v = np.ones_like(gt, bool)
    s = sf.summarize_near_field([sf.near_field_counts(gt, pred, v, cfg)], cfg)["sf2"]
    assert s["band_0-1"]["miss_rate"] == pytest.approx(0.10)
    assert s["band_1-2"]["miss_rate"] == pytest.approx(0.50)
    assert s["band_2-4"]["miss_rate"] == 0.0
    assert s["below_1"]["miss_rate"] == pytest.approx(0.10)
    # the 0.5 m pixels predicted at 1.0 m are still below 2 m -> only the 50 band-1-2 misses count
    assert s["below_2"]["miss_rate"] == pytest.approx(50 / 200)


def test_sf2_gate_against_reference(cfg):
    gt, v = _two_depth()
    cur = sf.summarize_near_field([sf.near_field_counts(gt, gt * 2.0, v, cfg)], cfg)  # 1 m -> 2 m = miss
    ref = sf.summarize_near_field([sf.near_field_counts(gt, gt, v, cfg)], cfg)
    no_ref = sf.gate_sf2(cur, None, cfg)
    assert {g["status"] for g in no_ref} == {"not_evaluated"}
    g = {x["metric"]: x for x in sf.gate_sf2(cur, {"metrics": {"near_field": ref}}, cfg)}
    k = "miss_rise_vs_R0_pts@band_1-2"
    assert g[k]["value"] == pytest.approx(100.0) and g[k]["status"] == "fail"
    assert g["miss_rise_vs_R0_pts@band_0-1"]["status"] == "not_evaluated"   # no 0-1 m pixels
    with pytest.raises(ValueError):
        sf.gate_sf2(cur, {"metrics": {}}, cfg)


def test_near_field_signed_bias_sign_convention(cfg):
    gt, v = _two_depth()
    s = sf.summarize_near_field([sf.near_field_counts(gt, gt * 1.1, v, cfg)], cfg)
    assert s["near_bias"]["mean_signed_log"] == pytest.approx(math.log(1.1))   # > 0: too far
    g = sf.gate_near_bias(s, cfg)[0]
    assert g["status"] == "report_only"
    c = sf.load_config(overrides={"near_bias": {"max_abs_bias_log": 0.05}})
    assert sf.gate_near_bias(s, c)[0]["status"] == "fail"


# ── SF3 / SF4: ROI and conformal alert ──────────────────────────────────────

def test_roi_bounds_and_corridor_events(cfg):
    assert sf.roi_bounds(480, 640, cfg) == (168, 480, 160, 480)
    h, w = 48, 64
    y0, y1, x0, x1 = sf.roi_bounds(h, w, cfg)
    assert (y0, y1, x0, x1) == (17, 48, 16, 48)
    v = np.ones((h, w), bool)
    inside = np.full((h, w), 5.0)
    inside[30:40, 25:35] = 0.8                                   # 100 px = 10 % of the ROI
    outside = np.full((h, w), 5.0)
    outside[0:10, 0:10] = 0.8                                    # top-left: outside the ROI
    a = sf.corridor_stats(inside, inside, v, cfg)
    b = sf.corridor_stats(outside, outside, v, cfg)
    assert a["eligible"] and a["gt_q"] == pytest.approx(0.8) and a["pred_q"] == pytest.approx(0.8)
    assert b["gt_q"] == pytest.approx(5.0) and b["n_roi"] == (y1 - y0) * (x1 - x0)
    sparse_v = np.zeros((h, w), bool)
    sparse_v[40, 20:30] = True                                   # 10 of 992 ROI px valid
    assert not sf.corridor_stats(inside, inside, sparse_v, cfg)["eligible"]
    far = inside.copy()
    far[y0:y1, x0:x1] = np.inf                                   # non-finite predictions
    assert sf.corridor_stats(inside, far, v, cfg)["pred_q"] == float("inf")
    with pytest.raises(ValueError):
        sf.corridor_stats(inside, inside[:-1], v, cfg)


def test_conformal_threshold_is_the_exact_order_statistic():
    scores = np.arange(1, 100, dtype=float)                      # n = 99 events
    r = sf.conformal_threshold(scores, 0.05)
    assert r["k"] == 95 and r["tau"] == 95.0 and r["status"] == "ok"
    # ocmr_conformal takes one rank higher (96): conservative on continuous scores
    gt = np.full(99, 0.5)
    o = _fnr_controlled_alert(gt, scores, gt, scores, 1.0, 0.05)
    assert o["tau_m"] == 96.0
    assert sf.conformal_threshold(scores[:10], 0.05)["status"] == "insufficient_calibration_events"
    tiny = sf.conformal_threshold(scores[:20], 0.001)
    assert tiny["status"] == "insufficient_for_target" and tiny["tau"] == float("inf")


def test_conformal_coverage_monte_carlo():
    rng = np.random.default_rng(0)
    misses = []
    for _ in range(400):
        cal = rng.lognormal(0.0, 0.5, 99)
        test = rng.lognormal(0.0, 0.5, 2000)
        tau = sf.conformal_threshold(cal, 0.05)["tau"]
        misses.append(np.mean(test > tau))                       # alert iff score <= tau
    m = float(np.mean(misses))
    assert 0.045 <= m <= 0.053                                   # exact rate (n+1-k)/(n+1) = .05


def test_conformal_ties_inclusive_alert_keeps_the_bound():
    """Quantised scores (ties): the inclusive rule keeps P(miss) <= eps; the strict rule of
    ocmr_conformal (pred < tau) misses every tied event."""
    cal = np.full(40, 1.0)
    test = np.full(500, 1.0)
    tau = sf.conformal_threshold(cal, 0.05)["tau"]
    assert np.mean(test > tau) == 0.0
    gt = np.full(40, 0.5)
    assert _fnr_controlled_alert(gt, cal, np.full(500, 0.5), test, 1.0, 0.05)["test_fnr"] == 1.0
    rng = np.random.default_rng(1)
    misses = []
    for _ in range(300):
        c = np.round(rng.uniform(0.3, 1.5, 60), 1)
        t = np.round(rng.uniform(0.3, 1.5, 1000), 1)
        misses.append(np.mean(t > sf.conformal_threshold(c, 0.05)["tau"]))
    assert np.mean(misses) <= 0.05


def _corr_records(rng, n_event, n_non, prefix, noise=0.3):
    out = []
    for i in range(n_event):
        g = rng.uniform(0.3, 0.9)
        out.append({"id": f"{prefix}e{i}", "gt_q": g, "pred_q": g * math.exp(rng.normal(0, noise)),
                    "eligible": True})
    for i in range(n_non):
        g = rng.uniform(2.5, 6.0)
        out.append({"id": f"{prefix}n{i}", "gt_q": g, "pred_q": g * math.exp(rng.normal(0, noise)),
                    "eligible": True})
    return out


def test_sf3_alert_rates_and_gates(cfg):
    rng = np.random.default_rng(2)
    cal = _corr_records(rng, 60, 60, "c")
    test = _corr_records(rng, 200, 200, "t") + [{"id": "x", "gt_q": float("nan"),
                                                "pred_q": 1.0, "eligible": False}]
    r = sf.sf3_corridor_alert(cal, test, cfg)
    assert r["n_ineligible"] == 1 and r["n_test"] == 400
    t1 = r["per_threshold"]["1"]
    assert t1["status"] == "ok" and t1["n_events"] == 60 and t1["n_test_events"] == 200
    tg = np.array([x["gt_q"] for x in test[:-1]])
    tp = np.array([x["pred_q"] for x in test[:-1]])
    ev, al = tg < 1.0, tp <= t1["tau"]
    assert t1["test_miss_rate"] == pytest.approx((ev & ~al).sum() / ev.sum())
    assert t1["test_false_alarm_rate"] == pytest.approx((~ev & al).sum() / (~ev).sum())
    lo, hi = t1["test_miss_ci95"]
    assert lo <= t1["test_miss_rate"] <= hi
    gates = sf.gate_sf3(r, cfg)
    assert len(gates) == 4 and all(g["status"] in sf.GATE_STATUSES for g in gates)
    assert all(g["detail"]["split_kind"].startswith("within-dataset") for g in gates)


def test_sf4_regression_after_recalibration(cfg):
    rng = np.random.default_rng(3)
    cal = _corr_records(rng, 60, 60, "c")
    test = _corr_records(rng, 200, 200, "t")
    f = sf.sf3_corridor_alert(cal, test, cfg)
    q = sf.sf3_corridor_alert(cal, test, cfg)
    g = sf.gate_sf4(f, q, cfg)
    assert {x["status"] for x in g} == {"pass"} and all(x["value"] == 0.0 for x in g)
    worse_test = [dict(x, pred_q=x["pred_q"] * (3.0 if x["gt_q"] < 1 else 1.0)) for x in test]
    q2 = sf.sf3_corridor_alert(cal, worse_test, cfg)
    g2 = {x["metric"]: x for x in sf.gate_sf4(f, q2, cfg)}
    assert g2["miss_rise_pts@1m"]["value"] > 1.0 and g2["miss_rise_pts@1m"]["status"] == "fail"
    with pytest.raises(ValueError, match="same calibration and test"):
        sf.gate_sf4(f, sf.sf3_corridor_alert(cal, test[:-1], cfg), cfg)


def test_split_calibration_is_deterministic_and_about_20pct():
    keys = [f"img{i}" for i in range(2000)]
    m = sf.split_calibration(keys)
    assert np.array_equal(m, sf.split_calibration(keys))
    assert 0.16 < m.mean() < 0.24


# ── int8 gates I1-I4, I6 ─────────────────────────────────────────────────────

def _maps(h=40, w=50):
    gt = np.full((h, w), 5.0)
    gt[: h // 2] = 1.0                                           # top half near (< 2 m)
    return gt, np.log(gt)


def test_int8_agreement_i1_i2_i3(cfg):
    gt, a = _maps()
    b = a.copy()
    b[: 20] += 0.005                                             # near shift
    far_idx = np.flatnonzero(gt.ravel() > 2)
    b.flat[far_idx[:5]] += 0.5                                   # 0.5 % of all pixels off by 1.65x
    ag = sf.Int8Agreement(cfg)
    ag.add("i0", a, b, gt=gt, valid=np.ones_like(gt, bool))
    s = ag.summary()
    assert s["I1_frac_within"] == pytest.approx(1 - 5 / 2000)
    assert s["I2_p_abs_dlog_near"] == pytest.approx(0.005, abs=1.5e-4)
    assert s["I3_mean_signed_shift_near"] == pytest.approx(0.005)
    assert s["near_ref"] == ["gt"] and s["n_near_pixels"] == 1000
    g = {x["gate"]: x for x in sf.int8_gates(s, cfg)}
    assert g["I1"]["status"] == g["I2"]["status"] == g["I3"]["status"] == "pass"
    # without GT, "near" falls back to the fp32 prediction (< 2 m): same pixels here
    ag2 = sf.Int8Agreement(cfg)
    ag2.add("i0", a, b)
    assert ag2.summary()["near_ref"] == ["fp32"] and ag2.summary()["n_near_pixels"] == 1000
    # a +0.02 near-field shift fails I3 (> .01) but not I2 (ln 1.1 = .095)
    b3 = a.copy()
    b3[:20] += 0.02
    ag3 = sf.Int8Agreement(cfg)
    ag3.add("i0", a, b3, gt=gt)
    g3 = {x["gate"]: x for x in sf.int8_gates(ag3.summary(), cfg)}
    assert g3["I3"]["status"] == "fail" and g3["I2"]["status"] == "pass"
    assert {x["status"] for x in sf.int8_gates(ag3.summary(), cfg) if x["gate"] == "I4"} == {"not_evaluated"}


def test_int8_nonfinite_pixels_fail_i2(cfg):
    gt, a = _maps()
    b = a.copy()
    b.flat[:30] = np.nan                                         # 3 % of near pixels non-finite
    ag = sf.Int8Agreement(cfg)
    ag.add("i0", a, b, gt=gt)
    s = ag.summary()
    assert s["n_nonfinite"] == 30 and s["n_near_nonfinite"] == 30
    assert s["I2_p_abs_dlog_near"] == float("inf")
    assert s["I3_mean_signed_shift_near"] == pytest.approx(0.0)  # finite near pixels only
    g = {x["gate"]: x for x in sf.int8_gates(s, cfg)}
    assert g["I2"]["status"] == "fail" and g["I1"]["value"] == pytest.approx(1 - 30 / 2000)
    with pytest.raises(ValueError):
        ag.add("i1", a, b[:-1])


def test_abs_histogram_quantile_is_conservative_within_one_bin():
    rng = np.random.default_rng(4)
    v = np.abs(rng.normal(0, 0.05, 50000))
    h = sf.AbsHistogram(2.0, 1e-4)
    h.add(v[:20000])
    h.add(-v[20000:])                                            # sign is ignored
    for q in (0.5, 0.95, 0.99):
        ref = float(np.quantile(v, q, method="inverted_cdf"))
        got = h.quantile(q)
        assert ref <= got <= ref + 1e-4 + 1e-12
    h.add(np.array([5.0]))
    assert h.quantile(1.0) == 5.0                                # overflow bin -> largest value
    assert sf.AbsHistogram().quantile(0.5) is None


def _rows(d1, arel=0.2, bf=0.1, n=40, prefix="i"):
    return [{"id": f"{prefix}{k}", "d1": d1, "arel": arel, sf.BF_KEY: bf} for k in range(n)]


def test_int8_gate_i4(cfg):
    g = {x["metric"]: x for x in sf.int8_gates({}, cfg, _rows(0.70), _rows(0.68, 0.21, 0.085))}
    assert g["d1_drop_pts"]["value"] == pytest.approx(2.0) and g["d1_drop_pts"]["status"] == "fail"
    assert g["d1_drop_ub95_pts"]["status"] == "fail"
    assert g["arel_rise"]["value"] == pytest.approx(0.01) and g["arel_rise"]["status"] == "fail"
    assert g["bf_rel_drop"]["value"] == pytest.approx(0.15) and g["bf_rel_drop"]["status"] == "fail"
    same = {x["metric"]: x for x in sf.int8_gates({}, cfg, _rows(0.70), _rows(0.70))}
    assert all(same[m]["status"] == "pass"
               for m in ("d1_drop_pts", "d1_drop_ub95_pts", "arel_rise", "bf_rel_drop"))
    rng = np.random.default_rng(5)
    f = [{"id": f"i{k}", "d1": float(rng.uniform(0.5, 0.9)), "arel": 0.2} for k in range(60)]
    q = [dict(r, d1=r["d1"] - 0.005) for r in f]
    cl = {f"i{k}": str(k // 6) for k in range(60)}
    gc = {x["metric"]: x for x in sf.int8_gates({}, cfg, f, q, clusters=cl)}
    assert gc["d1_drop_pts"]["value"] == pytest.approx(0.5)
    assert gc["d1_drop_pts"]["detail"]["n_clusters"] == 10
    assert gc["bf_rel_drop"]["status"] == "not_evaluated"
    with pytest.raises(ValueError, match="id sets differ"):
        sf.int8_gates({}, cfg, _rows(0.7), _rows(0.7, prefix="j"))


def test_gate_i6_summarises_safety_regressions(cfg):
    def g(gate, metric, status):
        return {"gate": gate, "metric": metric, "status": status}
    ok = [g("SF1", "int8_rise_pts[int8]", "pass"), g("SF4", "miss_rise_pts@1m", "pass"),
          g("SF1", "over_rate@<2m", "fail")]                     # absolute SF1 is not I6
    assert sf.gate_i6(ok, cfg)["status"] == "pass"
    assert sf.gate_i6(ok + [g("SF4", "false_alarm_rise_pts@2m", "fail")], cfg)["status"] == "fail"
    assert sf.gate_i6(ok + [g("SF4", "x", "not_evaluated")], cfg)["status"] == "not_evaluated"
    assert sf.gate_i6([], cfg)["status"] == "not_evaluated"


# ── SF5 perturbations ───────────────────────────────────────────────────────

def _rgb(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(40, 220, w)[None, :, None] * np.ones((h, 1, 3))
    return np.clip(x + rng.normal(0, 10, (h, w, 3)), 0, 255).astype(np.uint8)


def test_perturbations(cfg):
    pytest.importorskip("cv2")
    rgb = _rgb()
    gt = np.linspace(1, 5, 48 * 64).reshape(48, 64).astype(np.float32)
    v = np.ones_like(gt, bool)
    dark, g, vv = sf.perturb_sample(rgb, gt, v, "low_light", cfg, seed=1)
    assert dark.dtype == np.uint8 and dark.mean() < 0.6 * rgb.mean() and g is gt
    assert np.array_equal(dark, sf.perturb_sample(rgb, gt, v, "low_light", cfg, seed=1)[0])
    blur = sf.perturb_sample(rgb, gt, v, "motion_blur", cfg)[0].astype(float)
    assert np.abs(np.diff(blur, axis=1)).mean() < np.abs(np.diff(rgb.astype(float), axis=1)).mean()
    jp = sf.perturb_sample(rgb, gt, v, "jpeg_q40", cfg)[0]
    assert jp.shape == rgb.shape and not np.array_equal(jp, rgb)
    for kind in ("roll_p15", "roll_m15"):
        r, g2, v2 = sf.perturb_sample(rgb, gt, v, kind, cfg)
        assert r.shape == rgb.shape and g2.shape == gt.shape
        assert not v2[0, 0] and not v2[-1, -1] and v2[24, 32]         # corners invalid
        assert g2[24, 32] == pytest.approx(gt[24, 32], rel=0.05)
    with pytest.raises(ValueError):
        sf.perturb_sample(rgb, gt, v, "fog", cfg)


def test_sf5_gate(cfg):
    clean = [{"id": f"i{k}", "d1": 0.8} for k in range(10)]
    rows = {"clean": clean, "low_light": [{"id": f"i{k}", "d1": 0.7} for k in range(10)],
            "jpeg_q40": [{"id": f"i{k}", "d1": 0.79} for k in range(1, 10)]}
    g = {x["metric"]: x for x in sf.gate_sf5(rows, cfg)}
    assert g["d1_drop_pts@low_light"]["value"] == pytest.approx(10.0)
    assert g["d1_drop_pts@low_light"]["status"] == "fail"
    assert g["d1_drop_pts@jpeg_q40"]["status"] == "pass"
    assert g["d1_drop_pts@jpeg_q40"]["detail"]["n_unpaired"] == 1
    assert g["d1_drop_pts@motion_blur"]["status"] == "not_evaluated"


# ── SF6 temporal ─────────────────────────────────────────────────────────────

def test_temporal_stability_never_pairs_across_videos():
    r = sf.temporal_stability({"a": [0.0, 0.01, 0.02, 0.2], "b": [1.0, 1.0]})
    assert r["n_pairs"] == 4
    assert r["p95_abs_change"] == pytest.approx(np.quantile([0.01, 0.01, 0.18, 0.0], 0.95))
    assert r["per_video"]["a"]["n_pairs"] == 3
    assert r["p95_abs_second_diff"] == pytest.approx(np.quantile([0.0, 0.17], 0.95))
    nf = sf.temporal_stability({"a": [0.0, float("nan"), 0.0]})
    assert nf["p95_abs_change"] == float("inf")


def test_gate_sf6_requires_verified_holdout(cfg):
    res = sf.temporal_stability({"a": [0.0, 0.01, 0.02, 0.03]})
    res["fps"] = {"a": {"fps_ok": True}}
    assert sf.gate_sf6(res, cfg)[0]["status"] == "not_evaluated"
    res["held_out_verified"] = True
    g = sf.gate_sf6(res, cfg, reference={"metrics": {"temporal": {"p95_abs_change": 0.02}}})
    assert g[0]["status"] == "pass" and g[1]["status"] == "report_only"
    assert g[1]["value"] == pytest.approx(res["p95_abs_change"] - 0.02)
    res["fps"] = {"a": {"fps_ok": False}}
    assert sf.gate_sf6(res, cfg)[0]["status"] == "not_evaluated"
    bad = sf.temporal_stability({"a": [0.0, float("inf")]})
    bad.update({"held_out_verified": True, "fps": {"a": {"fps_ok": True}}})
    assert sf.gate_sf6(bad, cfg)[0]["status"] == "fail"


def test_frame_stride():
    assert sf.frame_stride(30.0, 5.0) == 6 and sf.frame_stride(29.97, 5.0) == 6
    assert sf.frame_stride(4.0, 5.0) == 1
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            sf.frame_stride(bad, 5.0)


class _VideoModel:
    name = "fake-video"

    def predict(self, rgb, K):
        return {"depth": 0.5 + rgb[..., 0].astype(np.float32) / 50.0}


def _write_video(path, n=30, fps=25.0):
    cv2 = pytest.importorskip("cv2")
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (48, 32))
    if not wr.isOpened():
        pytest.skip("OpenCV build cannot write MJPG video")
    for i in range(n):
        wr.write(np.full((32, 48, 3), 60 + 3 * i, np.uint8))
    wr.release()


def test_video_sampling_holdout_check_and_run_temporal(tmp_path, cfg):
    vid = tmp_path / "Dollar_Tree.avi"
    _write_video(vid)
    frames = list(sf.iter_video_frames(str(vid), 5.0))
    assert [f[0] for f in frames] == [0, 5, 10, 15, 20, 25]
    assert frames[1][1] == pytest.approx(0.2)
    with pytest.raises(FileNotFoundError):
        list(sf.iter_video_frames(str(tmp_path / "missing.mp4")))
    roles = tmp_path / "roles.yaml"
    roles.write_text(
        "sf6_temporal:\n  videos:\n    Dollar_Tree:\n      path: " + str(vid).replace("\\", "/")
        + "\n      sha256: " + sf._sha256(str(vid)) + "\n", encoding="utf-8")
    hold = sf.sf6_holdout_videos(str(roles))
    res = sf.run_temporal(_VideoModel(), [str(vid)], cfg, holdout=hold)
    assert res["held_out_verified"] is True and res["n_pairs"] == 5
    assert res["fps"]["Dollar_Tree.avi"]["effective_fps"] == pytest.approx(5.0)
    copy = tmp_path / "renamed.avi"
    shutil.copy(vid, copy)
    ver = sf.verify_sf6_videos([str(copy)], hold)
    assert ver[str(copy)] == {"held_out": True, "name": "Dollar_Tree", "matched_by": "sha256"}
    other = tmp_path / "other.avi"
    _write_video(other, n=12)
    res2 = sf.run_temporal(_VideoModel(), [str(other)], cfg, holdout=hold)
    assert res2["held_out_verified"] is False
    assert sf.gate_sf6(res2, cfg)[0]["status"] == "not_evaluated"
    assert sf.run_temporal(_VideoModel(), [str(vid)], cfg)["held_out_verified"] is None
    with pytest.raises(FileNotFoundError):
        sf.sf6_holdout_videos(str(tmp_path / "nope.yaml"))


# ── sparse anchors ───────────────────────────────────────────────────────────

def test_sparse_anchor_track_recovers_global_scale(cfg):
    rng = np.random.default_rng(6)
    gt = rng.uniform(1.0, 5.0, (40, 50))
    v = np.ones_like(gt, bool)
    pred = gt * 2.0
    out = sf.sparse_anchor_align(gt, pred, v, 150, 0.0, 0.0, np.random.default_rng(0))
    aligned, mask, info = out
    assert mask.sum() == 150 and info["log_scale"] == pytest.approx(-math.log(2.0))
    assert np.allclose(aligned, gt)
    c = sf.load_config(overrides={"anchor": {"sweep_points": [10, 150], "sweep_noise": [0.0]}})
    acc = sf.AnchorAccumulator(c)
    acc.add("i0", gt, pred, v)
    s = acc.summary()
    assert s["zero_shot_eligible"] is False and s["monocular"] is False
    assert s["d1"]["raw"] == 0.0 and s["d1"]["n150_sigma0"] == 1.0
    assert "n150_sigma0.02" in s["d1"]                            # the default config is added
    assert sf.sparse_anchor_align(gt, pred, np.zeros_like(v), 150, 0, 0, rng) is None


# ── end-to-end driver on synthetic samples ──────────────────────────────────

class _Sample:
    def __init__(self, i):
        self.id = f"s{i:03d}"
        self.rgb = _rgb(seed=i)
        base = _RgbDepth.depth_of(self.rgb)
        self.gt = (base * (1.0 + 0.05 * ((i % 5) - 2))).astype(np.float32)
        if i % 3 == 0:
            self.gt[30:40, 28:36] = 0.6                          # obstacle in the corridor
        self.valid = np.ones(self.gt.shape, bool)
        self.K = None
        self.max_depth = 10.0


class _RgbDepth:
    def __init__(self, name="fake-fp32", gain=1.0):
        self.name, self.gain = name, gain

    @staticmethod
    def depth_of(rgb):
        return 0.4 + rgb[..., 0].astype(np.float32) / 255.0 * 4.0

    def predict(self, rgb, K):
        return {"depth": self.depth_of(rgb) * self.gain}


def test_run_safety_eval_end_to_end():
    pytest.importorskip("cv2")
    c = sf.load_config(overrides={"sf3": {"min_cal_events": 2},
                                  "anchor": {"sweep_points": [50], "sweep_noise": [0.0]}})
    samples = [_Sample(i) for i in range(30)]
    rep = sf.run_safety_eval(_RgbDepth(), samples, c, int8_model=_RgbDepth("fake-int8", 1.01))
    assert rep["claim_level"] == sf.RESEARCH_ONLY_LABEL and rep["owner_confirmed"] is False
    assert rep["meta"]["n_images"] == 30 and rep["meta"]["camera_mode"] == "none"
    gates = {g["gate"] for g in rep["gates"]}
    assert {"SF1", "SF2", "NB", "SF3", "SF4", "SF5", "I1", "I2", "I3", "I4", "I6"} <= gates
    assert all(g["status"] in sf.GATE_STATUSES for g in rep["gates"])
    i1 = next(g for g in rep["gates"] if g["gate"] == "I1")
    assert i1["status"] == "pass" and i1["value"] == pytest.approx(1.0)  # 1 % gain << 1.25x
    i3 = next(g for g in rep["gates"] if g["gate"] == "I3")
    assert i3["value"] == pytest.approx(math.log(1.01), rel=1e-3) and i3["status"] == "pass"
    assert rep["metrics"]["sparse_anchor"]["zero_shot_eligible"] is False
    assert len(rep["metrics"]["rows"]) == 30 and sf.BF_KEY in rep["metrics"]["rows"][0]
    assert rep["counts"]["not_evaluated"] >= 1 and rep["complete"] is False   # no R0 reference
    json.dumps(rep)                                              # plain-JSON serialisable
    with pytest.raises(ValueError):
        sf.run_safety_eval(_RgbDepth(), samples[:1], c, camera_mode="gt")


def test_datasets_root_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="not reachable"):
        sf.require_datasets_root("ibims", str(tmp_path / "Z_missing"))
    with pytest.raises(SystemExit, match="dataset folder missing"):
        sf.require_datasets_root("ibims", str(tmp_path))
    assert sf.require_datasets_root("tum", str(tmp_path)) == str(tmp_path)


def test_cli_print_config(capsys):
    assert sf.main(["print-config"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == sf.SAFETY_CONFIG_VERSION
