"""Tests for the V2 teacher-label pipeline: scripts/v2/fuse_composite.py (DCT-II composite solver,
robust affine fit, low-pass agreement mask, A8 label variants, L0 scoring, registry record) and
the shared labelling framework in scripts/v2/label_unidepth.py / label_moge.py (run with a fake
teacher; no model is loaded).

Env: depth or depth_cmp (numpy, scipy, opencv, pyyaml). CPU-only, offline, synthetic data,
temporary directories only. Plan 9.5 row: solver residual, mean-log preservation, edge survival
>= 85%.
"""
import csv
import json
import math
import os
import string

import cv2
import numpy as np
import pytest
from scipy.fft import dctn, idctn

from scripts.v2 import fuse_composite as fc
from scripts.v2 import label_unidepth as lu
from scripts.v2.label_moge import moge_postprocess

LAM48 = (math.pi / 48.0) ** 2


def _rand_maps(h=37, w=53, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.5, 0.4, (h, w)), rng.normal(0.8, 0.6, (h, w))


def _basis(h, w, ky, kx):
    c = np.zeros((h, w))
    c[ky, kx] = 1.0
    return idctn(c, type=2, norm="ortho")


def _room(h=480, w=640):
    """Piecewise-smooth synthetic indoor scene (metres): floor ramp, back wall, two boxes."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.full((h, w), 6.0)
    floor = yy > 0.55 * h
    d[floor] = 6.0 - 4.5 * (yy[floor] - 0.55 * h) / (0.45 * h)
    d[(yy > 0.30 * h) & (yy < 0.75 * h) & (xx > 0.15 * w) & (xx < 0.35 * w)] = 2.2
    d[(yy > 0.40 * h) & (yy < 0.85 * h) & (xx > 0.60 * w) & (xx < 0.80 * w)] = 3.1
    return d


# ── DCT-II solver ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lam", [0.0, 1e-3, LAM48, 0.5, 10.0])
def test_solver_satisfies_normal_equations(lam):
    U, M = _rand_maps()
    C = fc.dct_composite(U, M, lam)
    assert fc.normal_equation_residual(C, U, M, lam) < 1e-9


def test_dct_diagonalises_the_neumann_laplacian():
    h, w = 12, 17
    mu = fc.neumann_eigenvalues(h, w)
    for ky, kx in ((0, 0), (0, 3), (5, 0), (7, 11), (11, 16)):
        b = _basis(h, w, ky, kx)
        np.testing.assert_allclose(fc.neumann_laplacian(b), mu[ky, kx] * b, atol=1e-12)


def test_solver_is_the_minimiser():
    U, M = _rand_maps(seed=3)
    C = fc.dct_composite(U, M, LAM48)
    e0 = fc.composite_energy(C, U, M, LAM48)
    rng = np.random.default_rng(1)
    for _ in range(5):
        assert fc.composite_energy(C + 1e-3 * rng.normal(size=C.shape), U, M, LAM48) > e0


def test_exact_recovery_identical_and_shifted_moge():
    U, _ = _rand_maps(seed=5)
    np.testing.assert_allclose(fc.dct_composite(U, U.copy(), LAM48), U, atol=1e-12)
    # MoGe off by a constant log shift has identical gradients -> composite is exactly V2-L
    np.testing.assert_allclose(fc.dct_composite(U, U + 0.7, LAM48), U, atol=1e-12)


@pytest.mark.parametrize("lam", [0.0, LAM48, 3.0])
def test_mean_log_depth_comes_only_from_v2l(lam):
    U, M = _rand_maps(h=40, w=64, seed=7)
    C = fc.dct_composite(U, M + 2.5, lam)
    assert abs(C.mean() - U.mean()) < 1e-12


def test_lam_limits():
    U, M = _rand_maps(seed=9)
    np.testing.assert_array_equal(fc.dct_composite(U, M, math.inf), U)
    np.testing.assert_allclose(fc.dct_composite(U, M, 0.0), M - M.mean() + U.mean(), atol=1e-12)
    with pytest.raises(ValueError):
        fc.dct_composite(U, M, -1.0)
    with pytest.raises(ValueError):
        fc.dct_composite(U, M[:, :-1], 1.0)


def test_single_frequency_blend_is_exact():
    h, w, ky, kx = 30, 48, 4, 9
    mu = fc.neumann_eigenvalues(h, w)[ky, kx]
    b = _basis(h, w, ky, kx)
    for lam in (0.1 * mu, mu, 10 * mu):
        C = fc.dct_composite(0.3 * b, 1.7 * b, lam)
        coef = dctn(C, type=2, norm="ortho")[ky, kx]
        assert coef == pytest.approx((lam * 0.3 + mu * 1.7) / (lam + mu), abs=1e-12)
    # at lam == mu (the crossover) the two teachers get equal weight
    C = fc.dct_composite(0.0 * b, 2.0 * b, mu)
    assert dctn(C, type=2, norm="ortho")[ky, kx] == pytest.approx(1.0, abs=1e-12)


def test_crossover_low_from_v2l_high_from_moge():
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    low_u = 0.4 * np.cos(np.pi * (xx + 0.5) / w)              # 1280 px period: coarse layout
    low_m = -0.3 * np.cos(np.pi * (yy + 0.5) / h)             # MoGe's (wrong) coarse layout
    high = 0.05 * np.cos(np.pi * 160 * (xx + 0.5) / w)        # 8 px period: fine detail
    C = fc.dct_composite(1.0 + low_u, 1.0 + low_m + high, LAM48)
    resid = C - 1.0
    a_low_u = (resid * low_u).sum() / (low_u ** 2).sum()
    a_low_m = (resid * low_m).sum() / (low_m ** 2).sum()
    a_high = (resid * high).sum() / (high ** 2).sum()
    assert a_low_u > 0.99 and abs(a_low_m) < 0.02 and a_high > 0.99


def test_effective_parameters_scale_with_short_side():
    cfg = fc.CompositeConfig()
    cross, sigma, lam = cfg.effective(480, 640)
    assert (cross, sigma) == (48.0, 16.0) and lam == pytest.approx(LAM48)
    cross, sigma, _ = cfg.effective(576, 1024)
    assert cross == pytest.approx(57.6) and sigma == pytest.approx(19.2)
    cross, sigma, _ = fc.CompositeConfig(scale_with_resolution=False).effective(576, 1024)
    assert (cross, sigma) == (48.0, 16.0)


# ── robust affine fit ────────────────────────────────────────────────────────

def test_irls_exact_and_robust():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 2.5, 5000)
    f = fc.robust_affine_fit(x, 1.1 * x + 0.3)
    assert f.ok and not f.clamped and f.a == pytest.approx(1.1, abs=1e-6) and f.b == pytest.approx(0.3, abs=1e-6)
    y = 0.95 * x - 0.2
    out = rng.random(x.size) < 0.25
    y[out] += rng.uniform(0.5, 1.5, out.sum())
    f = fc.robust_affine_fit(x, y)
    assert f.a == pytest.approx(0.95, abs=0.01) and f.b == pytest.approx(-0.2, abs=0.01)


def test_irls_clamp_flat_and_tiny():
    x = np.linspace(0, 2, 1000)
    y = 2.0 * x + 1.0
    f = fc.robust_affine_fit(x, y)
    assert f.clamped and f.a == 1.25 and f.a_raw == pytest.approx(2.0, abs=1e-6)
    assert f.b == pytest.approx(float(np.median(y - 1.25 * x)))
    f = fc.robust_affine_fit(np.full(100, 0.3), np.linspace(0, 1, 100))
    assert f.ok and f.a == 1.0
    assert not fc.robust_affine_fit(np.arange(5.0), np.arange(5.0)).ok


# ── agreement mask and edge statistics ───────────────────────────────────────

def test_edge_pair_survival_hand_count():
    L = np.zeros((3, 4))
    L[:, 2:] = 1.0                                           # 3 strong horizontal pairs (col 1|2)
    keep = np.ones((3, 4), bool)
    assert fc.edge_pair_survival(L, keep) == 1.0
    keep[0, 2] = False
    assert fc.edge_pair_survival(L, keep) == pytest.approx(2 / 3)
    assert math.isnan(fc.edge_pair_survival(np.zeros((3, 4)), keep))


def test_global_scale_gap_is_aligned_away_but_s8_punishes_it():
    dU = _room()
    dM = 1.4 * dU                                            # ln 1.4 > ln 1.25 everywhere
    r = fc.fuse_one(dU, dM)
    assert r.status == "ok" and r.fit.a == pytest.approx(1.0, abs=1e-3)
    assert r.stats["masked_frac"] == 0.0
    np.testing.assert_allclose(r.L_C, np.log(dU), atol=1e-6)
    assert fc.s8_weight(dU, dM).mean() == pytest.approx(math.exp(-math.log(1.4) / 0.2), rel=1e-6)


def test_coarse_disagreement_is_masked_locally():
    dU = _room()
    dM = dU.copy()
    dM[300:420, 420:540] *= 2.0                              # 120 px blob, coarse layout error
    r = fc.fuse_one(dU, dM)
    assert not r.keep[360, 480]                              # blob centre masked
    assert r.keep[100, 100] and r.keep[460, 30]              # far away kept
    assert 0.01 < r.stats["masked_frac"] < 0.15


def test_edge_misalignment_not_masked_and_survival_above_85pct():
    dU = _room()
    dM = np.roll(dU, 1, axis=1)                              # every edge 1 px off
    r = fc.fuse_one(dU, dM)
    assert r.stats["masked_frac"] < 0.01
    assert r.stats["edge_pair_survival"] >= 0.85
    # the rejected per-pixel test removes the edges the composite takes from MoGe
    assert r.stats["edge_pair_survival_pixel_test"] < r.stats["edge_pair_survival"]
    edges = fc.strong_edge_pixels(r.L_C) & r.valid_U
    assert r.keep[edges].mean() > fc.s8_weight(dU, dM)[edges].mean()


def test_plan_edge_survival_on_realistic_teacher_pair():
    """Plan 9.5: edge survival >= 85% for a MoGe map that is sharper, slightly warped and
    affinely off from a blurrier V2-L map of the same scene."""
    h, w = 480, 640
    sharp = _room(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dU = np.exp(cv2.GaussianBlur(np.log(sharp), (0, 0), 2.0))            # V2-L: blurrier edges
    warp = np.exp(0.08 * np.sin(2 * np.pi * xx / w) * np.cos(np.pi * yy / h))
    dM = np.exp(0.9 * np.log(sharp * warp) - 0.15)                      # MoGe: sharp, warped, affine off
    r = fc.fuse_one(dU, dM)
    assert r.status == "ok" and not r.fit.clamped and r.fit.a == pytest.approx(1 / 0.9, abs=0.05)
    assert r.stats["edge_pair_survival"] >= 0.85
    assert r.stats["masked_frac"] < 0.05
    assert abs(r.L_C.mean() - r.L_U.mean()) < 1e-9
    # the composite's edges sit where the sharp scene's edges are; V2-L's blur smears each jump
    # over several >10% pairs (more strong-edge pixels, in the wrong places), so compare by IoU
    sharp_e = fc.strong_edge_pixels(np.log(sharp))

    def iou(e):
        return (e & sharp_e).sum() / max(1, (e | sharp_e).sum())

    assert iou(fc.strong_edge_pixels(r.L_C)) >= 0.9
    assert iou(fc.strong_edge_pixels(r.L_C)) > iou(fc.strong_edge_pixels(np.log(dU))) + 0.3


def test_invalid_v2l_pixels_never_kept_and_low_overlap_fallback():
    dU = _room()
    dU[:40, :] = 0.0                                         # V2-L invalid band
    dM = dU.copy()
    dM[dU == 0] = 5.0
    r = fc.fuse_one(dU, dM, valid_u=dU > 0)
    assert not r.keep[:40].any() and (r.depth_composite()[:40] == 0).all()
    assert np.isfinite(r.L_C).all()
    dM2 = np.zeros_like(dU)
    dM2[:10, :10] = 2.0                                      # MoGe almost entirely invalid
    r2 = fc.fuse_one(dU, dM2, valid_u=dU > 0, valid_m=dM2 > 0)
    assert r2.status == "v2l_only_low_overlap"
    np.testing.assert_array_equal(r2.L_C, r2.L_U)


# ── labelling framework (fake teacher) ───────────────────────────────────────

def _write_img(path, seed, h=60, w=80):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _fake_teacher(scale=1.0, sharp=True):
    def predict(rgb):
        h, w = rgb.shape[:2]
        d = _room(h, w) * scale
        if not sharp:
            d = np.exp(cv2.GaussianBlur(np.log(d), (0, 0), 1.5))
        return {"depth_m": d.astype(np.float32), "valid": np.ones((h, w), bool), "extra": {"fake": True}}
    return predict


def test_read_items_keep_column_verbatim_ids_and_errors(tmp_path):
    for n in ("a", "b - Copy", "c"):
        _write_img(tmp_path / f"{n}.jpg", 1)
    p = tmp_path / "list.csv"
    p.write_text("id,img,keep\na,a.jpg,1\nb - Copy,b - Copy.jpg,1\nc,c.jpg,0\n", encoding="utf-8")
    items = lu.read_items(str(p))
    assert [it.id for it in items] == ["a", "b - Copy"]                 # verbatim, keep honoured
    assert len(lu.read_items(str(p), keep_column="ignore")) == 3
    assert [it.id for it in lu.read_items(str(tmp_path))] == ["a", "b - Copy", "c"]
    bad = tmp_path / "dup.csv"
    bad.write_text("id,img\na,a.jpg\na,c.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lu.read_items(str(bad))
    unsafe = tmp_path / "unsafe.csv"
    unsafe.write_text("id,img\nx:y,a.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lu.read_items(str(unsafe))


def test_check_drives_fails_loudly_on_missing_drive():
    if os.name != "nt":
        pytest.skip("drive letters are Windows-only")
    free = [d for d in string.ascii_uppercase[3:] if not os.path.exists(f"{d}:\\")]
    if not free:
        pytest.skip("no unused drive letter")
    with pytest.raises(SystemExit, match="not available"):
        lu.check_drives([f"{free[-1]}:/Datasets/SUNRGBD/x.jpg"])


def test_run_labelling_resumable_and_format(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for k in range(3):
        _write_img(imgs / f"im{k}.jpg", k, h=600, w=1600)                # longest side > 1024
    out = tmp_path / "v2l"
    items = lu.read_items(str(imgs))
    s1 = lu.run_labelling(items, str(out), _fake_teacher(), {"checkpoint": "fake"}, warmup=0, log=lambda m: None)
    assert s1["n_labelled"] == 3 and s1["n_failed"] == 0 and s1["img_per_s_model"] > 0
    png = cv2.imread(str(out / "depth_mm" / "im0.png"), cv2.IMREAD_UNCHANGED)
    ref = lu.resize_longest_side(lu.depth_m_to_png16(_room(600, 1600).astype(np.float32)), 1024, cv2.INTER_NEAREST)
    assert png.dtype == np.uint16 and png.shape == (384, 1024)
    np.testing.assert_array_equal(png, ref)                               # same chain as outputs/depth_mm
    meta = json.loads((out / "meta" / "im0.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and meta["orig_hw"] == [600, 1600] and meta["label_valid"]
    assert meta["png_sha256"] == lu.sha256_file(str(out / "depth_mm" / "im0.png"))
    with open(out / "done.csv", newline="", encoding="utf-8") as f:
        assert [r["id"] for r in csv.DictReader(f)] == ["im0", "im1", "im2"]
    s2 = lu.run_labelling(items, str(out), _fake_teacher(), {"checkpoint": "fake"}, warmup=0, log=lambda m: None)
    assert s2["n_labelled"] == 0 and s2["n_already_done"] == 3
    # a sidecar that is not "ok" is relabelled
    meta["status"] = "partial"
    (out / "meta" / "im1.json").write_text(json.dumps(meta), encoding="utf-8")
    s3 = lu.run_labelling(items, str(out), _fake_teacher(), {"checkpoint": "fake"}, warmup=0, log=lambda m: None)
    assert s3["n_labelled"] == 1


def test_done_id_conflicts(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_img(a / "x.jpg", 1)
    out = str(tmp_path / "lab")
    lu.run_labelling(lu.read_items(str(a)), out, _fake_teacher(), {}, warmup=0, log=lambda m: None)
    cv2.imwrite(str(b / "x.jpg"), cv2.imread(str(a / "x.jpg")))           # same bytes? re-encode may differ
    import shutil
    shutil.copyfile(a / "x.jpg", b / "x.jpg")                             # moved copy: same SHA-1 -> accepted
    s = lu.run_labelling(lu.read_items(str(b)), out, _fake_teacher(), {}, warmup=0, log=lambda m: None)
    assert s["n_already_done"] == 1
    _write_img(b / "x.jpg", 99)                                           # different image, same id
    with pytest.raises(SystemExit, match="different"):
        lu.run_labelling(lu.read_items(str(b)), out, _fake_teacher(), {}, warmup=0, log=lambda m: None)


def test_moge_postprocess_records_invalid_pixels():
    d = np.full((20, 30), 2.0, np.float32)
    d[0, :5] = np.inf
    d[1, :5] = -1.0
    mask = np.ones((20, 30), bool)
    mask[2, :10] = False
    r = moge_postprocess(d, mask, (40, 60))
    ex = r["extra"]
    assert ex["nonfinite_frac"] == pytest.approx(10 / 600, abs=1e-6)  # stored rounded to 6 d.p.
    assert ex["mask_invalid_frac"] == pytest.approx(10 / 600, abs=1e-6)  # stored rounded to 6 d.p.
    assert ex["invalid_frac"] == pytest.approx(20 / 600, abs=1e-6)
    assert r["depth_m"].shape == (40, 60) and np.isfinite(r["depth_m"]).all()
    assert not r["valid"][0:2, 0:10].any() and r["valid"][10:, 20:].all()
    png = lu.encode_label(r["depth_m"], r["valid"])
    assert (png[~r["valid"]] == 0).all() and (png[r["valid"]] == 2000).all()


# ── fuse_dir, A8 variants, leak guard ────────────────────────────────────────

def _make_teacher_dirs(tmp_path, ids_seed=(0, 1)):
    imgs = tmp_path / "imgs"
    imgs.mkdir(exist_ok=True)
    for k in ids_seed:
        _write_img(imgs / f"f{k}.jpg", k, h=120, w=160)
    items = lu.read_items(str(imgs))
    v2l, moge = str(tmp_path / "v2l"), str(tmp_path / "moge")
    lu.run_labelling(items, v2l, _fake_teacher(sharp=False), {"checkpoint": "fake-v2l"}, warmup=0, log=lambda m: None)
    lu.run_labelling(items, moge, _fake_teacher(scale=0.8), {"checkpoint": "fake-moge"}, warmup=0, log=lambda m: None)
    return v2l, moge, items


def test_fuse_dir_and_label_variants(tmp_path):
    v2l, moge, _ = _make_teacher_dirs(tmp_path)
    out = str(tmp_path / "comp")
    s = fc.fuse_dir(v2l, moge, out, roles_path=None, log=lambda m: None)
    assert s["n_fused"] == 2 and s["n_v2l_only_fallback"] == 0
    d_a, v_a = fc.load_composite_label(out, "f0", "l1a")
    d_b, v_b = fc.load_composite_label(out, "f0", "l1b")
    np.testing.assert_array_equal(d_a, d_b)
    assert v_b.sum() <= v_a.sum() and v_a.all()
    # composite keeps V2-L's metric scale (MoGe is 0.8x) to within PNG mm quantisation
    dU = lu.read_label_png(os.path.join(v2l, "depth_mm", "f0.png"))
    assert abs(np.log(d_a).mean() - np.log(dU).mean()) < 1e-3
    meta = json.loads(open(os.path.join(out, "meta", "f0.json"), encoding="utf-8").read())
    assert meta["inputs"]["v2l_teacher"] == "fake-v2l" and meta["roles_checked"] is False
    assert fc.fuse_dir(v2l, moge, out, roles_path=None, log=lambda m: None)["n_fused"] == 0    # resumable
    with pytest.raises(ValueError):
        fc.load_composite_label(out, "f0", "s8")


def test_fuse_dir_legacy_flat_v2l_cache(tmp_path):
    v2l, moge, _ = _make_teacher_dirs(tmp_path)
    flat = tmp_path / "flat"
    flat.mkdir()
    for i in ("f0", "f1"):
        os.replace(os.path.join(v2l, "depth_mm", i + ".png"), flat / f"{i}.png")
    assert fc.LabelDir(str(flat)).layout == "flat" and fc.LabelDir(str(flat)).ids() == {"f0", "f1"}
    s = fc.fuse_dir(str(flat), moge, str(tmp_path / "comp"), roles_path=None, log=lambda m: None)
    assert s["n_fused"] == 2 and s["v2l_layout"] == "flat"


def test_fuse_dir_refuses_dev_and_test_ids(tmp_path):
    v2l, moge, _ = _make_teacher_dirs(tmp_path)
    roles = tmp_path / "roles.yaml"
    roles.write_text("dev:\n  d_sun: [f1]\ntest:\n  frozen_own: []\n  video_own: []\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="dev/test"):
        fc.fuse_dir(v2l, moge, str(tmp_path / "comp"), roles_path=str(roles), log=lambda m: None)
    s = fc.fuse_dir(v2l, moge, str(tmp_path / "comp"), ids=["f0"], roles_path=str(roles), log=lambda m: None)
    assert s["n_fused"] == 1 and s["roles_checked"] is True


# ── L0 scoring and registry ──────────────────────────────────────────────────

def _make_l0_fixture(tmp_path, n=3):
    """GT = sharp room; V2-L = blurred GT at 1.05x scale; MoGe = sharp GT at 0.7x scale.

    (0.8x would sit exactly on the d1 = 1.25 boundary, where mm PNG rounding decides the pixel.)"""
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    rows = []
    v2l, moge = str(tmp_path / "v2l"), str(tmp_path / "moge")
    for k in range(n):
        i = f"sun_{k:06d}"
        g = _room() * (1.0 + 0.1 * k)
        cv2.imwrite(str(gt_dir / f"{i}.png"), lu.depth_m_to_png16(g.astype(np.float32)))
        u = np.exp(cv2.GaussianBlur(np.log(g * 1.05), (0, 0), 3.0))
        for root, d, ck in ((v2l, u, "fake-v2l"), (moge, g * 0.7, "fake-moge")):
            os.makedirs(os.path.join(root, "depth_mm"), exist_ok=True)
            os.makedirs(os.path.join(root, "meta"), exist_ok=True)
            cv2.imwrite(os.path.join(root, "depth_mm", i + ".png"), lu.depth_m_to_png16(d.astype(np.float32)))
            with open(os.path.join(root, "meta", i + ".json"), "w", encoding="utf-8") as f:
                json.dump({"status": "ok", "teacher": {"checkpoint": ck}}, f)
        rows.append({"id": i, "img": "", "gt_depth": str(gt_dir / f"{i}.png"), "dmin": "0.2", "dmax": "8.0",
                     "source": "sun"})
    lst = tmp_path / "l0.csv"
    with open(lst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(lst), v2l, moge


def test_run_l0_composite_takes_moge_edges_and_v2l_scale(tmp_path):
    lst, v2l, moge = _make_l0_fixture(tmp_path)
    clusters = {"sun_000000": "c:sun_000000", "sun_000001": "c:sun_000000", "sun_000002": "sun_000002"}
    res = fc.run_l0(lst, v2l, moge, n_boot=200, clusters=clusters, log=lambda m: None)
    assert res["n_images"] == 3 and res["bf_corrected"] is True
    assert res["bf_function"] == "benchmark.protocol_v2.boundary_f1_si_v2"
    g = res["gates"]
    assert set(g) == {"G1_d1", "G2_d1_ssi", "G3_bf", "G4_edge_pair_survival"}
    assert g["G3_bf"]["composite"] > g["G3_bf"]["v2l"] + 0.005               # sharper edges from MoGe
    assert g["G1_d1"]["composite"] >= g["G1_d1"]["v2l"] - 0.2                # scale from V2-L kept
    assert res["aggregate"]["moge_raw@moge_valid"]["d1"] < 0.5               # raw MoGe is 0.7x: wrong scale
    assert res["bootstrap"]["resampling"] == "cluster"
    assert res["paired_bootstrap"]["composite-v2l:d1"]["n_clusters"] == 2
    assert {"l1a", "l1b", "pixels_pooled", "l1b_label_gates_descriptive"} <= set(res["a8"])
    assert "bf_legacy" in res["per_image"][0]["composite"]
    json.dumps(res, allow_nan=False)                                         # strictly JSON-serialisable


def test_run_l0_refuses_silent_subset(tmp_path):
    lst, v2l, moge = _make_l0_fixture(tmp_path)
    os.remove(os.path.join(moge, "meta", "sun_000001.json"))
    with pytest.raises(SystemExit, match="lack a V2-L or MoGe label"):
        fc.run_l0(lst, v2l, moge, n_boot=50, log=lambda m: None)


def test_l0_list_seeded_and_dsun_loader(tmp_path):
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "img", "depth", "conf", "is_gt", "dmin", "dmax"])
        for k in range(20):
            w.writerow([f"sun_{k:06d}", "sun", f"i{k}.jpg", f"d{k}.png", "", 1, 0.2, 8.0])
    roles = tmp_path / "roles.yaml"
    ids = [f"sun_{k:06d}" for k in range(2, 20)]
    roles.write_text("dev:\n  d_sun: [" + ", ".join(ids) + "]\n  d_sun_clusters: [[sun_000003, sun_000007]]\n",
                     encoding="utf-8")
    ds = fc.load_dsun(str(roles))
    assert ds["ids"] == ids and ds["clusters"]["sun_000007"] == ds["clusters"]["sun_000003"] == "c:sun_000003"
    assert ds["clusters"]["sun_000004"] == "sun_000004"
    a = fc.build_l0_list(ds["ids"], str(man), n=5)
    b = fc.build_l0_list(ds["ids"], str(man), n=5)
    assert [r["id"] for r in a] == [r["id"] for r in b] and len({r["id"] for r in a}) == 5
    assert all(r["id"] in ids for r in a)
    with pytest.raises(SystemExit):
        fc.build_l0_list(ds["ids"], str(man), n=50)
    with pytest.raises(SystemExit):
        fc.load_dsun(str(tmp_path / "missing.yaml"))


def test_register_l0_root_record(tmp_path, monkeypatch):
    pytest.importorskip("benchmark.registry_v2")
    from benchmark import registry_v2
    monkeypatch.setenv("UNIDEPTH_REGISTRY_V2", str(tmp_path / "reg.jsonl"))
    lst, v2l, moge = _make_l0_fixture(tmp_path, n=2)
    res = fc.run_l0(lst, v2l, moge, n_boot=50, log=lambda m: None)
    rp = tmp_path / "L0.json"
    rp.write_text(json.dumps(res), encoding="utf-8")
    rec = fc.register_l0(res, str(rp))
    assert rec["type"] == "root" and rec["run_id"] == "L0" and rec["status"] in ("accepted", "rejected")
    ok, problems = registry_v2.verify_chain()
    assert ok, problems
    with pytest.raises(ValueError):
        fc.register_l0(res, str(rp))                                         # duplicate id without supersede
    rec2 = fc.register_l0(res, str(rp), supersede=True)
    assert rec2["supersedes"] == rec["record_hash"]
