"""b20 C3a ownar + C3b fov: geometry, K updates, pad masking, s5 rate. CPU only."""
import csv
import json
import os

import cv2
import numpy as np
import pytest
import torch

from distill.dataset import MultiSourceDataset
from distill.losses import eg_ssi_loss
from _orig import OUTPUTS, load_orig


def _stub(crop_p=0.5, fov_p=0.25, th=480, tw=640):
    ds = MultiSourceDataset.__new__(MultiSourceDataset)
    ds.crop_p, ds.fov_p, ds.fov_zmin, ds.fov_zmax = crop_p, fov_p, 0.6, 0.95
    ds.th, ds.tw = th, tw
    return ds


def _backproject(j, i, d, K):
    fx, fy, cx, cy = K
    return np.stack([(j + 0.5 - cx) / fx * d, (i + 0.5 - cy) / fy * d, d], axis=-1)


# ── C3a ownar ───────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("hw,train", [((480, 853), True), ((700, 500), True), ((700, 500), False),
                                      ((400, 900), True), ((300, 500), False), ((640, 640), False)])
def test_ownar_backproject(hw, train):
    np.random.seed(0)
    ds = _stub()
    h, w = hw
    idx = np.arange(h * w, dtype=np.float64).reshape(h, w)
    depth = (1.0 + idx * 1e-5).astype(np.float32)               # unique value -> source pixel id
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    weight = np.ones((h, w), np.float32)
    pad = np.zeros((h, w), np.uint8)
    K_src = (0.9 * w, 0.9 * w, 0.52 * w, 0.47 * h)
    rgb2, dep2, w2, pad2, K_out, _ = ds._crop_or_pad_to(480, 640, rgb, depth, weight, pad, K_src, train)
    assert rgb2.shape == (480, 640, 3) and dep2.shape == (480, 640) and pad2.shape == (480, 640)
    real = pad2 == 0
    assert np.all(dep2[~real] == 0) and np.all(w2[~real] == 0)
    if (h < 480) or (w < 640):
        assert (~real).any()
    ii, jj = np.nonzero(real)
    d = dep2[ii, jj].astype(np.float64)
    src = np.rint((d - 1.0) / 1e-5).astype(np.int64)
    si, sj = src // w, src % w
    assert np.all(rgb2[ii, jj] == rgb[si, sj])                  # same pixel content
    p_out = _backproject(jj, ii, d, K_out)
    p_src = _backproject(sj, si, d, K_src)
    assert np.abs(p_out - p_src).max() < 1e-4


def _write_png16(p, depth_m):
    cv2.imwrite(p, np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))


def _make_tiny_corpus(tmp):
    """data_gt (stretched 640x480 own copies + nyu) and a v2 cache with aspect-true own rows."""
    out = os.path.join(tmp, "outputs")
    dg = os.path.join(out, "data_gt")
    cache = os.path.join(out, "v2", "cache", "own")
    for d in (dg, cache):
        os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(0)
    gt_rows, cache_rows, intr = [], [], {}
    own_sizes = {"o_wide": (480, 853), "o_tall": (967, 640), "o_narrow": (600, 400), "o_missing": None,
                 "o_sq": (640, 640)}
    for rid, cs in own_sizes.items():
        img = rng.randint(0, 255, (480, 640, 3)).astype(np.uint8)
        dep = rng.uniform(1.0, 5.0, (480, 640)).astype(np.float32)
        ip, dp = os.path.join(dg, rid + ".jpg"), os.path.join(dg, rid + ".png")
        cv2.imwrite(ip, img)
        _write_png16(dp, dep)
        kn = [0.8, 1.05, 0.5, 0.5]
        gt_rows.append(dict(id=rid, source="own", img=ip, depth=dp, conf="", is_gt="0", dmin="0.2",
                            dmax="30.0", fx=kn[0], fy=kn[1], cx=kn[2], cy=kn[3]))
        intr[rid] = kn
        if cs is None:
            continue
        hc, wc = cs
        cimg = rng.randint(0, 255, (hc, wc, 3)).astype(np.uint8)
        cdep = rng.uniform(1.0, 5.0, (hc, wc)).astype(np.float32)
        cip, cdp = os.path.join(cache, rid + ".jpg"), os.path.join(cache, rid + ".png")
        cv2.imwrite(cip, cimg)
        _write_png16(cdp, cdep)
        cache_rows.append(dict(id=rid, source="own", img=cip, depth=cdp, conf="", is_gt="0", dmin="0.2",
                               dmax="30.0", orig_w=wc * 2, orig_h=hc * 2, label_w=wc, label_h=hc,
                               cover_w=wc, cover_h=hc, scale=0.5))
    for n in range(3):
        img = rng.randint(0, 255, (480, 640, 3)).astype(np.uint8)
        dep = rng.uniform(1.0, 5.0, (480, 640)).astype(np.float32)
        ip, dp = os.path.join(dg, f"nyu_{n}.jpg"), os.path.join(dg, f"nyu_{n}.png")
        cv2.imwrite(ip, img)
        _write_png16(dp, dep)
        gt_rows.append(dict(id=f"nyu_{n}", source="nyu", img=ip, depth=dp, conf="", is_gt="1", dmin="0.2",
                            dmax="10.0", fx=0.81, fy=1.08, cx=0.5, cy=0.5))
    for path, rows in ((os.path.join(dg, "manifest.csv"), gt_rows),
                       (os.path.join(cache, "manifest.csv"), cache_rows)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with open(os.path.join(out, "intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(intr, f)
    cfg = {"data_gt": {"root": dg, "manifest": os.path.join(dg, "manifest.csv"), "val_frac": 0.03,
                       "seed": 1234},
           "student": {"train_size": [480, 640], "use_intrinsics": True, "depth_min_m": 0.2,
                       "depth_max_m": 30.0},
           "output": {"root": out},
           "distill": {"crop_context": {"p": 0.5, "min_scale": 0.5}, "range_weight_alpha": 1.0,
                       "fov_uncrop": {"p": 0.25, "zmin": 0.6, "zmax": 0.95}, "own_source": "v2cache",
                       "v2cache_manifest": os.path.join(cache, "manifest.csv"), "loss": {}}}
    return cfg


def test_ownar_dataset_pad_invalid(tmp_path):
    cfg = _make_tiny_corpus(str(tmp_path))
    for split, aug in (("train", True), ("val", False)):
        ds = MultiSourceDataset(cfg, split, augment=aug)
        assert ds.emit_pad
        np.random.seed(3)
        seen_pad = False
        for _ in range(6):
            for i in range(len(ds)):
                s = ds[i]
                assert s["image"].shape == (3, 480, 640) and s["pad"].shape == (1, 480, 640)
                assert s["pad"].dtype == torch.uint8
                padm = s["pad"].bool()
                assert not (s["valid"] & padm).any()               # pad is never valid
                assert float(s["weight"][padm].abs().sum()) == 0.0
                assert torch.isfinite(s["intrinsics"]).all()
                seen_pad |= bool(padm.any())
        if aug:
            assert seen_pad                                         # o_narrow (400 wide) and/or fov
    ds = MultiSourceDataset(cfg, "train", augment=True)
    assert ds.n_own_fallback + ds.n_own_cache == sum(1 for r in ds.rows if r["source"] == "own")


# ── C3b fov ─────────────────────────────────────────────────────────────────────────────────
def test_uncrop_backproject():
    np.random.seed(5)
    ds = _stub()
    H, W = 480, 640
    K = (0.81, 1.08, 0.51, 0.53)
    for _ in range(20):
        rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.full((H, W), 2.0, np.float32)
        weight = np.ones((H, W), np.float32)
        pad = np.zeros((H, W), np.uint8)
        rgb2, dep2, w2, pad2, K2, _ = ds._fov_uncrop(rgb, depth, weight, pad, K)
        real = (pad2 == 0)
        assert np.all(dep2[real] == 2.0) and np.all(dep2[~real] == 0) and np.all(w2[~real] == 0)
        assert (~real).any()

        def extent(mask, k):
            ii, jj = np.nonzero(mask)
            x = ((jj + 0.5) / W - k[2]) / k[0] * 2.0
            y = ((ii + 0.5) / H - k[3]) / k[1] * 2.0
            return np.array([x.min(), x.max(), y.min(), y.max()])

        e0 = extent(np.ones((H, W), bool), K)
        e1 = extent(real, K2)
        span = np.array([e0[1] - e0[0]] * 2 + [e0[3] - e0[2]] * 2)
        assert np.all(np.abs(e1 - e0) / span < 0.01), (e0, e1)


def test_s5_rate_unchanged():
    ds = _stub(crop_p=0.5, fov_p=0.25)
    np.random.seed(0)
    modes = [ds._draw_geom_mode() for _ in range(10000)]
    s5 = sum(m == "s5" for m in modes) / 1e4
    fov = sum(m == "fov" for m in modes) / 1e4
    assert abs(s5 - 0.50) <= 0.01 and abs(fov - 0.25) <= 0.01
    # without fov the draw is the legacy one, consuming the RNG stream identically
    ds0 = _stub(crop_p=0.5, fov_p=0.0)
    np.random.seed(1)
    a = [ds0._draw_geom_mode() == "s5" for _ in range(500)]
    np.random.seed(1)
    b = [bool(np.random.rand() < 0.5) for _ in range(500)]
    assert a == b
    ds_off = _stub(crop_p=0.0, fov_p=0.0)
    np.random.seed(2)
    st = np.random.get_state()[2]
    assert all(ds_off._draw_geom_mode() is None for _ in range(10))
    assert np.random.get_state()[2] == st                         # no RNG consumed when off


def test_egssi_pad():
    orig = load_orig("distill/losses.py", "orig_b10_losses")
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 1, 64, 80, generator=g)
    tgt = 1.0 + 0.2 * torch.randn(2, 1, 64, 80, generator=g)
    mask = torch.rand(2, 1, 64, 80, generator=g) > 0.2
    w = mask.float() * torch.rand(2, 1, 64, 80, generator=g)
    for m in (mask, w):
        ref = orig.eg_ssi_loss(pred, tgt, m)
        assert torch.equal(eg_ssi_loss(pred, tgt, m), ref)
        assert torch.equal(eg_ssi_loss(pred, tgt, m, pad=None), ref)
        assert torch.equal(eg_ssi_loss(pred, tgt, m, pad=torch.zeros_like(tgt)), ref)
    # a zero-filled pad band creates a huge target edge; with pad given it no longer sets emax
    padm = torch.zeros(2, 1, 64, 80)
    padm[..., :, :12] = 1
    t2 = tgt * (1 - padm)
    m2 = mask & (padm == 0)
    with_pad = eg_ssi_loss(pred, t2, m2, pad=padm)
    without = eg_ssi_loss(pred, t2, m2)
    clean = eg_ssi_loss(pred, tgt, m2)          # same interior, no artificial border
    assert abs(float(with_pad) - float(clean)) < abs(float(without) - float(clean))


# ── real data (read-only) ───────────────────────────────────────────────────────────────────
def test_ownar_real_cache_rows():
    from common.io_utils import load_config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "config.yaml"))
    man = os.path.join(OUTPUTS, "v2", "cache", "own", "manifest.csv")
    if not (os.path.exists(cfg["data_gt"]["manifest"]) and os.path.exists(man)):
        pytest.skip("real corpus not available")
    cfg["student"]["use_intrinsics"] = True
    cfg["distill"].update({"crop_context": {"p": 0.5, "min_scale": 0.5}, "own_source": "v2cache",
                           "v2cache_manifest": man,
                           "fov_uncrop": {"p": 0.25, "zmin": 0.6, "zmax": 0.95}})
    ds = MultiSourceDataset(cfg, "train", augment=True)
    assert ds.n_own_cache > 7000 and ds.n_own_fallback <= 70
    own_idx = [i for i, r in enumerate(ds.rows) if r["source"] == "own"]
    np.random.seed(0)
    for i in own_idx[:6]:
        s = ds[i]
        assert s["image"].shape == (3, 480, 640)
        assert not (s["valid"] & s["pad"].bool()).any()
        fx, fy, cx, cy = [float(v) for v in s["intrinsics"]]
        assert fx > 0 and fy > 0 and -0.5 < cx < 1.5 and -0.5 < cy < 1.5
    # aspect-true cache rows keep an isotropic pixel focal and a plausible cstm c
    from distill.student import _cstm_term
    cached = [i for i in own_idx if ds.rows[i]["id"] in ds.v2cache]
    rng = np.random.RandomState(0)
    ratios, cs = [], []
    for i in rng.choice(cached, 60, replace=False):
        k = ds[int(i)]["intrinsics"]
        ratios.append(float(k[0] * 640 / (k[1] * 480)))
        cs.append(float(_cstm_term(k[None].double(), 480, 640, 518.9)))
    assert 0.95 < np.median(ratios) < 1.05
    assert np.all(np.abs(np.array(cs)) < 1.5)


# ── end-to-end K consistency of the whole pipeline on a tilted plane (folded from reviewer
# scratch). Every own cache / data_gt row stores the depth of ONE plane n.X = 1 rendered with its
# own K; after ownar crop/pad -> fov uncrop or s5 crop+resize -> flip, back-projecting each valid
# output pixel with the OUTPUT intrinsics must land on the same plane (or its mirror if flipped).
_PLANE_N = np.array([0.15, -0.10, 0.30])


def _plane_depth(h, w, K_px):
    fx, fy, cx, cy = K_px
    jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    den = _PLANE_N[0] * (jj - cx) / fx + _PLANE_N[1] * (ii - cy) / fy + _PLANE_N[2]
    return (1.0 / den).astype(np.float32)


def _plane_png16(p, d):
    cv2.imwrite(p, np.clip(np.rint(d * 1000.0), 0, 65535).astype(np.uint16))


def _plane_corpus(tmp):
    out = os.path.join(tmp, "outputs")
    dg = os.path.join(out, "data_gt")
    cache = os.path.join(out, "v2", "cache", "own")
    os.makedirs(dg, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    gt_rows, cache_rows, intr = [], [], {}
    sizes = {"o_wide": (480, 853), "o_tall": (853, 640), "o_sq": (640, 640), "o_narrow": (600, 400),
             "o_fallback": None}
    for rid, cs in sizes.items():
        kn = [0.78, 1.02, 0.53, 0.47]
        # stretched data_gt copy at 640x480 with the same normalised K
        Kd = (kn[0] * 640, kn[1] * 480, kn[2] * 640, kn[3] * 480)
        ip, dp = os.path.join(dg, rid + ".jpg"), os.path.join(dg, rid + ".png")
        cv2.imwrite(ip, np.full((480, 640, 3), 128, np.uint8))
        _plane_png16(dp, _plane_depth(480, 640, Kd))
        gt_rows.append(dict(id=rid, source="own", img=ip, depth=dp, conf="", is_gt="0", dmin="0.2",
                            dmax="30.0", fx=kn[0], fy=kn[1], cx=kn[2], cy=kn[3]))
        intr[rid] = kn
        if cs is None:
            continue
        hc, wc = cs
        Kc = (kn[0] * wc, kn[1] * hc, kn[2] * wc, kn[3] * hc)
        cip, cdp = os.path.join(cache, rid + ".jpg"), os.path.join(cache, rid + ".png")
        cv2.imwrite(cip, np.full((hc, wc, 3), 128, np.uint8))
        _plane_png16(cdp, _plane_depth(hc, wc, Kc))
        cache_rows.append(dict(id=rid, source="own", img=cip, depth=cdp, conf="", is_gt="0",
                               dmin="0.2", dmax="30.0", orig_w=wc, orig_h=hc, label_w=wc,
                               label_h=hc, cover_w=wc, cover_h=hc, scale=1.0))
    for n in range(2):
        kn = [0.81, 1.08, 0.51, 0.53]
        Kd = (kn[0] * 640, kn[1] * 480, kn[2] * 640, kn[3] * 480)
        ip, dp = os.path.join(dg, f"nyu_{n}.jpg"), os.path.join(dg, f"nyu_{n}.png")
        cv2.imwrite(ip, np.full((480, 640, 3), 128, np.uint8))
        _plane_png16(dp, _plane_depth(480, 640, Kd))
        gt_rows.append(dict(id=f"nyu_{n}", source="nyu", img=ip, depth=dp, conf="", is_gt="1",
                            dmin="0.2", dmax="10.0", fx=kn[0], fy=kn[1], cx=kn[2], cy=kn[3]))
    for path, rows in ((os.path.join(dg, "manifest.csv"), gt_rows),
                       (os.path.join(cache, "manifest.csv"), cache_rows)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with open(os.path.join(out, "intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(intr, f)
    return {"data_gt": {"root": dg, "manifest": os.path.join(dg, "manifest.csv"),
                        "val_frac": 0.03, "seed": 1234},
            "student": {"train_size": [480, 640], "use_intrinsics": True, "depth_min_m": 0.2,
                        "depth_max_m": 30.0},
            "output": {"root": out},
            "distill": {"crop_context": {"p": 0.5, "min_scale": 0.5}, "range_weight_alpha": 1.0,
                        "fov_uncrop": {"p": 0.25, "zmin": 0.6, "zmax": 0.95},
                        "own_source": "v2cache",
                        "v2cache_manifest": os.path.join(cache, "manifest.csv"), "loss": {}}}


def _plane_residual(s, mirror):
    d = s["depth_m"][0].numpy().astype(np.float64)
    v = s["valid"][0].numpy()
    fx, fy, cx, cy = [float(t) for t in s["intrinsics"]]
    H, W = d.shape
    ii, jj = np.nonzero(v)
    x = ((jj + 0.5) / W - cx) / fx * d[ii, jj]
    y = ((ii + 0.5) / H - cy) / fy * d[ii, jj]
    z = d[ii, jj]
    n0 = -_PLANE_N[0] if mirror else _PLANE_N[0]
    r = n0 * x + _PLANE_N[1] * y + _PLANE_N[2] * z - 1.0
    return np.abs(r)


def test_pipeline_plane_consistency(tmp_path):
    cfg = _plane_corpus(str(tmp_path))
    worst = {}
    for split, aug in (("train", True), ("val", False)):
        ds = MultiSourceDataset(cfg, split, augment=aug)
        np.random.seed(0)
        for rep in range(40 if aug else 1):
            for i in range(len(ds)):
                s = ds[i]
                r = min(np.percentile(_plane_residual(s, False), 99), np.percentile(_plane_residual(s, True), 99))
                key = (split, ds.rows[i]["id"])
                worst[key] = max(worst.get(key, 0.0), float(r))
    for k, v in sorted(worst.items()):
        print(k, f"p99 |n.X-1| = {v:.2e}")
    assert max(worst.values()) < 5e-3, worst


def test_without_use_intrinsics_k_never_reaches_model(tmp_path):
    """Why distill/train.py asserts --use-intrinsics for cstm/dsh/fov/ownar: without it the
    dataset applies the fov / ownar geometry (pad emitted) but drops the updated K."""
    cfg = _plane_corpus(str(tmp_path))
    cfg["student"]["use_intrinsics"] = False
    ds = MultiSourceDataset(cfg, "train", augment=True)
    np.random.seed(0)
    s = ds[0]
    assert "pad" in s and "intrinsics" not in s


# ── real-data b20 batches through Windows DataLoader workers (folded from reviewer scratch) ──
def _b20_real_cfg():
    import copy
    from distill.train import CSTM_FC_PX, FOV_UNCROP
    from _orig import CKPTS
    p = os.path.join(CKPTS, "b10", "best.pt")
    if not os.path.exists(p):
        pytest.skip("no b10 ckpt")
    cfg = copy.deepcopy(torch.load(p, map_location="cpu", weights_only=False)["cfg"])
    cfg["student"].update(pretrained_backbone=False, scale_head=True, cstm_fc=CSTM_FC_PX, rep4=True)
    d = cfg["distill"]
    d["fov_uncrop"] = dict(FOV_UNCROP)
    d["own_source"] = "v2cache"
    d["v2cache_manifest"] = os.path.join(cfg["output"]["root"], "v2", "cache", "own",
                                         "manifest.csv").replace("\\", "/")
    if not os.path.exists(d["v2cache_manifest"]):
        pytest.skip("no v2 own cache")
    return cfg


def test_b20_dataloader_workers_windows():
    from torch.utils.data import DataLoader, Subset
    cfg = _b20_real_cfg()
    ds = MultiSourceDataset(cfg, "train", augment=True)
    own_c = [i for i, r in enumerate(ds.rows) if r["source"] == "own" and r["id"] in ds.v2cache][:6]
    own_f = [i for i, r in enumerate(ds.rows) if r["source"] == "own" and r["id"] not in ds.v2cache][:2]
    other = [i for i, r in enumerate(ds.rows) if r["source"] != "own"][:6]
    sub = Subset(ds, own_c + own_f + other)
    ld = DataLoader(sub, batch_size=4, shuffle=False, num_workers=2, drop_last=False,
                    persistent_workers=False)
    n = 0
    for batch in ld:
        n += batch["image"].shape[0]
        assert "pad" in batch and batch["pad"].dtype == torch.uint8
        assert batch["pad"].shape == batch["valid"].shape
        assert not (batch["valid"] & (batch["pad"] > 0)).any()
        for k in ("image", "log_depth", "depth_m", "weight", "intrinsics"):
            assert torch.isfinite(batch[k]).all(), k
    assert n == len(sub)
