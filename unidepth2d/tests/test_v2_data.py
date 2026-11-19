"""distill/v2/data.py and configs/v2 rung configs (plan 9.5 test_v2_data: aspect, crop bounds,
role disjointness, blacklist, leak blocker). CPU, offline, synthetic files only."""
import csv
import os

import cv2
import numpy as np
import pytest
import torch

from distill.v2 import data as dm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_row(root, src, rid, h, w, depth_m, is_gt, dmax, k=None):
    for sub in ("images", "depth_mm"):
        os.makedirs(os.path.join(root, src, sub), exist_ok=True)
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([(xx * 255 // max(1, w - 1)), (yy * 255 // max(1, h - 1)), np.full_like(xx, 128)], -1)
    img = os.path.join(root, src, "images", rid + ".jpg")
    dep = os.path.join(root, src, "depth_mm", rid + ".png")
    cv2.imwrite(img, cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(dep, np.round(depth_m * 1000).astype(np.uint16))
    k = k or ("", "", "", "")
    return {"id": rid, "source": src, "img": img, "depth": dep, "conf": "", "is_gt": str(int(is_gt)),
            "dmin": "0.2", "dmax": str(dmax), "fx": k[0], "fy": k[1], "cx": k[2], "cy": k[3]}


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def world(tmp_path):
    """data_gt-like manifest (own/nyu/sun at 480x640) + an own cover cache at 480x853."""
    gt = tmp_path / "data_gt"
    rows = []
    ramp = np.tile(np.linspace(0.5, 6.0, 640, dtype=np.float32), (480, 1))
    for i in range(4):
        rows.append(_write_row(str(gt), "own", f"own{i}", 480, 640, ramp, 0, 30.0, (0.8, 1.0, 0.5, 0.5)))
        rows.append(_write_row(str(gt), "nyu", f"nyu{i}", 480, 640, ramp, 1, 10.0, (0.8, 1.0, 0.5, 0.5)))
        rows.append(_write_row(str(gt), "sun", f"sun{i}", 480, 640, ramp * 1.5, 1, 8.0))   # 0.75-9 m
    fields = ["id", "source", "img", "depth", "conf", "is_gt", "dmin", "dmax", "fx", "fy", "cx", "cy"]
    _write_csv(str(gt / "manifest.csv"), rows, fields)
    cache = tmp_path / "cache"
    crows = []
    cramp = np.tile(np.linspace(0.5, 6.0, 853, dtype=np.float32), (480, 1))
    for i in range(4):
        crows.append(_write_row(str(cache), "own", f"own{i}", 480, 853, cramp, 0, 30.0))
    _write_csv(str(cache / "own" / "manifest.csv"), crows, fields[:8])
    roles = {
        "frozen": True,
        "paths": {"manifest": str(gt / "manifest.csv"), "cover_cache": str(cache)},
        "train": {"own": ["own0", "own1", "own2"], "nyu": ["nyu0", "nyu1", "nyu2"],
                  "pool": {"status": "not built", "by_pool": {}}},
        "val": {"own": [], "nyu": ["nyu3"]},
        "dev": {"d_sun": ["sun0", "sun1", "sun2"], "d_sun_clusters": []},
        "test": {"frozen_own": ["own3"], "video_own": []},
        "calibration": {"own": ["own0"]},
        "sf6_temporal": {"own_ids": []},
        "blacklist": {"sun_8x": ["sun3"]},
        "nyu_family_own": {"ids": []},
        "excluded": {"own": {}, "nyu": {}},
    }
    return roles


def test_splits_follow_roles_and_blacklist(world):
    tr = dm.V2Dataset(world, "train", {})
    assert sorted(tr.ids) == ["nyu0", "nyu1", "nyu2", "own0", "own1", "own2"]
    dev = dm.V2Dataset(world, "dev", {})
    assert dev.ids == ["sun0", "sun1", "sun2"] and "sun3" not in dev.ids     # 8x row blacklisted
    assert dm.V2Dataset(world, "frozen_own", {}).ids == ["own3"]


@pytest.mark.parametrize("held", ["dev.d_sun", "test.frozen_own", "sf6_temporal.own_ids",
                                  "blacklist.sun_8x", "nyu_family_own"])
def test_leak_blocker_refuses_overlap(world, held):
    sec, _, key = held.partition(".")
    if sec == "nyu_family_own":
        world["nyu_family_own"]["ids"] = ["own1"]
    else:
        world[sec][key] = list(world[sec][key]) + ["own1"]
    with pytest.raises(dm.LeakError):
        dm.V2Dataset(world, "train", {})


def test_leak_blocker_excluded_ids(world):
    world["excluded"]["nyu"] = {"nyu2": "a3 hit vs nyu_labelled"}
    with pytest.raises(dm.LeakError):
        dm.V2Dataset(world, "train", {})


def test_unfrozen_roles_and_unknown_keys_refused(world):
    with pytest.raises(ValueError):
        dm.V2Dataset(world, "train", {"zoom": 0.5})
    with pytest.raises(ValueError):
        dm.V2Dataset(world, "dev", {"augment": True})
    world["frozen"] = False
    with pytest.raises(ValueError):
        dm.V2Dataset(world, "train", {})


def test_item_contract_and_validity(world):
    ds = dm.V2Dataset(world, "dev", {"intrinsics": True})
    it = ds[0]
    assert it["image"].shape == (3, 480, 640) and it["image"].dtype == torch.float32
    assert it["target_log"].shape == (1, 480, 640) and it["valid"].dtype == torch.bool
    assert it["is_gt"] is True and it["dmax"] == 8.0 and it["source"] == "sun"
    d = torch.exp(it["target_log"][it["valid"]])
    assert float(d.min()) >= 0.2 - 1e-3 and float(d.max()) <= 8.0 + 1e-3
    assert not bool(it["valid"][0, :, -1].any())                             # 9 m > dmax 8: masked
    assert bool(it["valid"][0, :, 0].all())
    assert torch.allclose(it["intrinsics"], torch.tensor(list(dm.DEFAULT_INTR)))   # no K -> default


def test_cover_eval_is_centre_crop_and_keeps_aspect(world):
    ds = dm.V2Dataset(world, "frozen_own", {"pipeline": "cover", "intrinsics": True})
    it = ds[0]
    raw_rgb, raw_d, _ = ds.load_raw(0)
    assert raw_d.shape == (480, 853)                                         # 16:9 cover image
    x0 = (853 - 640) // 2
    exp = np.log(raw_d[:, x0:x0 + 640])
    np.testing.assert_allclose(it["target_log"][0].numpy(), exp, atol=1e-6)  # no resize: aspect kept
    fx, fy, cx, cy = it["intrinsics"].tolist()
    assert fx == pytest.approx(0.8 * 853 / 640, rel=1e-6) and fy == pytest.approx(1.0)
    assert cx == pytest.approx((0.5 * 853 - x0) / 640, rel=1e-6)


def test_cover_train_crop_bounds(world):
    ds = dm.V2Dataset(world, "train", {"pipeline": "cover", "zoom_crop": {"p": 0.0}, "flip_p": 0.0,
                                       "jitter": {"p": 0.0}, "sources": ["own"]})
    _, raw_d, _ = ds.load_raw(0)
    full = np.log(raw_d[0])                                                   # depth ramps along x
    np.random.seed(0)
    starts = set()
    for _ in range(40):
        row = ds[0]["target_log"][0, 0].numpy()
        x0 = int(np.argmin(np.abs(full - row[0])))
        np.testing.assert_allclose(row, full[x0:x0 + 640], atol=1e-6)        # a real window
        assert 0 <= x0 <= 853 - 640
        starts.add(x0)
    assert len(starts) > 5                                                     # actually random


def test_zoom_crop_and_flip_update_intrinsics():
    k = dm.crop_intrinsics([0.8, 1.0, 0.5, 0.5], 64, 48, 320, 240, 640, 480)
    assert k == pytest.approx([1.6, 2.0, 0.8, 0.8])                          # (.5 - .1) / .5


def test_squash_flip_mirrors_cx(world):
    ds = dm.V2Dataset(world, "train", {"intrinsics": True, "zoom_crop": {"p": 0.0}, "flip_p": 1.0,
                                       "jitter": {"p": 0.0}, "sources": ["nyu"]})
    it = ds[0]
    assert it["intrinsics"].tolist() == pytest.approx([0.8, 1.0, 0.5, 0.5])
    world["train"]["nyu"] = ["nyu0"]
    _, raw_d, _ = ds.load_raw(0)
    assert float(it["target_log"][0, 0, 0]) == pytest.approx(float(np.log(raw_d[0, -1])), abs=1e-6)


def test_strong_photometric_only_on_pseudo_rows(world):
    ds = dm.V2Dataset(world, "train", {"photometric": "strong", "zoom_crop": {"p": 0.0}, "flip_p": 0.0,
                                       "jitter": {"p": 0.0}, "sources": ["nyu"]})
    ref = dm.V2Dataset(world, "train", {"zoom_crop": {"p": 0.0}, "flip_p": 0.0, "jitter": {"p": 0.0},
                                        "sources": ["nyu"]})
    np.random.seed(1)
    assert torch.equal(ds[0]["image"], ref[0]["image"])                      # GT rows untouched
    np.random.seed(1)
    out = dm.strong_photometric(np.full((8, 8, 3), 100, np.uint8))
    assert out.dtype == np.uint8 and out.shape == (8, 8, 3)


def test_sampler_weights_groups_and_determinism(world):
    ds = dm.V2Dataset(world, "train", {})
    a = dm.make_sampler(ds, {"own": 0.5, "nyu": 0.5}, 4000, seed=7)
    assert a == dm.make_sampler(ds, {"own": 0.5, "nyu": 0.5}, 4000, seed=7)
    assert a != dm.make_sampler(ds, {"own": 0.5, "nyu": 0.5}, 4000, seed=8)
    frac = np.mean([ds.sources[i] == "own" for i in a])
    assert 0.46 < frac < 0.54 and len(a) == 4000
    assert {ds.sources[i] for i in dm.make_sampler(ds, {"own": 1.0, "nyu": 0.0}, 50, 0)} == {"own"}
    with pytest.raises(ValueError, match="x_indoor"):
        dm.make_sampler(ds, {"own+x_indoor": 0.5, "nyu": 0.5}, 10, 0)


def test_cover_cache_missing_is_an_error(world, tmp_path):
    world["paths"]["cover_cache"] = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="build_cover_cache"):
        dm.V2Dataset(world, "train", {"pipeline": "cover"})


# ── rung configs ─────────────────────────────────────────────────────────────

TIER1 = ["R0", "R1a", "R1b", "R2", "R3", "R4", "R4d", "R5a-cover", "R5a", "R5b", "R5c", "K1", "C0", "C1"]


def test_tier1_configs_resolve_and_obey_one_factor_rule():
    tr = pytest.importorskip("distill.v2.train")
    d = os.path.join(ROOT, "configs", "v2", "tier1")
    cfgs = {r: tr.load_config(os.path.join(d, f"{r}.yaml")) for r in TIER1}
    for r, c in cfgs.items():
        assert c["rung"] == r
        tr.validate_config(c)
        if c["record_type"] == "ablation":
            res = tr.check_one_factor(cfgs[c["parent"]], c)
            assert res["ok"], (r, res["problems"])
        dm.resolve_config(c["data"]["dataset"], "train")
        dm.resolve_config(c["eval"]["dataset"], c["eval"]["split"])
    assert cfgs["R5a"]["data"]["dataset"]["zoom_crop"]["p"] == 0.0
    assert cfgs["R5a-cover"]["data"]["dataset"]["pipeline"] == "cover"
    assert cfgs["R0"]["data"]["draw_weights"] == {"own": 0.5, "nyu": 0.5}
