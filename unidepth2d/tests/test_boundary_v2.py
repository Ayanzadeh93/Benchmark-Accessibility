"""A04: corrected boundary F1 in benchmark/run_bench.py (--boundary).

Unit tests are CPU/numpy only. The last test re-runs run_bench on IBims-1 for the b20dws
checkpoint into a tmp dir and checks the LEGACY key against the stored results_bench JSON;
it is skipped (never faked) when the checkpoint, the stored JSON or the IBims data is missing.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark import protocol, protocol_v2 as pv2  # noqa: E402
from benchmark.run_bench import BF1_LEGACY_NOTE, boundary_row, boundary_summary  # noqa: E402

H, W = 48, 64


def _step_gt(col=32, near=1.0, far=3.0):
    gt = np.full((H, W), near, np.float64)
    gt[:, col:] = far
    return gt


def test_no_edge_prediction_scores_zero_and_is_counted():
    gt = _step_gt()
    valid = np.ones((H, W), bool)
    flat = np.full((H, W), 2.0)                         # prediction without a single edge
    assert protocol.boundary_f1_si(gt, flat, valid) is None    # legacy drops the image
    row = boundary_row("img0", gt, flat, valid)
    assert row["id"] == "img0"
    assert row["bf1_si_legacy"] is None
    assert row[pv2.BF_KEY] == 0.0                       # v2: scored 0, not None
    assert row["bf1_depthpro_masked"] == 0.0
    assert row["bf1_depthpro_nomask"] == 0.0
    agg = boundary_summary([row])["aggregate"]
    assert agg[pv2.BF_KEY] == {"bf1": 0.0, "n_images": 1, "n_contributing": 1,
                               "n_no_gt_boundary": 0}
    assert agg["bf1_si_legacy"]["n_contributing"] == 0 and agg["bf1_si_legacy"]["bf1"] is None


def test_n_contributing_equals_image_count_and_rows_have_ids():
    valid = np.ones((H, W), bool)
    rows = []
    for i in range(5):
        gt = _step_gt(col=10 + 8 * i)
        pred = gt if i % 2 == 0 else np.full((H, W), 1.5)  # perfect / edgeless predictions
        rows.append(boundary_row(f"id{i}", gt, pred, valid))
    s = boundary_summary(rows)
    assert [r["id"] for r in s["per_image"]] == [f"id{i}" for i in range(5)]
    for key in (pv2.BF_KEY, "bf1_depthpro_masked", "bf1_depthpro_nomask"):
        assert s["aggregate"][key]["n_contributing"] == 5 == s["aggregate"][key]["n_images"]
    # perfect predictions score 1, edgeless 0 -> v2 mean 3/5; legacy only sees the perfect ones
    assert s["aggregate"][pv2.BF_KEY]["bf1"] == pytest.approx(0.6)
    assert s["aggregate"]["bf1_si_legacy"]["bf1"] == pytest.approx(1.0)
    assert s["aggregate"]["bf1_si_legacy"]["n_contributing"] == 3
    assert "LEGACY" in s["definitions"]["bf1_si_legacy"]


def test_rows_match_protocol_functions_exactly():
    rng = np.random.default_rng(0)
    gt = np.exp(rng.normal(0.5, 0.4, (H, W)))
    pred = np.exp(rng.normal(0.5, 0.4, (H, W)))
    valid = rng.random((H, W)) > 0.2
    row = boundary_row(7, gt, pred, valid)
    assert row["id"] == "7"
    assert row["bf1_si_legacy"] == protocol.boundary_f1_si(gt, pred, valid)
    assert row[pv2.BF_KEY] == pv2.boundary_f1_si_v2(gt, pred, valid, both_empty="zero")
    assert row["bf1_depthpro_masked"] == pv2.si_boundary_f1_depthpro(pred, gt, valid=valid)
    assert row["bf1_depthpro_nomask"] == pv2.si_boundary_f1_depthpro(pred, gt, valid=None)


def test_no_gt_boundary_image_is_excluded_and_counted():
    valid = np.ones((H, W), bool)
    flat_gt = np.full((H, W), 2.0)
    row = boundary_row("flat", flat_gt, _step_gt(), valid)
    assert row[pv2.BF_KEY] is None
    agg = boundary_summary([row, boundary_row("ok", _step_gt(), _step_gt(), valid)])["aggregate"]
    assert agg[pv2.BF_KEY]["n_contributing"] == 1 and agg[pv2.BF_KEY]["n_no_gt_boundary"] == 1


STORED = os.path.join(ROOT, "results_bench", "ibims__student-b20dws_last.json")
# results_bench names the row after the checkpoint's parent dir: student-b20dws_last
CKPT = os.path.join(ROOT, "outputs", "checkpoints", "b20dws_last", "best.pt")
ORIG_BRANCH = "b20-canonscale"        # the original (pre-A04/A05) code
# The stored value (2026-11-23) is not bit-reproducible on today's stack even with the ORIGINAL
# code: 2026-11-25 re-runs give 0.0318815 (cuda) / 0.0317619 (cpu) for both old and new code.
# The legacy metric thresholds per-pixel ratios, so 1e-5-level depth drift moves it by ~1e-4.
STORED_TOL = 1.5e-4


def _ibims_ok():
    from benchmark.datasets_tier1 import iter_ibims
    try:
        next(iter(iter_ibims(limit=1)))
        return None
    except (OSError, FileNotFoundError, StopIteration, ImportError) as e:
        return str(e)


def _run_original(tmp_path, ckpt, device):
    """run_bench from the ORIGINAL branch (git archive -> tmp) on the same device."""
    import io
    import subprocess
    import tarfile
    try:
        tar = subprocess.run(["git", "-C", ROOT, "archive", "--format=tar", ORIG_BRANCH,
                              "benchmark", "distill", "common", "edge"],
                             check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"original branch {ORIG_BRANCH} unavailable: {e}")
    src = tmp_path / "orig"
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(src, filter="data")
    subprocess.run([sys.executable, "-m", "benchmark.run_bench", "--dataset", "ibims",
                    "--boundary", "--model", f"student:{ckpt}"],
                   cwd=src, check=True, capture_output=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    with open(src / "results_bench" / "ibims__student-b20dws_last.json", encoding="utf-8") as f:
        return json.load(f)


def test_legacy_value_matches_stored_b20dws_json(tmp_path, monkeypatch):
    if not os.path.isfile(STORED):
        pytest.skip(f"stored result missing: {STORED}")
    if not os.path.isfile(CKPT):
        pytest.skip(f"checkpoint missing: {CKPT}")
    err = _ibims_ok()
    if err:
        pytest.skip(f"IBims-1 unavailable: {err}")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(STORED, encoding="utf-8") as f:
        ref = json.load(f)
    assert abs(ref["boundary_f1_si"] - 0.03185) < 5e-6

    from benchmark import run_bench
    new_dir = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["run_bench", "--dataset", "ibims", "--boundary",
                                      "--device", device, "--model", f"student:{CKPT}",
                                      "--out-dir", str(new_dir)])
    run_bench.main()
    out = json.load(open(new_dir / "ibims__student-b20dws_last.json", encoding="utf-8"))
    assert out["num_images"] == ref["num_images"] == 100
    assert out["boundary_f1_si_legacy_note"] == BF1_LEGACY_NOTE
    # (1) legacy key == the stored results_bench value (to the documented stack drift)
    assert abs(out["boundary_f1_si"] - ref["boundary_f1_si"]) < STORED_TOL
    # (2) legacy key == the ORIGINAL code on the same device, bit-for-bit
    orig = _run_original(tmp_path, CKPT, device)
    assert out["boundary_f1_si"] == orig["boundary_f1_si"]
    assert out["depth"] == orig["depth"]
    # (3) additive keys are consistent with the legacy key and count every image
    agg = out["boundary_f1"]["aggregate"]
    assert agg["bf1_si_legacy"]["bf1"] == out["boundary_f1_si"]
    assert agg[pv2.BF_KEY]["n_images"] == 100
    assert len(out["boundary_f1"]["per_image"]) == 100
    assert len({r["id"] for r in out["boundary_f1"]["per_image"]}) == 100
    assert json.load(open(STORED, encoding="utf-8")) == ref      # stored file untouched
