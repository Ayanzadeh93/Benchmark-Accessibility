"""Bit-parity golden for the eval path (A04/A05 guard): capture today's outputs, then
assert that later code reproduces them exactly under the default flags.

  python -m scripts.parity_golden capture --out <DIR>
  python -m scripts.parity_golden compare --ref <DIR>

What is captured, per student checkpoint (outputs/checkpoints/{b10,b20core,b20dws,b1_mix}/last.pt,
built exactly as benchmark/run_bench.py does via StudentModelRunner, incl. rep4 fusion):
  * inputs: 4 seeded random 480x640 uint8 RGB images (K = NYU GT K) and up to 4 IBims-1
    images from benchmark.datasets_tier1.iter_ibims (K = the image's GT calib);
  * each input twice: with GT K and with K=None (intrinsics=None);
  * the raw model.forward output (all channels) and the runner.predict depth, with sha256;
  * on IBims: benchmark metrics on the run_bench-resized prediction
    (protocol.per_image_depth_metrics, protocol.boundary_f1_si legacy, protocol.aggregate).

Nothing outside --out is written. Default device is CPU with a fixed thread count so the
capture is reproducible bit-for-bit; compare re-uses the device/threads stored in the ref.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark import protocol  # noqa: E402
from benchmark.datasets_tier1 import NYU_K, iter_ibims  # noqa: E402
from benchmark.run_bench import StudentModelRunner  # noqa: E402

CKPTS = ("b10", "b20core", "b20dws", "b1_mix")
N_RANDOM, N_IBIMS = 4, 4
RAND_HW = (480, 640)
RAND_SEED = 20261125


def sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def inputs():
    """[(id, rgb uint8 HxWx3, K 3x3, sample-or-None)] - deterministic."""
    out = []
    for i in range(N_RANDOM):
        rng = np.random.default_rng(RAND_SEED + i)
        rgb = rng.integers(0, 256, size=(*RAND_HW, 3), dtype=np.uint8)
        out.append((f"rand{i}", rgb, np.asarray(NYU_K, dtype=np.float64), None))
    try:
        samples = list(iter_ibims(limit=N_IBIMS))
    except (OSError, FileNotFoundError, ImportError) as e:  # data missing -> reported, never faked
        samples = []
        print(f"[parity] IBims unavailable ({type(e).__name__}: {e}); random inputs only")
    for s in samples:
        out.append((f"ibims:{s.id}", s.rgb, s.K, s))
    return out


def run_one(runner: StudentModelRunner, rgb, K):
    """runner.predict(rgb, K) while recording the raw forward (input, intrinsics, output)."""
    rec = {}
    orig = runner.model.forward

    def wrapped(x, intrinsics=None, *a, **kw):
        y = orig(x, intrinsics, *a, **kw)
        rec["x"] = x.detach().float().cpu().numpy()
        rec["k"] = None if intrinsics is None else intrinsics.detach().float().cpu().numpy()
        rec["fwd"] = y.detach().float().cpu().numpy()
        return y

    runner.model.forward = wrapped
    try:
        depth = runner.predict(rgb, K)["depth"]
    finally:
        del runner.model.forward          # restore the class method
    return depth, rec


def metrics(sample, depth):
    gt, valid = sample.gt, sample.valid
    h, w = gt.shape
    pred = depth
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    row = protocol.per_image_depth_metrics(gt, pred, valid, max_depth=sample.max_depth)
    bf1 = protocol.boundary_f1_si(gt, pred, valid)
    return {"pred_resized_sha": sha(pred), "depth": row, "boundary_f1_si_legacy": bf1}


def collect(device: str):
    arrays, table, metric_rows = {}, {}, {}
    ins = inputs()
    for ck in CKPTS:
        path = os.path.join(ROOT, "outputs", "checkpoints", ck, "last.pt")
        if not os.path.isfile(path):
            print(f"[parity] MISSING checkpoint {path}; skipped")
            continue
        torch.manual_seed(0)
        runner = StudentModelRunner(path, device)
        agg = {"GTK": [], "noK": []}
        for iid, rgb, K, sample in ins:
            for mode, kk in (("GTK", K), ("noK", None)):
                depth, rec = run_one(runner, rgb, kk)
                key = f"{ck}|{iid}|{mode}"
                arrays[key + "|depth"] = depth
                arrays[key + "|fwd"] = rec["fwd"]
                table[key + "|depth"] = sha(depth)
                table[key + "|fwd"] = sha(rec["fwd"])
                table[key + "|input"] = sha(rec["x"])
                table[key + "|intr"] = "None" if rec["k"] is None else sha(rec["k"])
                if sample is not None:
                    m = metrics(sample, depth)
                    metric_rows[key] = m
                    if m["depth"] is not None:
                        agg[mode].append(m["depth"])
        for mode, rows in agg.items():
            if rows:
                metric_rows[f"{ck}|ibims-agg|{mode}"] = {"depth": protocol.aggregate(rows)}
        del runner
    return arrays, table, metric_rows


def _setup(device: str, threads: int):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    if device.startswith("cuda"):
        torch.backends.cudnn.deterministic = True


def capture(out: str, device: str, threads: int):
    _setup(device, threads)
    os.makedirs(out, exist_ok=True)
    arrays, table, mrows = collect(device)
    np.savez_compressed(os.path.join(out, "outputs.npz"),
                        **{k.replace("|", "__").replace(":", "~"): v for k, v in arrays.items()})
    manifest = {"device": device, "threads": threads, "torch": torch.__version__,
                "numpy": np.__version__, "ckpts": list(CKPTS), "sha256": table,
                "metrics": mrows}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    n_ib = len({k.split("|")[1] for k in table if "|ibims:" in k})
    print(f"[parity] captured {len(table)} hashes, {len(mrows)} metric rows "
          f"({n_ib} IBims images) -> {out}")


def compare(ref: str, device: str | None, threads: int | None) -> int:
    with open(os.path.join(ref, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    device = device or man["device"]
    threads = threads or man["threads"]
    _setup(device, threads)
    ref_arr = np.load(os.path.join(ref, "outputs.npz"))
    arrays, table, mrows = collect(device)

    fails, rows = 0, []
    for key in sorted(set(man["sha256"]) | set(table)):
        a, b = man["sha256"].get(key), table.get(key)
        ok = a == b
        if ok and (key.endswith("|depth") or key.endswith("|fwd")):
            ok = np.array_equal(ref_arr[key.replace("|", "__").replace(":", "~")],
                                arrays[key], equal_nan=True)
        rows.append((key, ok))
        fails += not ok
    dump = lambda d: json.dumps(d, sort_keys=True)  # noqa: E731  exact float repr
    for key in sorted(set(man["metrics"]) | set(mrows)):
        ok = dump(man["metrics"].get(key)) == dump(mrows.get(key))
        rows.append(("metric " + key, ok))
        fails += not ok

    # compact table: one line per (ckpt, input kind, K mode)
    groups = {}
    for key, ok in rows:
        parts = key.replace("metric ", "").split("|")
        kind = "rand" if parts[1].startswith("rand") else ("ibims-agg" if parts[1] == "ibims-agg"
                                                            else "ibims")
        g = (parts[0], kind, parts[2], "metric" if key.startswith("metric ") else "tensor")
        n, p = groups.get(g, (0, 0))
        groups[g] = (n + 1, p + ok)
    print(f"{'ckpt':9s} {'input':10s} {'K':4s} {'kind':6s} {'n':>3s}  result")
    for g in sorted(groups):
        n, p = groups[g]
        print(f"{g[0]:9s} {g[1]:10s} {g[2]:4s} {g[3]:6s} {n:3d}  {'PASS' if p == n else 'FAIL'}")
    for key, ok in rows:
        if not ok:
            print("  FAIL", key)
    print(f"[parity] {'PASS' if fails == 0 else 'FAIL'}: {len(rows) - fails}/{len(rows)} "
          f"identical (device={device}, threads={threads})")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cpu")
    c.add_argument("--threads", type=int, default=4)
    r = sub.add_parser("compare")
    r.add_argument("--ref", required=True)
    r.add_argument("--device", default=None)
    r.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a.out, a.device, a.threads)
        return 0
    return compare(a.ref, a.device, a.threads)


if __name__ == "__main__":
    sys.exit(main())
