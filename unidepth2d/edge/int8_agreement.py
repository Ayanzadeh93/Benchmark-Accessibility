"""Compare an int8 ONNX student with its fp32 parent on images the calibration never saw.

  python -m edge.int8_agreement outputs/edge/b1_mix outputs/edge/S16_prox

This measures how far quantization moves the prediction and how fast each file runs on the
CPU. It is NOT an accuracy result: the reference is the fp32 model, not ground truth, and the
images come from the training corpus. Benchmark accuracy comes from benchmark/run_bench.py.

Held-out rule: edge.ort_int8 calibrates on the first N rows of a seeded shuffle of the `own`
source; this tool uses the rows right after them, so no image is shared with calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from edge.ort_int8 import DEFAULT_INTR, calibration_rows, preprocess


def agreement(log_a: np.ndarray, log_b: np.ndarray) -> dict:
    """Per-pixel agreement between two log-depth maps (natural log of metres)."""
    d = np.abs(np.asarray(log_a, dtype=np.float64) - np.asarray(log_b, dtype=np.float64))
    return {
        "mean_abs_log": float(d.mean()),
        "p95_abs_log": float(np.quantile(d, 0.95)),
        "frac_within_1pct": float((d < np.log(1.01)).mean()),
        "frac_within_5pct": float((d < np.log(1.05)).mean()),
        "frac_within_delta1": float((d < np.log(1.25)).mean()),
    }


def _session(path: str, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _latency_ms(sess, feed: dict, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        sess.run(["log_depth"], feed)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(["log_depth"], feed)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000.0)


def evaluate_dir(folder: str, manifest: str, n_calib: int, n_eval: int, seed: int, threads: int) -> dict:
    fp_path, q_path = os.path.join(folder, "fp32.onnx"), os.path.join(folder, "int8-headfp.onnx")
    with open(os.path.join(folder, "int8-headfp.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta["quant"]["calibration_images"] != n_calib or meta["quant"]["calibration_seed"] != seed:
        raise SystemExit(f"{folder}: calibration used {meta['quant']['calibration_images']} images, "
                         f"seed {meta['quant']['calibration_seed']}; pass matching --calib/--seed")
    h, w = meta["train_size"]
    rows = calibration_rows(manifest, n_calib + n_eval, seed)[n_calib:]
    fp, q = _session(fp_path, threads), _session(q_path, threads)
    per_image, feed = [], None
    for r in rows:
        bgr = cv2.imread(r["img"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        feed = {"image": preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), h, w)}
        if "intrinsics" in meta["inputs"]:
            try:
                k = [float(r["fx"]), float(r["fy"]), float(r["cx"]), float(r["cy"])]
            except (KeyError, ValueError, TypeError):
                k = list(DEFAULT_INTR)
            feed["intrinsics"] = np.asarray([k], dtype=np.float32)
        per_image.append(agreement(fp.run(["log_depth"], feed)[0], q.run(["log_depth"], feed)[0]))
    if not per_image:
        raise SystemExit(f"{folder}: no readable held-out images")
    out = {k: float(np.mean([p[k] for p in per_image])) for k in per_image[0]}
    out.update({
        "model": meta["name"], "n_images": len(per_image), "threads": threads,
        "fp32_mb": meta.get("fp32_file_mb"), "int8_mb": meta.get("file_mb"),
        "fp32_cpu_ms": _latency_ms(fp, feed), "int8_cpu_ms": _latency_ms(q, feed),
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--manifest", default="outputs/data_gt/manifest.csv")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--eval", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", default="outputs/edge/int8_agreement.json")
    a = ap.parse_args()
    res = [evaluate_dir(f, a.manifest, a.calib, a.eval, a.seed, a.threads) for f in a.folders]
    for r in res:
        print(json.dumps(r, indent=2))
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.json)


if __name__ == "__main__":
    main()
