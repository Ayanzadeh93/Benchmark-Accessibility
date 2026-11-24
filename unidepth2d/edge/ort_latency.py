"""CPU latency of exported student ONNX files (compute-bound, unlike batch-1 GPU timing).

    python -m edge.ort_latency --onnx outputs/edge/b10_last/fp32.onnx --onnx ... --threads 4 --threads 1
Reports median / p90 ms over --iters runs after --warmup runs, 640x480 input, batch 1.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort


def bench(path: str, threads: int, warmup: int, iters: int) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    feeds = {}
    for i in s.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        if len(shape) == 2 and shape[-1] == 4:
            feeds[i.name] = np.array([[0.8107, 1.0822, 0.5087, 0.5286]], np.float32)
        else:
            feeds[i.name] = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    for _ in range(warmup):
        s.run(None, feeds)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        s.run(None, feeds)
        t.append(1000 * (time.perf_counter() - t0))
    t = np.array(t)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", action="append", required=True)
    ap.add_argument("--threads", action="append", type=int, default=[])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    for th in (a.threads or [4]):
        for p in a.onnx:
            r = bench(p, th, a.warmup, a.iters)
            print(f"threads={th}  {p:55s} median={r['median_ms']:7.1f} ms  p90={r['p90_ms']:7.1f} ms", flush=True)


if __name__ == "__main__":
    main()
