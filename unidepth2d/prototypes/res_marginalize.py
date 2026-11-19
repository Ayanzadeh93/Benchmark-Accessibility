"""Training-free resolution marginalization (A3-lite) — a boundary-quality/compute lever.

UniDepthV2 exposes an internal pixel budget via ``resolution_level`` (a slice of its
[0.2, 0.6] MP range) and a per-pixel ERROR-like ``confidence`` head (higher = less
certain; PROTOCOL_NOTES §2b #3). Running the SAME frozen model at several resolution
levels gives several depth maps whose errors are partly independent (coarse levels are
smooth/globally-consistent, fine levels sharpen edges). We fuse them in LOG-depth space
weighted by per-pixel PRECISION (inverse confidence), so each pixel takes its most
trustworthy scale:

    logD_fused(x) = Σ_r w_r(x)·logD_r(x) / Σ_r w_r(x),   w_r(x) = 1 / (conf_r(x) + eps)

No training, no extra parameters — only extra forward passes. We report the resulting
boundary-F1 / ARel against the extra COMPUTE (wall-clock ms + #passes), so the fusion is
presented honestly as a quality↔latency trade-off, not a free lunch. Three fusion modes
are compared: each single level, the uniform average ("marginalize", precision-free), and
the precision-weighted fusion (the method). A frozen well-calibrated model should gain
most on boundary-F1 (sharp levels win at edges) while ARel barely moves — the intended
signature.

Primary target: official UniDepthV2 (has confidence + resolution_level). A student with an
uncertainty head is also supported via an input-SCALE sweep with sigma-precision; a student
without one falls back to the uniform average (logged).

  python -m prototypes.res_marginalize --dataset ibims --model v2-l --levels 3,6,9 --boundary
  python -m prototypes.res_marginalize --dataset nyu  --model v2-l --levels 2,5,9

Outputs: results_bench/resmarg__<dataset>__<model>.json (per-level + fused metrics,
compute proxy, boundary-F1 delta).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from benchmark import protocol
from benchmark.datasets_tier1 import LOADERS
from benchmark.run_bench import OFFICIAL, build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_bench")
_EPS = 1e-3


def _to_gt(arr, h, w):
    if arr is None or arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def official_passes(m, rgb, K, levels, h, w):
    """Run one official model at several resolution_levels. Returns list of (depth, conf, ms)."""
    outs = []
    for lvl in levels:
        m.model.resolution_level = int(lvl)      # narrows the internal [0.2,0.6]MP budget
        t0 = time.time()
        pred = m.predict(rgb, K)
        ms = 1000.0 * (time.time() - t0)
        outs.append((_to_gt(pred["depth"], h, w), _to_gt(pred["conf"], h, w), ms))
    return outs


def student_passes(m, rgb, K, scales, h, w):
    """Run a student at several input scales (multi-resolution). Uses the uncertainty head
    for per-pixel precision when present, else confidence is None (→ uniform fusion)."""
    from distill.dataset import IMAGENET_MEAN, IMAGENET_STD
    outs = []
    base_h, base_w = m.th, m.tw
    for sc in scales:
        ih, iw = int(round(base_h * sc)), int(round(base_w * sc))
        rz = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA)
        rz = (rz.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(rz.transpose(2, 0, 1)))[None].to(m.device)
        intr = None
        if m.use_intr and K is not None:
            oh, ow = rgb.shape[:2]
            intr = torch.tensor([[K[0, 0] / ow, K[1, 1] / oh, K[0, 2] / ow, K[1, 2] / oh]],
                                dtype=torch.float32, device=m.device)
        t0 = time.time()
        with torch.inference_mode():
            if getattr(m.model, "uncertainty_head", False):
                depth, sigma = m.model.infer_with_uncertainty(t, intr)
                conf = sigma[0, 0].float().cpu().numpy() if sigma is not None else None
            else:
                depth, conf = m.model.infer_metres(t, intr), None
            depth = depth[0, 0].float().cpu().numpy()
        ms = 1000.0 * (time.time() - t0)
        outs.append((_to_gt(depth, h, w), _to_gt(conf, h, w), ms))
    return outs


def fuse(passes, mode: str):
    """Fuse a list of (depth, conf, ms) into one depth map.

    mode='uniform' : equal-weight mean of log-depth (precision-free marginalization).
    mode='prec'    : per-pixel precision weights 1/(conf+eps); falls back to uniform for
                     any pass whose conf is None (so a mixed set still fuses sensibly).
    """
    logs = [np.log(np.clip(d.astype(np.float64), _EPS, None)) for d, _, _ in passes]
    if mode == "uniform" or all(c is None for _, c, _ in passes):
        fused = np.mean(logs, axis=0)
        return np.exp(fused)
    ws = []
    for _, c, _ in passes:
        if c is None:
            ws.append(np.ones_like(logs[0]))
        else:
            ws.append(1.0 / (np.clip(c.astype(np.float64), 0.0, None) + _EPS))
    wsum = np.sum(ws, axis=0)
    fused = np.sum([w * lg for w, lg in zip(ws, logs)], axis=0) / np.clip(wsum, 1e-9, None)
    return np.exp(fused)


def _row(gt, pred, valid, max_depth, want_boundary):
    r = protocol.per_image_depth_metrics(gt, pred, valid, max_depth=max_depth)
    if r is None:
        return None
    keep = {k: r[k] for k in ("d1", "arel", "silog", "medianlog")}
    if want_boundary:
        b = protocol.boundary_f1_si(gt, pred, valid)
        keep["boundary_f1"] = b if b is not None else float("nan")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--model", required=True, help="v2-s|v2-b|v2-l|v1-l|student:<ckpt>")
    ap.add_argument("--levels", default="3,6,9",
                    help="official: resolution_level list; student: input-scale*10 list")
    ap.add_argument("--variant", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--boundary", action="store_true", help="compute boundary-F1 (dense GT sets)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(args.model, device, use_gt_cam=False)
    is_official = args.model in OFFICIAL
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    scales = [max(0.25, lvl / 10.0) for lvl in levels]     # student input-scale interpretation
    os.makedirs(RESULTS_DIR, exist_ok=True)

    loader_kwargs = {}
    if args.variant:
        loader_kwargs["variant"] = args.variant
    if args.limit:
        loader_kwargs["limit"] = args.limit

    variant_names = [f"L{lvl}" for lvl in levels] + ["fuse_uniform", "fuse_prec"]
    acc = {v: [] for v in variant_names}
    ms_acc = {f"L{lvl}": [] for lvl in levels}
    n = 0
    for sample in LOADERS[args.dataset](**loader_kwargs):
        gt, valid, K = sample.gt, sample.valid, sample.K
        h, w = gt.shape
        passes = (official_passes(m, sample.rgb, K, levels, h, w) if is_official
                  else student_passes(m, sample.rgb, K, scales, h, w))
        for lvl, (d, _c, ms) in zip(levels, passes):
            r = _row(gt, d, valid, sample.max_depth, args.boundary)
            if r is not None:
                acc[f"L{lvl}"].append(r)
            ms_acc[f"L{lvl}"].append(ms)
        for mode, name in (("uniform", "fuse_uniform"), ("prec", "fuse_prec")):
            fused = fuse(passes, mode)
            r = _row(gt, fused, valid, sample.max_depth, args.boundary)
            if r is not None:
                acc[name].append(r)
        n += 1
        if n % 50 == 0:
            print(f"  {n} images", flush=True)

    total_ms = float(sum(np.mean(v) for v in ms_acc.values() if v))
    summary = {
        "dataset": args.dataset, "model": m.name, "mode": "official" if is_official else "student",
        "levels": levels, "num_images": n,
        "compute": {
            "per_level_ms": {k: round(float(np.mean(v)), 2) for k, v in ms_acc.items() if v},
            "fused_ms": round(total_ms, 2), "n_passes": len(levels),
        },
        "metrics": {v: protocol.aggregate(rows) for v, rows in acc.items() if rows},
    }
    # boundary-F1 headline delta: best single level vs precision fusion
    if args.boundary and summary["metrics"].get("fuse_prec"):
        singles = [summary["metrics"][f"L{lvl}"].get("boundary_f1", float("nan")) for lvl in levels]
        best_single = np.nanmax(singles) if singles else float("nan")
        fused_bf1 = summary["metrics"]["fuse_prec"].get("boundary_f1", float("nan"))
        summary["boundary_f1_gain_vs_best_single"] = round(float(fused_bf1 - best_single), 4)

    out_path = os.path.join(RESULTS_DIR, f"resmarg__{args.dataset}__{m.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== resolution marginalization: {m.name} on {args.dataset} ({n} imgs) ===")
    for v in variant_names:
        mv = summary["metrics"].get(v)
        if not mv:
            continue
        bf = f" bF1={mv.get('boundary_f1'):.4f}" if "boundary_f1" in mv else ""
        print(f"[{v:12s}] ARel={mv.get('arel'):.4f} d1={mv.get('d1'):.4f} "
              f"silog={mv.get('silog'):.3f}{bf}")
    if "boundary_f1_gain_vs_best_single" in summary:
        print(f"boundary-F1 gain (prec-fuse vs best single level): "
              f"{summary['boundary_f1_gain_vs_best_single']:+.4f}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
