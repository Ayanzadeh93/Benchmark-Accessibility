"""Benchmark eval driver: model(s) x dataset -> UniDepthV2's exact metrics (2D + F_A + rho_A).

  conda run -n depth python -m bench.eval --dataset nyu --model vitl --model vits --limit 654
  conda run -n depth python -m bench.eval --dataset sun --model student:outputs/checkpoints/S10_ultra_L/best.pt

Same discipline as our fixed gt_eval: no scale/median alignment on the RAW metrics (alignment only
feeds the explicit *_ssi/*_si variants, per the paper). Appends JSON + prints a compact table.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from bench.datasets import get_loader
from bench.metrics import backproject, eval_depth_2d, f1_3d, reduce_dataset
from bench.refmodel import build_model
from bench.rho import rho_from_numpy

# Compact headline set printed to console (full dict goes to JSON).
HEADLINE = ["d1", "d1_ssi", "arel", "arel_ssi", "silog", "rmse", "f1_3d", "rho_a"]


def eval_model_on_dataset(model, loader, limit, device):
    rows = []
    n = 0
    for s in loader(limit):
        rgb, gt, valid, K, max_d = s["rgb"], s["depth"], s["valid"], s["K"], s["max_depth"]
        H, W = gt.shape[:2]
        pr = model.predict(rgb, gt_K=K)
        pred = pr["depth_m"]
        if pred.shape != (H, W):
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        gt_t = torch.from_numpy(gt).float().to(device)
        pred_t = torch.from_numpy(np.clip(pred, 1e-3, None)).float().to(device)
        mask_t = torch.from_numpy(valid).bool().to(device)
        m = eval_depth_2d(gt_t, pred_t, mask_t, max_depth=max_d)
        if not m:
            continue
        # F_A: model's own 3D points if it has them (self-calibrated, camera-inclusive); else
        # back-project the model's depth with GT K (labelled the same but K-substituted).
        if K is not None:
            gt_pts = backproject(gt_t, torch.from_numpy(K).float().to(device))
            if pr.get("points") is not None:
                pred_pts = torch.from_numpy(pr["points"]).float().to(device)
                if pred_pts.shape[-2:] != (H, W):
                    pred_pts = torch.nn.functional.interpolate(
                        pred_pts[None], size=(H, W), mode="nearest")[0]
            else:
                pred_pts = backproject(pred_t, torch.from_numpy(K).float().to(device))
            try:
                m["f1_3d"] = f1_3d(gt_pts, pred_pts, mask_t, max_depth=max_d)
            except Exception:
                pass
        # rho_A: only for models that output rays and a non-approx GT K.
        if pr.get("rays") is not None and K is not None and not s.get("K_approx", False):
            try:
                m["rho_a"] = rho_from_numpy(pr["rays"], K, device=device)
            except Exception:
                pass
        rows.append(m)
        n += 1
        if n % 100 == 0:
            print(f"    {model.name}: {n} imgs")
    return reduce_dataset(rows), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resolution-level", type=int, default=5)
    ap.add_argument("--out", default="outputs/checkpoints/bench_results.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = get_loader(args.dataset)

    results = {}
    for spec in args.model:
        model = build_model(spec, device, args.resolution_level)
        print(f"[{args.dataset}] {model.name} ...")
        metrics, n = eval_model_on_dataset(model, loader, args.limit, device)
        metrics["_num_images"] = n
        results[model.name] = metrics
        del model
        torch.cuda.empty_cache()

    print(f"\n=== {args.dataset} (exact UniDepthV2 metrics) ===")
    hdr = "model".ljust(26) + "".join(h.rjust(10) for h in HEADLINE)
    print(hdr)
    for name, m in results.items():
        row = name[:26].ljust(26) + "".join(
            (f"{m[h]:.4f}" if h in m else "-").rjust(10) for h in HEADLINE)
        print(row)

    entry = {"dataset": args.dataset, "resolution_level": args.resolution_level, "models": results}
    out = args.out if os.path.isabs(args.out) else os.path.join(os.getcwd(), args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allr = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    allr.append(entry)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(allr, f, indent=2)
    print("appended ->", out)


if __name__ == "__main__":
    main()
