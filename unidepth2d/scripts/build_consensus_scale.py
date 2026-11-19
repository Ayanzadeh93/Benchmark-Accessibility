"""Consensus-Scale Distillation (CSD) targets — denoise the teacher's per-image metric scale.

Motivation (measured, docs/SCALE_GAP.md + benchmark/scale_gap.py): in metric distillation the
student loses SCALE, not structure (IBims: student d1 .69 vs d1_ssi .93). Only ~59% of the
student's per-image scale error is inherited from the teacher — the rest is introduced by
distillation. But the teacher's own per-image scale is itself noisy: re-running it under a
different resolution budget or crop shifts its global log-scale by a measurable amount.

This script queries the teacher under N views (resolution levels + intrinsics-consistent crops)
and estimates, per image, the robust CONSENSUS offset of the teacher's scale relative to the
single cached pseudo-label pass:

    off_v = median(log D_v) - median(log D_cache | region_v)      # per view
    delta = median_v(off_v)                                        # consensus correction
    sigma = 1.4826 * MAD_v(off_v)                                  # teacher scale disagreement

`delta` refines the pseudo-label's global scale (recipe tag `csd`); `sigma` is a per-image
teacher-reliability signal for weighting the scale loss (recipe tag `rel`) — a principled
alternative to the confidence head, which is error-like and collapsed training (PROTOCOL_NOTES
§2b#3). Cost is entirely offline; student inference is unchanged.

  python -m scripts.build_consensus_scale --source own
  python -m scripts.build_consensus_scale --source own --limit 50      # smoke test

Writes <data_gt.root>/<source>_consensus_scale.json:
  {"<id>": {"delta": float, "sigma": float, "n": int}, ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch

from common.io_utils import load_config

# Views: resolution levels are full-frame (direct comparison); crops are (scale, cx, cy) with
# the fractional top-left offset, compared against the same region of the cached map.
RES_LEVELS = (0, 5, 9)
CROPS = ((0.75, 0.0, 0.0), (0.75, 0.25, 0.25))
_LOG_MIN, _LOG_MAX = np.log(0.2), np.log(60.0)


def _log_median(depth: np.ndarray) -> float:
    """Robust per-image log-scale: median of log-depth over plausible pixels."""
    v = depth[np.isfinite(depth) & (depth > 0.2) & (depth < 60.0)]
    if v.size < 200:
        return float("nan")
    return float(np.median(np.log(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default="own", help="manifest source to process")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="override output json path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg["data_gt"]["root"]
    manifest = cfg["data_gt"].get("manifest", os.path.join(root, "manifest.csv"))
    out_path = args.out or os.path.join(root, f"{args.source}_consensus_scale.json")

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == args.source]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"no rows for source={args.source}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from benchmark.run_bench import OfficialModel
    models = {r: OfficialModel("v2-l", device, use_gt_cam=False, res_level=r) for r in RES_LEVELS}
    base = models[RES_LEVELS[-1]]

    out, t0 = {}, time.time()
    for i, r in enumerate(rows):
        bgr = cv2.imread(r["img"], cv2.IMREAD_COLOR)
        d16 = cv2.imread(r["depth"], cv2.IMREAD_UNCHANGED)
        if bgr is None or d16 is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cache = d16.astype(np.float32) / 1000.0
        h, w = cache.shape[:2]
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        s_cache_full = _log_median(cache)
        if not np.isfinite(s_cache_full):
            continue

        offs = []
        for lvl, m in models.items():                       # full-frame resolution views
            d = m.predict(rgb, None)["depth"]
            s = _log_median(d)
            if np.isfinite(s):
                offs.append(s - s_cache_full)
        for sc, fx, fy in CROPS:                            # crop views vs the same region
            ch, cw = int(h * sc), int(w * sc)
            oy, ox = int(fy * (h - ch)), int(fx * (w - cw))
            d = base.predict(np.ascontiguousarray(rgb[oy:oy + ch, ox:ox + cw]), None)["depth"]
            s_v = _log_median(d)
            s_c = _log_median(cache[oy:oy + ch, ox:ox + cw])
            if np.isfinite(s_v) and np.isfinite(s_c):
                offs.append(s_v - s_c)

        if len(offs) < 3:
            continue
        a = np.asarray(offs, dtype=np.float64)
        delta = float(np.median(a))
        sigma = float(1.4826 * np.median(np.abs(a - delta)))
        out[r["id"]] = {"delta": round(delta, 6), "sigma": round(sigma, 6), "n": int(a.size)}

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {el/(i+1)*(len(rows)-i-1)/60:.0f}min",
                  flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    d = np.array([v["delta"] for v in out.values()])
    s = np.array([v["sigma"] for v in out.values()])
    print(f"wrote {out_path}  n={len(out)}")
    print(f"  delta: mean {d.mean():+.4f}  std {d.std():.4f}  |delta|>0.05: {(np.abs(d)>0.05).mean():.1%}")
    print(f"  sigma: mean {s.mean():.4f}  median {np.median(s):.4f}  p90 {np.percentile(s,90):.4f}")


if __name__ == "__main__":
    main()
