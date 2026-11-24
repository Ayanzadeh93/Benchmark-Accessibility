"""Cache the assistant's (S3-mix-L / b2_mix) depth predictions over the whole manifest (S8).

Writes data_gt/<source>/assist_mm/<id>.png (16-bit mm, 640x480) for every manifest row.
The training dataset uses these to compute per-pixel teacher-assistant agreement weights.

  python -m scripts.cache_assistant --config config.yaml --ckpt outputs/checkpoints/b2_mix/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch

from common.io_utils import load_config
from distill.dataset import IMAGENET_MEAN, IMAGENET_STD
from distill.student import build_student, needs_intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default="outputs/checkpoints/b2_mix/best.pt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_m = ck.get("cfg", cfg)
    model = build_student(cfg_m).to(device).eval()
    model.load_state_dict(ck["model"])
    s = cfg_m["student"]
    th, tw = int(s["train_size"][0]), int(s["train_size"][1])
    use_intr = needs_intrinsics(s, legacy_ray=False)
    default_intr = torch.tensor([[0.8, 1.08, 0.5, 0.5]], dtype=torch.float32, device=device)

    root = cfg["data_gt"]["root"]
    manifest = cfg["data_gt"].get("manifest", os.path.join(root, "manifest.csv"))
    rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    print(f"caching assistant predictions for {len(rows)} rows from {args.ckpt}")

    done = skipped = 0
    t0 = time.time()
    for r in rows:
        out_dir = os.path.join(root, r["source"], "assist_mm")
        os.makedirs(out_dir, exist_ok=True)
        out_p = os.path.join(out_dir, r["id"] + ".png")
        if os.path.exists(out_p):
            skipped += 1
            continue
        bgr = cv2.imread(r["img"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]
        rz = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rz = (rz - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(np.transpose(rz, (2, 0, 1)))).float()[None].to(device)
        intr = None
        if use_intr:
            try:
                intr = torch.tensor([[float(r["fx"]), float(r["fy"]), float(r["cx"]), float(r["cy"])]],
                                    dtype=torch.float32, device=device)
            except (TypeError, ValueError):
                intr = default_intr
        with torch.no_grad():
            dm = model.infer_metres(t, intr)[0, 0].cpu().numpy()
        dm = cv2.resize(dm, (ow, oh), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_p, (np.clip(dm, 0, 65.535) * 1000.0).astype(np.uint16))
        done += 1
        if done % 2000 == 0:
            print(f"  {done} done ({done / max(1e-6, time.time() - t0):.1f}/s)")
    print(f"DONE cached={done} skipped_existing={skipped} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
