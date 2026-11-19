"""Side-by-side RGB | teacher depth | student depth panels for eyeballing quality.

Teacher and student are colorized with the SAME per-image depth scale so colors are directly
comparable. Also stamps per-image AbsRel. Saves one panel per image + a contact sheet.

  python -m distill.visualize --config config.yaml --split test --n 12
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from common.io_utils import load_config
from distill.dataset import IMAGENET_MEAN, IMAGENET_STD
from distill.student import build_student

DISP_H, DISP_W = 384, 512


def colorize(depth_m, vmin, vmax, cmap=cv2.COLORMAP_INFERNO):
    d = np.clip((depth_m - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cmap)


def label_strip(width, text, h=26):
    strip = np.full((h, width, 3), 30, np.uint8)
    cv2.putText(strip, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = args.ckpt or os.path.join(cfg["distill"]["checkpoints_dir"], "best.pt")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_student(cfg).to(device).eval()
    model.load_state_dict(ck["model"])

    with open(cfg["splits"]["file"], "r", encoding="utf-8") as f:
        stems = json.load(f)[args.split][: args.n]

    out_dir = os.path.join(cfg["output"]["root"], "viz")
    os.makedirs(out_dir, exist_ok=True)
    images_dir = cfg["data"]["images_dir"]
    depth_dir = cfg["output"]["depth_dir"]
    exts = cfg["data"]["image_exts"]

    panels = []
    for stem in stems:
        img_path = next((os.path.join(images_dir, stem + e)
                         for e in exts if os.path.exists(os.path.join(images_dir, stem + e))), None)
        if img_path is None:
            continue
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_disp = cv2.resize(rgb, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)

        teach = cv2.imread(os.path.join(depth_dir, stem + ".png"), cv2.IMREAD_UNCHANGED)
        teach_m = cv2.resize(teach.astype(np.float32) / 1000.0, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)

        x = (rgb_disp.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(x, (2, 0, 1)))).float()[None].to(device)
        with torch.no_grad():
            stud_m = model.infer_metres(x)[0, 0].cpu().numpy()

        # shared scale from teacher robust percentiles
        vmin, vmax = np.percentile(teach_m, 2), np.percentile(teach_m, 98)
        teach_c = colorize(teach_m, vmin, vmax)
        stud_c = colorize(stud_m, vmin, vmax)

        v = teach_m > 0.2
        absrel = float(np.mean(np.abs(stud_m[v] - teach_m[v]) / teach_m[v]))

        row = np.hstack([cv2.cvtColor(rgb_disp, cv2.COLOR_RGB2BGR), teach_c, stud_c])
        labels = np.hstack([
            label_strip(DISP_W, "RGB"),
            label_strip(DISP_W, f"TEACHER UniDepthV2  [{vmin:.1f}-{vmax:.1f}m]"),
            label_strip(DISP_W, f"STUDENT 5.1M  AbsRel={absrel:.3f}"),
        ])
        panel = np.vstack([labels, row])
        cv2.imwrite(os.path.join(out_dir, f"panel_{stem[:24]}.png"), panel)
        panels.append(panel)

    if panels:
        sheet = np.vstack(panels)
        sheet_path = os.path.join(out_dir, "contact_sheet.png")
        cv2.imwrite(sheet_path, sheet)
        print(f"wrote {len(panels)} panels + contact sheet -> {sheet_path}")
    else:
        print("no panels produced")


if __name__ == "__main__":
    main()
