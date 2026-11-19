"""Draw detection boxes + student-measured distance labels on images, beside the student depth map.

  python -m distill.annotate_objects --config config.yaml --split test --n 3
Outputs annotated panels to outputs/viz_objects/.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from common import depth_ops as D
from common.io_utils import load_config
from distill.predict_objects import load_student, student_depth

PALETTE = [(66, 135, 245), (46, 204, 113), (241, 196, 15), (231, 76, 60), (155, 89, 182),
           (26, 188, 156), (230, 126, 34), (52, 152, 219), (211, 84, 0), (127, 140, 141)]


def color_for(key: str):
    return PALETTE[(hash(key) % len(PALETTE))]


def colorize(depth_m, vmin, vmax):
    d = np.clip((depth_m - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw_label(img, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(y, th + 8)
    x0 = min(x, img.shape[1] - tw - 8)
    x0 = max(x0, 0)
    cv2.rectangle(img, (x0, y0 - th - 8), (x0 + tw + 8, y0), color, -1)
    cv2.putText(img, text, (x0 + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stem", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(cfg["distill"]["checkpoints_dir"], "best.pt")
    model, use_intr, intr_map = load_student(cfg, ckpt, device)

    images_dir, exts, objects_dir = cfg["data"]["images_dir"], cfg["data"]["image_exts"], cfg["output"]["objects_dir"]
    if args.stem:
        stems = [args.stem]
    else:
        allstems = json.load(open(cfg["splits"]["file"], encoding="utf-8"))[args.split]
        stems = []
        for st in allstems:
            jp = os.path.join(objects_dir, st + ".json")
            if os.path.exists(jp) and json.load(open(jp, encoding="utf-8")).get("num_objects", 0) >= 3:
                stems.append(st)
            if len(stems) >= args.n:
                break

    out_dir = os.path.join(cfg["output"]["root"], "viz_objects")
    os.makedirs(out_dir, exist_ok=True)

    for stem in stems:
        img_path = next((os.path.join(images_dir, stem + e)
                         for e in exts if os.path.exists(os.path.join(images_dir, stem + e))), None)
        if img_path is None:
            continue
        depth_m, (ow, oh) = student_depth(model, img_path, cfg, device, use_intr, intr_map, stem)
        bgr = cv2.imread(img_path)

        oj = json.load(open(os.path.join(objects_dir, stem + ".json"), encoding="utf-8"))
        gw = oj.get("depth_grid_w", oj.get("image_w", ow))
        gh = oj.get("depth_grid_h", oj.get("image_h", oh))
        sx, sy = ow / float(gw), oh / float(gh)

        for o in oj["objects"]:
            x1, y1, x2, y2 = [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(o["bbox_xyxy"])]
            name = o.get("class_name") or f"class{o.get('class_id')}"
            od = D.sample_object_distance(depth_m, (x1, y1, x2, y2), inset_frac=0.20)
            if od is None:
                continue
            col = color_for(name)
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
            draw_label(bgr, int(x1), int(y1), f"{name} {od.distance_m:.1f}m/{od.distance_ft:.0f}ft", col)

        # depth beside it (shared robust scale)
        vmin, vmax = np.percentile(depth_m, 2), np.percentile(depth_m, 98)
        dcol = colorize(depth_m, vmin, vmax)
        H = 520
        def rz(im):
            return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
        panel = np.hstack([rz(bgr), rz(dcol)])
        cv2.imwrite(os.path.join(out_dir, f"obj_{stem[:26]}.png"), panel)
        print(f"wrote obj_{stem[:26]}.png  ({oj['num_objects']} objects)")


if __name__ == "__main__":
    main()
