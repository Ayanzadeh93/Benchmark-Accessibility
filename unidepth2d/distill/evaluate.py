"""Evaluate the student against the UniDepth v2 teacher.

Two views:
  1. Dense depth agreement over valid pixels: AbsRel, RMSE, delta1.
  2. Per-object distance agreement (the task metric): sample the student depth at the SAME frozen
     detection boxes with the SAME 5-point recipe, compare to the teacher's cached object distance.
     Reports MAE overall and within 5 m (mirrors the source paper's 0.31 m / 0.22 m reporting).

  python -m distill.evaluate --config config.yaml --split test --ckpt outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from common import depth_ops as D
from common.io_utils import get_detections, load_config
from distill.dataset import IMAGENET_MEAN, IMAGENET_STD
from distill.student import build_student, needs_intrinsics


def _prep_image(img_path, th, tw):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]
    rz = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rz = (rz - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(np.ascontiguousarray(np.transpose(rz, (2, 0, 1)))).float()
    return t, (orig_w, orig_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test", "train", "all"])
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = cfg["distill"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt_path = args.ckpt or os.path.join(cfg["distill"]["checkpoints_dir"], "best.pt")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_model = ck.get("cfg", cfg)  # build the architecture that was actually trained
    model = build_student(cfg_model).to(device).eval()
    model.load_state_dict(ck["model"])

    use_intr = needs_intrinsics(cfg_model["student"], legacy_ray=False)
    intrinsics_map = {}
    if use_intr:
        ipath = os.path.join(cfg["output"]["root"], "intrinsics.json")
        if os.path.exists(ipath):
            with open(ipath, "r", encoding="utf-8") as f:
                intrinsics_map = json.load(f)

    with open(cfg["splits"]["file"], "r", encoding="utf-8") as f:
        sp = json.load(f)
    stems = (sp["train"] + sp["val"] + sp["test"]) if args.split == "all" else sp[args.split]
    # '--split all' includes TRAINING frames the model was fit on: its numbers are leakage-inflated
    # and must never be read as held-out accuracy (audit finding — this produced the misleading
    # eval_all.json 0.0585 AbsRel vs the honest test 0.135). Warn loudly and stamp the output.
    if args.split == "all":
        print("WARNING: --split all evaluates on TRAIN+VAL+TEST (includes training frames). "
              "These numbers are LEAKAGE-INFLATED and are NOT a valid measure of generalization. "
              "Use --split test for reporting.")

    s = cfg["student"]
    th, tw = int(s["train_size"][0]), int(s["train_size"][1])
    images_dir = cfg["data"]["images_dir"]
    depth_dir = cfg["output"]["depth_dir"]
    objects_dir = cfg["output"]["objects_dir"]
    exts = cfg["data"]["image_exts"]

    # dense metrics accumulators
    abs_rel, rmse, d1, npix = 0.0, 0.0, 0.0, 0
    # per-object accumulators
    obj_abs_err, obj_n = 0.0, 0
    obj_abs_err_5m, obj_n_5m = 0.0, 0

    def find_image(stem):
        for e in exts:
            p = os.path.join(images_dir, stem + e)
            if os.path.exists(p):
                return p
        return None

    for stem in stems:
        img_path = find_image(stem)
        if img_path is None:
            continue
        t, (ow, oh) = _prep_image(img_path, th, tw)
        intr = None
        if use_intr:
            vals = intrinsics_map.get(stem, [0.7, 0.9, 0.5, 0.5])
            intr = torch.tensor([vals], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_m_small = model.infer_metres(t[None].to(device), intr)[0, 0].cpu().numpy()  # (th,tw)

        # ── dense agreement vs teacher depth (compare at teacher-map resolution) ──
        tp = os.path.join(depth_dir, stem + ".png")
        teach_png = cv2.imread(tp, cv2.IMREAD_UNCHANGED)
        if teach_png is not None:
            teach_m = teach_png.astype(np.float32) / 1000.0
            pred_up = cv2.resize(pred_m_small, (teach_m.shape[1], teach_m.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            v = np.isfinite(teach_m) & (teach_m >= s["depth_min_m"]) & (teach_m <= s["depth_max_m"])
            if v.sum() > 0:
                pt, tt = pred_up[v], teach_m[v]
                abs_rel += float((np.abs(pt - tt) / tt).sum())
                rmse += float(((pt - tt) ** 2).sum())
                ratio = np.maximum(pt / tt, tt / pt)
                d1 += float((ratio < 1.25).sum())
                npix += int(v.sum())

        # ── per-object distance agreement (paper recipe) ──
        # Sample the student depth on the SAME pixel grid the teacher's stored boxes use:
        #   yolo source (sample_objects.py) -> depth_grid_w/h ; run_teacher/ensemble -> image_w/h.
        # Mismatching these frames samples the wrong locations (was a real eval bug).
        obj_json = os.path.join(objects_dir, stem + ".json")
        if os.path.exists(obj_json):
            with open(obj_json, "r", encoding="utf-8") as f:
                oj = json.load(f)
            teach_objs = oj["objects"]
            if "depth_grid_w" in oj and "depth_grid_h" in oj:
                gw, gh = int(oj["depth_grid_w"]), int(oj["depth_grid_h"])
            elif "image_w" in oj and "image_h" in oj:
                gw, gh = int(oj["image_w"]), int(oj["image_h"])
            else:
                gw, gh = ow, oh
            pred_grid = cv2.resize(pred_m_small, (gw, gh), interpolation=cv2.INTER_LINEAR)
            for to in teach_objs:
                od = D.sample_object_distance(
                    pred_grid, tuple(to["bbox_xyxy"]),
                    inset_frac=float(cfg["teacher"].get("sample_inset_frac", 0.20)),
                )
                if od is None:
                    continue
                err = abs(od.distance_m - to["distance_m"])
                obj_abs_err += err
                obj_n += 1
                if to["distance_m"] <= 5.0:
                    obj_abs_err_5m += err
                    obj_n_5m += 1

    results = {
        "split": args.split,
        "ckpt": ckpt_path,
        "num_images": len(stems),
        "leakage_warning": ("TRAIN-INCLUSIVE: includes training frames; NOT held-out accuracy"
                            if args.split == "all" else None),
        "dense_vs_teacher": {
            "AbsRel": round(abs_rel / max(1, npix), 4),
            "RMSE_m": round((rmse / max(1, npix)) ** 0.5, 4),
            "delta1": round(d1 / max(1, npix), 4),
            "num_pixels": npix,
        },
        "per_object_vs_teacher": {
            "MAE_m": round(obj_abs_err / max(1, obj_n), 4),
            "MAE_m_within_5m": round(obj_abs_err_5m / max(1, obj_n_5m), 4),
            "num_objects": obj_n,
        },
    }
    print(json.dumps(results, indent=2))
    out = os.path.join(cfg["distill"]["checkpoints_dir"], f"eval_{args.split}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
