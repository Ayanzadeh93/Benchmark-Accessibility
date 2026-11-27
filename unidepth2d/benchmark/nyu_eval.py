"""Evaluate the UniDepth v2 TEACHER and our distilled STUDENT against NYU Depth v2 REAL ground truth.

This is a true metric evaluation (no median/scale alignment) on 654 held-out NYU validation images
with Kinect depth. Answers: how accurate is the teacher in absolute meters, and how much does the
5.1M student retain vs. real ground truth (not just vs. the teacher).

  python -m benchmark.nyu_eval --config config.yaml --limit 5     # quick check (also downloads ~1GB)
  python -m benchmark.nyu_eval --config config.yaml               # full 654-image eval
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
from distill.student import DEFAULT_K, build_student, needs_intrinsics
from teacher.unidepth_teacher import UniDepthTeacher

MIN_D, MAX_D = 1e-3, 10.0  # NYU standard depth range (meters)


def eigen_crop_mask(h, w):
    """Standard NYU 'Eigen' evaluation crop."""
    m = np.zeros((h, w), dtype=bool)
    if (h, w) == (480, 640):
        m[45:471, 41:601] = True
    else:
        m[:] = True
    return m


def metrics(gt, pred):
    """Per-image depth metrics over the valid pixels (both metric — no scaling)."""
    thresh = np.maximum(gt / pred, pred / gt)
    d1 = float((thresh < 1.25).mean())
    d2 = float((thresh < 1.25 ** 2).mean())
    d3 = float((thresh < 1.25 ** 3).mean())
    absrel = float((np.abs(gt - pred) / gt).mean())
    rmse = float(np.sqrt(((gt - pred) ** 2).mean()))
    rmse_log = float(np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean()))
    log10 = float(np.abs(np.log10(gt) - np.log10(pred)).mean())
    return dict(AbsRel=absrel, RMSE=rmse, RMSE_log=rmse_log, log10=log10, d1=d1, d2=d2, d3=d3)




def student_intrinsics(mode, s_cfg, oh, ow, device):
    """A05 eval-time K for the student (normalised fx/W, fy/H, cx/W, cy/H of the ORIGINAL
    image) or None. mode "none" (default) = historical behaviour (no K passed); "gt" = NYU
    GT K; "default" = DEFAULT_K passed explicitly. Students that read no K always get None."""
    if mode == "none" or not needs_intrinsics(s_cfg):
        return None
    if mode == "gt":
        from benchmark.datasets_tier1 import NYU_K      # NYU Kinect RGB K (pixels, 640x480)
        k = [NYU_K[0, 0] / ow, NYU_K[1, 1] / oh, NYU_K[0, 2] / ow, NYU_K[1, 2] / oh]
    elif mode == "default":
        k = list(DEFAULT_K)
    else:
        raise ValueError(f"unknown student K mode {mode!r}")
    return torch.tensor([k], dtype=torch.float32, device=device)


def student_depth(model, rgb, cfg, device, k_mode="none", s_cfg=None):
    th, tw = int(cfg["student"]["train_size"][0]), int(cfg["student"]["train_size"][1])
    oh, ow = rgb.shape[:2]
    rz = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rz = (rz - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(np.ascontiguousarray(np.transpose(rz, (2, 0, 1)))).float()[None].to(device)
    intr = student_intrinsics(k_mode, s_cfg if s_cfg is not None else cfg["student"], oh, ow, device)
    with torch.no_grad():
        dm = (model.infer_metres(t) if intr is None else model.infer_metres(t, intr))[0, 0].cpu().numpy()
    return cv2.resize(dm, (ow, oh), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--no-crop", action="store_true", help="disable the Eigen crop")
    ap.add_argument("--student-k", default="none", choices=["none", "gt", "default"],
                    help="A05: K passed to the student: none (historical; FiLM off), gt (NYU "
                         "GT K) or default (assumed DEFAULT_K)")
    ap.add_argument("--out", default="", help="result JSON path (default: <checkpoints_dir>/"
                                               "eval_nyu_gt.json, suffixed for --student-k)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from datasets import load_dataset
    # STREAMING: iterate parquet shards on the fly (no multi-GB Arrow cache build).
    print("streaming NYU v2 validation (fetches ~1GB of parquet shards on the fly)...")
    ds = load_dataset("vikhyatk/nyu_depth_v2", split="validation", streaming=True)
    n = args.limit if args.limit else 654
    print(f"evaluating on up to {n} images")

    teacher = UniDepthTeacher(cfg["teacher"]["checkpoints"], device=device)
    teacher.load()
    ckpt = args.ckpt or os.path.join(cfg["distill"]["checkpoints_dir"], "best.pt")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    s_cfg = ck.get("cfg", cfg)["student"]
    student = build_student(ck.get("cfg", cfg)).to(device).eval()
    student.load_state_dict(ck["model"])

    acc = {"teacher": [], "student": []}
    for i, s in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        rgb = np.array(s["image"])
        gt = np.array(s["depth_map"]).astype(np.float32)
        h, w = gt.shape[:2]

        t_pred = teacher.predict(rgb, infer_longest_side=640)["depth_m"]
        s_pred = student_depth(student, rgb, cfg, device, args.student_k, s_cfg)
        t_pred = cv2.resize(t_pred, (w, h), interpolation=cv2.INTER_LINEAR)
        s_pred = cv2.resize(s_pred, (w, h), interpolation=cv2.INTER_LINEAR)

        valid = (gt > MIN_D) & (gt < MAX_D)
        if not args.no_crop:
            valid &= eigen_crop_mask(h, w)
        if valid.sum() < 100:
            continue
        gtv = gt[valid]
        for name, pred in (("teacher", t_pred), ("student", s_pred)):
            pv = np.clip(pred[valid], MIN_D, MAX_D)
            acc[name].append(metrics(gtv, pv))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}")

    def summarize(rows):
        keys = rows[0].keys()
        return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}

    result = {
        "dataset": "NYU Depth v2 (validation, real Kinect GT)",
        "num_images": len(acc["teacher"]),
        "eigen_crop": not args.no_crop,
        "metric_eval": "no scale alignment (both models are metric)",
        "teacher_UniDepthV2_ViTL": summarize(acc["teacher"]),
        "student_5p1M": summarize(acc["student"]),
    }
    if args.student_k != "none":
        result["student_k"] = {"gt": "NYU GT K", "default": "assumed default K"}[args.student_k]
    print(json.dumps(result, indent=2))
    sfx = "" if args.student_k == "none" else f"__k{args.student_k}"
    out = args.out or os.path.join(cfg["distill"]["checkpoints_dir"], f"eval_nyu_gt{sfx}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
