"""Evaluate any model (teacher ViT-L / official ViT-S / a student ckpt) against REAL sensor GT.

⚠️ SUPERSEDED FOR PAPER NUMBERS: this legacy harness streams HF repacks (vikhyatk NYU /
wyrx SUN), clips predictions before metrics, and uses min-depth 1e-3 — all protocol
deviations fixed in the v2 harness (benchmark/run_bench.py + benchmark/protocol.py +
benchmark/datasets_tier1.py). Its numbers must never be mixed with v2/paper tables
(PROTOCOL_NOTES §3-4). Kept only for regression reference against the pre-fix results.

Datasets:
  nyu  — NYU Depth v2 validation (654, Kinect), HF vikhyatk/nyu_depth_v2, STREAMED, Eigen crop, 10m cap.
  sun  — SUN RGB-D test (5.0k, Kinect-family), HF wyrx/SUNRGBD_seg default/test, STREAMED, 8m cap.

Models (--model, repeatable):
  teacher            UniDepth v2 ViT-L (the distillation teacher)
  vits               official UniDepth v2 ViT-S (34M) — the "just use the small official model" baseline
  student:<ckpt>     any student checkpoint (architecture rebuilt from the ckpt's saved cfg)

  python -m benchmark.gt_eval --config config.yaml --dataset nyu --model teacher --model student:outputs/checkpoints/b1_mix/best.pt
  python -m benchmark.gt_eval --config config.yaml --dataset sun --limit 500 --model vits

True metric evaluation: no scale/median alignment. Results appended to outputs/checkpoints/gt_eval_results.json.
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

DATASETS = {
    "nyu": dict(repo="vikhyatk/nyu_depth_v2", config=None, split="validation",
                depth_key="depth_map", max_d=10.0, eigen_crop=True, n_default=654),
    "sun": dict(repo="wyrx/SUNRGBD_seg", config="default", split="test",
                depth_key="depth", max_d=8.0, eigen_crop=False, n_default=5000),
}
MIN_D = 1e-3


def eigen_crop_mask(h, w):
    m = np.zeros((h, w), dtype=bool)
    if (h, w) == (480, 640):
        m[45:471, 41:601] = True
    else:
        m[:] = True
    return m


def metrics(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    return dict(
        AbsRel=float((np.abs(gt - pred) / gt).mean()),
        RMSE=float(np.sqrt(((gt - pred) ** 2).mean())),
        RMSE_log=float(np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())),
        log10=float(np.abs(np.log10(gt) - np.log10(pred)).mean()),
        d1=float((thresh < 1.25).mean()),
        d2=float((thresh < 1.25 ** 2).mean()),
        d3=float((thresh < 1.25 ** 3).mean()),
    )


def _decode_sun_depth(arr_u16):
    plain = arr_u16.astype(np.float32) / 1000.0
    med = float(np.median(plain[plain > 0])) if (plain > 0).any() else 0.0
    if 0.4 <= med <= 8.0:
        return plain
    return np.bitwise_or(np.right_shift(arr_u16, 3), np.left_shift(arr_u16, 13)).astype(np.float32) / 1000.0


class TeacherModel:
    def __init__(self, cfg, device, checkpoint=None):
        from teacher.unidepth_teacher import UniDepthTeacher
        ckpts = [checkpoint] if checkpoint else cfg["teacher"]["checkpoints"]
        self.t = UniDepthTeacher(ckpts, device=device)
        self.t.load()
        self.name = self.t.loaded_checkpoint

    def predict(self, rgb):
        return self.t.predict(rgb, infer_longest_side=640)["depth_m"]


class StudentModel:
    def __init__(self, cfg, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_model = ck.get("cfg", cfg)
        self.model = build_student(cfg_model).to(device).eval()
        self.model.load_state_dict(ck["model"])
        s = cfg_model["student"]
        self.th, self.tw = int(s["train_size"][0]), int(s["train_size"][1])
        self.use_intr = bool(s.get("use_intrinsics", False))
        self.device = device
        parent = os.path.basename(os.path.dirname(ckpt_path))
        self.name = f"student:{parent if parent != 'checkpoints' else 'ckpt'}"
        # default intrinsics for unknown eval cameras (NYU Kinect close to defaults anyway)
        self.default_intr = torch.tensor([[0.8, 1.08, 0.5, 0.5]], dtype=torch.float32, device=device)

    def predict(self, rgb):
        oh, ow = rgb.shape[:2]
        rz = cv2.resize(rgb, (self.tw, self.th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rz = (rz - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(np.transpose(rz, (2, 0, 1)))).float()[None].to(self.device)
        with torch.no_grad():
            dm = self.model.infer_metres(t, self.default_intr if self.use_intr else None)[0, 0].cpu().numpy()
        return cv2.resize(dm, (ow, oh), interpolation=cv2.INTER_LINEAR)


def build_model(spec, cfg, device):
    if spec == "teacher":
        return TeacherModel(cfg, device)
    if spec == "vits":
        return TeacherModel(cfg, device, checkpoint="lpiccinelli/unidepth-v2-vits14")
    if spec.startswith("student:"):
        return StudentModel(cfg, spec.split(":", 1)[1], device)
    raise ValueError(f"unknown model spec {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--model", action="append", required=True,
                    help="teacher | vits | student:<ckpt>  (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = DATASETS[args.dataset]

    models = [build_model(m, cfg, device) for m in args.model]

    from datasets import load_dataset
    kw = dict(split=spec["split"], streaming=True)
    ds = load_dataset(spec["repo"], spec["config"], **kw) if spec["config"] else load_dataset(spec["repo"], **kw)
    n_target = args.limit or spec["n_default"]
    print(f"[{args.dataset}] streaming up to {n_target} images; models: {[m.name for m in models]}")

    acc = {m.name: [] for m in models}
    n_done = 0
    for i, s in enumerate(ds):
        if n_done >= n_target:
            break
        rgb = np.array(s["image"].convert("RGB") if hasattr(s["image"], "convert") else s["image"])
        raw = np.array(s[spec["depth_key"]])
        gt = _decode_sun_depth(raw.astype(np.uint16)) if args.dataset == "sun" else raw.astype(np.float32)
        h, w = gt.shape[:2]
        valid = (gt > MIN_D) & (gt < spec["max_d"])
        if spec["eigen_crop"]:
            valid &= eigen_crop_mask(h, w)
        if valid.sum() < 100:
            continue
        gtv = gt[valid]
        for m in models:
            pred = m.predict(rgb)
            if pred.shape[:2] != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            acc[m.name].append(metrics(gtv, np.clip(pred[valid], MIN_D, spec["max_d"])))
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{n_target}")

    def summarize(rows):
        return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}

    result = {
        "dataset": args.dataset,
        "num_images": n_done,
        "metric_eval": "no scale alignment",
        "models": {name: summarize(rows) for name, rows in acc.items() if rows},
    }
    print(json.dumps(result, indent=2))
    out = os.path.join(cfg["distill"]["checkpoints_dir"], "gt_eval_results.json")
    all_results = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    all_results.append(result)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("appended ->", out)


if __name__ == "__main__":
    main()
