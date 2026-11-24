"""Paper-grade benchmark runner: official UniDepth checkpoints + students, Tier-1 sets.

  python -m benchmark.run_bench --dataset sun --model v2-l --model v2-s
  python -m benchmark.run_bench --dataset nyu --model v2-l --model student:outputs/checkpoints/b1_mix/best.pt
  python -m benchmark.run_bench --dataset ibims --model v2-l --boundary

Protocol (docs/PROTOCOL_NOTES.md): predictions from native `model.infer()` (no external
resize; the model handles its 0.2-0.6MP budget), bilinear-resized to GT resolution,
NEVER clipped; per-image metrics from benchmark/protocol.py (bit-checked vs official);
unweighted mean over images. Optional: F_A + rho_A (needs model points + GT K),
AUSE/nAUSE/Spearman (needs confidence), boundary F1 (dense GT only).

Zero-shot means NO GT camera is passed (self-calibration), matching Tables I-III.
Pass --gt-cam to probe the GT-camera-prompted variant (reported separately, never mixed).

Results: results_bench/<dataset>__<model>__<tag>.json (aggregate + metadata
+ optional per-image rows) and one line appended to results_bench/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import time

import cv2
import numpy as np
import torch

from benchmark import protocol
from benchmark.datasets_tier1 import LOADERS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_bench")

OFFICIAL = {
    "v2-s": ("UniDepthV2", "lpiccinelli/unidepth-v2-vits14"),
    "v2-b": ("UniDepthV2", "lpiccinelli/unidepth-v2-vitb14"),
    "v2-l": ("UniDepthV2", "lpiccinelli/unidepth-v2-vitl14"),
    "v1-l": ("UniDepthV1", "lpiccinelli/unidepth-v1-vitl14"),
}


class OfficialModel:
    """Official UniDepth checkpoint through its native infer()."""

    def __init__(self, spec: str, device: str, use_gt_cam: bool = False,
                 res_level: int | None = None):
        cls_name, hub = OFFICIAL[spec]
        import unidepth.models as um
        self.model = getattr(um, cls_name).from_pretrained(hub).to(device).eval()
        if res_level is not None:
            # narrows the internal pixel budget to slice res_level of [0.2,0.6]MP
            self.model.resolution_level = int(res_level)
        self.device = device
        self.name = spec + ("+gtcam" if use_gt_cam else "") + \
            (f"+r{res_level}" if res_level is not None else "")
        self.use_gt_cam = use_gt_cam

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, K: np.ndarray | None):
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(self.device)
        cam = None
        if self.use_gt_cam and K is not None:
            cam = torch.from_numpy(K.astype(np.float32)).to(self.device)
        pred = self.model.infer(t, cam) if cam is not None else self.model.infer(t)

        def grab(key):
            v = pred.get(key)
            if v is None:
                return None
            v = v.squeeze().float().cpu().numpy()
            return v

        depth = grab("depth")                      # (H,W)
        conf = grab("confidence")                  # (H,W) or None
        points = grab("points")                    # (3,H,W) or None
        return {"depth": depth, "conf": conf, "points": points}


class StudentModelRunner:
    """Registry student: fixed train-size inference, bilinear upsample (deploy mode)."""

    def __init__(self, ckpt_path: str, device: str):
        from distill.student import build_student, needs_intrinsics
        from distill.dataset import IMAGENET_MEAN, IMAGENET_STD
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        self.model = build_student(cfg).to(device).eval()
        self.model.load_state_dict(ck["model"])
        s = cfg["student"]
        self.th, self.tw = int(s["train_size"][0]), int(s["train_size"][1])
        self.use_intr = needs_intrinsics(s)
        # b20 rep4: evaluate the FUSED deploy graph (what ONNX / Core ML ship), so student: and
        # q:<bits> (which quantises self.model's Conv weights) measure the shipped convs, not the
        # separate DW3x3 / DW1x1 / PW1x1 training branches. No-op (no forward, no RNG) otherwise.
        from distill.rep import fuse_with_parity, has_unfused_rep
        if has_unfused_rep(self.model):
            g = torch.Generator().manual_seed(0)
            x = torch.randn(1, 3, self.th, self.tw, generator=g).to(device)
            k = torch.tensor([[0.8, 1.08, 0.5, 0.5]], device=device) if self.use_intr else None
            fuse_with_parity(self.model, x, k)
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        self.device = device
        parent = os.path.basename(os.path.dirname(ckpt_path))
        self.name = f"student-{parent if parent != 'checkpoints' else 'base'}"

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, K: np.ndarray | None):
        oh, ow = rgb.shape[:2]
        rz = cv2.resize(rgb, (self.tw, self.th), interpolation=cv2.INTER_AREA)
        rz = (rz.astype(np.float32) / 255.0 - self.mean) / self.std
        t = torch.from_numpy(np.ascontiguousarray(rz.transpose(2, 0, 1)))[None].to(self.device)
        intr = None
        if self.use_intr and K is not None:
            # normalized (fx/W, fy/H, cx/W, cy/H) of the ORIGINAL image
            intr = torch.tensor([[K[0, 0] / ow, K[1, 1] / oh,
                                  K[0, 2] / ow, K[1, 2] / oh]],
                                dtype=torch.float32, device=self.device)
        depth = self.model.infer_metres(t, intr)[0, 0].float().cpu().numpy()
        return {"depth": depth, "conf": None, "points": None}


class HybridSSDI:
    """Scale-Structure Decoupled Inference: student structure + big-model scale.

    Spec: hybrid:<student_ckpt>+<donor>+r<level>, e.g.
      hybrid:outputs/checkpoints/S21_scale2/best.pt+v2-l+r0

    Rationale (scale_gap/view_var/transplant probes, 2026-07-20): the student's
    zero-shot metric error is a per-scene GLOBAL log-scale latent (view-stable,
    K-uncorrelated) that in-domain training cannot supply OOD, while a large
    model reads it correctly even at its LOWEST resolution budget. The hybrid
    keeps the student's full-res structure and rescales it by one scalar per
    image from a cheap donor pass. Deployment-honest: the shift never sees GT —
    median per-pixel log-ratio over the mutual indoor working range 0.2-8 m of
    the two PREDICTIONS (fallback: all finite pixels). Scale is temporally
    stable, so the donor can refresh at ~1 Hz while the student runs per-frame
    (donor cost amortizes away); worst-case latency = student + donor.
    """

    RANGE_LO, RANGE_HI, MIN_PX = 0.2, 8.0, 500

    def __init__(self, spec_rest: str, device: str):
        parts = spec_rest.split("+")
        if len(parts) != 3 or not parts[2].startswith("r"):
            raise SystemExit("hybrid spec must be <student_ckpt>+<donor>+r<level>")
        ckpt, donor, rlev = parts[0], parts[1], int(parts[2][1:])
        self.student = StudentModelRunner(ckpt, device)
        self.donor = OfficialModel(donor, device, use_gt_cam=False, res_level=rlev)
        self.name = f"ssdi-{self.student.name.replace('student-', '')}+{self.donor.name}"

    def predict(self, rgb: np.ndarray, K: np.ndarray | None):
        ps = self.student.predict(rgb, K)["depth"]
        pt = self.donor.predict(rgb, K)["depth"]
        if pt.shape != ps.shape:
            pt = cv2.resize(pt, (ps.shape[1], ps.shape[0]), interpolation=cv2.INTER_LINEAR)
        dep = (np.isfinite(ps) & np.isfinite(pt)
               & (ps > self.RANGE_LO) & (ps < self.RANGE_HI)
               & (pt > self.RANGE_LO) & (pt < self.RANGE_HI))
        if dep.sum() < self.MIN_PX:
            dep = np.isfinite(ps) & (ps > 0) & np.isfinite(pt) & (pt > 0)
        shift = float(np.median(np.log(pt[dep]) - np.log(ps[dep]))) if dep.any() else 0.0
        return {"depth": ps * float(np.exp(shift)), "conf": None, "points": None}


class QuantWeightStudent(StudentModelRunner):
    """Registry student with N-bit WEIGHTS (simulated: quantize-dequantize, edge/quant_sim.py).

    Spec: q:<bits>[pt]:<ckpt>   e.g. q:8:outputs/checkpoints/b1_mix/best.pt  (per-channel)
                                     q:8pt:...                                (per-tensor)
    Activations and biases stay float - this is the "8-bit weights" storage scheme.
    """

    def __init__(self, spec_rest: str, device: str):
        mode, ckpt = spec_rest.split(":", 1)
        per_channel = not mode.endswith("pt")
        bits = int(mode[:-2] if mode.endswith("pt") else mode)
        super().__init__(ckpt, device)
        from edge.quant_sim import quantize_weights_
        self.quant_stats = quantize_weights_(self.model, bits=bits, per_channel=per_channel)
        self.name = f"{self.name}+w{bits}{'' if per_channel else 'pt'}"
        print(f"[{self.name}] {self.quant_stats}")


EXTERNAL = {
    # 2025 zero-shot metric baselines. Need the `depth_cmp` env (moge, unik3d installed);
    # imported lazily so the `depth` env keeps working without them.
    "moge2-s": ("moge", "Ruicheng/moge-2-vits-normal"),
    "moge2-b": ("moge", "Ruicheng/moge-2-vitb-normal"),
    "moge2-l": ("moge", "Ruicheng/moge-2-vitl-normal"),
    "unik3d-s": ("unik3d", "lpiccinelli/unik3d-vits"),
    "unik3d-b": ("unik3d", "lpiccinelli/unik3d-vitb"),
    "unik3d-l": ("unik3d", "lpiccinelli/unik3d-vitl"),
}


class ExternalModel:
    """MoGe-2 / UniK3D through their native infer(), zero-shot (no GT camera).

    Both run at resolution_level 9 (their max token/pixel budget), matching the +r9
    policy used for the official UniDepthV2 rows. Output depth is camera z in metres.
    MoGe-2 runs with apply_mask=False so sky/invalid pixels keep a finite depth (the
    harness never masks predictions). Confidence is not passed: neither head is the
    log-error regressor that protocol.ause_delta1 assumes.
    """

    def __init__(self, spec: str, device: str):
        kind, hub = EXTERNAL[spec]
        self.kind, self.device, self.name = kind, device, spec + "+r9"
        if kind == "moge":
            from moge.model.v2 import MoGeModel
            self.model = MoGeModel.from_pretrained(hub).to(device).eval()
        else:
            from unik3d.models import UniK3D
            self.model = UniK3D.from_pretrained(hub).to(device).eval()
            self.model.resolution_level = 9

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, K: np.ndarray | None):
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(self.device)
        if self.kind == "moge":
            out = self.model.infer(t.float() / 255.0, resolution_level=9, apply_mask=False)
            depth = out["depth"].float().cpu().numpy()                       # (H,W)
            points = out["points"].float().permute(2, 0, 1).cpu().numpy()    # (3,H,W)
        else:
            out = self.model.infer(t)                                        # uint8 (3,H,W)
            depth = out["depth"].squeeze().float().cpu().numpy()
            points = out["points"].squeeze(0).float().cpu().numpy()
        return {"depth": depth, "conf": None, "points": points}


def build_model(spec: str, device: str, use_gt_cam: bool, res_level: int | None = None):
    if spec in OFFICIAL:
        return OfficialModel(spec, device, use_gt_cam, res_level)
    if spec in EXTERNAL:
        return ExternalModel(spec, device)
    if spec.startswith("student:"):
        return StudentModelRunner(spec.split(":", 1)[1], device)
    if spec.startswith("hybrid:"):
        return HybridSSDI(spec.split(":", 1)[1], device)
    if spec.startswith("q:"):
        return QuantWeightStudent(spec.split(":", 1)[1], device)
    if spec.startswith("onnx:"):
        from edge.onnx_runner import OnnxStudent     # needs onnxruntime (depth_cmp env)
        return OnnxStudent(spec.split(":", 1)[1], device)
    raise SystemExit(f"unknown model spec: {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--model", action="append", required=True,
                    help="v2-s|v2-b|v2-l|v1-l|student:<ckpt> (repeatable)")
    ap.add_argument("--variant", default="", help="dataset variant (nyu: filled|raw; sun: bfx|raw)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt-cam", action="store_true", help="prompt official models with GT K")
    ap.add_argument("--res-level", type=int, default=None,
                    help="official models: resolution_level 0..9 (internal pixel budget)")
    ap.add_argument("--boundary", action="store_true", help="compute boundary F1 (dense GT)")
    ap.add_argument("--no-3d", action="store_true", help="skip F_A/chamfer even if possible")
    ap.add_argument("--save-per-image", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [build_model(m, device, args.gt_cam, args.res_level) for m in args.model]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    loader_kwargs = {}
    if args.variant:
        loader_kwargs["variant"] = args.variant
    if args.limit:
        loader_kwargs["limit"] = args.limit

    acc = {m.name: {"depth": [], "f3d": [], "ray_err": [], "unc": [], "bf1": []}
           for m in models}
    n_done = 0
    t0 = time.time()
    for sample in LOADERS[args.dataset](**loader_kwargs):
        gt, valid, K = sample.gt, sample.valid, sample.K
        h, w = gt.shape
        gt_rays = protocol.rays_from_K(K, h, w) if K is not None else None
        for m in models:
            out = m.predict(sample.rgb, K)
            pred = out["depth"]
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            row = protocol.per_image_depth_metrics(gt, pred, valid,
                                                   max_depth=sample.max_depth)
            if row is None:
                continue
            acc[m.name]["depth"].append(row)

            pts = out["points"]
            if pts is not None and K is not None and not args.no_3d:
                if pts.shape[-2:] != (h, w):
                    pts = np.stack([cv2.resize(pts[c], (w, h), interpolation=cv2.INTER_LINEAR)
                                    for c in range(3)], 0)
                r3d = protocol.eval_3d_image(gt, pts, valid, K,
                                             sample.min_depth, sample.max_depth)
                if r3d:
                    acc[m.name]["f3d"].append(r3d)
                pred_rays = pts / np.linalg.norm(pts, axis=0, keepdims=True).clip(1e-9)
                acc[m.name]["ray_err"].append(
                    protocol.mean_ray_angular_error_deg(pred_rays, gt_rays))
            if out["conf"] is not None:
                conf = out["conf"]
                if conf.shape != (h, w):
                    conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
                # UniDepthV2's "confidence" head is trained to regress the log-depth
                # ERROR (config loss target |log d - log d*|): higher = LESS certain.
                # It is therefore already an uncertainty map — pass it as-is.
                u = protocol.ause_delta1(gt, pred, conf, valid,
                                         max_depth=sample.max_depth)
                if u:
                    acc[m.name]["unc"].append(u)
            if args.boundary:
                bf1 = protocol.boundary_f1_si(gt, pred, valid)
                if bf1 is not None:
                    acc[m.name]["bf1"].append(bf1)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done} images ({time.time()-t0:.0f}s)", flush=True)

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    for m in models:
        a = acc[m.name]
        result = {
            "dataset": args.dataset,
            "variant": args.variant or "default",
            "model": m.name,
            "num_images": len(a["depth"]),
            "protocol": "official-mirror v1 (protocol.py, no pred clip, per-image mean)",
            "gt_cam_prompted": args.gt_cam,
            "date": stamp,
            "depth": protocol.aggregate(a["depth"]),
            "f3d": protocol.aggregate(a["f3d"]) if a["f3d"] else None,
            "rho_a": protocol.rho_a(a["ray_err"]) if a["ray_err"] else None,
            "uncertainty": protocol.aggregate(a["unc"]) if a["unc"] else None,
            "boundary_f1_si": float(np.mean(a["bf1"])) if a["bf1"] else None,
        }
        if getattr(m, "quant_stats", None):
            result["quantization"] = m.quant_stats
        if args.save_per_image:
            result["per_image"] = a["depth"]
        tag = f"__{args.tag}" if args.tag else ""
        var = f"-{args.variant}" if args.variant else ""
        out_path = os.path.join(RESULTS_DIR,
                                f"{args.dataset}{var}__{m.name}{tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        d = result["depth"]
        print(f"[{m.name}] {args.dataset}{var} n={result['num_images']} "
              f"d1={d.get('d1'):.4f} d1_ssi={d.get('d1_ssi'):.4f} "
              f"arel={d.get('arel'):.4f} F1={result['f3d']['F1'] if result['f3d'] else float('nan')} "
              f"rho_a={result['rho_a']}")
        print("wrote", out_path)

        csv_path = os.path.join(RESULTS_DIR, "summary.csv")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            if new:
                wcsv.writerow(["date", "dataset", "variant", "model", "n",
                               "d1", "d1_ssi", "arel", "arel_ssi", "rmse", "silog",
                               "tau", "F1_3d", "rho_a", "ause", "nause", "spearman",
                               "boundary_f1", "gt_cam"])
            unc = result["uncertainty"] or {}
            wcsv.writerow([stamp, args.dataset, args.variant or "default", m.name,
                           result["num_images"],
                           d.get("d1"), d.get("d1_ssi"), d.get("arel"), d.get("arel_ssi"),
                           d.get("rmse"), d.get("silog"), d.get("tau"),
                           (result["f3d"] or {}).get("F1"), result["rho_a"],
                           unc.get("ause"), unc.get("nause"), unc.get("spearman"),
                           result["boundary_f1_si"], args.gt_cam])


if __name__ == "__main__":
    main()
