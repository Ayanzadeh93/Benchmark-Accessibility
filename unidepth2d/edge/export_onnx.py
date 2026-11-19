"""Export a student checkpoint to ONNX at its fixed training resolution, plus a metadata sidecar.

  python -m edge.export_onnx --ckpt outputs/checkpoints/b1_mix/best.pt
  python -m edge.export_onnx --ckpt outputs/checkpoints/S16_prox/best.pt --out outputs/edge/S16_prox/fp32.onnx

Graph contract (what every downstream tool relies on):
  input  "image"       float32 (1, 3, H, W)  ImageNet-normalised RGB at the student's train size
  input  "intrinsics"  float32 (1, 4)        (fx/W, fy/H, cx/W, cy/H) - only if the student uses
                                             FiLM intrinsics or the ray-map input
  output "log_depth"   float32 (1, 1, H, W)  natural-log metres, clamped to [log dmin, log dmax]
The exp() is left to the caller on purpose: an 8-bit output tensor would otherwise band the
depth map (256 levels over the whole range), so the head can stay float and exp runs outside.

The sidecar <name>.json records the checkpoint, its student config and the input contract so
edge/onnx_runner.py can evaluate the file through benchmark/run_bench.py (spec onnx:<path>).
After export the script runs a parity check against PyTorch on a random input and on a
structured one, and fails loudly if the max log-depth difference exceeds --parity-tol.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """student(image[, intrinsics]) -> clamped log-depth (B,1,H,W)."""

    def __init__(self, student, uses_intrinsics: bool):
        super().__init__()
        self.student = student
        self.uses_intrinsics = uses_intrinsics
        self.log_min = float(student.log_min)
        self.log_max = float(student.log_max)

    def forward(self, image, intrinsics=None):
        out = self.student(image, intrinsics if self.uses_intrinsics else None)
        return out[:, :1].clamp(self.log_min, self.log_max)


def load_student(ckpt_path: str, device: str = "cpu"):
    """Return (student in eval mode, checkpoint cfg dict)."""
    from distill.student import build_student
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    cfg["student"]["pretrained_backbone"] = False      # weights come from the checkpoint
    model = build_student(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg


def student_name(ckpt_path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    return f"student-{parent if parent != 'checkpoints' else 'base'}"


def export(ckpt_path: str, out_path: str, opset: int = 18, parity_tol: float = 1e-3) -> dict:
    model, cfg = load_student(ckpt_path)
    s = cfg["student"]
    h, w = int(s["train_size"][0]), int(s["train_size"][1])
    uses_intr = bool(s.get("use_intrinsics", False)) or bool(s.get("ray_map", False))
    wrap = ExportWrapper(model, uses_intr).eval()

    g = torch.Generator().manual_seed(0)
    image = torch.randn(1, 3, h, w, generator=g)
    intr = torch.tensor([[0.8, 1.08, 0.5, 0.5]], dtype=torch.float32)
    args = (image, intr) if uses_intr else (image,)
    input_names = ["image", "intrinsics"] if uses_intr else ["image"]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    exporter = "dynamo"
    try:
        torch.onnx.export(wrap, args, out_path, input_names=input_names,
                          output_names=["log_depth"], opset_version=opset, dynamo=True,
                          external_data=False)
    except Exception as e:  # noqa: BLE001 - fall back to the TorchScript exporter
        print(f"[export] dynamo exporter failed ({type(e).__name__}: {str(e)[:200]}); "
              "falling back to the TorchScript exporter")
        exporter = "torchscript"
        torch.onnx.export(wrap, args, out_path, input_names=input_names,
                          output_names=["log_depth"], opset_version=min(opset, 17),
                          dynamo=False, do_constant_folding=True)

    # ── parity: PyTorch vs ONNX Runtime (CPU) on two inputs ─────────────────────────────
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ramp = torch.linspace(-2, 2, h * w).view(1, 1, h, w).expand(1, 3, h, w).contiguous()
    worst = 0.0
    for x in (image, ramp):
        with torch.inference_mode():
            ref = wrap(x, intr).numpy() if uses_intr else wrap(x).numpy()
        feed = {"image": x.numpy()}
        if uses_intr:
            feed["intrinsics"] = intr.numpy()
        got = sess.run(["log_depth"], feed)[0]
        if got.shape != ref.shape:
            raise SystemExit(f"parity FAILED: shape {got.shape} vs {ref.shape}")
        worst = max(worst, float(np.abs(got - ref).max()))
    if not np.isfinite(worst) or worst > parity_tol:
        raise SystemExit(f"parity FAILED: max |log-depth diff| = {worst:.3e} > {parity_tol:.1e}")

    meta = {
        "name": student_name(ckpt_path),
        "ckpt": os.path.abspath(ckpt_path).replace("\\", "/"),
        "student": s,
        "train_size": [h, w],
        "inputs": input_names,
        "output": "log_depth",
        "log_min": wrap.log_min,
        "log_max": wrap.log_max,
        "precision": "fp32",
        "exporter": exporter,
        "opset": opset,
        "parity_max_abs_logdepth": worst,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "file_mb": round(os.path.getsize(out_path) / 1e6, 3),
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {out_path}  {meta['file_mb']} MB  exporter={exporter}  "
          f"parity max|dlog|={worst:.2e}  inputs={input_names}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="", help="default: outputs/edge/<ckpt parent>/fp32.onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    a = ap.parse_args()
    out = a.out or os.path.join("outputs", "edge",
                                student_name(a.ckpt).replace("student-", ""), "fp32.onnx")
    export(a.ckpt, out, a.opset, a.parity_tol)


if __name__ == "__main__":
    main()
