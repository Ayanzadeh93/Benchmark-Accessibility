"""Export a student checkpoint to Core ML (fp16) for the Mac / ANE (b20 spec section 5, step 6d).

Run on macOS with coremltools installed (``pip install coremltools``):

  python -m edge.export_coreml --ckpt outputs/checkpoints/b20_last/best.pt \
      --out outputs/edge/b20/b20_fp16.mlpackage --fixed-k 0.81 1.08 0.5 0.53

What it does
  1. loads the checkpoint, re-parameterises rep4 blocks (``model.reparameterize()``) and asserts
     fused-vs-unfused parity (max |d log D| < 1e-4);
  2. optional ``--fixed-k fx/W fy/H cx/W cy/H``: wraps the model with that camera baked in as a
     constant buffer, so the ray map, FiLM (gamma, beta), the dsh camera embedding and the cstm
     term are computed from constants and constant-folded by the converter -> the Core ML graph
     has a single "image" input (the spec's fold_camera kernel folding is NOT done; K is baked
     as a constant input instead, which removes the same runtime ops);
     without it the graph keeps an "intrinsics" (1,4) input;
  3. torch.jit.trace -> coremltools.convert(compute_precision=FLOAT16, compute_units=ALL);
  4. optional ``--w8``: 8-bit weight-only (linear symmetric) quantisation, keeping the dsh scale
     head (scale_mlp / k_embed weights) in fp16 when the coremltools version exposes weight
     metadata (otherwise a warning is printed);
  5. on macOS: fp16 parity vs PyTorch fp32 on a random input and, with ``--parity-images DIR``,
     on up to 50 images (resized to the train size, ImageNet-normalised, same K).

Graph contract (same as edge/export_onnx.py):
  input  "image"       float (1,3,H,W) ImageNet-normalised RGB at the student's train size
  input  "intrinsics"  float (1,4) (fx/W, fy/H, cx/W, cy/H)   [absent with --fixed-k]
  output "log_depth"   float (1,1,H,W) natural-log metres, clamped to [log dmin, log dmax]

``--save-traced PATH`` also writes the traced TorchScript module, so the trace can be made on
any machine and converted on the Mac. Without coremltools the script stops after tracing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Only the scale_mlp.0 matrix (496 -> 128) needs excluding: the other scale-head tensors
# ((1,128), (16,4) and the biases) are below coremltools' default 2048-element threshold and are
# never quantised. 1-D bias shapes must NOT be listed: after BN folding every encoder conv with
# 128 / 16 output channels has such a bias, and matching it would leave that conv in fp16.
SCALE_HEAD_SHAPES = {(128, 496)}


def scale_head_ops(meta) -> list:
    """Names of the ops to keep in fp16, from coremltools ``get_weights_metadata`` output
    ({weight name: metadata with .val and .child_ops}): consumers of 2-D scale-head matrices."""
    names = []
    for _, md in meta.items():
        shape = tuple(np.shape(md.val))
        if len(shape) == 2 and shape in SCALE_HEAD_SHAPES:
            names += [child.name for child in getattr(md, "child_ops", [])]
    return names


class CoreMLWrapper(nn.Module):
    """student(image[, intrinsics]) -> clamped log-depth (1,1,H,W); K optionally a constant."""

    def __init__(self, student, uses_intrinsics: bool, fixed_k=None):
        super().__init__()
        self.student = student
        self.uses_intrinsics = uses_intrinsics
        self.log_min = float(student.log_min)
        self.log_max = float(student.log_max)
        self.fixed = fixed_k is not None
        if self.fixed:
            self.register_buffer("k_const", torch.tensor([list(fixed_k)], dtype=torch.float32))

    def forward(self, image, intrinsics=None):
        k = self.k_const if self.fixed else intrinsics
        out = self.student(image, k if self.uses_intrinsics else None)
        return out[:, :1].clamp(self.log_min, self.log_max)


def _load(ckpt):
    from distill.rep import fuse_with_parity
    from edge.export_onnx import load_student
    model, cfg = load_student(ckpt)
    s = cfg["student"]
    h, w = int(s["train_size"][0]), int(s["train_size"][1])
    from distill.student import needs_intrinsics
    uses_intr = needs_intrinsics(s)
    x = torch.randn(1, 3, h, w, generator=torch.Generator().manual_seed(0))
    k = torch.tensor([[0.8, 1.08, 0.5, 0.5]])
    fuse_with_parity(model, x, k if uses_intr else None)
    return model.eval(), cfg, (h, w), uses_intr


def _load_image(p, h, w):
    import cv2
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
    img = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None])


def _w8(mlmodel, ct):
    from coremltools.optimize.coreml import (OpLinearQuantizerConfig, OptimizationConfig,
                                             linear_quantize_weights)
    glob_cfg = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    op_cfgs = {}
    try:   # keep the dsh scale head in fp16 (spec C1 / section 3)
        from coremltools.optimize.coreml import get_weights_metadata
        meta = get_weights_metadata(mlmodel, weight_threshold=1)
        keep = scale_head_ops(meta)
        for op_name in keep:
            op_cfgs[op_name] = None
        # expected: exactly one linear op (scale_mlp.0); anything else means a wrong match
        print(f"[w8] keeping {len(keep)} scale-head op(s) in fp16: {keep}")
    except Exception as e:  # noqa: BLE001
        print(f"[w8] WARNING: could not exclude the scale head ({type(e).__name__}: {e}); "
              "quantising every weight above the default threshold")
    cfg = OptimizationConfig(global_config=glob_cfg, op_name_configs=op_cfgs or None)
    return linear_quantize_weights(mlmodel, config=cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="", help="default: outputs/edge/<ckpt parent>/fp16.mlpackage")
    ap.add_argument("--fixed-k", type=float, nargs=4, default=None,
                    metavar=("FX_W", "FY_H", "CX_W", "CY_H"),
                    help="bake this normalised camera into the graph (single image input)")
    ap.add_argument("--w8", action="store_true", help="8-bit weight-only quantisation")
    ap.add_argument("--save-traced", default="", help="also save the traced TorchScript here")
    ap.add_argument("--parity-images", default="", help="dir of images for fp16 parity (<=50)")
    ap.add_argument("--min-target", default="macOS13", help="coremltools deployment target")
    a = ap.parse_args()

    model, cfg, (h, w), uses_intr = _load(a.ckpt)
    wrap = CoreMLWrapper(model, uses_intr, a.fixed_k).eval()
    image = torch.randn(1, 3, h, w, generator=torch.Generator().manual_seed(0))
    k_in = torch.tensor([a.fixed_k or [0.8, 1.08, 0.5, 0.5]], dtype=torch.float32)
    example = (image,) if (wrap.fixed or not uses_intr) else (image, k_in)
    with torch.no_grad():
        traced = torch.jit.trace(wrap, example, check_trace=True)
        ref = wrap(*example)
        got = traced(*example)
    assert float((ref - got).abs().max()) < 1e-5, "trace parity failed"
    if a.save_traced:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_traced)), exist_ok=True)
        traced.save(a.save_traced)
        print(f"[trace] saved {a.save_traced}")

    try:
        import coremltools as ct
    except ImportError:
        print("coremltools is not installed in this environment; traced OK. Run this script on "
              "the Mac (pip install coremltools) to produce the .mlpackage.")
        sys.exit(2 if not a.save_traced else 0)

    parent = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out = a.out or os.path.join("outputs", "edge", parent,
                                ("w8_" if a.w8 else "") + "fp16" + ("_fixedK" if wrap.fixed else "")
                                + ".mlpackage")
    inputs = [ct.TensorType(name="image", shape=tuple(image.shape), dtype=np.float32)]
    if not wrap.fixed and uses_intr:
        inputs.append(ct.TensorType(name="intrinsics", shape=(1, 4), dtype=np.float32))
    target = getattr(ct.target, a.min_target)
    mlmodel = ct.convert(traced, inputs=inputs,
                         outputs=[ct.TensorType(name="log_depth", dtype=np.float32)],
                         convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                         compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=target)
    if a.w8:
        mlmodel = _w8(mlmodel, ct)
    mlmodel.short_description = f"unidepth2d student {parent} (log-depth, fp16)"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    mlmodel.save(out)
    print(f"[coreml] saved {out}")

    report = {"ckpt": os.path.abspath(a.ckpt), "out": os.path.abspath(out), "fixed_k": a.fixed_k,
              "w8": a.w8, "train_size": [h, w], "inputs": [i.name for i in inputs]}
    if platform.system() == "Darwin":
        feeds = [image.numpy()]
        if a.parity_images:
            files = sorted(p for e in ("*.png", "*.jpg", "*.jpeg")
                           for p in glob.glob(os.path.join(a.parity_images, e)))[:50]
            feeds += [_load_image(p, h, w) for p in files]
        worst, errs = 0.0, []
        for x in feeds:
            xt = torch.from_numpy(x)
            with torch.no_grad():
                r = (wrap(xt) if (wrap.fixed or not uses_intr) else wrap(xt, k_in)).numpy()
            feed = {"image": x}
            if "intrinsics" in report["inputs"]:
                feed["intrinsics"] = k_in.numpy()
            g = mlmodel.predict(feed)["log_depth"]
            e = float(np.abs(np.asarray(g, dtype=np.float32) - r).max())
            errs.append(e)
            worst = max(worst, e)
        report.update(parity_max_abs_logdepth=worst, parity_n=len(feeds),
                      parity_median=float(np.median(errs)), parity_pass=worst < 0.01)
        print(f"[parity] fp16 vs torch fp32 over {len(feeds)} inputs: max|dlogD|={worst:.4f} "
              f"(spec gate < 0.01: {'PASS' if worst < 0.01 else 'FAIL'})")
    else:
        print("[parity] skipped (Core ML prediction needs macOS)")
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
