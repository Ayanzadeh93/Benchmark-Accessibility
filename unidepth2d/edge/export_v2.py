"""Static ONNX export for V2 (CoDeQ) students and for legacy students with a constant default K.

  python -m edge.export_v2 --ckpt outputs/v2/runs/<run_id>/final_ema.pt            # CoDeQ
  python -m edge.export_v2 --legacy --ckpt outputs/checkpoints/b1_mix/best.pt      # legacy, default K

Env: depth_cmp (needs onnx + onnxruntime; the main `depth` env has neither).

CONTRACT (what downstream tools - edge/weight_int8.py, edge/ptq.py, edge/onnx_runner.py - rely on)
  * One graph input  "image"     float32 (1, 3, H, W), ImageNet-normalised RGB, static H, W.
  * One graph output "log_depth" float32 (1, 1, H, W), natural-log metres.
      - CoDeQ: UNCLAMPED (plan section 3.3: exp and clamp stay outside the graph).
      - legacy: clamped to [log dmin, log dmax] exactly as edge/export_onnx.py does, so a legacy
        default-K file differs from the legacy GT-K file only in how intrinsics enter.
  * No dynamic dimension anywhere on inputs or outputs (`check_static_shapes`).
  * CoDeQ graphs contain only ALLOWED_OPS_V2 = Conv, Relu, Add, Resize, DepthToSpace
    (`check_allowlist`). If torch's exporter emits a `Constant` node (Resize scales) it is
    folded into an initializer (`fold_constants_to_initializers`), and `Identity` nodes that
    re-emit deduplicated initializers are bypassed (`remove_identities`) - both exact.
    A config that legitimately needs AveragePool (run T2-13 onward) passes it via `extra_ops`.
  * Legacy students (FiLM intrinsics and/or ray map) take K as a CONSTANT buffer, default
    DEFAULT_K = (fx/W, fy/H, cx/W, cy/H) = (0.8, 1.08, 0.5, 0.5); the sidecar records
    camera_mode = "default-k". FiLM stays ON (plan 'Where reviewers disagreed' 1: K=None
    skips FiLM and collapses S16). GT intrinsics are never baked in.
  * Parity: max |log_depth(ORT CPU) - log_depth(PyTorch)| over the parity inputs must be
    below `parity_tol` (default 1e-3, plan 8.6); export raises ExportError otherwise.
  * Sidecar <file>.json has the edge/export_onnx.py keys (name, ckpt, train_size, inputs, output,
    log_min, log_max, precision, exporter, opset, parity_max_abs_logdepth, params_M, file_mb) plus
    family, camera_mode, default_K, op_histogram, allowlist_ok, static_shapes_ok, sha256, so the
    file runs through benchmark/run_bench.py as spec onnx:<path> unchanged.

CoDeQ checkpoints: `load_v2_checkpoint` expects the distill/v2/train.py layout
{"model"|"ema": state_dict, "cfg": {... "model": <build_v2_model cfg>}}; anything else can be
passed explicitly with --model-cfg (a JSON/YAML file holding the build_v2_model cfg).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ALLOWED_OPS_V2: Tuple[str, ...] = ("Conv", "Relu", "Add", "Resize", "DepthToSpace")
DEFAULT_K: Tuple[float, float, float, float] = (0.8, 1.08, 0.5, 0.5)
LOG_MIN, LOG_MAX = math.log(0.2), math.log(30.0)


class ExportError(RuntimeError):
    """Raised when an exported graph violates the contract (parity, allowlist, static shapes)."""


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────────────────

class V2ExportWrapper(nn.Module):
    """image (B,3,H,W) -> log_depth (B,1,H,W), unclamped. Pure pass-through of a CoDeQ model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model(image)
        return out[:, :1] if out.shape[1] != 1 else out


class LegacyDefaultKWrapper(nn.Module):
    """Legacy student with intrinsics baked in as a constant: image -> clamped log_depth.

    `student(image, K)` must accept normalised intrinsics (B,4). K is a registered buffer, so the
    FiLM MLP (and the ray map, if any) is evaluated on a constant and folded at export.
    """

    def __init__(self, student: nn.Module, K: Sequence[float] = DEFAULT_K,
                 log_min: Optional[float] = None, log_max: Optional[float] = None):
        super().__init__()
        self.student = student
        self.register_buffer("K", torch.tensor([list(map(float, K))], dtype=torch.float32))
        self.log_min = float(student.log_min if log_min is None else log_min)
        self.log_max = float(student.log_max if log_max is None else log_max)

    def forward(self, image):
        k = self.K.expand(image.shape[0], 4)
        out = self.student(image, k)
        return out[:, :1].clamp(self.log_min, self.log_max)


# ──────────────────────────────────────────────────────────────────────────────
# Graph checks
# ──────────────────────────────────────────────────────────────────────────────

def _load(model_or_path):
    import onnx
    return onnx.load(model_or_path) if isinstance(model_or_path, (str, os.PathLike)) else model_or_path


def op_histogram(model_or_path) -> Dict[str, int]:
    """{op_type: count} over the top-level graph."""
    m = _load(model_or_path)
    return dict(sorted(collections.Counter(n.op_type for n in m.graph.node).items()))


def check_allowlist(model_or_path, allowed: Iterable[str] = ALLOWED_OPS_V2,
                    extra_ops: Iterable[str] = ()) -> Tuple[bool, Dict[str, int]]:
    """(ok, {offending op: count}). ok iff every node's op_type is in allowed + extra_ops."""
    allow = set(allowed) | set(extra_ops)
    bad = {k: v for k, v in op_histogram(model_or_path).items() if k not in allow}
    return (not bad), bad


def check_static_shapes(model_or_path) -> Tuple[bool, List[str]]:
    """(ok, problems). Every graph input/output dim must be a positive integer (no dim_param)."""
    m = _load(model_or_path)
    problems = []
    for vi in list(m.graph.input) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if not dims:
            problems.append(f"{vi.name}: no shape")
        for i, d in enumerate(dims):
            if d.dim_param or d.dim_value <= 0:
                problems.append(f"{vi.name}[{i}] = {d.dim_param or d.dim_value!r}")
    return (not problems), problems


def fold_constants_to_initializers(model):
    """Move every top-level `Constant` node's tensor into graph.initializer (in place).

    Only `value` constants are handled (what torch emits for Resize scales/sizes); a Constant
    using another attribute (value_float, ...) is left untouched and will show in the op
    histogram, so the allowlist test still catches it.
    """
    from onnx import numpy_helper
    keep = []
    have = {i.name for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            arr = numpy_helper.to_array(n.attribute[0].t)
            if n.output[0] not in have:
                model.graph.initializer.append(numpy_helper.from_array(arr, n.output[0]))
                have.add(n.output[0])
            continue
        keep.append(n)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return model


def remove_identities(model):
    """Bypass every top-level `Identity` node whose output is not a graph output (in place).

    The TorchScript exporter deduplicates byte-identical initializers (e.g. the all-zero
    BN-folded biases of an untrained model) and re-emits them through Identity nodes; rewiring
    the consumers to the Identity input is exact. Returns the number of nodes removed.
    """
    outs = {o.name for o in model.graph.output}
    alias = {}
    keep = []
    for n in model.graph.node:
        if n.op_type == "Identity" and len(n.input) == 1 and n.output[0] not in outs:
            alias[n.output[0]] = n.input[0]
            continue
        keep.append(n)

    def resolve(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    for n in keep:
        for i, x in enumerate(n.input):
            if x in alias:
                n.input[i] = resolve(x)
    removed = len(model.graph.node) - len(keep)
    if removed:
        del model.graph.node[:]
        model.graph.node.extend(keep)
        dead = set(alias)
        vis = [v for v in model.graph.value_info if v.name not in dead]
        del model.graph.value_info[:]
        model.graph.value_info.extend(vis)
    return removed


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Export + parity
# ──────────────────────────────────────────────────────────────────────────────

def parity_inputs(h: int, w: int, seed: int = 0) -> List[np.ndarray]:
    """Two deterministic parity inputs: N(0,1) noise and a smooth ramp (both ImageNet-normalised scale)."""
    g = np.random.default_rng(seed)
    noise = g.standard_normal((1, 3, h, w), dtype=np.float32)
    ramp = np.linspace(-2.0, 2.0, h * w, dtype=np.float32).reshape(1, 1, h, w)
    return [noise, np.ascontiguousarray(np.repeat(ramp, 3, axis=1))]


def onnx_parity(wrapper: nn.Module, onnx_path: str, inputs: Sequence[np.ndarray],
                input_name: str = "image", output_name: str = "log_depth") -> float:
    """Max |ORT - PyTorch| over `inputs` (CPU, fp32). Raises ExportError on shape mismatch or NaN."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for x in inputs:
        with torch.inference_mode():
            ref = wrapper(torch.from_numpy(np.ascontiguousarray(x))).float().numpy()
        got = sess.run([output_name], {input_name: x.astype(np.float32)})[0]
        if got.shape != ref.shape:
            raise ExportError(f"parity: shape {got.shape} vs PyTorch {ref.shape}")
        if not np.isfinite(got).all():
            raise ExportError("parity: ONNX output contains NaN/Inf")
        worst = max(worst, float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max()))
    return worst


def export_module(wrapper: nn.Module, out_path: str, input_hw: Tuple[int, int], opset: int = 18,
                  exporter: str = "torchscript", input_name: str = "image",
                  output_name: str = "log_depth") -> str:
    """Export `wrapper(image)` statically at (1,3,H,W). Returns the exporter actually used.

    torchscript first (it produced the measured E3 graph); dynamo is the fallback.
    """
    import onnx
    h, w = int(input_hw[0]), int(input_hw[1])
    wrapper = wrapper.eval()
    dummy = torch.zeros(1, 3, h, w)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    order = ["torchscript", "dynamo"] if exporter == "torchscript" else ["dynamo", "torchscript"]
    last_err = None
    used = None
    for ex in order:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if ex == "torchscript":
                    torch.onnx.export(wrapper, (dummy,), out_path, input_names=[input_name],
                                      output_names=[output_name], opset_version=opset,
                                      dynamo=False, do_constant_folding=True)
                else:
                    torch.onnx.export(wrapper, (dummy,), out_path, input_names=[input_name],
                                      output_names=[output_name], opset_version=opset,
                                      dynamo=True, external_data=False)
            used = ex
            break
        except Exception as e:  # noqa: BLE001 - try the other exporter, report both on failure
            last_err = e
            print(f"[export_v2] {ex} exporter failed ({type(e).__name__}: {str(e)[:200]})")
    if used is None:
        raise ExportError(f"both exporters failed; last error: {last_err!r}")
    m = onnx.load(out_path)
    fold_constants_to_initializers(m)
    remove_identities(m)
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return used


def _write_meta(out_path: str, meta: dict) -> dict:
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def export_v2(model: nn.Module, out_path: str, input_hw: Tuple[int, int] = (480, 640),
              opset: int = 18, parity_tol: float = 1e-3, extra_ops: Iterable[str] = (),
              name: str = "codeq", ckpt: str = "", model_cfg: Optional[dict] = None,
              enforce_allowlist: bool = True, exporter: str = "torchscript") -> dict:
    """Export a CoDeQ nn.Module (eval mode) and enforce allowlist, static shapes and parity.

    Returns the sidecar meta dict (also written next to the .onnx).
    """
    wrap = V2ExportWrapper(model.eval()).eval()
    used = export_module(wrap, out_path, input_hw, opset, exporter)
    ok_ops, bad = check_allowlist(out_path, ALLOWED_OPS_V2, extra_ops)
    ok_shapes, shape_problems = check_static_shapes(out_path)
    if enforce_allowlist and not ok_ops:
        raise ExportError(f"allowlist violated: {bad} (allowed {ALLOWED_OPS_V2} + {tuple(extra_ops)})")
    if not ok_shapes:
        raise ExportError(f"dynamic shapes: {shape_problems}")
    worst = onnx_parity(wrap, out_path, parity_inputs(*input_hw))
    if not np.isfinite(worst) or worst > parity_tol:
        raise ExportError(f"parity FAILED: max |dlog| = {worst:.3e} > {parity_tol:.1e}")
    meta = {
        "name": name, "family": "codeq", "ckpt": ckpt.replace("\\", "/"),
        "model_cfg": model_cfg or {}, "train_size": [int(input_hw[0]), int(input_hw[1])],
        "inputs": ["image"], "output": "log_depth",
        "log_min": float(getattr(model, "log_min", LOG_MIN)),
        "log_max": float(getattr(model, "log_max", LOG_MAX)),
        "output_clamped_in_graph": False, "camera_mode": "none", "default_K": None,
        "precision": "fp32", "exporter": used, "opset": opset,
        "parity_max_abs_logdepth": worst, "parity_tol": parity_tol,
        "op_histogram": op_histogram(out_path), "allowlist": list(ALLOWED_OPS_V2) + list(extra_ops),
        "allowlist_ok": ok_ops, "allowlist_offending": bad, "static_shapes_ok": ok_shapes,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
        "file_mb": round(os.path.getsize(out_path) / 1e6, 3), "sha256": sha256_file(out_path),
    }
    _write_meta(out_path, meta)
    print(f"[export_v2] {out_path} {meta['file_mb']} MB exporter={used} parity={worst:.2e} "
          f"ops={meta['op_histogram']}")
    return meta


def export_legacy_default_k(ckpt_path: str, out_path: str, K: Sequence[float] = DEFAULT_K,
                            opset: int = 18, parity_tol: float = 1e-3,
                            exporter: str = "torchscript") -> dict:
    """Export a legacy distill/student.py checkpoint with K baked in as a constant (camera_mode default-k)."""
    from edge.export_onnx import load_student, student_name
    student, cfg = load_student(ckpt_path)
    s = cfg["student"]
    h, w = int(s["train_size"][0]), int(s["train_size"][1])
    uses_k = bool(s.get("use_intrinsics", False)) or bool(s.get("ray_map", False))
    wrap = LegacyDefaultKWrapper(student, K).eval()
    used = export_module(wrap, out_path, (h, w), opset, exporter)
    ok_shapes, shape_problems = check_static_shapes(out_path)
    if not ok_shapes:
        raise ExportError(f"dynamic shapes: {shape_problems}")
    import onnx
    n_inputs = len(onnx.load(out_path).graph.input)
    if n_inputs != 1:
        raise ExportError(f"default-K export must have exactly one input, got {n_inputs}")
    worst = onnx_parity(wrap, out_path, parity_inputs(h, w))
    if not np.isfinite(worst) or worst > parity_tol:
        raise ExportError(f"parity FAILED: max |dlog| = {worst:.3e} > {parity_tol:.1e}")
    ok_ops, bad = check_allowlist(out_path)
    meta = {
        "name": student_name(ckpt_path) + "@defaultK", "family": "legacy",
        "ckpt": os.path.abspath(ckpt_path).replace("\\", "/"), "ckpt_sha256": sha256_file(ckpt_path),
        "student": s, "train_size": [h, w], "inputs": ["image"], "output": "log_depth",
        "log_min": wrap.log_min, "log_max": wrap.log_max, "output_clamped_in_graph": True,
        "camera_mode": "default-k" if uses_k else "none",
        "default_K": [float(k) for k in K] if uses_k else None,
        "precision": "fp32", "exporter": used, "opset": opset,
        "parity_max_abs_logdepth": worst, "parity_tol": parity_tol,
        "op_histogram": op_histogram(out_path), "allowlist_ok": ok_ops,
        "allowlist_offending": bad, "allowlist_note": "allowlist is a CoDeQ contract; reported only for legacy",
        "static_shapes_ok": ok_shapes,
        "params_M": round(sum(p.numel() for p in student.parameters()) / 1e6, 3),
        "file_mb": round(os.path.getsize(out_path) / 1e6, 3), "sha256": sha256_file(out_path),
    }
    _write_meta(out_path, meta)
    print(f"[export_v2] legacy {out_path} {meta['file_mb']} MB exporter={used} parity={worst:.2e} "
          f"camera_mode={meta['camera_mode']}")
    return meta


def load_v2_checkpoint(ckpt_path: str, model_cfg: Optional[dict] = None,
                       prefer_ema: bool = True) -> Tuple[nn.Module, dict]:
    """Build a CoDeQ model from a distill/v2/train.py checkpoint (see module docstring)."""
    from distill.v2.model import build_v2_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = model_cfg
    if cfg is None:
        c = ck.get("cfg", {}) if isinstance(ck, dict) else {}
        cfg = ck.get("model_cfg") or (c.get("model") if isinstance(c, dict) else None) or c
    if not isinstance(cfg, dict) or "variant" not in cfg:
        raise ExportError(f"{ckpt_path}: no build_v2_model cfg with 'variant'; pass --model-cfg")
    cfg = dict(cfg)
    cfg["pretrained"] = False
    model = build_v2_model(cfg)
    sd = None
    if prefer_ema and isinstance(ck.get("ema"), dict):
        sd = ck["ema"].get("module", ck["ema"].get("state_dict", ck["ema"]))
    if sd is None:
        sd = ck.get("model", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=True)
    return model.eval(), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--legacy", action="store_true", help="legacy distill/student.py checkpoint")
    ap.add_argument("--model-cfg", default="", help="JSON/YAML build_v2_model cfg (CoDeQ only)")
    ap.add_argument("--hw", type=int, nargs=2, default=(480, 640))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    ap.add_argument("--extra-op", action="append", default=[], help="e.g. AveragePool for T2-13+")
    a = ap.parse_args()
    if a.legacy:
        from edge.export_onnx import student_name
        out = a.out or os.path.join("outputs", "edge", student_name(a.ckpt).replace("student-", ""),
                                    "fp32_defaultK.onnx")
        export_legacy_default_k(a.ckpt, out, DEFAULT_K, a.opset, a.parity_tol)
        return
    cfg = None
    if a.model_cfg:
        with open(a.model_cfg, encoding="utf-8") as f:
            if a.model_cfg.endswith((".yaml", ".yml")):
                import yaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
    model, cfg = load_v2_checkpoint(a.ckpt, cfg)
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out = a.out or os.path.join("outputs", "v2", "edge", run, "fp32.onnx")
    export_v2(model, out, tuple(a.hw), a.opset, a.parity_tol, a.extra_op,
              name=f"codeq-{cfg.get('variant')}-{run}", ckpt=os.path.abspath(a.ckpt), model_cfg=cfg)


if __name__ == "__main__":
    main()
