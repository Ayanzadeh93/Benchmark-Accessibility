"""Eager quantization-aware training (QAT) for CoDeQ (V2 plan section 8.5, runs Q1/Q2).

Why eager: ``prepare_qat_fx`` on stock timm MobileNetV4 inlines BatchNorm with training=True, so
44 of 45 BN layers keep updating (plan 8.5). This module never uses FX or torch.ao.quantization;
it rewrites a plain float model in place of its modules.

CONTRACT
  prepare_qat(model, cfg=None, example_input=None) -> (qat_model, report)
      Deep-copies ``model`` (the caller's float model is never modified), then
      1. folds every BatchNorm2d into the Conv2d that feeds it (W' = W*g/sqrt(v+eps),
         b' = (b-mean)*g/sqrt(v+eps) + beta) and replaces the BN with nn.Identity. Conv->BN pairs
         are found from the real dataflow of one eval forward on ``example_input`` (forward hooks on
         every leaf module), not from module names, and the fold is verified numerically in
         float64 against the unfolded model. A BN that is not fed by exactly one conv (or whose conv
         also feeds another module) raises unless cfg.strict_bn=False, in which case it becomes a
         FrozenBN (fixed per-channel affine), so no BN statistic can ever update during QAT;
      2. replaces every Conv2d with QConv2d: per-channel symmetric int8 weight fake-quant
         (grid [-127, 127], scale = max|w_c|/127 recomputed every forward, straight-through
         estimator; the same grid as edge/quant_sim.quantize_tensor and ORT symmetric int8);
      3. attaches per-tensor asymmetric uint8 activation fake-quant (ActFakeQuant: EMA min/max
         observer, momentum 0.01, range always contains 0) to the output of every ReLU/ReLU6, every
         Add module (class name in cfg.add_class_names), every Resize module (nn.Upsample* or class
         name in cfg.resize_class_names), the model input (cfg.quant_input) and every conv whose
         output is not consumed only by ReLU (cfg.quant_conv_outputs; e.g. linear bottleneck
         projections and head convs, whose outputs ONNX Runtime U8S8 QDQ also quantizes). The tensor
         that leaves the model is fake-quantized to cfg.output_bits when cfg.quant_output.
         Resize outputs (cfg.resize_share_input_qparams=True, default) reuse the qparams of the
         quantization point that feeds the Resize, with no observer of their own
         (SharedFakeQuant): ONNX Runtime's QDQResize quantizes a Resize output "same as input", so
         this is what the deployed int8 graph computes. False gives each Resize its own observer.
      Modules whose path matches a cfg.exclude fnmatch pattern stay float (no weight or output
      fake-quant): the "keep at most 5 sensitive layers in float" step of the recovery ladder.
      ``report`` lists every fold and quantization point, unhandled leaf types, shared modules and
      ``functional_inputs``: modules fed by a tensor no hooked module produced (an inline ``a + b``,
      ``F.interpolate``, ``torch.cat`` ...). Such functional ops get NO quantization point; the
      CoDeQ graph has none (tested). cfg.strict_graph=True turns a non-empty list into an error.
  QATConfig: dataclass of every knob above plus the plan schedule (epochs=6, lr 1e-4 -> 1e-5
      cosine, observers frozen from epoch index 4 on, i.e. after 4 completed epochs, AdamW
      wd .01, grad clip 1.0, EMA .999, AMP off, twin weights 1.0 / 10.0).
      QATConfig.from_dict(d) rejects unknown keys.
  Observers update ONLY when the ActFakeQuant module is in training mode and not frozen, so an
      eval forward (validation, D_SUN, benchmarks) can never calibrate the ranges.
  freeze_observers(m), set_observers(m, on), set_fake_quant(m, on), qat_epoch_start(m, epoch, cfg),
  init_observers(m, image_batches), uninitialized_points(m)
  observer_state_names(m) -> set of state_dict keys that belong to observers (for any external EMA)
  make_ema(m), ema_update_(ema, m, decay): EMA of parameters and non-observer float buffers;
      observer buffers/state are COPIED from the live model, never averaged ("excluded from EMA").
  make_float_twin(float_model): frozen eval-mode deep copy (p_f in the QAT loss).
  QATLoss(base_loss, w_abs=1.0, w_mean=10.0, mean_mode="per_image")(pred_q, pred_f, batch)
      -> (L + w_abs*mean|p_q - p_f| + w_mean*mean_b(mean_hw(p_q - p_f))^2, terms).
      base_loss follows distill/v2/losses.V2Loss: base_loss(pred_log, batch) -> (loss, terms)
      (a bare tensor return is accepted). mean_mode="batch" squares one mean over the batch.
  qat_lr(step, total_steps, lr_start, lr_end): cosine schedule, no warm-up.
  qat_finetune(float_model, batches, base_loss, cfg, ...) -> dict(model, ema, twin, history,
      report): the whole Q1 schedule in one loop (AMP forced off). ``batches(epoch)`` yields dicts
      with "image" (B,3,H,W) plus whatever base_loss reads (V2Dataset items collated).
      ``init_batches`` (optional, own calibration rows only) set the ranges before step 0.
      The final-epoch EMA is the artefact (plan rule: no dataset selects a checkpoint).
  strip_qat(qat_model) -> plain float model (folded convs, no fake-quant, no hooks); its output
      equals the QAT model with fake-quant disabled. This is what gets exported to ONNX.
  activation_ranges(qat_model) -> {point: {kind, module, rmin, rmax, scale, zero_point, bits,
      initialized, frozen, shares_qparams_with}}, point = "<module path>:out" or ":input".
  onnx_tensor_ranges(onnx_model, ranges, scope_prefix=None) -> {onnx tensor name: (rmin, rmax)},
      ready for edge/calib_chunked.overrides_from_ranges -> TensorQuantOverrides of the ONNX
      Runtime quantizer (plan 8.5 "pass QAT ranges through TensorQuantOverrides"). It reads node /
      tensor names of the TorchScript exporter ("/decoder/coarse/coarse.0/act/Relu") from any
      object shaped like onnx.ModelProto; it does not import onnx. scope_prefix=None auto-detects
      a wrapper prefix such as "model." (edge/export_v2.V2ExportWrapper). Unmapped points raise.

Env: conda "depth" (torch only; timm only if the model needs it). CPU and CUDA. Tests:
tests/test_qat.py (CPU, offline; its ONNX mapping test needs onnx, i.e. env depth_cmp, and is
skipped elsewhere).
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

RELU_TYPES = (nn.ReLU, nn.ReLU6)
RESIZE_TYPES = (nn.Upsample, nn.UpsamplingBilinear2d, nn.UpsamplingNearest2d)
# Modules that return their input object unchanged in eval: invisible to the dataflow trace.
TRANSPARENT_TYPES = (nn.Identity, nn.Dropout, nn.Dropout2d)
# Pure data movement: values stay on the producer's grid, so no new quantization point.
MOVEMENT_TYPES = (nn.PixelShuffle, nn.PixelUnshuffle, nn.Flatten)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class QATConfig:
    weight_bits: int = 8
    act_bits: int = 8
    output_bits: int = 8
    observer_momentum: float = 0.01
    quant_input: bool = True
    quant_conv_outputs: bool = True
    quant_output: bool = True
    exclude: Tuple[str, ...] = ()
    add_class_names: Tuple[str, ...] = ("Add",)
    resize_class_names: Tuple[str, ...] = ("Up2", "Resize", "Upsample2x", "Interpolate")
    resize_share_input_qparams: bool = True   # mirror ORT QDQResize ("output same as input")
    strict_bn: bool = True
    strict_graph: bool = False             # raise if a functional op feeds a module
    fold_verify_tol: float = 1e-6          # relative to max|ref| in float64
    example_hw: Tuple[int, int] = (64, 64)
    # schedule (plan 8.5)
    epochs: int = 6
    freeze_observers_after: int = 4        # observers frozen from epoch index 4 on (0-based)
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    twin_w_abs: float = 1.0
    twin_w_mean: float = 10.0
    twin_mean_mode: str = "per_image"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "QATConfig":
        if d is None:
            return cls()
        if isinstance(d, cls):
            return d
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise KeyError(f"unknown QATConfig keys: {bad}")
        kw = dict(d)
        for k in ("exclude", "add_class_names", "resize_class_names", "example_hw"):
            if k in kw and kw[k] is not None:
                kw[k] = tuple(kw[k])
        return cls(**kw)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, tuple):
                out[k] = list(v)
        return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# fake-quant primitives
# ─────────────────────────────────────────────────────────────────────────────────────────────
def fake_quant_weight(w: torch.Tensor, bits: int = 8, axis: int = 0) -> torch.Tensor:
    """Per-channel symmetric fake-quant on the restricted grid [-(2^(b-1)-1), 2^(b-1)-1].

    scale_c = max|w_c| / qmax (detached), straight-through gradient. All-zero channels stay zero.
    """
    qmax = 2 ** (bits - 1) - 1
    dims = [d for d in range(w.dim()) if d != axis]
    amax = w.detach().abs().amax(dim=dims) if dims else w.detach().abs()
    scale = (amax / qmax).to(torch.float32).clamp_min(1e-12)
    zp = torch.zeros_like(scale, dtype=torch.int32)
    return torch.fake_quantize_per_channel_affine(w, scale, zp, axis, -qmax, qmax)


def _act_qparams(min_val: torch.Tensor, max_val: torch.Tensor, qmin: int, qmax: int):
    rmin = torch.clamp(min_val, max=0.0)
    rmax = torch.clamp(max_val, min=0.0)
    scale = ((rmax - rmin) / float(qmax - qmin)).clamp_min(1e-8).to(torch.float32)
    zp = (qmin - torch.round(rmin / scale)).clamp(qmin, qmax).to(torch.int32)
    return scale, zp


class ActFakeQuant(nn.Module):
    """Per-tensor asymmetric unsigned fake-quant (uint8 by default) with an EMA min/max observer.

    - Observes only in training mode and while not frozen; the first observed batch sets the range.
    - Before any observation the module is a pass-through (nothing to quantize against).
    - Range always contains 0 (zero exactly representable), zero_point in [0, 2^bits - 1].
    - Flags live in Python attributes (no GPU sync per call) and persist through get/set_extra_state.
    """

    def __init__(self, bits: int = 8, momentum: float = 0.01, kind: str = ""):
        super().__init__()
        self.bits = int(bits)
        self.qmin, self.qmax = 0, 2 ** self.bits - 1
        self.momentum = float(momentum)
        self.kind = kind
        self.register_buffer("min_val", torch.tensor(0.0))
        self.register_buffer("max_val", torch.tensor(0.0))
        self.observer_on = True
        self.fq_on = True
        self.initialized = False

    def extra_repr(self) -> str:
        return (f"kind={self.kind}, bits={self.bits}, observer_on={self.observer_on}, "
                f"fq_on={self.fq_on}, initialized={self.initialized}")

    def get_extra_state(self):
        return {"observer_on": bool(self.observer_on), "fq_on": bool(self.fq_on),
                "initialized": bool(self.initialized), "bits": int(self.bits)}

    def set_extra_state(self, state):
        self.observer_on = bool(state["observer_on"])
        self.fq_on = bool(state["fq_on"])
        self.initialized = bool(state["initialized"])
        if int(state.get("bits", self.bits)) != self.bits:
            raise ValueError(f"bits mismatch: checkpoint {state['bits']} vs module {self.bits}")

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        xd = x.detach()
        bmin = xd.amin().to(self.min_val.dtype)
        bmax = xd.amax().to(self.max_val.dtype)
        if not self.initialized:
            self.min_val.copy_(bmin)
            self.max_val.copy_(bmax)
            self.initialized = True
        else:
            self.min_val.add_(self.momentum * (bmin - self.min_val))
            self.max_val.add_(self.momentum * (bmax - self.max_val))

    def qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return _act_qparams(self.min_val, self.max_val, self.qmin, self.qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.observer_on:
            self.observe(x)
        if not (self.fq_on and self.initialized):
            return x
        scale, zp = self.qparams()
        return torch.fake_quantize_per_tensor_affine(x, scale, zp, self.qmin, self.qmax)


class SharedFakeQuant(nn.Module):
    """Fake-quant with ANOTHER point's qparams and no observer (Resize output, ORT QDQResize).

    The source ActFakeQuant is held as a plain attribute, not a submodule, so it is not
    duplicated in state_dict / named_modules. copy.deepcopy of the whole model keeps the link
    inside the copy (deepcopy memo), so EMA copies and float twins stay self-consistent.
    """

    def __init__(self, source: ActFakeQuant, source_point: str, kind: str = "resize"):
        super().__init__()
        object.__setattr__(self, "_src", source)
        self.source_point = source_point
        self.kind = kind
        self.fq_on = True

    @property
    def source(self) -> ActFakeQuant:
        return self._src

    def extra_repr(self) -> str:
        return f"kind={self.kind}, shares={self.source_point}, fq_on={self.fq_on}"

    def get_extra_state(self):
        return {"fq_on": bool(self.fq_on), "source_point": self.source_point}

    def set_extra_state(self, state):
        if state.get("source_point", self.source_point) != self.source_point:
            raise ValueError(f"shared fake-quant source mismatch: {state['source_point']} vs "
                             f"{self.source_point}")
        self.fq_on = bool(state["fq_on"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self._src
        if not (self.fq_on and src.initialized):
            return x
        scale, zp = src.qparams()
        return torch.fake_quantize_per_tensor_affine(x, scale, zp, src.qmin, src.qmax)


class QConv2d(nn.Conv2d):
    """nn.Conv2d whose weight is per-channel symmetric fake-quantized in every forward.

    Subclasses nn.Conv2d so the parameter keys (".weight", ".bias") are unchanged; the on/off
    flag and bit width persist as ``_extra_state`` (bits are checked on load).
    """

    w_bits: int = 8
    w_fq_on: bool = True

    def get_extra_state(self):
        return {"w_fq_on": bool(self.w_fq_on), "w_bits": int(self.w_bits)}

    def set_extra_state(self, state):
        if int(state.get("w_bits", self.w_bits)) != int(self.w_bits):
            raise ValueError(f"weight bits mismatch: checkpoint {state['w_bits']} vs {self.w_bits}")
        self.w_fq_on = bool(state["w_fq_on"])

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, bits: int = 8) -> "QConv2d":
        q = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                conv.dilation, conv.groups, bias=conv.bias is not None,
                padding_mode=conv.padding_mode, device=conv.weight.device,
                dtype=conv.weight.dtype)
        with torch.no_grad():
            q.weight.copy_(conv.weight)
            if conv.bias is not None:
                q.bias.copy_(conv.bias)
        q.w_bits = int(bits)
        q.w_fq_on = True
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quant_weight(self.weight, self.w_bits, 0) if self.w_fq_on else self.weight
        return self._conv_forward(x, w, self.bias)

    def to_float(self) -> nn.Conv2d:
        c = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                      self.padding, self.dilation, self.groups, bias=self.bias is not None,
                      padding_mode=self.padding_mode, device=self.weight.device,
                      dtype=self.weight.dtype)
        with torch.no_grad():
            c.weight.copy_(self.weight)
            if self.bias is not None:
                c.bias.copy_(self.bias)
        return c


class FrozenBN(nn.Module):
    """BatchNorm2d frozen to its eval-mode affine (used only when strict_bn=False)."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        std = torch.sqrt(bn.running_var.double() + bn.eps)
        g = bn.weight.detach().double() if bn.affine else torch.ones_like(std)
        b = bn.bias.detach().double() if bn.affine else torch.zeros_like(std)
        scale = g / std
        self.register_buffer("scale", scale.float().view(1, -1, 1, 1))
        self.register_buffer("shift", (b - bn.running_mean.double() * scale).float().view(1, -1, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift


# ─────────────────────────────────────────────────────────────────────────────────────────────
# BN folding
# ─────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a new Conv2d (with bias) equal to bn(conv(x)) with bn in eval mode."""
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm without running statistics cannot be folded")
    if bn.num_features != conv.out_channels:
        raise ValueError(f"BN features {bn.num_features} != conv out_channels {conv.out_channels}")
    std = torch.sqrt(bn.running_var.double() + bn.eps)
    g = bn.weight.double() if bn.affine else torch.ones_like(std)
    beta = bn.bias.double() if bn.affine else torch.zeros_like(std)
    scale = g / std
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode, device=conv.weight.device,
                      dtype=conv.weight.dtype)
    w = conv.weight.double() * scale.view(-1, *([1] * (conv.weight.dim() - 1)))
    b0 = conv.bias.double() if conv.bias is not None else torch.zeros_like(std)
    fused.weight.copy_(w.to(conv.weight.dtype))
    fused.bias.copy_(((b0 - bn.running_mean.double()) * scale + beta).to(conv.weight.dtype))
    return fused


def _flatten_tensors(obj) -> List[torch.Tensor]:
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in _flatten_tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _flatten_tensors(o)]
    return []


def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    parent._modules[leaf] = new


def _default_example(model: nn.Module, hw: Tuple[int, int]) -> torch.Tensor:
    p = next(iter(model.parameters()), None)
    dev = p.device if p is not None else torch.device("cpu")
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 3, hw[0], hw[1], generator=g).to(dev)


@torch.no_grad()
def trace_dataflow(model: nn.Module, example_input: torch.Tensor) -> Dict[str, Any]:
    """One eval forward with hooks on every leaf module.

    Returns {"consumers": {producer: set(consumer)}, "inputs_from": {consumer: set(producer|None)},
    "calls": Counter(name), "types": {name: module}, "output_producer": name|None,
    "input_consumers": set(names)}. ``None`` as producer means a graph input or a tensor made by a
    functional op the hooks cannot see. Transparent modules (Identity/Dropout) are skipped so the
    tensor keeps its real producer. Every tensor is kept alive during the trace so ids are unique.
    """
    was_training = model.training
    model.eval()
    leaves = {n: m for n, m in model.named_modules() if len(m._modules) == 0 and n}
    records: List[Tuple[str, List[int], List[int]]] = []
    keep: List[torch.Tensor] = []
    handles = []

    def mk(name):
        def hook(mod, args, out):
            ins, outs = _flatten_tensors(args), _flatten_tensors(out)
            keep.extend(ins)
            keep.extend(outs)
            records.append((name, [id(t) for t in ins], [id(t) for t in outs]))
        return hook

    for n, m in leaves.items():
        if isinstance(m, TRANSPARENT_TYPES):
            continue
        handles.append(m.register_forward_hook(mk(n)))
    try:
        keep.append(example_input)
        out = model(example_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    out_t = _flatten_tensors(out)
    keep.extend(out_t)

    producer_of: Dict[int, str] = {}
    consumers: Dict[str, set] = defaultdict(set)
    inputs_from: Dict[str, set] = defaultdict(set)
    input_consumers: set = set()
    calls: Counter = Counter()
    in_id = id(example_input)
    for name, ins, outs in records:
        calls[name] += 1
        for tid in ins:
            p = producer_of.get(tid)
            inputs_from[name].add(p)
            if p is not None:
                consumers[p].add(name)
            elif tid == in_id:
                input_consumers.add(name)
        for tid in outs:
            producer_of[tid] = name
    output_producer = producer_of.get(id(out_t[0])) if out_t else None
    # consumers of a tensor that is neither the graph input nor a hooked module's output:
    # a functional op (inline add, F.interpolate, cat, ...) sits in between.
    functional_inputs = sorted(n for n, ps in inputs_from.items()
                               if None in ps and n not in input_consumers)
    del keep
    return {"consumers": consumers, "inputs_from": inputs_from, "calls": calls,
            "types": leaves, "output_producer": output_producer,
            "input_consumers": input_consumers, "functional_inputs": functional_inputs}


def _max_abs_out(a, b) -> Tuple[float, float]:
    ta, tb = _flatten_tensors(a), _flatten_tensors(b)
    if len(ta) != len(tb):
        raise RuntimeError("folded model returned a different number of tensors")
    diff = max(float((x - y).abs().max()) for x, y in zip(ta, tb)) if ta else 0.0
    ref = max(float(x.abs().max()) for x in ta) if ta else 0.0
    return diff, ref


@torch.no_grad()
def fold_bn_(model: nn.Module, example_input: Optional[torch.Tensor] = None,
             strict: bool = True, verify: bool = True, verify_tol: float = 1e-6,
             example_hw: Tuple[int, int] = (64, 64)) -> Dict[str, Any]:
    """Fold every Conv2d->BatchNorm2d pair of ``model`` IN PLACE. Returns a report dict.

    A BN qualifies when its only producer is one Conv2d and that conv feeds nothing but this BN.
    Remaining BNs raise (strict) or become FrozenBN. With ``verify`` the folded model is compared
    with the unfolded one in float64 (fallback float32 at tol 1e-3) and a mismatch raises.
    """
    x = example_input if example_input is not None else _default_example(model, example_hw)
    ref_model = copy.deepcopy(model).eval() if verify else None
    flow = trace_dataflow(model, x)
    types, consumers, inputs_from = flow["types"], flow["consumers"], flow["inputs_from"]
    bns = [n for n, m in types.items() if isinstance(m, nn.BatchNorm2d)]
    pairs, bad = [], []
    for bn_name in bns:
        prods = inputs_from.get(bn_name, set())
        if flow["calls"][bn_name] == 0:
            bad.append((bn_name, "never called in the example forward"))
            continue
        if len(prods) != 1 or None in prods:
            bad.append((bn_name, f"producers {sorted(map(str, prods))}"))
            continue
        conv_name = next(iter(prods))
        conv = types.get(conv_name)
        # exact types only: Conv2d subclasses (e.g. timm Conv2dSame) have their own forward
        if type(conv) not in (nn.Conv2d, QConv2d):
            bad.append((bn_name, f"producer {conv_name} is {type(conv).__name__}, not nn.Conv2d"))
            continue
        if consumers.get(conv_name, set()) != {bn_name}:
            bad.append((bn_name, f"{conv_name} also feeds {sorted(consumers[conv_name] - {bn_name})}"))
            continue
        if flow["calls"][conv_name] != flow["calls"][bn_name]:
            bad.append((bn_name, f"{conv_name} called {flow['calls'][conv_name]}x vs BN "
                                 f"{flow['calls'][bn_name]}x"))
            continue
        pairs.append((conv_name, bn_name))
    if bad and strict:
        lines = "\n  ".join(f"{n}: {why}" for n, why in bad)
        raise RuntimeError(f"{len(bad)} BatchNorm2d cannot be folded (strict_bn=True):\n  {lines}")
    for conv_name, bn_name in pairs:
        conv, bn = types[conv_name], types[bn_name]
        _set_module(model, conv_name, fold_conv_bn(conv, bn))
        _set_module(model, bn_name, nn.Identity())
    for bn_name, _ in bad:
        _set_module(model, bn_name, FrozenBN(types[bn_name]))
    # BNs that were not leaves of the trace cannot exist (BN has no children), but BNs never
    # reached by the example forward are in ``bad`` and are frozen/raised above.
    report = {"bn_total": len(bns), "bn_folded": len(pairs), "bn_frozen_unfolded": [n for n, _ in bad],
              "folded_pairs": pairs, "fold_verify_max_abs": None, "fold_verify_dtype": None}
    if verify:
        report.update(_verify_equal(ref_model, model, x, verify_tol))
    return report


@torch.no_grad()
def _verify_equal(ref_model: nn.Module, model: nn.Module, x: torch.Tensor, tol: float) -> Dict[str, Any]:
    was_training = model.training
    try:
        a, b = copy.deepcopy(ref_model).double().eval(), copy.deepcopy(model).double().eval()
        diff, ref = _max_abs_out(a(x.double()), b(x.double()))
        dtype, eff_tol = "float64", tol
    except Exception:  # noqa: BLE001 - some models hard-code float32 constants
        a, b = ref_model.eval(), model.eval()
        diff, ref = _max_abs_out(a(x), b(x))
        dtype, eff_tol = "float32", max(tol, 1e-3)
    model.train(was_training)
    if not math.isfinite(diff) or diff > eff_tol * max(1.0, ref):
        raise RuntimeError(f"BN folding changed the output: max|diff|={diff:.3e} "
                           f"(ref max {ref:.3e}, {dtype}, tol {eff_tol:g})")
    return {"fold_verify_max_abs": diff, "fold_verify_dtype": dtype}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# instrumentation
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _apply_out_fq(mod, args, out):
    return mod.qat_out(out)


def _apply_in_fq(mod, args):
    if args and torch.is_tensor(args[0]):
        return (mod.qat_in(args[0]),) + tuple(args[1:])
    return None


def _kind(m: nn.Module, cfg: QATConfig) -> str:
    if type(m) is nn.Conv2d:            # subclasses (Conv2dSame, QConv2d) are not rewritten
        return "conv"
    if isinstance(m, RELU_TYPES):
        return "relu"
    name = type(m).__name__
    if name in cfg.add_class_names:
        return "add"
    if isinstance(m, RESIZE_TYPES) or name in cfg.resize_class_names:
        return "resize"
    if isinstance(m, nn.BatchNorm2d):
        return "bn"
    if isinstance(m, FrozenBN):
        return "frozen_bn"
    if isinstance(m, TRANSPARENT_TYPES):
        return "transparent"
    if isinstance(m, MOVEMENT_TYPES):
        return "movement"
    return "other"


def _excluded(path: str, cfg: QATConfig) -> bool:
    return any(fnmatch.fnmatchcase(path, pat) or path.startswith(pat + ".") for pat in cfg.exclude)


def prepare_qat(model: nn.Module, cfg: Optional[Any] = None,
                example_input: Optional[torch.Tensor] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """Deep-copy, fold BN, insert weight and activation fake-quant. See module docstring."""
    cfg = QATConfig.from_dict(cfg) if not isinstance(cfg, QATConfig) else cfg
    m = copy.deepcopy(model)
    was_training = model.training
    x = example_input if example_input is not None else _default_example(m, cfg.example_hw)
    fold = fold_bn_(m, x, strict=cfg.strict_bn, verify=True, verify_tol=cfg.fold_verify_tol)

    flow = trace_dataflow(m, x)
    types, consumers, inputs_from = flow["types"], flow["consumers"], flow["inputs_from"]
    final = flow["output_producer"]
    if flow["functional_inputs"] and cfg.strict_graph:
        raise RuntimeError("functional ops feed these modules and would get no quantization point "
                           f"(strict_graph=True): {flow['functional_inputs']}")
    points: Dict[str, str] = {}
    fq_of: Dict[str, ActFakeQuant] = {}          # module path -> its own ActFakeQuant
    excluded, uncalled, unhandled = [], [], Counter()
    shared = sorted(n for n, c in flow["calls"].items() if c > 1)
    shared_resize: Dict[str, str] = {}
    n_wq = 0

    def attach(name: str, mod: nn.Module, kind: str) -> None:
        bits = cfg.output_bits if name == final else cfg.act_bits
        fq = ActFakeQuant(bits, cfg.observer_momentum, kind)
        mod.add_module("qat_out", fq)
        mod.register_forward_hook(_apply_out_fq)
        points[f"{name}:out"] = kind
        fq_of[name] = fq

    resizes: List[Tuple[str, nn.Module]] = []
    for name, mod in types.items():           # pass 1: conv / relu / add
        kind = _kind(mod, cfg)
        if kind == "other":
            unhandled[type(mod).__name__] += 1
        if kind not in ("conv", "relu", "add", "resize"):
            continue
        if flow["calls"][name] == 0:
            uncalled.append(name)
            continue
        if _excluded(name, cfg):
            excluded.append(name)
            continue
        if kind in ("resize", "movement"):
            resizes.append((name, mod))
            continue
        if kind == "conv":
            q = QConv2d.from_conv(mod, cfg.weight_bits)
            _set_module(m, name, q)
            mod = q
            n_wq += 1
            cons = consumers.get(name, set())
            only_relu = bool(cons) and all(isinstance(types[c], RELU_TYPES) for c in cons)
            if only_relu and name != final:
                continue
            if not cfg.quant_conv_outputs and name != final:
                continue
            kind = "conv_out"
        if name == final and not cfg.quant_output:
            continue
        attach(name, mod, kind)

    for name, mod in resizes:                 # pass 2: resize (may share its producer's qparams)
        if name == final and not cfg.quant_output:
            continue
        prods = inputs_from.get(name, set())
        src = next(iter(prods)) if len(prods) == 1 else None
        if (cfg.resize_share_input_qparams and name != final and src in fq_of
                and fq_of[src].bits == cfg.act_bits):
            mod.add_module("qat_out", SharedFakeQuant(fq_of[src], f"{src}:out", "resize"))
            mod.register_forward_hook(_apply_out_fq)
            points[f"{name}:out"] = "resize"
            shared_resize[f"{name}:out"] = f"{src}:out"
        else:
            attach(name, mod, "resize")

    if cfg.quant_input:
        m.add_module("qat_in", ActFakeQuant(cfg.act_bits, cfg.observer_momentum, "input"))
        m.register_forward_pre_hook(_apply_in_fq)
        points[":input"] = "input"

    m.train(was_training)
    report = {
        "fold": {k: v for k, v in fold.items() if k != "folded_pairs"},
        "n_folded_pairs": len(fold["folded_pairs"]),
        "conv_weight_quantized": n_wq,
        "act_points": dict(Counter(points.values())),
        "act_point_names": sorted(points),
        "resize_shares_qparams_with": shared_resize,
        "final_output_producer": final,
        "excluded": excluded,
        "uncalled_in_example_forward": uncalled,
        "shared_modules_called_more_than_once": shared,
        "functional_inputs": flow["functional_inputs"],
        "unhandled_leaf_types": dict(unhandled),
        "cfg": cfg.to_dict(),
    }
    return m, report


# ─────────────────────────────────────────────────────────────────────────────────────────────
# observer / fake-quant control
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _fq_modules(model: nn.Module) -> List[ActFakeQuant]:
    return [mm for mm in model.modules() if isinstance(mm, ActFakeQuant)]


def set_observers(model: nn.Module, enabled: bool) -> int:
    mods = _fq_modules(model)
    for q in mods:
        q.observer_on = bool(enabled)
    return len(mods)


def freeze_observers(model: nn.Module) -> int:
    """Stop every range update; fake-quant keeps running with the frozen ranges."""
    return set_observers(model, False)


def observers_frozen(model: nn.Module) -> bool:
    mods = _fq_modules(model)
    return bool(mods) and all(not q.observer_on for q in mods)


def set_fake_quant(model: nn.Module, enabled: bool) -> int:
    """Toggle weight and activation fake-quant together (observers are left as they are)."""
    n = 0
    for mm in model.modules():
        if isinstance(mm, (ActFakeQuant, SharedFakeQuant)):
            mm.fq_on = bool(enabled)
            n += 1
        elif isinstance(mm, QConv2d):
            mm.w_fq_on = bool(enabled)
            n += 1
    return n


def uninitialized_points(model: nn.Module) -> List[str]:
    """Module paths of ActFakeQuant observers that have never seen data (pass-through)."""
    return sorted(n for n, mm in model.named_modules()
                  if isinstance(mm, ActFakeQuant) and not mm.initialized)


def qat_epoch_start(model: nn.Module, epoch: int, cfg: Optional[Any] = None) -> Dict[str, Any]:
    """Call at the start of every QAT epoch (0-based). Freezes observers once epoch >= cfg value."""
    cfg = QATConfig.from_dict(cfg) if not isinstance(cfg, QATConfig) else cfg
    frozen = epoch >= cfg.freeze_observers_after
    set_observers(model, not frozen)
    return {"epoch": int(epoch), "observers_frozen": frozen}


@torch.no_grad()
def init_observers(model: nn.Module, batches: Iterable[torch.Tensor]) -> int:
    """Initialise ranges from a few batches of images (own training rows only, plan 5.1 / P6).

    Runs forwards with only the ActFakeQuant modules in training mode; returns #batches used.
    """
    was = model.training
    model.eval()
    mods = _fq_modules(model)
    for q in mods:
        q.train()
    n = 0
    for xb in batches:
        model(xb)
        n += 1
    model.train(was)
    return n


def observer_state_names(model: nn.Module) -> set:
    """state_dict keys owned by observers (min/max buffers and extra state)."""
    names = set()
    for mname, mm in model.named_modules():
        if isinstance(mm, ActFakeQuant):
            pre = f"{mname}." if mname else ""
            names.update({pre + "min_val", pre + "max_val", pre + "_extra_state"})
    return names


def activation_ranges(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Per quantization point: range and uint8 qparams, ready for TensorQuantOverrides.

    A shared point (Resize output) reports its source's range and names it in
    ``shares_qparams_with``; ``frozen`` then means the source observer is frozen.
    """
    out = {}
    for mname, mm in model.named_modules():
        if not isinstance(mm, (ActFakeQuant, SharedFakeQuant)):
            continue
        owner = mname.rsplit(".", 1)[0] if "." in mname else ""
        point = ":input" if mm.kind == "input" else f"{owner}:out"
        src = mm.source if isinstance(mm, SharedFakeQuant) else mm
        scale, zp = src.qparams()
        out[point] = {
            "kind": mm.kind, "module": owner, "bits": src.bits,
            "rmin": float(torch.clamp(src.min_val, max=0.0)),
            "rmax": float(torch.clamp(src.max_val, min=0.0)),
            "scale": float(scale), "zero_point": int(zp),
            "initialized": bool(src.initialized), "frozen": not src.observer_on,
            "shares_qparams_with": mm.source_point if isinstance(mm, SharedFakeQuant) else None,
        }
    return out


def _onnx_module_path(scoped: str) -> Optional[str]:
    """TorchScript-exporter scope -> torch module path, or None.

    "/decoder/coarse/coarse.0/coarse.0.1/act/Relu"            -> "decoder.coarse.0.1.act"
    "/decoder/up_c/Resize_output_0"                           -> "decoder.up_c"
    Container children appear as "<parent>.<idx>" components that repeat the parent's name.
    """
    if not scoped.startswith("/"):
        return None
    comps = [c for c in scoped.split("/") if c]
    if len(comps) < 2:
        return ""                                   # op at the root module
    comps = comps[:-1]                              # drop "<Op>" / "<Op>_output_<k>"
    parts: List[str] = []
    prev = None
    for c in comps:
        if prev is not None and c.startswith(prev + "."):
            parts.append(c[len(prev) + 1:])
        else:
            parts.append(c)
        prev = c
    return ".".join(parts)


def onnx_tensor_ranges(onnx_model: Any, ranges: Dict[str, Dict[str, Any]],
                       scope_prefix: Optional[str] = None,
                       skip_uninitialized: bool = True) -> Dict[str, Tuple[float, float]]:
    """Map activation_ranges() points onto tensor names of an ONNX graph exported from
    strip_qat(model) by the TorchScript exporter. Returns {tensor: (rmin, rmax)}.

    ``onnx_model`` is anything shaped like onnx.ModelProto (graph.node[*].name/.op_type/.output,
    graph.input[*].name, graph.initializer[*].name). ``scope_prefix`` is the module path of the
    QAT model inside the exported wrapper ("model." for edge/export_v2.V2ExportWrapper); None
    tries "" and every one-component prefix seen in the graph and keeps the one that maps all
    points. The point of a module is the output of the LAST non-Constant node in its scope (the
    exporter emits nodes in topological order). Raises KeyError listing unmapped points.
    """
    g = onnx_model.graph
    inits = {i.name for i in g.initializer}
    graph_inputs = [i.name for i in g.input if i.name not in inits]
    last_out: Dict[str, str] = {}
    for node in g.node:
        if node.op_type == "Constant" or not node.output:
            continue
        path = _onnx_module_path(node.name) if node.name else None
        if path is None:
            path = _onnx_module_path(node.output[0])
        if path is not None:
            last_out[path] = node.output[0]

    pts = {p: r for p, r in ranges.items() if r["initialized"] or not skip_uninitialized}

    def try_prefix(pre: str) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
        got, miss = {}, []
        for p, r in pts.items():
            if p == ":input":
                if not graph_inputs:
                    miss.append(p)
                    continue
                got[graph_inputs[0]] = (r["rmin"], r["rmax"])
                continue
            mod = pre + p[:-len(":out")]
            t = last_out.get(mod)
            if t is None:
                miss.append(p)
            else:
                got[t] = (r["rmin"], r["rmax"])
        return got, miss

    cands = [scope_prefix] if scope_prefix is not None else (
        [""] + sorted({k.split(".")[0] + "." for k in last_out if k}))
    best = None
    for pre in cands:
        got, miss = try_prefix(pre)
        if not miss:
            return got
        if best is None or len(miss) < len(best[1]):
            best = (pre, miss)
    raise KeyError(f"{len(best[1])} QAT points not found in the ONNX graph (prefix {best[0]!r}): "
                   f"{best[1][:10]}")


def strip_qat(model: nn.Module) -> nn.Module:
    """Plain float copy: QConv2d -> Conv2d, fake-quant modules and their hooks removed."""
    m = copy.deepcopy(model)
    if isinstance(m, QConv2d):
        return m.to_float()
    for name, mm in list(m.named_modules()):
        if isinstance(mm, QConv2d):
            _set_module(m, name, mm.to_float())   # drops its qat_out child and hook with it
    for mm in m.modules():
        for hid, fn in list(mm._forward_hooks.items()):
            if fn is _apply_out_fq:
                del mm._forward_hooks[hid]
        for hid, fn in list(mm._forward_pre_hooks.items()):
            if fn is _apply_in_fq:
                del mm._forward_pre_hooks[hid]
        for key in ("qat_out", "qat_in"):
            if key in mm._modules:
                del mm._modules[key]
    return m


# ─────────────────────────────────────────────────────────────────────────────────────────────
# EMA, float twin, loss, schedule
# ─────────────────────────────────────────────────────────────────────────────────────────────
def make_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update_(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA over parameters and non-observer float buffers; observer state copied verbatim."""
    obs = observer_state_names(model)
    msd = model.state_dict()
    for k, v in ema.state_dict().items():
        if k in obs or not torch.is_tensor(v):
            continue
        src = msd[k]
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(src.detach().to(v.dtype), alpha=1.0 - decay)
        else:
            v.copy_(src)
    for (_, e), (_, s) in zip(ema.named_modules(), model.named_modules()):
        if isinstance(s, ActFakeQuant):
            e.min_val.copy_(s.min_val)
            e.max_val.copy_(s.max_val)
            e.set_extra_state(s.get_extra_state())


def make_float_twin(float_model: nn.Module) -> nn.Module:
    """Frozen eval-mode copy of the float model: p_f in the QAT loss."""
    t = copy.deepcopy(float_model).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


class QATLoss:
    """L_QAT = L(p_q) + w_abs*mean|p_q - p_f| + w_mean*(mean(p_q - p_f))^2 (plan section 4).

    p_f is detached. mean_mode "per_image" squares the per-image mean signed shift and averages
    over the batch (penalises per-image scale drift); "batch" squares one mean over everything.
    Twin terms use every pixel: agreement with the float model needs no GT validity.
    """

    def __init__(self, base_loss: Callable, w_abs: float = 1.0, w_mean: float = 10.0,
                 mean_mode: str = "per_image"):
        if mean_mode not in ("per_image", "batch"):
            raise ValueError(f"mean_mode must be per_image or batch, got {mean_mode}")
        self.base_loss, self.w_abs, self.w_mean, self.mean_mode = base_loss, w_abs, w_mean, mean_mode

    def __call__(self, pred_q: torch.Tensor, pred_f: torch.Tensor, batch: Dict[str, Any]):
        res = self.base_loss(pred_q, batch)
        base, terms = (res if isinstance(res, tuple) else (res, {}))
        terms = dict(terms)
        d = pred_q - pred_f.detach()
        l_abs = d.abs().mean()
        mean = d.flatten(1).mean(dim=1) if self.mean_mode == "per_image" else d.mean()
        l_mean = (mean ** 2).mean()
        total = base + self.w_abs * l_abs + self.w_mean * l_mean
        terms.update({"qat_base": float(base.detach()), "qat_twin_abs": float(l_abs.detach()),
                      "qat_twin_mean2": float(l_mean.detach()), "qat_total": float(total.detach())})
        return total, terms


def qat_lr(step: int, total_steps: int, lr_start: float = 1e-4, lr_end: float = 1e-5) -> float:
    """Cosine from lr_start (step 0) to lr_end (last step); no warm-up (fine-tuning)."""
    if total_steps <= 1:
        return lr_start
    t = min(max(step, 0), total_steps - 1) / (total_steps - 1)
    return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * t))


def _to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def qat_finetune(float_model: nn.Module, batches: Callable[[int], Iterable[Dict[str, Any]]],
                 base_loss: Callable, cfg: Optional[Any] = None, device: str = "cpu",
                 example_input: Optional[torch.Tensor] = None,
                 log: Optional[Callable[[str], None]] = None,
                 steps_per_epoch: Optional[int] = None,
                 init_batches: Optional[Iterable[torch.Tensor]] = None) -> Dict[str, Any]:
    """Run the plan's QAT schedule. AMP is forced off. Returns model/ema/twin/history/report.

    ``steps_per_epoch`` (if given) sizes the cosine schedule; otherwise it is learned from the
    first epoch's length and the remaining epochs are assumed to match. ``init_batches`` (image
    tensors from OWN calibration rows only, plan 5.1 / P6) initialise the observers before the
    first step; without them the first training batch sets every range. The caller's float model
    is deep-copied (prepare_qat) and only moved to ``device``.
    """
    cfg = QATConfig.from_dict(cfg) if not isinstance(cfg, QATConfig) else cfg
    float_model = float_model.to(device)
    ex = example_input.to(device) if example_input is not None else None
    qmodel, report = prepare_qat(float_model, cfg, ex)
    if init_batches is not None:
        report["observer_init_batches"] = init_observers(qmodel, (xb.to(device) for xb in init_batches))
    twin = make_float_twin(float_model)
    ema = make_ema(qmodel)
    qloss = QATLoss(base_loss, cfg.twin_w_abs, cfg.twin_w_mean, cfg.twin_mean_mode)
    opt = torch.optim.AdamW([p for p in qmodel.parameters() if p.requires_grad],
                            lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    history: List[Dict[str, Any]] = []
    step, total = 0, (steps_per_epoch or 0) * cfg.epochs
    dev_type = torch.device(device).type
    lr = cfg.lr_start
    for epoch in range(cfg.epochs):
        st = qat_epoch_start(qmodel, epoch, cfg)
        qmodel.train()
        n_ep, sums = 0, Counter()
        for batch in batches(epoch):
            lr = qat_lr(step, total, cfg.lr_start, cfg.lr_end) if total else cfg.lr_start
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            b = _to_device(batch, device)
            with torch.autocast(device_type=dev_type, enabled=False):
                pred_q = qmodel(b["image"])
                with torch.no_grad():
                    pred_f = twin(b["image"])
                loss, terms = qloss(pred_q.float(), pred_f.float(), b)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite QAT loss at epoch {epoch} step {step}: {terms}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(qmodel.parameters(), cfg.grad_clip)
            opt.step()
            ema_update_(ema, qmodel, cfg.ema_decay)
            sums.update({k: v for k, v in terms.items() if isinstance(v, (int, float))})
            n_ep += 1
            step += 1
        if total == 0 and epoch == 0:
            total = n_ep * cfg.epochs          # schedule sized from the first epoch
        if epoch == 0:
            missing = uninitialized_points(qmodel)
            if missing:
                raise RuntimeError(f"{len(missing)} observers saw no data in epoch 0: {missing[:5]}")
        ranges = activation_ranges(qmodel)
        rec = {**st, "steps": n_ep, "lr_last": lr if n_ep else None,
               "range_checksum": sum(r["rmax"] - r["rmin"] for r in ranges.values()),
               **{k: v / max(n_ep, 1) for k, v in sums.items()}}
        history.append(rec)
        if log:
            log(json.dumps(rec))
    ema.eval()
    return {"model": qmodel, "ema": ema, "twin": twin, "history": history, "report": report}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# CLI: instrumentation report for a V2 model (no training)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Print the QAT folding/instrumentation report.")
    ap.add_argument("--variant", default="N", help="CoDeQ variant P|N|S (distill.v2.model)")
    ap.add_argument("--decoder-mode", default="ps")
    ap.add_argument("--hw", type=int, nargs=2, default=[64, 64])
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args(argv)
    from distill.v2.model import build_v2_model  # noqa: WPS433 - optional, other agent's module
    model = build_v2_model({"variant": a.variant, "decoder_mode": a.decoder_mode,
                            "pretrained": False}).eval()
    cfg = QATConfig(exclude=tuple(a.exclude), example_hw=tuple(a.hw))
    _, rep = prepare_qat(model, cfg)
    rep.pop("cfg")
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
