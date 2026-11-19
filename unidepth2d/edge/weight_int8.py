"""Weight-only int8 ONNX transform (artefact "W8", plan 8.1): int8 weights + DequantizeLinear.

  python -m edge.weight_int8 outputs/edge/b1_mix/fp32_defaultK.onnx            # -> w8.onnx next to it

Env: depth_cmp (onnx, onnxruntime).

CONTRACT
  * Every float weight initializer consumed ONLY as the weight input of Conv / ConvTranspose /
    Gemm (input B) / MatMul (2-D input B) is replaced by three initializers
        <w>_q8 int8, <w>_s8 float32 scale (per output channel), <w>_z8 int8 zeros
    and a DequantizeLinear(<w>_q8, <w>_s8, <w>_z8, axis=<out-channel axis>) node whose output
    feeds the consumer(s). Biases, BN-folded constants, Resize scales etc. stay float.
  * Quantization is symmetric, restricted range: q = clip(round(w / s), -127, 127) with
    s = max|w_channel| / 127; an all-zero channel gets s = 1 (q = 0, so it dequantizes to 0).
    This is the same grid as edge/quant_sim.py (8 bits, per channel), so the simulated and the
    ONNX W8 numbers are directly comparable.
  * Output-channel axis: Conv 0, ConvTranspose 1, Gemm transB=0 -> 1 / transB=1 -> 0, MatMul 1.
  * Requires opset >= 13 (per-axis DequantizeLinear). Activations stay float: this is a SIZE
    option, not a speed option (plan E10: it ran slower than fp32 on x86).
  * `dequantize_weights(model)` reconstructs {original weight name: float32 array} from a W8
    model, which is what the round-trip test checks (|w - dq(q(w))| <= s/2 per channel).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

QMAX = 127
DEFAULT_OPS = ("Conv", "ConvTranspose", "Gemm", "MatMul")
_SUFFIX_Q, _SUFFIX_S, _SUFFIX_Z, _SUFFIX_DQ = "_q8", "_s8", "_z8", "_dq8"


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            from onnx import helper
            return helper.get_attribute_value(a)
    return default


def _weight_axis(node, w_ndim: int) -> Optional[int]:
    """Output-channel axis of `node`'s weight input, or None if the node/shape is unsupported."""
    t = node.op_type
    if t == "Conv":
        return 0
    if t == "ConvTranspose":
        return 1
    if t == "Gemm" and w_ndim == 2:
        return 0 if int(_attr(node, "transB", 0)) else 1
    if t == "MatMul" and w_ndim == 2:
        return 1
    return None


def quantize_per_channel(w: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-channel int8: returns (q int8, scale float32 of shape (w.shape[axis],))."""
    w = np.asarray(w, dtype=np.float32)
    red = tuple(i for i in range(w.ndim) if i != axis)
    amax = np.abs(w).max(axis=red) if red else np.abs(w)
    scale = (amax / QMAX).astype(np.float32)
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    shape = [1] * w.ndim
    shape[axis] = -1
    q = np.clip(np.rint(w / scale.reshape(shape)), -QMAX, QMAX).astype(np.int8)
    return q, scale


def dequantize_per_channel(q: np.ndarray, scale: np.ndarray, axis: int) -> np.ndarray:
    shape = [1] * q.ndim
    shape[axis] = -1
    return q.astype(np.float32) * np.asarray(scale, np.float32).reshape(shape)


def quantize_weights_int8(src, dst: Optional[str] = None, op_types: Iterable[str] = DEFAULT_OPS,
                          min_elements: int = 0) -> Tuple[object, dict]:
    """Return (W8 ModelProto, stats). Writes `dst` (plus a sidecar .json if src has one) when given.

    min_elements: weights with fewer elements stay float (0 = quantize all eligible weights).
    """
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(src) if isinstance(src, (str, os.PathLike)) else src
    opset = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
    if opset < 13:
        raise ValueError(f"per-axis DequantizeLinear needs opset >= 13, model has {opset}")
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    ops = set(op_types)

    consumers = collections.defaultdict(list)
    for n in g.node:
        for idx, name in enumerate(n.input):
            if name:
                consumers[name].append((n, idx))
    graph_outputs = {o.name for o in g.output}

    new_inits, dq_nodes, replaced = [], [], {}
    bytes_before = bytes_after = 0
    n_skipped_shared = 0
    for name, init in inits.items():
        if init.data_type != onnx.TensorProto.FLOAT or name in graph_outputs:
            continue
        uses = consumers.get(name, [])
        if not uses:
            continue
        axes = set()
        ok = True
        w = None
        for node, idx in uses:
            weight_idx = 1
            if node.op_type not in ops or idx != weight_idx:
                ok = False
                break
            if w is None:
                w = numpy_helper.to_array(init)
            ax = _weight_axis(node, w.ndim)
            if ax is None:
                ok = False
                break
            axes.add(ax)
        if not ok or len(axes) != 1:
            # count only real weights left float (used as a weight input somewhere); biases and
            # other non-weight inputs of Conv/Gemm are float by design and are not counted
            if any(n.op_type in ops and i == 1 for n, i in uses):
                n_skipped_shared += 1
            continue
        if w.size < min_elements:
            continue
        axis = axes.pop()
        q, s = quantize_per_channel(w, axis)
        z = np.zeros_like(s, dtype=np.int8)
        new_inits += [numpy_helper.from_array(q, name + _SUFFIX_Q),
                      numpy_helper.from_array(s, name + _SUFFIX_S),
                      numpy_helper.from_array(z, name + _SUFFIX_Z)]
        dq_nodes.append(helper.make_node("DequantizeLinear", [name + _SUFFIX_Q, name + _SUFFIX_S,
                                                              name + _SUFFIX_Z],
                                         [name + _SUFFIX_DQ], name=name + "_DequantizeLinear8",
                                         axis=axis))
        replaced[name] = axis
        bytes_before += w.size * 4
        bytes_after += q.size + s.size * 4 + z.size

    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in replaced:
                n.input[i] = inp + _SUFFIX_DQ
    keep = [i for i in g.initializer if i.name not in replaced]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    nodes = dq_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(model)

    stats = {"n_weights_int8": len(replaced), "n_skipped_shared_or_unsupported": n_skipped_shared,
             "weight_bytes_fp32": int(bytes_before), "weight_bytes_int8": int(bytes_after),
             "axes": dict(collections.Counter(replaced.values())), "op_types": sorted(ops)}
    if dst:
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        onnx.save(model, dst)
        stats["file_mb"] = round(os.path.getsize(dst) / 1e6, 3)
        if isinstance(src, (str, os.PathLike)):
            stats["src_file_mb"] = round(os.path.getsize(src) / 1e6, 3)
            side = os.path.splitext(str(src))[0] + ".json"
            if os.path.exists(side):
                with open(side, encoding="utf-8") as f:
                    meta = json.load(f)
                meta.update({"precision": "w8", "fp32_file_mb": meta.get("file_mb"),
                             "file_mb": stats["file_mb"],
                             "quant": {"tool": "edge.weight_int8", "scheme": "weight-only int8, "
                                       "symmetric per-channel, DequantizeLinear", **stats}})
                with open(os.path.splitext(dst)[0] + ".json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
    return model, stats


def dequantize_weights(model) -> Dict[str, np.ndarray]:
    """{original weight name: float32 array} reconstructed from a W8 model's int8 initializers."""
    from onnx import numpy_helper
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type == "DequantizeLinear" and n.output[0].endswith(_SUFFIX_DQ):
            axis = int(_attr(n, "axis", 1))
            q, s = inits[n.input[0]], inits[n.input[1]]
            out[n.output[0][: -len(_SUFFIX_DQ)]] = dequantize_per_channel(q, s, axis)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("onnx")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.onnx), "w8.onnx")
    _, stats = quantize_weights_int8(a.onnx, out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
