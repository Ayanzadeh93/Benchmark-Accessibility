"""Audit what ONNX Runtime actually executes after graph optimization (plan 8.2 step 5, gate I5).

  python -m edge.verify_runtime outputs/v2/edge/<run>/u8s8.onnx [--json audit.json]

Env: depth_cmp (onnxruntime CPU).

A QDQ file only says what MAY run in integer arithmetic. ORT fuses DequantizeLinear -> op ->
QuantizeLinear patterns into integer kernels (QLinearConv, com.microsoft QLinearAdd, ...) only
where it has a kernel; everything else runs in float between a DQ/Q pair. So the audit is done
on the OPTIMIZED graph that ORT writes with SessionOptions.optimized_model_filepath.

CONTRACT  audit(path, level="all") -> dict with
  source_ops       {op: count} of the file on disk
  runtime_ops      {op: count} after ORT optimization (CPU EP, this machine)
  conv_total       Conv + ConvTranspose nodes in the source graph (excluding none)
  conv_integer     integer conv kernels at runtime (QLinearConv, ConvInteger)
  add_total / add_integer   Add vs QLinearAdd
  float_nodes      [{name, op_type}] every runtime node that consumes or produces a float tensor,
                   excluding QuantizeLinear / DequantizeLinear (listed as qdq_boundaries)
  float_op_counts  {op: count} of float_nodes
  float_conv / float_add     float Conv-like / Add-like nodes left at runtime
  i5_pass          True iff float_conv == 0 and float_add == 0 (plan 8.3 I5: all Conv and Add
                   run as QLinear kernels; the remaining float nodes are LISTED, not failed)
  pure_integer     True iff float_nodes is empty (Resize paths make this False - plan E4)
Data types are propagated through the optimized graph by `infer_dtypes` (ORT writes contrib ops
that onnx shape inference cannot type).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import tempfile
from typing import Dict, List, Optional

FLOAT_T = {1, 10, 11, 16}          # FLOAT, FLOAT16, DOUBLE, BFLOAT16
INT_T = {2, 3, 4, 5, 6, 7, 12, 13, 17, 18, 19, 20, 21, 22}
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
INTEGER_CONV = {"QLinearConv", "ConvInteger", "NhwcQLinearConv"}
INTEGER_ADD = {"QLinearAdd"}
CONV_LIKE = {"Conv", "ConvTranspose", "FusedConv", "NhwcFusedConv", "Conv_nchwc"}
ADD_LIKE = {"Add", "Sum"}
PASS_THROUGH = {"Reshape", "Transpose", "Resize", "MaxPool", "Concat", "Slice", "Squeeze",
                "Unsqueeze", "Flatten", "DepthToSpace", "SpaceToDepth", "Identity", "Split",
                "Pad", "Gather", "Expand", "Tile", "Upsample", "ReorderInput", "ReorderOutput",
                "NhwcMaxPool", "GatherElements"}
# integer kernels whose output dtype is the dtype of their output zero point (input index)
_QLINEAR_OUT_ZP = {"QLinearConv": 7, "QLinearAdd": 7, "QLinearMul": 7, "QLinearMatMul": 7,
                   "QLinearSigmoid": 4, "QLinearLeakyRelu": 4, "QLinearSoftmax": 4,
                   "QLinearAveragePool": 4, "QLinearGlobalAveragePool": 4, "QLinearConcat": 1,
                   "QLinearReduceMean": 4, "QLinearWhere": 8, "QGemm": 8, "NhwcQLinearConv": 7}


def optimized_model(path: str, level: str = "all", out_path: Optional[str] = None,
                    threads: int = 1):
    """Create a CPU session with the given optimization level and return the optimized ModelProto."""
    import onnx
    import onnxruntime as ort
    lv = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
          "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[level]
    tmp = None
    if out_path is None:
        tmp = tempfile.TemporaryDirectory(prefix="ort_opt_")
        out_path = os.path.join(tmp.name, "optimized.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = lv
    so.optimized_model_filepath = out_path
    so.intra_op_num_threads = threads
    ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    m = onnx.load(out_path)
    if tmp is not None:
        tmp.cleanup()
    return m


def infer_dtypes(model) -> Dict[str, int]:
    """Propagate TensorProto element types through a (possibly contrib-op) graph."""
    from onnx import helper
    dt: Dict[str, int] = {}
    for i in model.graph.initializer:
        dt[i.name] = i.data_type
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        et = vi.type.tensor_type.elem_type
        if et:
            dt.setdefault(vi.name, et)
    for n in model.graph.node:            # graph is topologically sorted by ORT
        ins = [dt.get(x) for x in n.input]
        t = n.op_type
        if t == "QuantizeLinear":
            od = None
            for a in n.attribute:
                if a.name == "output_dtype" and helper.get_attribute_value(a):
                    od = int(helper.get_attribute_value(a))
            outs = [od or (ins[2] if len(ins) > 2 and ins[2] else 2)]
        elif t == "DequantizeLinear":
            outs = [ins[1] if len(ins) > 1 and ins[1] else 1]
        elif t in _QLINEAR_OUT_ZP:
            k = _QLINEAR_OUT_ZP[t]
            outs = [ins[k] if len(ins) > k and ins[k] in INT_T else 1]
        elif t in ("ConvInteger", "MatMulInteger"):
            outs = [6]
        elif t in ("MatMulIntegerToFloat", "DynamicQuantizeMatMul"):
            outs = [1]
        elif t in ("DynamicQuantizeLinear",):
            outs = [2, 1, 2]
        elif t in ("Shape", "Size", "NonZero"):
            outs = [7]
        elif t == "Cast":
            to = [helper.get_attribute_value(a) for a in n.attribute if a.name == "to"]
            outs = [int(to[0]) if to else None]
        elif t in PASS_THROUGH:
            outs = [ins[0] if ins else None]
        else:
            fl = [x for x in ins if x in FLOAT_T]
            outs = [fl[0] if fl else (ins[0] if ins else None)]
        for j, o in enumerate(n.output):
            if o and o not in dt:
                dt[o] = outs[j] if j < len(outs) else outs[0]
    return dt


def _is_float_node(n, dt: Dict[str, int]) -> bool:
    if n.op_type in QDQ_OPS:
        return False
    if n.op_type in _QLINEAR_OUT_ZP or n.op_type in ("ConvInteger", "MatMulInteger"):
        return False        # integer kernel; its float inputs are scales
    if n.op_type in ("MatMulIntegerToFloat", "DynamicQuantizeMatMul"):
        return False
    data_in = [x for x in n.input if x]
    return any(dt.get(x) in FLOAT_T for x in list(data_in) + list(n.output))


def audit(path: str, level: str = "all", keep_optimized: Optional[str] = None) -> dict:
    """See module docstring."""
    import onnx
    src = onnx.load(path)
    src_ops = collections.Counter(n.op_type for n in src.graph.node)
    opt = optimized_model(path, level, keep_optimized)
    rt_ops = collections.Counter(n.op_type for n in opt.graph.node)
    dt = infer_dtypes(opt)
    float_nodes: List[dict] = []
    for n in opt.graph.node:
        if _is_float_node(n, dt):
            float_nodes.append({"name": n.name, "op_type": n.op_type, "domain": n.domain or "ai.onnx"})
    fcount = collections.Counter(f["op_type"] for f in float_nodes)
    float_conv = sum(v for k, v in fcount.items() if k in CONV_LIKE)
    float_add = sum(v for k, v in fcount.items() if k in ADD_LIKE)
    res = {
        "path": os.path.abspath(path).replace("\\", "/"), "optimization_level": level,
        "source_ops": dict(sorted(src_ops.items())), "runtime_ops": dict(sorted(rt_ops.items())),
        "conv_total": int(src_ops.get("Conv", 0) + src_ops.get("ConvTranspose", 0)),
        "conv_integer": int(sum(rt_ops.get(k, 0) for k in INTEGER_CONV)),
        "add_total": int(src_ops.get("Add", 0)),
        "add_integer": int(sum(rt_ops.get(k, 0) for k in INTEGER_ADD)),
        "qdq_boundaries": {"QuantizeLinear": int(rt_ops.get("QuantizeLinear", 0)),
                           "DequantizeLinear": int(rt_ops.get("DequantizeLinear", 0))},
        "float_nodes": float_nodes, "float_op_counts": dict(sorted(fcount.items())),
        "float_conv": int(float_conv), "float_add": int(float_add),
        "i5_pass": bool(float_conv == 0 and float_add == 0),
        "pure_integer": not float_nodes,
    }
    try:
        import onnxruntime as ort
        res["onnxruntime"] = ort.__version__
    except Exception:  # noqa: BLE001
        pass
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("onnx")
    ap.add_argument("--level", default="all", choices=["basic", "extended", "all"])
    ap.add_argument("--json", default="")
    ap.add_argument("--keep-optimized", default="")
    a = ap.parse_args()
    r = audit(a.onnx, a.level, a.keep_optimized or None)
    short = {k: v for k, v in r.items() if k != "float_nodes"}
    print(json.dumps(short, indent=2))
    for f in r["float_nodes"]:
        print(f"  float: {f['op_type']:<20} {f['name']}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
