"""U8S8 static QDQ post-training quantization, int8 gates I1-I6 and the X0 run (plan 8.2-8.4, A14).

  python -m edge.ptq quantize --onnx outputs/v2/edge/<run>/fp32.onnx [--method minmax|percentile]
         [--output-precision uint8|int16|float_tail] [--roles configs/v2/roles.yaml]
  python -m edge.ptq x0 [--models b1_mix S16_prox] [--out outputs/v2/x0_current_models.json]

Env: depth_cmp (onnx 1.23.2, onnxruntime 1.32.0 CPU). CPU only.

PROCEDURE (plan 8.2; each step is a function below)
  1. `prepare`: onnxruntime quant_pre_process (symbolic shape inference; falls back to
     skip_symbolic_shape when the symbolic pass fails).
  2. Calibration: edge.calib_chunked.collect_ranges_chunked - repeated collect_data on chunks of
     at most 8 images. CalibMaxIntermediateOutputs is NEVER set (plan E5: ORT 1.32 silently
     drops chunks); `quantize_u8s8` rejects it in extra_options. Calibration images are the own
     TRAINING calibration ids of configs/v2/roles.yaml (calibration.own), validated by
     calib_chunked.calibration_ids_from_roles. Benchmark, D_SUN, validation and frozen-test
     images never calibrate.
  3. MinMax and Percentile 99.99 are compared on the own VALIDATION slice (roles val.own) only;
     `select_method` keeps the higher I1 (ties -> MinMax).
  4. `quantize_u8s8`: the chunked ranges go to quantize_static twice over - as its calibration
     cache (so ORT does not recalibrate) and as TensorQuantOverrides (rmin/rmax per tensor, ORT's
     own Relu/Clip range rule pre-applied) - with QDQ format, uint8 asymmetric activations, int8
     symmetric per-channel weights (U8S8).
  5. Output-precision ablation (run O1), `OUTPUT_PRECISIONS`:
       uint8       everything quantized, the graph output included.
       int16       every activation tensor produced by the TAIL (the last Conv(s) through to the
                   graph output, `tail_nodes`) is QUInt16 via TensorQuantOverrides quant_type
                   (ORT bumps the opset to 21 for 16-bit Q/DQ). No x86 integer kernel exists for
                   these, so the tail Convs run in float at runtime (audit lists them).
       float_tail  the tail nodes are excluded from quantization (the edge/ort_int8 "head float"
                   rule): float output, tail Convs in float.
     `select_output_precision` applies the O1 rule: keep uint8 unless its D1 is more than 0.3
     points below the best alternative or its corrected boundary F1 more than 5 % below it.
  6. Gates (benchmark/safety.py is the single implementation; this module only wires it):
       I1-I4  safety.Int8Agreement + safety.int8_gates (I4 needs per-image reference rows).
       I5     edge.verify_runtime.audit on the ORT-optimized graph (all Conv and Add integer;
              remaining float nodes are listed).
       I6     safety regression gates measured in the same pass: SF1 [int8] incl. the int8 rise
              rule, SF3 [int8] and SF4 (quantization regression of the corridor alert, each
              artefact recalibrated). SF2/SF5/SF6 need R0, perturbation or video runs and are
              not part of this pass (reported as not evaluated).
     The recovery trigger of plan 8.4 (`needs_recovery`): D1 drop > 1.0 point, or I1/I2/I3 fail.

X0 (amendment A14, `run_x0`): for S16_prox and b1_mix, export with DEFAULT K baked in (camera
mode "default-k", never GT K), build weight-only int8 (edge/weight_int8.py, the guaranteed
artefact) and U8S8 PTQ, decide method / output precision / recovery on the own validation slice,
run the sensitivity ladder (edge/sensitivity.py) on failures, then report I1-I6 on the frozen own
test (test.frozen_own) and, when Z: is reachable, I1-I4 on IBims-1. Frozen-test and IBims numbers
are reported only; they select nothing. Own-image "GT" is the UniDepthV2-L pseudo-label cache
(teacher agreement, not sensor GT) and every own-image result says so.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from edge.calib_chunked import (MAX_CHUNK, PERCENTILE, CalibResult, CalibrationSourceError,
                                assert_no_max_intermediate, calibration_ids_from_roles,
                                collect_ranges_chunked, load_roles, manifest_feeds,
                                overrides_from_ranges, read_manifest, roles_ids,
                                save_ranges, save_ranges_cache)

OUTPUT_PRECISIONS = ("uint8", "int16", "float_tail")
METHODS = ("minmax", "percentile")
DEFAULT_MANIFEST = os.path.join("outputs", "data_gt", "manifest.csv")
DEFAULT_ROLES = os.path.join("configs", "v2", "roles.yaml")
X0_MODELS = {"b1_mix": os.path.join("outputs", "checkpoints", "b1_mix", "best.pt"),
             "S16_prox": os.path.join("outputs", "checkpoints", "S16_prox", "best.pt")}
OWN_REFERENCE = ("UniDepthV2-L pseudo-label cache (outputs/data_gt/own/depth_mm): teacher "
                 "agreement, NOT sensor ground truth")
O1_MAX_D1_LOSS_PTS = 0.3
O1_MAX_BF_REL_LOSS = 0.05
RECOVERY_D1_DROP_PTS = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(fp32_path: str, pre_path: Optional[str] = None) -> str:
    """quant_pre_process(fp32) -> <stem>.pre.onnx (symbolic shapes, else skip_symbolic_shape)."""
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre_path = pre_path or os.path.splitext(fp32_path)[0] + ".pre.onnx"
    try:
        quant_pre_process(fp32_path, pre_path, skip_symbolic_shape=False)
    except Exception as e:  # noqa: BLE001 - symbolic shape inference is optional
        print(f"[ptq] pre-process with symbolic shapes failed ({type(e).__name__}); retrying without")
        quant_pre_process(fp32_path, pre_path, skip_symbolic_shape=True)
    return pre_path


def tail_nodes(model) -> List[str]:
    """Nodes from the last Conv (inclusive) on every path back from a graph output.

    Same walk as edge/ort_int8.head_nodes: backwards from each output through producers, stopping
    a branch at its first Conv. For CoDeQ "ps": Add, Resize, coarse Conv, DepthToSpace, residual
    Conv. For a legacy student: Clip/Slice, head Conv.
    """
    producer = {o: n for n in model.graph.node for o in n.output}
    names: List[str] = []
    for out in model.graph.output:
        frontier, seen = [out.name], set()
        while frontier:
            t = frontier.pop()
            n = producer.get(t)
            if n is None or n.name in seen:
                continue
            seen.add(n.name)
            if n.name not in names:
                names.append(n.name)
            if n.op_type == "Conv":
                continue
            frontier.extend(i for i in n.input if i in producer)
    return names


def tail_tensors(model, nodes: Optional[Sequence[str]] = None) -> List[str]:
    """Float activation tensors PRODUCED by the tail nodes (what the int16 variant widens)."""
    nodes = set(tail_nodes(model) if nodes is None else nodes)
    inits = {i.name for i in model.graph.initializer}
    out: List[str] = []
    for n in model.graph.node:
        if n.name in nodes:
            out.extend(o for o in n.output if o and o not in inits)
    return out


def _ensure_node_names(model) -> int:
    """Give every unnamed node a unique name (nodes_to_exclude and audits work by name)."""
    used = {n.name for n in model.graph.node if n.name}
    k = 0
    for i, n in enumerate(model.graph.node):
        if not n.name:
            nm = f"{n.op_type}_{i}"
            while nm in used:
                nm += "_"
            n.name = nm
            used.add(nm)
            k += 1
    return k


# ──────────────────────────────────────────────────────────────────────────────
# Quantization
# ──────────────────────────────────────────────────────────────────────────────

def quantize_u8s8(pre_path: str, out_path: str, ranges: Dict[str, Tuple[float, float]],
                  method: str, output_precision: str = "uint8",
                  nodes_to_exclude: Sequence[str] = (), per_channel: bool = True,
                  use_overrides: bool = True, extra_options: Optional[dict] = None,
                  meta: Optional[dict] = None, quiet: bool = True) -> dict:
    """Static QDQ U8S8 quantization of the pre-processed float model with precomputed ranges.

    ranges: {tensor: (rmin, rmax)} from collect_ranges_chunked (same pre-processed model).
    Writes out_path plus a sidecar <out>.json (the fp32 sidecar `meta` updated with "quant").
    Returns the sidecar dict.
    """
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from edge.calib_chunked import _ort_method
    if output_precision not in OUTPUT_PRECISIONS:
        raise ValueError(f"output_precision must be one of {OUTPUT_PRECISIONS}")
    assert_no_max_intermediate(extra_options)
    model = onnx.load(pre_path)
    tail = tail_nodes(model)
    exclude = list(dict.fromkeys(list(nodes_to_exclude) + (tail if output_precision == "float_tail" else [])))
    qtypes = None
    if output_precision == "int16":
        qtypes = {t: QuantType.QUInt16 for t in tail_tensors(model, tail)}
    opts = {"ActivationSymmetric": False, "WeightSymmetric": True, **(extra_options or {})}
    if use_overrides:
        opts["TensorQuantOverrides"] = overrides_from_ranges(model, ranges, qtypes, exclude)
    elif qtypes:
        opts["TensorQuantOverrides"] = {t: [{"quant_type": q}] for t, q in qtypes.items()}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="ptq_cache_", ignore_cleanup_errors=True) as td:
        cache = save_ranges_cache(ranges, method, os.path.join(td, "calib_cache.json"))
        sink = io.StringIO()
        ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
        with ctx:
            quantize_static(pre_path, out_path, calibration_data_reader=None,
                            quant_format=QuantFormat.QDQ, per_channel=per_channel,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            calibrate_method=_ort_method(method), nodes_to_exclude=exclude,
                            extra_options=opts, calibration_cache_path=cache)
    out_meta = dict(meta or {})
    out_meta.update({
        "precision": f"int8-u8s8-{output_precision}",
        "fp32_file_mb": (meta or {}).get("file_mb"),
        "file_mb": round(os.path.getsize(out_path) / 1e6, 3),
        "sha256": _sha256(out_path),
        "quant": {"tool": "edge.ptq.quantize_u8s8 (onnxruntime.quantize_static)", "format": "QDQ",
                  "activation_type": "QUInt8 asymmetric", "weight_type": "QInt8 symmetric",
                  "per_channel": per_channel, "calibration_method": method,
                  "percentile": PERCENTILE if method == "percentile" else None,
                  "output_precision": output_precision, "tail_nodes": tail,
                  "int16_tensors": sorted(qtypes) if qtypes else [],
                  "nodes_excluded": exclude, "n_nodes_excluded": len(exclude),
                  "tensor_quant_overrides": bool(use_overrides),
                  "calib_max_intermediate_outputs": None,
                  "onnxruntime": _ort_version(), "build_s": round(time.time() - t0, 1)},
    })
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)
    return out_meta


def _ort_version() -> Optional[str]:
    try:
        import onnxruntime as ort
        return ort.__version__
    except Exception:  # noqa: BLE001
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Runners and evaluation
# ──────────────────────────────────────────────────────────────────────────────

class OrtLogDepth:
    """run_bench-style runner for a single-input ONNX student (CoDeQ or default-K legacy).

    predict(rgb uint8 HxWx3, K) -> {"depth": metres at model resolution}; K is IGNORED because the
    graph has no intrinsics input (camera_mode "none" for CoDeQ, "default-k" for legacy files).
    Preprocessing is edge/ort_int8.preprocess (INTER_AREA to the model size, ImageNet norm), the
    same as edge/onnx_runner.OnnxStudent. Predictions are memoised per image id when `cache`.
    """

    def __init__(self, onnx_path: str, name: Optional[str] = None, threads: int = 0,
                 hw: Optional[Tuple[int, int]] = None, log_clamp: Optional[Tuple[float, float]] = None,
                 cache: bool = False):
        import onnxruntime as ort
        side = os.path.splitext(onnx_path)[0] + ".json"
        self.meta = {}
        if os.path.exists(side):
            with open(side, encoding="utf-8") as f:
                self.meta = json.load(f)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        if len(ins) != 1:
            raise ValueError(f"{onnx_path}: expected one graph input, got {[i.name for i in ins]}")
        self.input_name = ins[0].name
        shp = ins[0].shape
        self.h, self.w = hw or (int(shp[2]), int(shp[3]))
        lo = self.meta.get("log_min", math.log(0.2))
        hi = self.meta.get("log_max", math.log(30.0))
        self.log_clamp = log_clamp or (float(lo), float(hi))
        self.name = name or self.meta.get("name", os.path.basename(onnx_path))
        self.path = onnx_path
        self._cache: Optional[Dict[str, np.ndarray]] = {} if cache else None
        self._last_id: Optional[str] = None

    def log_depth(self, feed: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: feed})[0][0, 0]

    def predict(self, rgb: np.ndarray, K=None, sample_id: Optional[str] = None) -> dict:
        from edge.ort_int8 import preprocess
        key = sample_id if sample_id is not None else None
        if self._cache is not None and key is not None and key in self._cache:
            return {"depth": self._cache[key], "conf": None, "points": None}
        ld = self.log_depth(preprocess(rgb, self.h, self.w)).astype(np.float32)
        d = np.exp(np.clip(ld, *self.log_clamp))
        if self._cache is not None and key is not None:
            self._cache[key] = d
        return {"depth": d, "conf": None, "points": None}


class _IdBound:
    """Adapter so safety.run_safety_eval's model.predict(rgb, K) can use the per-id cache."""

    def __init__(self, runner: OrtLogDepth):
        self.runner, self.name, self._sid = runner, runner.name, None

    def bind(self, sid):
        self._sid = sid

    def predict(self, rgb, K=None):
        return self.runner.predict(rgb, K, sample_id=self._sid)


def own_samples(ids: Sequence[str], manifest: Dict[str, dict]) -> Iterator:
    """benchmark.datasets_tier1.EvalSample for own rows: rgb, teacher depth (mm png), valid."""
    import cv2
    from benchmark.datasets_tier1 import EvalSample
    for i in ids:
        r = manifest[i]
        bgr = cv2.imread(r["img"], cv2.IMREAD_COLOR)
        d16 = cv2.imread(r["depth"], cv2.IMREAD_UNCHANGED)
        if bgr is None or d16 is None:
            raise FileNotFoundError(f"{i}: unreadable image or depth ({r['img']}, {r['depth']})")
        gt = d16.astype(np.float32) / 1000.0
        lo, hi = float(r.get("dmin") or 0.2), float(r.get("dmax") or 30.0)
        valid = np.isfinite(gt) & (gt >= lo) & (gt <= hi)
        yield EvalSample(id=i, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gt=gt, valid=valid, K=None,
                         min_depth=lo, max_depth=hi)


class _BindingSamples:
    """Iterates samples and binds each sample id to the runners before safety uses them."""

    def __init__(self, samples: Iterable, runners: Sequence[_IdBound]):
        self.samples, self.runners = samples, runners

    def __iter__(self):
        for s in self.samples:
            for r in self.runners:
                r.bind(str(s.id))
            yield s


def evaluate_int8(fp32: OrtLogDepth, q: OrtLogDepth, samples: Iterable, cfg: Optional[dict] = None,
                  clusters: Optional[dict] = None, boundary: bool = True) -> dict:
    """One safety pass (benchmark.safety.run_safety_eval) of int8 `q` against its fp32 parent.

    Returns the safety report (gates I1-I4, SF1/SF3 for both, SF1[int8] rise, SF4). No
    perturbations, no anchors. Predictions of `fp32` are memoised by id when it was built with
    cache=True, so one fp32 pass serves every artefact.
    """
    from benchmark import safety
    cfg = cfg or safety.default_config()
    a, b = _IdBound(fp32), _IdBound(q)
    rep = safety.run_safety_eval(a, _BindingSamples(samples, [a, b]), cfg, int8_model=b,
                                 clusters=clusters, perturb=False, anchors=False, boundary=boundary,
                                 camera_mode="default-k" if fp32.meta.get("camera_mode") == "default-k"
                                 else "none", log_clamp=fp32.log_clamp)
    return rep


def gate_summary(report: dict) -> dict:
    """Compact view of a safety report: {gate/metric: (value, status)} plus I1-I6 verdicts."""
    gates = report.get("gates", [])
    flat = {}
    for g in gates:
        key = f"{g['gate']}:{g['metric']}"
        flat[key] = {"value": g["value"], "threshold": g["threshold"], "status": g["status"]}

    def status(prefix, metric_filter=None):
        sel = [g for g in gates if g["gate"] == prefix and (metric_filter is None or metric_filter(g))]
        if not sel:
            return "not_evaluated"
        st = [g["status"] for g in sel]
        if "fail" in st:
            return "fail"
        if all(s == "pass" for s in st):
            return "pass"
        return "partial" if "pass" in st else st[0]

    m = report.get("metrics", {})
    rows_f, rows_q = m.get("rows", []), m.get("int8_rows", [])
    d1f = float(np.mean([r["d1"] for r in rows_f]) * 100) if rows_f else None
    d1q = float(np.mean([r["d1"] for r in rows_q]) * 100) if rows_q else None
    from benchmark.protocol_v2 import BF_KEY

    def bf(rows):
        v = [r.get(BF_KEY) for r in rows if r.get(BF_KEY) is not None]
        return float(np.mean(v)) if v else None

    # I6 = the quantization-REGRESSION safety gates: SF1 int8 rise and SF4 (corridor alert,
    # per-artefact recalibration). Absolute SF1/SF3 statuses of each artefact are listed apart.
    i6_parts = [g for g in gates if (g["gate"] == "SF1" and g["metric"].startswith("int8_rise"))
                or g["gate"] == "SF4"]
    i6 = "not_evaluated"
    if i6_parts:
        st = [g["status"] for g in i6_parts]
        i6 = "fail" if "fail" in st else ("pass" if all(s == "pass" for s in st) else "partial")
    absolute = {f"{g['gate']}:{g['metric']}": g["status"] for g in gates
                if g["gate"] in ("SF1", "SF3") and not g["metric"].startswith("int8_rise")}
    agree = m.get("int8_agreement", {})
    return {"n_images": report.get("meta", {}).get("n_images"),
            "I1_frac_within_1.25x": agree.get("I1_frac_within"),
            "I2_p99_abs_dlog_near": agree.get("I2_p_abs_dlog_near"),
            "I3_mean_shift_near": agree.get("I3_mean_signed_shift_near"),
            "near_ref": agree.get("near_ref"),
            "d1_fp32": d1f, "d1_int8": d1q, "d1_drop_pts": (d1f - d1q) if d1f is not None and d1q is not None else None,
            "bf_fp32": bf(rows_f), "bf_int8": bf(rows_q),
            "I1": status("I1"), "I2": status("I2"), "I3": status("I3"), "I4": status("I4"),
            "I6_safety_regression": i6, "safety_absolute": absolute, "gates": flat}


def needs_recovery(summary: dict) -> bool:
    """Plan 8.4 trigger: D1 drop > 1.0 point, or I1/I2/I3 not passing."""
    if any(summary.get(k) != "pass" for k in ("I1", "I2", "I3")):
        return True
    dd = summary.get("d1_drop_pts")
    return dd is not None and dd > RECOVERY_D1_DROP_PTS


def select_method(summaries: Dict[str, dict]) -> str:
    """Higher I1 on the validation slice; ties (within 1e-6) -> minmax."""
    best, best_v = "minmax", summaries.get("minmax", {}).get("I1_frac_within_1.25x") or -1.0
    for m, s in summaries.items():
        v = s.get("I1_frac_within_1.25x") or -1.0
        if v > best_v + 1e-6:
            best, best_v = m, v
    return best


def select_output_precision(summaries: Dict[str, dict]) -> Tuple[str, str]:
    """O1 rule: keep uint8 unless D1 loss > 0.3 pt or BF loss > 5 % vs the best alternative."""
    u = summaries.get("uint8")
    if not u:
        raise ValueError("uint8 summary required")
    alts = {k: v for k, v in summaries.items() if k != "uint8" and v}
    if not alts:
        return "uint8", "no alternative evaluated"
    best = max(alts, key=lambda k: (alts[k].get("d1_int8") or -1e9))
    b = alts[best]
    d1_loss = (b.get("d1_int8") or 0.0) - (u.get("d1_int8") or 0.0)
    bf_b, bf_u = b.get("bf_int8"), u.get("bf_int8")
    bf_loss = ((bf_b - bf_u) / bf_b) if (bf_b and bf_u is not None and bf_b > 0) else 0.0
    if d1_loss > O1_MAX_D1_LOSS_PTS or bf_loss > O1_MAX_BF_REL_LOSS:
        return best, (f"uint8 loses {d1_loss:.3f} D1 pts / {100 * bf_loss:.1f}% BF vs {best} "
                      f"(limits {O1_MAX_D1_LOSS_PTS} pt / {100 * O1_MAX_BF_REL_LOSS:.0f}%)")
    return "uint8", (f"uint8 within limits: D1 loss {d1_loss:.3f} pt, BF loss {100 * bf_loss:.1f}% "
                     f"vs best alternative {best}")


# ──────────────────────────────────────────────────────────────────────────────
# Roles: calibration, validation slice, frozen test
# ──────────────────────────────────────────────────────────────────────────────

def resolve_roles(roles_path: str, manifest_path: str, n_val: int, seed: int = 2027) -> dict:
    """Calibration ids (validated allowlist), a seeded val.own slice and test.frozen_own."""
    roles = load_roles(roles_path)
    calib = calibration_ids_from_roles(roles_path, manifest_path)
    val = roles_ids(roles, "val", "own")
    frozen = roles_ids(roles, "test", "frozen_own")
    val_slice = sorted(val)
    random.Random(seed).shuffle(val_slice)
    val_slice = sorted(val_slice[:n_val])
    overlap = (set(calib) | set(val_slice)) & set(frozen)
    if overlap or set(calib) & set(val_slice):
        raise CalibrationSourceError(f"role overlap between calibration/val/frozen: {sorted(overlap)[:5]}")
    return {"source": os.path.abspath(roles_path).replace("\\", "/"),
            "roles_sha256": _sha256(roles_path), "provisional": False,
            "calibration": calib, "val_slice": val_slice, "val_total": len(val),
            "frozen_test": frozen,
            "ids_sha256": {k: hashlib.sha256("\n".join(v).encode()).hexdigest()
                           for k, v in (("calibration", calib), ("val_slice", val_slice),
                                        ("frozen_test", frozen))}}


# ──────────────────────────────────────────────────────────────────────────────
# One model through the whole chain
# ──────────────────────────────────────────────────────────────────────────────

def _calib_feeds(ids, man, hw, input_name="image"):
    return manifest_feeds([man[i] for i in ids], hw[0], hw[1], input_name)


def _val_summary(fp32: OrtLogDepth, qpath: str, val_ids, man, cfg) -> dict:
    q = OrtLogDepth(qpath)
    rep = evaluate_int8(fp32, q, own_samples(val_ids, man), cfg)
    return gate_summary(rep)


def chain_one_model(name: str, fp32_path: str, roles: dict, man: Dict[str, dict], out_dir: str,
                    methods: Sequence[str] = METHODS, precisions: Sequence[str] = OUTPUT_PRECISIONS,
                    sensitivity: bool = True, n_sens_images: int = 2, ibims: bool = True,
                    log: Callable[[str], None] = print) -> dict:
    """Steps 1-6 for one exported fp32 single-input model. Returns the per-model report."""
    from benchmark import safety
    from edge import sensitivity as sens
    from edge.verify_runtime import audit
    from edge.weight_int8 import quantize_weights_int8
    cfg = safety.default_config()
    with open(os.path.splitext(fp32_path)[0] + ".json", encoding="utf-8") as f:
        fmeta = json.load(f)
    hw = tuple(fmeta["train_size"])
    rep: dict = {"model": name, "fp32": {k: fmeta.get(k) for k in (
        "name", "ckpt", "ckpt_sha256", "camera_mode", "default_K", "parity_max_abs_logdepth",
        "file_mb", "sha256", "params_M", "exporter", "opset", "op_histogram")}}
    rep["fp32"]["path"] = fp32_path.replace("\\", "/")
    t_model = time.time()

    # ── weight-only int8 (guaranteed artefact) ──────────────────────────────────
    w8_path = os.path.join(out_dir, "w8.onnx")
    _, wstats = quantize_weights_int8(fp32_path, w8_path)
    rep["w8"] = {"path": w8_path.replace("\\", "/"), "file_mb": wstats.get("file_mb"),
                 "sha256": _sha256(w8_path), "stats": wstats}
    log(f"[x0:{name}] w8 {wstats.get('file_mb')} MB")

    # ── calibration ─────────────────────────────────────────────────────────────
    pre = prepare(fp32_path, os.path.join(out_dir, "fp32.pre.onnx"))
    import onnx
    pm = onnx.load(pre)
    if _ensure_node_names(pm):
        onnx.save(pm, pre)
    ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    rep["calibration"] = {"ids_source": roles["source"], "provisional": roles["provisional"],
                          "n_ids": len(roles["calibration"]),
                          "ids_sha256": roles["ids_sha256"]["calibration"], "methods": {}}
    for m in methods:
        t0 = time.time()
        res = collect_ranges_chunked(pre, _calib_feeds(roles["calibration"], man, hw), m,
                                     MAX_CHUNK, verbose=False)
        ranges[m] = res.ranges
        rpath = os.path.join(out_dir, f"ranges_{m}.json")
        save_ranges(rpath, res, {"calibration_ids_sha256": roles["ids_sha256"]["calibration"],
                                 "ids_source": roles["source"]})
        rep["calibration"]["methods"][m] = {"n_images": res.n_images, "n_chunks": res.n_chunks,
                                            "chunk_size": res.chunk_size, "n_tensors": len(res.ranges),
                                            "ranges_file": rpath.replace("\\", "/"),
                                            "seconds": round(time.time() - t0, 1)}
        log(f"[x0:{name}] calibrated {m}: {res.n_images} imgs in {res.n_chunks} chunks of <= "
            f"{res.chunk_size} ({time.time() - t0:.0f}s)")

    fp32 = OrtLogDepth(fp32_path, cache=True)
    val_ids = roles["val_slice"]

    def build(tag, m, prec, excl=()):
        path = os.path.join(out_dir, f"{tag}.onnx")
        qm = quantize_u8s8(pre, path, ranges[m], m, prec, excl, meta=fmeta)
        return path, qm

    # ── method selection on the validation slice (uint8 output) ──────────────
    variants: Dict[str, dict] = {}
    sel_m = {}
    for m in methods:
        tag = f"ptq_{m}_uint8"
        path, qm = build(tag, m, "uint8")
        s = _val_summary(fp32, path, val_ids, man, cfg)
        variants[tag] = {"path": path.replace("\\", "/"), "file_mb": qm["file_mb"], "sha256": qm["sha256"],
                         "method": m, "output_precision": "uint8", "nodes_excluded": [], "val": s}
        sel_m[m] = s
        log(f"[x0:{name}] val {tag}: I1={s['I1_frac_within_1.25x']:.4f} d1 {s['d1_fp32']:.2f}->{s['d1_int8']:.2f}")
    method = select_method(sel_m)
    rep["method_selection"] = {"slice": f"val.own ({len(val_ids)} of {roles['val_total']} ids)",
                               "rule": "higher I1 on the own validation slice; ties -> minmax",
                               "chosen": method,
                               "I1": {m: sel_m[m]["I1_frac_within_1.25x"] for m in sel_m}}

    # ── output precision ablation (O1) on the validation slice ───────────────
    prec_sum = {"uint8": sel_m[method]}
    for prec in precisions:
        if prec == "uint8":
            continue
        tag = f"ptq_{method}_{prec}"
        path, qm = build(tag, method, prec)
        s = _val_summary(fp32, path, val_ids, man, cfg)
        variants[tag] = {"path": path.replace("\\", "/"), "file_mb": qm["file_mb"], "sha256": qm["sha256"],
                         "method": method, "output_precision": prec,
                         "nodes_excluded": qm["quant"]["nodes_excluded"], "val": s}
        prec_sum[prec] = s
        log(f"[x0:{name}] val {tag}: I1={s['I1_frac_within_1.25x']:.4f} d1 {s['d1_int8']:.2f}")
    prec_choice, prec_reason = select_output_precision(prec_sum)
    rep["output_precision"] = {"slice": "val.own", "chosen": prec_choice, "reason": prec_reason,
                               "rule": f"keep uint8 unless D1 loss > {O1_MAX_D1_LOSS_PTS} pt or BF "
                                       f"loss > {100 * O1_MAX_BF_REL_LOSS:.0f}% (plan O1)",
                               "d1_int8": {k: v["d1_int8"] for k, v in prec_sum.items()},
                               "bf_int8": {k: v["bf_int8"] for k, v in prec_sum.items()},
                               "I1": {k: v["I1_frac_within_1.25x"] for k, v in prec_sum.items()}}
    chosen_tag = f"ptq_{method}_{prec_choice}"

    # ── recovery ladder on failure (plan 8.4 steps 1-2) ─────────────────────────
    final_tag = chosen_tag
    rep["recovery"] = {"triggered": needs_recovery(variants[chosen_tag]["val"]),
                       "trigger_rule": "D1 drop > 1.0 pt or I1/I2/I3 not pass (plan 8.4), on val.own"}
    if sensitivity and rep["recovery"]["triggered"]:
        t0 = time.time()
        qpath = variants[chosen_tag]["path"]
        sfeeds = list(_calib_feeds(roles["calibration"][:n_sens_images], man, hw))
        ranking = sens.layer_sensitivity(pre, qpath, sfeeds)
        rep["recovery"]["ranking_images"] = f"{len(sfeeds)} calibration images (own train)"
        rep["recovery"]["ranking_top20"] = ranking["nodes"][:20]
        rep["recovery"]["ranking_method"] = ranking["method"]

        def eval_fn(excl, k):
            tag = f"ptq_{method}_{prec_choice}_float{k}"
            path, qm = build(tag, method, prec_choice, excl)
            s = _val_summary(fp32, path, val_ids, man, cfg)
            variants[tag] = {"path": path.replace("\\", "/"), "file_mb": qm["file_mb"],
                             "sha256": qm["sha256"], "method": method, "output_precision": prec_choice,
                             "nodes_excluded": qm["quant"]["nodes_excluded"], "val": s,
                             "ladder_k": k}
            log(f"[x0:{name}] ladder k={k}: I1={s['I1_frac_within_1.25x']:.4f} d1 {s['d1_int8']:.2f}")
            return s, tag

        lad = sens.float_ladder([r["node"] for r in ranking["nodes"]], eval_fn, max_float=5,
                                passes=lambda s: not needs_recovery(s))
        rep["recovery"]["ladder"] = lad["rungs"]
        rep["recovery"]["ladder_passed"] = lad["passed"]
        final_tag = lad["best_tag"] if lad["best_tag"] else chosen_tag
        rep["recovery"]["seconds"] = round(time.time() - t0, 1)
        rep["recovery"]["final"] = final_tag
        # diagnostic only (not a ladder rung, may exceed 5 layers): which blocks break int8
        try:
            groups = sens.default_groups(onnx.load(pre))
            gres = sens.group_sensitivity(pre, ranges[method], method, prec_choice, groups,
                                          lambda p: _val_summary(fp32, p, val_ids[:16], man, cfg),
                                          os.path.join(out_dir, "groups"), quantize_u8s8)
            rep["recovery"]["group_diagnostic"] = gres
        except Exception as e:  # noqa: BLE001 - diagnostic must not kill the run
            rep["recovery"]["group_diagnostic"] = {"error": f"{type(e).__name__}: {e}"}
    rep["variants"] = variants
    rep["chosen_ptq"] = final_tag

    # ── runtime audit (I5) for every PTQ artefact ─────────────────────────────
    for tag, v in variants.items():
        a = audit(v["path"])
        v["runtime_audit"] = {k: a[k] for k in ("conv_total", "conv_integer", "add_total", "add_integer",
                                                "float_op_counts", "float_conv", "float_add",
                                                "i5_pass", "pure_integer", "qdq_boundaries")}
        v["runtime_audit"]["float_nodes"] = a["float_nodes"]

    # ── frozen own test: I1-I6, reported only ────────────────────────────────────
    frozen = roles["frozen_test"]
    rep["frozen_test"] = {"n_ids": len(frozen), "reference": OWN_REFERENCE,
                          "ids_sha256": roles["ids_sha256"]["frozen_test"],
                          "selects_nothing": True, "artefacts": {}}
    eval_tags = ["w8", f"ptq_{method}_uint8"]
    for t in (chosen_tag, final_tag):
        if t not in eval_tags:
            eval_tags.append(t)
    for tag in eval_tags:
        path = w8_path if tag == "w8" else variants[tag]["path"]
        t0 = time.time()
        r = evaluate_int8(fp32, OrtLogDepth(path), own_samples(frozen, man), cfg)
        s = gate_summary(r)
        if tag != "w8":
            s["I5"] = "pass" if variants[tag]["runtime_audit"]["i5_pass"] else "fail"
        else:
            s["I5"] = "n/a (weight-only: activations float by design)"
        rep["frozen_test"]["artefacts"][tag] = s
        with open(os.path.join(out_dir, f"frozen_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, default=str)
        log(f"[x0:{name}] frozen {tag}: I1={s['I1_frac_within_1.25x']:.4f} d1 {s['d1_fp32']:.2f}->"
            f"{s['d1_int8']:.2f} I4={s['I4']} ({time.time() - t0:.0f}s)")

    # ── IBims-1 when reachable (sensor GT, report only) ─────────────────────────
    if ibims:
        rep["ibims"] = _ibims_block(fp32_path, {t: (w8_path if t == "w8" else variants[t]["path"])
                                                for t in eval_tags}, cfg, out_dir, log, name)

    # ── quick latency (x86 proxy, noisy, informative only) ─────────────────────
    rep["latency_ms_4threads_median20"] = _quick_latency(
        {"fp32": fp32_path, "w8": w8_path, final_tag: variants[final_tag]["path"]}, hw)
    rep["seconds"] = round(time.time() - t_model, 1)
    return rep


def _ibims_block(fp32_path, paths: Dict[str, str], cfg, out_dir, log, name) -> dict:
    root = os.environ.get("DATASETS_ROOT", "Z:/Datasets")
    try:
        from benchmark.safety import require_datasets_root
        require_datasets_root("ibims", root)
    except SystemExit as e:
        return {"status": "not reachable", "detail": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"status": "not reachable", "detail": f"{type(e).__name__}: {e}"}
    from benchmark.datasets_tier1 import iter_ibims
    try:
        samples = list(iter_ibims())
    except Exception as e:  # noqa: BLE001 - Z: is unstable: report, do not fall back
        return {"status": "load failed", "detail": f"{type(e).__name__}: {e}"}
    fp = OrtLogDepth(fp32_path, cache=True)
    out = {"status": "evaluated", "n_images": len(samples), "reference": "IBims-1 sensor GT",
           "camera_mode": "default-k", "artefacts": {},
           "note": "report-only zero-shot set; legacy models only (A9 allows legacy rows); "
                   "selects nothing"}
    for tag, p in paths.items():
        r = evaluate_int8(fp, OrtLogDepth(p), samples, cfg)
        s = gate_summary(r)
        out["artefacts"][tag] = s
        with open(os.path.join(out_dir, f"ibims_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, default=str)
        log(f"[x0:{name}] ibims {tag}: I1={s['I1_frac_within_1.25x']:.4f} d1 {s['d1_fp32']:.2f}->"
            f"{s['d1_int8']:.2f}")
    return out


def _quick_latency(paths: Dict[str, str], hw, threads: int = 4, warmup: int = 3, iters: int = 20) -> dict:
    import onnxruntime as ort
    x = np.zeros((1, 3, hw[0], hw[1]), np.float32)
    out = {"note": f"x86 proxy, {threads} threads, unpinned, median of {iters}; noisy (other jobs "
                   "may share the CPU); not a latency result"}
    for k, p in paths.items():
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        for _ in range(warmup):
            s.run(None, {name: x})
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            s.run(None, {name: x})
            ts.append(time.perf_counter() - t0)
        out[k] = round(float(np.median(ts)) * 1000.0, 1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# X0 driver
# ──────────────────────────────────────────────────────────────────────────────

def run_x0(models: Sequence[str], out_json: str, out_root: str, roles_path: str, manifest_path: str,
           n_val: int = 64, sensitivity: bool = True, ibims: bool = True) -> dict:
    from edge.export_v2 import DEFAULT_K, export_legacy_default_k
    t0 = time.time()
    man = read_manifest(manifest_path)
    roles = resolve_roles(roles_path, manifest_path, n_val)
    res = {"run": "X0", "plan": "docs/V2_PLAN.md amendment A14; sections 8.2-8.4",
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "camera_mode": "default-k", "default_K": list(DEFAULT_K),
           "gates_label": "research-only: provisional thresholds (V2_PLAN 8.3/9.2), not owner-confirmed",
           "roles": {k: roles[k] for k in ("source", "roles_sha256", "provisional", "ids_sha256", "val_total")},
           "n": {"calibration": len(roles["calibration"]), "val_slice": len(roles["val_slice"]),
                 "frozen_test": len(roles["frozen_test"])},
           "decision_data": "val.own slice only (method, output precision, recovery ladder)",
           "report_only_data": "test.frozen_own (teacher labels) and IBims-1 (sensor GT, if reachable)",
           "env": _env(), "models": {}}
    for name in models:
        ckpt = X0_MODELS.get(name, name)
        out_dir = os.path.join(out_root, os.path.basename(os.path.dirname(os.path.abspath(ckpt))))
        os.makedirs(out_dir, exist_ok=True)
        fp32_path = os.path.join(out_dir, "fp32_defaultK.onnx")
        print(f"[x0] {name}: export default-K fp32 -> {fp32_path}", flush=True)
        export_legacy_default_k(ckpt, fp32_path, DEFAULT_K)
        res["models"][name] = chain_one_model(name, fp32_path, roles, man, out_dir,
                                              sensitivity=sensitivity, ibims=ibims,
                                              log=lambda s: print(s, flush=True))
        res["models"][name]["verdict"] = _verdict(res["models"][name])
        _write_json(out_json, res)
    res["seconds"] = round(time.time() - t0, 1)
    _write_json(out_json, res)
    return res


def _verdict(m: dict) -> dict:
    fz = m.get("frozen_test", {}).get("artefacts", {})
    w8 = fz.get("w8", {})
    q = fz.get(m.get("chosen_ptq"), {})
    w8_ok = all(w8.get(k) == "pass" for k in ("I1", "I2", "I3", "I4"))
    q_ok = all(q.get(k) == "pass" for k in ("I1", "I2", "I3", "I4", "I5"))
    return {"weight_only_int8": "passes I1-I4 on the frozen own test" if w8_ok else
            "FAILS at least one of I1-I4 on the frozen own test",
            "full_int8": (f"{m.get('chosen_ptq')} passes I1-I5" if q_ok else
                          f"FAILED: {m.get('chosen_ptq')} stays below the int8 gates "
                          "(labelled failed per A14; ship weight-only int8)"),
            "full_int8_gate_status": {k: q.get(k) for k in ("I1", "I2", "I3", "I4", "I5",
                                                             "I6_safety_regression")},
            "w8_gate_status": {k: w8.get(k) for k in ("I1", "I2", "I3", "I4", "I6_safety_regression")}}


def _env() -> dict:
    try:
        from benchmark.provenance import env_stamp
        return env_stamp(gpu=False)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize", help="calibrate (roles calibration ids) and quantize one model")
    q.add_argument("--onnx", required=True, help="fp32 single-input ONNX with sidecar .json")
    q.add_argument("--out", default="")
    q.add_argument("--method", default="minmax", choices=list(METHODS))
    q.add_argument("--output-precision", default="uint8", choices=list(OUTPUT_PRECISIONS))
    q.add_argument("--exclude", action="append", default=[], help="node to keep float")
    q.add_argument("--roles", default=DEFAULT_ROLES)
    q.add_argument("--manifest", default=DEFAULT_MANIFEST)
    x = sub.add_parser("x0", help="run X0 on the current legacy students (CPU)")
    x.add_argument("--models", nargs="+", default=list(X0_MODELS))
    x.add_argument("--out", default=os.path.join("outputs", "v2", "x0_current_models.json"))
    x.add_argument("--out-root", default=os.path.join("outputs", "v2", "edge", "x0"))
    x.add_argument("--roles", default=DEFAULT_ROLES)
    x.add_argument("--manifest", default=DEFAULT_MANIFEST)
    x.add_argument("--n-val", type=int, default=64)
    x.add_argument("--no-sensitivity", action="store_true")
    x.add_argument("--no-ibims", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "quantize":
        with open(os.path.splitext(a.onnx)[0] + ".json", encoding="utf-8") as f:
            fmeta = json.load(f)
        hw = tuple(fmeta["train_size"])
        ids = calibration_ids_from_roles(a.roles, a.manifest)
        man = read_manifest(a.manifest)
        pre = prepare(a.onnx)
        res = collect_ranges_chunked(pre, _calib_feeds(ids, man, hw), a.method, MAX_CHUNK)
        out = a.out or os.path.join(os.path.dirname(a.onnx), f"ptq_{a.method}_{a.output_precision}.onnx")
        meta = quantize_u8s8(pre, out, res.ranges, a.method, a.output_precision, a.exclude, meta=fmeta)
        meta["quant"]["calibration_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        meta["quant"]["calibration_ids_source"] = a.roles
        _write_json(os.path.splitext(out)[0] + ".json", meta)
        print(f"[ptq] wrote {out} ({meta['file_mb']} MB)")
        return 0
    run_x0(a.models, a.out, a.out_root, a.roles, a.manifest, a.n_val, not a.no_sensitivity, not a.no_ibims)
    print(f"[x0] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
