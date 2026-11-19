"""Chunked activation calibration for ONNX Runtime static quantization (plan 8.2, E5-E7).

  python -m edge.calib_chunked --onnx outputs/v2/edge/<run>/fp32.pre.onnx --method minmax \
      --roles configs/v2/roles.yaml --out outputs/v2/edge/<run>/ranges_minmax.json

Env: depth_cmp (onnxruntime 1.32 quantization tools).

WHY CHUNKED. Holding every intermediate tensor of 300 images at 480x640 does not fit in RAM, and
ORT 1.32's own memory knob is broken: with CalibMaxIntermediateOutputs / max_intermediate_outputs
= k, MinMaxCalibrater.collect_data CLEARS the collected outputs every k images WITHOUT merging
them, so full chunks are silently discarded (E5: true range [-5, 5] came back as [-1, 1]).
This module therefore NEVER sets that option (`assert_no_max_intermediate` rejects it) and
instead calls `calibrator.collect_data` repeatedly on chunks of at most MAX_CHUNK = 8 images.
MinMax merges exactly across calls (min of mins / max of maxes), so chunked MinMax ranges are
bit-identical to a single-shot run on the same images. Percentile merges histograms; the
result matches a single-shot run up to histogram re-binning (E6: [-5, 4.99] vs [-5, 5]).

CONTRACT
  * collect_ranges_chunked(model_path, feeds, method) -> CalibResult with
        .tensors_data  onnxruntime TensorsData (usable as a quantize_static calibration cache)
        .ranges        {tensor name: (rmin, rmax)} python floats
        .n_images, .n_chunks, .chunk_size, .method
  * Activations are calibrated ASYMMETRICALLY (symmetric=False) for both methods: U8 activations.
    (ORT's create_calibrator defaults Percentile to symmetric=True, which would put ReLU outputs
    in [-t, t] and waste half the uint8 grid.)
  * overrides_from_ranges(model, ranges) -> TensorQuantOverrides dict. It first applies ORT's own
    adjust_tensor_ranges rule (a Relu/Clip input with a single consumer takes the output's range;
    Softmax outputs are [0, 1]) so passing overrides changes nothing relative to ORT's internal
    handling of the same ranges (E7: overrides are honoured to within one quantization step).
  * Calibration-source allowlist (plan P6, section 5.1 "Calibration" role): calibration images are
    the own TRAINING calibration ids listed in configs/v2/roles.yaml - never frozen-test, D_SUN dev,
    validation or benchmark images. `validate_calibration_ids` raises CalibrationSourceError on any
    id that is not in the roles calibration list, is not source "own" in the manifest, or appears
    in a protected role. `provisional_calibration_ids` exists only for runs made BEFORE roles.yaml
    is frozen (X0): it applies the plan's rule (300 own training rows, >= 30 low-light, mean gray
    < 60) to an explicit exclusion set, and every caller must label the result provisional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

MAX_CHUNK = 8
FORBIDDEN_OPTIONS = ("CalibMaxIntermediateOutputs", "max_intermediate_outputs")
DEFAULT_ROLES = os.path.join("configs", "v2", "roles.yaml")
PERCENTILE = 99.99
NUM_BINS = 2048


class CalibrationSourceError(ValueError):
    """A calibration id violates the calibration-source allowlist."""


def assert_no_max_intermediate(options: Optional[dict]) -> None:
    """Reject the ORT option that silently drops calibration chunks (E5)."""
    bad = [k for k in FORBIDDEN_OPTIONS if options and k in options]
    if bad:
        raise ValueError(f"{bad} must never be set: ORT 1.32 discards full chunks without merging "
                         "them (plan E5). Use collect_ranges_chunked instead.")


class ListDataReader:
    """Minimal CalibrationDataReader over a list of feed dicts."""

    def __init__(self, feeds: Sequence[dict]):
        self.feeds = list(feeds)
        self._it = iter(self.feeds)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.feeds)

    def __len__(self):
        return len(self.feeds)


@dataclass
class CalibResult:
    method: str
    tensors_data: object
    ranges: Dict[str, Tuple[float, float]]
    n_images: int
    n_chunks: int
    chunk_size: int
    percentile: Optional[float] = None
    extra: dict = field(default_factory=dict)


def _ort_method(method: str):
    from onnxruntime.quantization import CalibrationMethod
    m = method.lower()
    if m == "minmax":
        return CalibrationMethod.MinMax
    if m == "percentile":
        return CalibrationMethod.Percentile
    raise ValueError(f"method must be minmax or percentile, got {method!r}")


def bytes_per_image(model_path: str) -> int:
    """Sum of float activation bytes for one image (shape-inferred); what Percentile holds per image."""
    import onnx
    from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
    from pathlib import Path
    m = load_model_with_shape_infer(Path(model_path))
    inits = {i.name for i in m.graph.initializer}
    total = 0
    for vi in list(m.graph.value_info) + list(m.graph.output):
        tt = vi.type.tensor_type
        if vi.name in inits or tt.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            continue
        dims = [d.dim_value for d in tt.shape.dim]
        if dims and all(d > 0 for d in dims):
            total += int(np.prod(dims)) * (4 if tt.elem_type == onnx.TensorProto.FLOAT else 2)
    return total


def choose_chunk_size(model_path: str, method: str, requested: int = MAX_CHUNK,
                      max_chunk_bytes: float = 3e9) -> int:
    """min(requested, MAX_CHUNK) and, for Percentile, small enough that one chunk fits max_chunk_bytes."""
    c = max(1, min(int(requested), MAX_CHUNK))
    if method.lower() == "percentile":
        per = bytes_per_image(model_path)
        if per > 0:
            c = max(1, min(c, int(max_chunk_bytes // per)))
    return c


def collect_ranges_chunked(model_path: str, feeds: Iterable[dict], method: str = "minmax",
                           chunk_size: int = MAX_CHUNK, percentile: float = PERCENTILE,
                           num_bins: int = NUM_BINS, op_types: Optional[Sequence[str]] = None,
                           workdir: Optional[str] = None, extra_options: Optional[dict] = None,
                           max_chunk_bytes: float = 3e9, verbose: bool = True) -> CalibResult:
    """Calibrate `model_path` (normally the quant_pre_process output) on `feeds`, chunk by chunk.

    feeds: iterable of {input name: float32 array}; consumed lazily, at most `chunk_size`
    (<= MAX_CHUNK) materialised at a time. op_types None = every float tensor, exactly what
    quantize_static calibrates by default.
    """
    from pathlib import Path
    from onnxruntime.quantization.calibrate import TensorsData, create_calibrator
    assert_no_max_intermediate(extra_options)
    cs = choose_chunk_size(model_path, method, chunk_size, max_chunk_bytes)
    opts = {"symmetric": False}
    if method.lower() == "percentile":
        opts.update({"percentile": float(percentile), "num_bins": int(num_bins)})
    tmp = tempfile.TemporaryDirectory(prefix="calib_chunked_", dir=workdir, ignore_cleanup_errors=True)
    try:
        cal = create_calibrator(Path(model_path), op_types, augmented_model_path=os.path.join(
            tmp.name, "augmented.onnx"), calibrate_method=_ort_method(method), extra_options=opts)
        n_img = n_chunks = 0
        chunk: List[dict] = []
        for feed in feeds:
            chunk.append(feed)
            if len(chunk) == cs:
                cal.collect_data(ListDataReader(chunk))
                n_img += len(chunk)
                n_chunks += 1
                chunk = []
                if verbose:
                    print(f"[calib_chunked] {method}: {n_img} images ({n_chunks} chunks of <= {cs})",
                          flush=True)
        if chunk:
            cal.collect_data(ListDataReader(chunk))
            n_img += len(chunk)
            n_chunks += 1
        if n_img == 0:
            raise ValueError("no calibration images")
        td = cal.compute_data()
        if not isinstance(td, TensorsData):
            raise TypeError(f"calibrator returned {type(td)}")
        del cal
    finally:
        tmp.cleanup()
    ranges = {}
    for name, d in td.data.items():
        lo, hi = d.range_value
        ranges[name] = (float(np.asarray(lo).min()), float(np.asarray(hi).max()))
    return CalibResult(method.lower(), td, ranges, n_img, n_chunks, cs,
                       float(percentile) if method.lower() == "percentile" else None)


def single_shot_ranges(model_path: str, feeds: Sequence[dict], method: str = "minmax",
                       percentile: float = PERCENTILE, num_bins: int = NUM_BINS,
                       extra_options: Optional[dict] = None) -> Dict[str, Tuple[float, float]]:
    """Reference: ONE collect_data call over all feeds (small tests only; holds everything in memory).

    extra_options are passed to ORT's create_calibrator unchecked - this is the ONLY place the
    forbidden max_intermediate_outputs option may be passed, so tests can reproduce E5.
    """
    from pathlib import Path
    from onnxruntime.quantization.calibrate import create_calibrator
    opts = {"symmetric": False, **(extra_options or {})}
    if method.lower() == "percentile":
        opts.update({"percentile": float(percentile), "num_bins": int(num_bins)})
    with tempfile.TemporaryDirectory(prefix="calib_ref_", ignore_cleanup_errors=True) as t:
        cal = create_calibrator(Path(model_path), None, augmented_model_path=os.path.join(t, "a.onnx"),
                                calibrate_method=_ort_method(method), extra_options=opts)
        cal.collect_data(ListDataReader(feeds))
        td = cal.compute_data()
        del cal
    return {k: (float(np.asarray(v.range_value[0]).min()), float(np.asarray(v.range_value[1]).max()))
            for k, v in td.data.items()}


def ort_adjusted_ranges(model, ranges: Dict[str, Tuple[float, float]],
                        exclude: Iterable[str] = ()) -> Dict[str, Tuple[float, float]]:
    """Apply ORT 1.32's adjust_tensor_ranges rule (base_quantizer.py) to plain ranges.

    Relu/Clip: if the input tensor has exactly one consumer, it takes the output tensor's range.
    Softmax: output range is [0, 1]. Nodes named in `exclude` (nodes_to_exclude) are skipped,
    as ORT skips nodes it does not quantize.
    """
    skip = set(exclude)
    out = dict(ranges)
    n_cons: Dict[str, int] = {}
    for n in model.graph.node:
        for i in n.input:
            if i:
                n_cons[i] = n_cons.get(i, 0) + 1
    for n in model.graph.node:
        if n.name in skip:
            continue
        if n.op_type in ("Clip", "Relu"):
            if n_cons.get(n.input[0], 0) != 1:
                continue
            if n.input[0] in out and n.output[0] in out:
                out[n.input[0]] = out[n.output[0]]
        elif n.op_type == "Softmax" and n.output[0] in out:
            out[n.output[0]] = (0.0, 1.0)
    return out


def overrides_from_ranges(model, ranges: Dict[str, Tuple[float, float]],
                          quant_types: Optional[Dict[str, object]] = None,
                          exclude: Iterable[str] = ()) -> Dict[str, list]:
    """TensorQuantOverrides {tensor: [{"rmin": f32, "rmax": f32[, "quant_type": QuantType]}]}.

    Only tensors present in `model` are emitted (ORT rejects unknown names). `exclude` = the
    nodes_to_exclude passed to quantize_static (see ort_adjusted_ranges).
    """
    import onnx
    if isinstance(model, (str, os.PathLike)):
        model = onnx.load(model)
    adj = ort_adjusted_ranges(model, ranges, exclude)
    names = {i.name for i in model.graph.input} | {o.name for o in model.graph.output}
    for n in model.graph.node:
        names.update(x for x in n.input if x)
        names.update(x for x in n.output if x)
    inits = {i.name for i in model.graph.initializer}
    ov = {}
    for k, (lo, hi) in adj.items():
        if k not in names or k in inits:
            continue
        d = {"rmin": np.array(lo, dtype=np.float32), "rmax": np.array(hi, dtype=np.float32)}
        if quant_types and k in quant_types:
            d["quant_type"] = quant_types[k]
        ov[k] = [d]
    return ov


def save_ranges(path: str, res: CalibResult, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"method": res.method, "percentile": res.percentile, "n_images": res.n_images,
               "n_chunks": res.n_chunks, "chunk_size": res.chunk_size,
               "max_intermediate_outputs": None, "ranges": {k: list(v) for k, v in res.ranges.items()},
               **(meta or {})}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def load_ranges(path: str) -> Tuple[Dict[str, Tuple[float, float]], dict]:
    """Inverse of save_ranges: ({tensor: (rmin, rmax)}, the rest of the payload)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rng = {k: (float(v[0]), float(v[1])) for k, v in d.pop("ranges").items()}
    return rng, d


def tensors_data_from_ranges(ranges: Dict[str, Tuple[float, float]], method: str):
    """A TensorsData holding ONLY (lowest, highest) float32 per tensor, labelled with `method`.

    Percentile histograms are dropped: quantize_static only reads range_value, and a 2048-bin
    histogram per tensor would make the cache tens of MB.
    """
    from onnxruntime.quantization.calibrate import TensorData, TensorsData
    return TensorsData(_ort_method(method), {
        k: TensorData(lowest=np.array(lo, dtype=np.float32), highest=np.array(hi, dtype=np.float32))
        for k, (lo, hi) in ranges.items()})


def save_ranges_cache(ranges: Dict[str, Tuple[float, float]], method: str, path: str) -> str:
    """Write an ORT calibration cache (quantize_static calibration_cache_path) from plain ranges."""
    from onnxruntime.quantization.calibrate import save_tensors_data
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    save_tensors_data(tensors_data_from_ranges(ranges, method), path)
    return path


def save_tensors_cache(res: CalibResult, path: str) -> str:
    """save_ranges_cache for a CalibResult."""
    return save_ranges_cache(res.ranges, res.method, path)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration-source allowlist
# ──────────────────────────────────────────────────────────────────────────────

# roles.yaml schema "v2-roles/1" (scripts/v2/build_roles.py): calibration.own, test.frozen_own,
# test.video_own, val.own, val.nyu, dev.d_sun, train.own. Older/alternative spellings are accepted
# only as a fallback so a schema bump fails loudly in validate_* rather than silently passing.
_CALIB_KEYS = (("calibration", "own"), ("calibration", "ids"), ("calibration_ids",))
_TRAIN_OWN_KEYS = (("train", "own"),)
_PROTECTED_KEYS = {
    "frozen_test": (("test", "frozen_own"), ("frozen_test", "ids"), ("frozen_test_ids",)),
    "video_test": (("test", "video_own"),),
    "dev": (("dev", "d_sun"), ("dev", "ids"), ("dev_ids",)),
    "val": (("val", "own"), ("val", "nyu"), ("val", "ids")),
}


def roles_ids(roles: dict, *path: str) -> List[str]:
    """The id list at roles[path[0]][path[1]]...; KeyError if it is missing or not a list."""
    ids = _id_list(_get(roles, path))
    if ids is None:
        raise KeyError(f"roles has no id list at {'.'.join(path)}")
    return ids


def _get(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _id_list(v) -> Optional[List[str]]:
    if isinstance(v, list) and all(isinstance(x, (str, int)) for x in v):
        return [str(x) for x in v]
    return None


def load_roles(path: str = DEFAULT_ROLES) -> dict:
    """Read configs/v2/roles.yaml. Missing file -> FileNotFoundError (never a silent fallback)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: dataset roles must be frozen before calibrating "
                                "(scripts/v2/build_roles.py)")
    import yaml
    with open(path, encoding="utf-8") as f:
        roles = yaml.safe_load(f)
    if not isinstance(roles, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return roles


def roles_calibration_ids(roles: dict) -> List[str]:
    for p in _CALIB_KEYS:
        ids = _id_list(_get(roles, p))
        if ids:
            return ids
    raise CalibrationSourceError("roles file has no calibration id list (looked for "
                                 + ", ".join(".".join(p) for p in _CALIB_KEYS) + ")")


def roles_protected_ids(roles: dict) -> Dict[str, Set[str]]:
    out = {}
    for role, paths in _PROTECTED_KEYS.items():
        s: Set[str] = set()
        for p in paths:
            ids = _id_list(_get(roles, p))
            if ids:
                s.update(ids)
        out[role] = s
    return out


def read_manifest(path: str) -> Dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def validate_calibration_ids(ids: Sequence[str], allowed_ids: Iterable[str],
                             manifest: Dict[str, dict],
                             protected: Optional[Dict[str, Set[str]]] = None,
                             allowed_sources: Sequence[str] = ("own",)) -> None:
    """Raise CalibrationSourceError unless every id is allowed, own-source and unprotected."""
    allowed = set(allowed_ids)
    problems = []
    for i in ids:
        if i not in allowed:
            problems.append(f"{i}: not in the roles calibration list")
            continue
        r = manifest.get(i)
        if r is None:
            problems.append(f"{i}: not in the manifest")
        elif r.get("source") not in allowed_sources:
            problems.append(f"{i}: source {r.get('source')!r} not in {tuple(allowed_sources)}")
        for role, s in (protected or {}).items():
            if i in s:
                problems.append(f"{i}: listed in protected role {role!r}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate calibration ids")
    if problems:
        raise CalibrationSourceError("calibration-source allowlist violated:\n  "
                                     + "\n  ".join(problems[:20])
                                     + (f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else ""))


def calibration_ids_from_roles(roles_path: str, manifest_path: str) -> List[str]:
    """The only sanctioned source of calibration ids: roles.yaml, validated against the manifest."""
    roles = load_roles(roles_path)
    ids = roles_calibration_ids(roles)
    allowed = set(ids)
    for p in _TRAIN_OWN_KEYS:           # calibration must also be a subset of train.own
        tr = _id_list(_get(roles, p))
        if tr is not None:
            allowed &= set(tr)
    validate_calibration_ids(ids, allowed, read_manifest(manifest_path), roles_protected_ids(roles))
    return ids


def mean_gray(path: str) -> float:
    import cv2
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return float(g.mean())


def provisional_calibration_ids(manifest: Dict[str, dict], exclude: Set[str], n: int = 300,
                                min_lowlight: int = 30, luma_thresh: float = 60.0,
                                seed: int = 2027) -> Tuple[List[str], dict]:
    """Plan-rule calibration set for runs before roles.yaml exists. ALWAYS label as provisional.

    Seeded: sort own ids not in `exclude`, shuffle with `seed`, scan in that order computing mean
    gray until `min_lowlight` dark (< luma_thresh) and n - min_lowlight other images are found.
    """
    cand = sorted(i for i, r in manifest.items() if r.get("source") == "own" and i not in exclude)
    random.Random(seed).shuffle(cand)
    dark, light, scanned = [], [], 0
    for i in cand:
        if len(dark) >= min_lowlight and len(light) >= n - min_lowlight:
            break
        scanned += 1
        g = mean_gray(manifest[i]["img"])
        if g < luma_thresh:
            if len(dark) < min_lowlight:
                dark.append(i)
        elif len(light) < n - min_lowlight:
            light.append(i)
    if len(dark) < min_lowlight:
        raise CalibrationSourceError(f"only {len(dark)} low-light own rows found (< {min_lowlight})")
    ids = sorted(dark + light)
    info = {"provisional": True, "rule": f"{n} own training rows, >= {min_lowlight} with mean gray "
            f"< {luma_thresh}, seed {seed}", "n_lowlight": len(dark), "n_scanned": scanned,
            "n_excluded": len(exclude), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    return ids, info


def manifest_feeds(rows: Sequence[dict], h: int, w: int, input_name: str = "image") -> Iterator[dict]:
    """Lazy {input_name: (1,3,h,w) float32} feeds, identical preprocessing to edge/onnx_runner.py."""
    import cv2
    from edge.ort_int8 import preprocess
    for r in rows:
        bgr = cv2.imread(r["img"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(r["img"])
        yield {input_name: preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), h, w)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--onnx", required=True, help="pre-processed float model (quant_pre_process output)")
    ap.add_argument("--method", default="minmax", choices=["minmax", "percentile"])
    ap.add_argument("--roles", default=DEFAULT_ROLES)
    ap.add_argument("--manifest", default="outputs/data_gt/manifest.csv")
    ap.add_argument("--hw", type=int, nargs=2, default=(480, 640))
    ap.add_argument("--chunk", type=int, default=MAX_CHUNK)
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = calibration_ids_from_roles(a.roles, a.manifest)
    man = read_manifest(a.manifest)
    res = collect_ranges_chunked(a.onnx, manifest_feeds([man[i] for i in ids], *a.hw), a.method,
                                 a.chunk, a.percentile)
    save_ranges(a.out, res, {"calibration_ids_source": a.roles, "n_ids": len(ids)})
    print(f"[calib_chunked] wrote {a.out}: {len(res.ranges)} tensors, {res.n_images} images")


if __name__ == "__main__":
    main()
