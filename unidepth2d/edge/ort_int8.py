"""Full 8-bit post-training quantization (weights AND activations) with ONNX Runtime.

  python -m edge.ort_int8 --onnx outputs/edge/b1_mix/fp32.onnx                 # int8, head float
  python -m edge.ort_int8 --onnx outputs/edge/b1_mix/fp32.onnx --quantize-head # everything int8
  python -m edge.ort_int8 --onnx ... --method percentile --calib 256

What it does: static QDQ quantization (per-channel int8 weights, uint8 activations), with
activation ranges calibrated on images drawn from the student's OWN training corpus.

POLICY (enforced in `calibration_rows`): calibration images come only from the `own` source of
the training manifest. Benchmark images (NYUv2 / SUN RGB-D / IBims-1 / KITTI) are never used
to calibrate, otherwise an "int8 zero-shot" number would have seen the test domain.

Head handling: by default the nodes from the last convolution to the graph output stay in
float. A depth map is a regression output; quantizing it to 256 levels over the whole
log-depth range (step ~0.02 in log = ~2% of depth) bands the map. `--quantize-head` turns
this off so the cost of a fully-integer graph can be measured as an ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ALLOWED_CALIB_SOURCES = ("own",)          # see POLICY in the module docstring
DEFAULT_INTR = (0.8, 1.08, 0.5, 0.5)


def preprocess(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """uint8 RGB (H0,W0,3) -> float32 (1,3,h,w), exactly as StudentModelRunner does."""
    rz = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    rz = (rz.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rz.transpose(2, 0, 1))[None]


def calibration_rows(manifest: str, n: int, seed: int = 0,
                     sources: tuple = ALLOWED_CALIB_SOURCES) -> List[dict]:
    """Seeded sample of n manifest rows from the allowed (training-only) sources."""
    bad = [s for s in sources if s not in ALLOWED_CALIB_SOURCES]
    if bad:
        raise ValueError(f"calibration source(s) {bad} not allowed; only {ALLOWED_CALIB_SOURCES} "
                         "(benchmark-domain images must never calibrate a zero-shot model)")
    with open(manifest, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] in sources]
    if not rows:
        raise ValueError(f"no rows from sources {sources} in {manifest}")
    rows.sort(key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    return rows[:n]


class ManifestCalibReader:
    """onnxruntime CalibrationDataReader over manifest rows (duck-typed: get_next/rewind)."""

    def __init__(self, rows: List[dict], h: int, w: int, input_names: List[str]):
        self.rows, self.h, self.w, self.input_names = rows, h, w, input_names
        self._it: Optional[Iterator[dict]] = None
        self.rewind()

    def _feeds(self) -> Iterator[Dict[str, np.ndarray]]:
        for r in self.rows:
            bgr = cv2.imread(r["img"], cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            feed = {"image": preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.h, self.w)}
            if "intrinsics" in self.input_names:
                try:
                    k = [float(r["fx"]), float(r["fy"]), float(r["cx"]), float(r["cy"])]
                except (KeyError, ValueError, TypeError):
                    k = list(DEFAULT_INTR)
                feed["intrinsics"] = np.asarray([k], dtype=np.float32)
            yield feed

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = self._feeds()


def head_nodes(model) -> List[str]:
    """Names of the nodes from the last Conv (inclusive) to the graph output.

    Walks backwards from each graph output through single-producer chains until the first
    Conv is met. Those nodes are kept in float when the head is excluded from quantization.
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
            names.append(n.name)
            if n.op_type == "Conv":
                continue                      # stop this branch at the last convolution
            frontier.extend(i for i in n.input if i in producer)
    return names


def quantize(onnx_path: str, out_path: str, manifest: str, n_calib: int = 128, seed: int = 0,
             method: str = "minmax", quantize_head: bool = False, per_channel: bool = True) -> dict:
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    meta_path = os.path.splitext(onnx_path)[0] + ".json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    h, w = meta["train_size"]

    pre = os.path.splitext(out_path)[0] + ".pre.onnx"
    try:
        quant_pre_process(onnx_path, pre, skip_symbolic_shape=False)
    except Exception as e:  # noqa: BLE001 - symbolic shape inference is optional
        print(f"[int8] pre-process with symbolic shapes failed ({type(e).__name__}); retrying without")
        quant_pre_process(onnx_path, pre, skip_symbolic_shape=True)

    model = onnx.load(pre)
    exclude = [] if quantize_head else head_nodes(model)
    rows = calibration_rows(manifest, n_calib, seed)

    class _Reader(ManifestCalibReader, CalibrationDataReader):
        pass

    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}
    quantize_static(
        pre, out_path, _Reader(rows, h, w, meta["inputs"]),
        quant_format=QuantFormat.QDQ, per_channel=per_channel,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=methods[method], nodes_to_exclude=exclude,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    os.remove(pre)

    out_meta = dict(meta)
    out_meta.update({
        "precision": "int8" + ("" if quantize_head else "-headfp"),
        "quant": {"tool": "onnxruntime.quantize_static", "format": "QDQ",
                  "per_channel": per_channel, "calibration_method": method,
                  "calibration_images": len(rows), "calibration_sources": list(ALLOWED_CALIB_SOURCES),
                  "calibration_seed": seed, "head_float": not quantize_head,
                  "excluded_nodes": len(exclude)},
        "file_mb": round(os.path.getsize(out_path) / 1e6, 3),
        "fp32_file_mb": meta.get("file_mb"),
    })
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)
    print(f"[int8] {out_path}  {out_meta['file_mb']} MB (fp32 {meta.get('file_mb')} MB)  "
          f"method={method} calib={len(rows)} head_float={not quantize_head}")
    return out_meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--onnx", required=True, help="fp32 ONNX written by edge.export_onnx")
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default="outputs/data_gt/manifest.csv")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="minmax", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--quantize-head", action="store_true")
    ap.add_argument("--per-tensor", action="store_true")
    a = ap.parse_args()
    tag = "int8" + ("" if a.quantize_head else "-headfp") + ("" if a.method == "minmax" else f"-{a.method}")
    out = a.out or os.path.join(os.path.dirname(a.onnx), f"{tag}.onnx")
    quantize(a.onnx, out, a.manifest, a.calib, a.seed, a.method, a.quantize_head, not a.per_tensor)


if __name__ == "__main__":
    main()
