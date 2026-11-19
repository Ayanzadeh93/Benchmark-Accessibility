"""Run an exported student (.onnx + sidecar .json) through the benchmark harness.

Spec in benchmark/run_bench.py:  onnx:<path/to/model.onnx>
The runner mirrors StudentModelRunner exactly (INTER_AREA resize to the train size, ImageNet
normalisation, normalised intrinsics of the ORIGINAL image) so that a PyTorch row and its
ONNX / int8 rows differ only by the model file. Inference runs on the ONNX Runtime CPU
provider: that is the edge-proxy configuration, and it makes int8 results device-independent.
"""

from __future__ import annotations

import json
import os

import numpy as np

from edge.ort_int8 import DEFAULT_INTR, preprocess


class OnnxStudent:
    def __init__(self, onnx_path: str, device: str = "cpu", threads: int = 0):
        import onnxruntime as ort
        with open(os.path.splitext(onnx_path)[0] + ".json", encoding="utf-8") as f:
            self.meta = json.load(f)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.h, self.w = self.meta["train_size"]
        self.uses_intr = "intrinsics" in self.meta["inputs"]
        self.log_min, self.log_max = float(self.meta["log_min"]), float(self.meta["log_max"])
        self.name = f"{self.meta['name']}+onnx-{self.meta.get('precision', 'fp32')}"
        q = self.meta.get("quant") or {}
        if q.get("calibration_method") not in (None, "minmax"):
            self.name += f"-{q['calibration_method']}"
        self.quant_stats = {"file_mb": self.meta.get("file_mb"),
                            "fp32_file_mb": self.meta.get("fp32_file_mb", self.meta.get("file_mb")),
                            "precision": self.meta.get("precision", "fp32"), **q}

    def predict(self, rgb: np.ndarray, K):
        oh, ow = rgb.shape[:2]
        feed = {"image": preprocess(rgb, self.h, self.w)}
        if self.uses_intr:
            if K is not None:
                k = [K[0, 0] / ow, K[1, 1] / oh, K[0, 2] / ow, K[1, 2] / oh]
            else:
                k = list(DEFAULT_INTR)
            feed["intrinsics"] = np.asarray([k], dtype=np.float32)
        log_depth = self.sess.run(["log_depth"], feed)[0][0, 0]
        depth = np.exp(np.clip(log_depth.astype(np.float32), self.log_min, self.log_max))
        return {"depth": depth, "conf": None, "points": None}
