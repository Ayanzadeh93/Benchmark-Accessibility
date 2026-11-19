"""Frozen UniDepth v2 metric-depth teacher.

Wraps ``unidepth.models.UniDepthV2``. Self-calibrates the camera from the image (no intrinsics
passed), so it handles the mixed/unknown cameras in the dataset. Output depth is METRIC METRES
(larger = farther). Depth is upsampled back to the ORIGINAL image resolution so detection boxes
(given in original pixels) can be sampled without any coordinate scaling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("unidepth2d.teacher")


class UniDepthTeacher:
    def __init__(self, checkpoints: List[str], device: str = "cuda"):
        self.checkpoints = checkpoints
        self.device = device
        self.model = None
        self.loaded_checkpoint: Optional[str] = None
        self._torch = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch

        self._torch = torch
        from unidepth.models import UniDepthV2

        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = device
        last_err: Optional[Exception] = None
        for ckpt in self.checkpoints:
            try:
                logger.info("Loading UniDepthV2 checkpoint: %s", ckpt)
                model = UniDepthV2.from_pretrained(ckpt)
                model = model.to(device).eval()
                self.model = model
                self.loaded_checkpoint = ckpt
                logger.info("Loaded %s on %s", ckpt, device)
                return
            except Exception as e:  # noqa: BLE001 - fall through to a smaller checkpoint
                logger.warning("Failed to load %s: %s", ckpt, str(e)[:200])
                last_err = e
        raise RuntimeError(f"Could not load any UniDepthV2 checkpoint {self.checkpoints}: {last_err}")

    def predict(self, rgb_hwc_uint8: np.ndarray, infer_longest_side: int = 1280) -> Dict[str, np.ndarray]:
        """rgb_hwc_uint8: (H, W, 3) uint8 RGB. Returns {'depth_m','conf'} at ORIGINAL resolution.

        The model runs at ``infer_longest_side`` (longer side) for speed/memory; both outputs are
        then resized back up to the input's original (H, W).
        """
        if self.model is None:
            self.load()
        torch = self._torch
        assert rgb_hwc_uint8.ndim == 3 and rgb_hwc_uint8.shape[2] == 3, "expected (H,W,3) RGB"
        orig_h, orig_w = rgb_hwc_uint8.shape[:2]

        rgb_infer = rgb_hwc_uint8
        if infer_longest_side and infer_longest_side > 0 and max(orig_h, orig_w) > infer_longest_side:
            scale = infer_longest_side / float(max(orig_h, orig_w))
            nh, nw = int(round(orig_h * scale)), int(round(orig_w * scale))
            rgb_infer = cv2.resize(rgb_hwc_uint8, (nw, nh), interpolation=cv2.INTER_AREA)

        t = torch.from_numpy(np.ascontiguousarray(rgb_infer)).permute(2, 0, 1)  # (3,h,w) uint8
        with torch.inference_mode():
            pred = self.model.infer(t.to(self.device))

        depth = pred["depth"].squeeze().float().cpu().numpy().astype(np.float32)  # (h,w) metres
        conf_t = pred.get("confidence")
        conf = (
            conf_t.squeeze().float().cpu().numpy().astype(np.float32)
            if conf_t is not None
            else None
        )

        if depth.shape[:2] != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if conf is not None:
                conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return {"depth_m": depth, "conf": conf}
