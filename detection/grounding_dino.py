"""GroundingDINO open-vocabulary detector wrapper (transformers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import nms_detections, xyxy_to_yolo_norm, clip01

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


@dataclass(frozen=True)
class GroundingDINOConfig:
    model_id: str = "IDEA-Research/grounding-dino-base"
    device: str = "auto"  # auto/cuda/cpu
    hf_token: Optional[str] = None


class GroundingDINODetector:
    """Open-vocabulary detection using GroundingDINO from HuggingFace transformers."""

    def __init__(self, cfg: GroundingDINOConfig):
        self.cfg = cfg
        self._enabled = False
        self.device = "cpu"
        self.processor = None
        self.model = None

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            if cfg.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = cfg.device
                if self.device == "cuda" and not torch.cuda.is_available():
                    logger.warning("GroundingDINO requested on CUDA but CUDA not available; using CPU.")
                    self.device = "cpu"

            kwargs = {}
            if cfg.hf_token:
                kwargs["token"] = cfg.hf_token

            self.processor = AutoProcessor.from_pretrained(cfg.model_id, **kwargs)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_id, **kwargs)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._enabled = True
            logger.info(f"[OK] GroundingDINO loaded: {cfg.model_id} on {self.device}")
        except Exception as e:
            logger.warning(f"GroundingDINO unavailable: {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _run_pass(
        self,
        image_pil: Image.Image,
        text_prompt: str,
        conf_threshold: float,
        text_threshold: float,
        target_size: Tuple[int, int],
    ) -> List[Tuple[np.ndarray, float, str]]:
        """Run a single prompt pass and return raw (box_xyxy, score, label) tuples."""
        if not self.enabled:
            return []

        import torch

        try:
            inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt")
            input_ids = inputs.get("input_ids")
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=float(conf_threshold),
                text_threshold=float(text_threshold),
                target_sizes=[target_size],
            )[0]

            labels = results.get("text_labels") or results.get("labels") or []
            if labels and isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], list):
                labels = labels[0]

            raw: List[Tuple[np.ndarray, float, str]] = []
            for idx, (box, score) in enumerate(zip(results.get("boxes", []), results.get("scores", []))):
                lbl = labels[idx] if idx < len(labels) else "object"
                if isinstance(box, torch.Tensor):
                    box = box.detach().cpu().numpy()
                raw.append((np.asarray(box), float(score), str(lbl)))
            return raw
        except Exception as e:
            logger.debug(f"GroundingDINO pass failed: {e}")
            return []

    def detect(
        self,
        image_path: str,
        class_names: List[str],
        conf_threshold: float = 0.15,
        text_threshold: float = 0.20,
        multi_prompt: bool = False,
        nms_iou: float = 0.5,
    ) -> Tuple[List[Detection], int, int]:
        """Detect requested classes in an image.

        Returns:
            detections, img_width, img_height
        """
        if not self.enabled:
            return [], 0, 0

        p = Path(image_path)
        image_pil = Image.open(p).convert("RGB")
        img_w, img_h = image_pil.size

        classes = [c.strip() for c in class_names if isinstance(c, str) and c.strip()]
        if not classes:
            return [], img_w, img_h

        # Prompt strategies: GroundingDINO usually prefers period-separated phrases.
        prompts: List[str] = [". ".join(classes) + "."]
        if multi_prompt:
            prompts.append(". ".join([f"a {c}" for c in classes]) + ".")
            prompts.append(", ".join(classes))

        raw_all: List[Tuple[np.ndarray, float, str]] = []
        for t in prompts:
            raw_all.extend(self._run_pass(image_pil, t, conf_threshold, text_threshold, (img_h, img_w)))

        dets: List[Detection] = []
        class_lookup = {c.lower(): (i, c) for i, c in enumerate(classes)}

        def _match_label(lbl: str) -> Tuple[int, str]:
            l = str(lbl).lower().strip()
            if l.startswith("a "):
                l = l[2:].strip()
            if l in class_lookup:
                return class_lookup[l]
            # fuzzy: containment
            for cls_l, (i, orig) in class_lookup.items():
                if cls_l in l or l in cls_l:
                    return i, orig
            return 0, classes[0]

        for box, score, lbl in raw_all:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            # clamp to image
            x1 = float(np.clip(x1, 0.0, float(img_w - 1)))
            y1 = float(np.clip(y1, 0.0, float(img_h - 1)))
            x2 = float(np.clip(x2, 0.0, float(img_w - 1)))
            y2 = float(np.clip(y2, 0.0, float(img_h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue

            cid, cname = _match_label(lbl)
            xc, yc, ww, hh = xyxy_to_yolo_norm(x1, y1, x2, y2, img_w, img_h)

            dets.append(
                {
                    "class_id": int(cid),
                    "class_name": str(cname),
                    "confidence": float(score),
                    "x_center": clip01(xc),
                    "y_center": clip01(yc),
                    "width": clip01(ww),
                    "height": clip01(hh),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        dets = nms_detections(dets, iou_threshold=float(nms_iou), class_aware=True)
        dets = [d for d in dets if float(d.get("confidence", 0.0)) >= float(conf_threshold)]
        return dets, img_w, img_h

