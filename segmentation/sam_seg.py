"""SAM3 (Segment Anything Model 3) wrapper for instance segmentation.

SAM3 supports **text prompts** (best match for this project, since we already have VLM
object lists) and also supports **visual/geometric prompts** (boxes/points).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

Detection = Dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAM3SegConfig:
    """Configuration for SAM3 segmentation."""

    device: str = "auto"  # auto|cuda|cpu
    checkpoint_path: Optional[str] = None  # If None, can load from HuggingFace (see load_from_hf)
    load_from_hf: bool = True
    compile: bool = False

    # SAM3Processor knobs (text-prompt mode)
    resolution: int = 1008
    text_confidence_threshold: float = 0.5

    # Post-processing knobs (for Sam3Processor)
    mask_threshold: float = 0.0
    max_hole_area: float = 256.0
    max_sprinkle_area: float = 0.0


class SAMSegDetector:
    """SAM3 predictor wrapper supporting text prompts (primary) and box prompts (fallback)."""

    def __init__(self, cfg: SAM3SegConfig):
        self.cfg = cfg
        self._enabled = False
        self.device: Union[int, str] = "cpu"
        # SAM3 uses Sam3Processor (not SAM3InteractiveImagePredictor which is for SAM1/SAM2)
        self.processor = None
        self.sam3_model = None

        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import os

            # If the user has a HuggingFace token in `.env` / env vars, forward it to the
            # variables used by huggingface_hub. SAM3 checkpoints are gated on HF.
            try:
                from config import get_huggingface_token

                token = get_huggingface_token()
                if token:
                    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
                    os.environ.setdefault("HF_TOKEN", token)
            except Exception:
                # Don't fail SAM3 init if config import isn't available.
                pass

            # Resolve device
            requested = str(cfg.device).lower()
            use_cuda = False
            if requested == "cuda":
                use_cuda = True
            elif requested == "cpu":
                use_cuda = False
            else:  # auto
                use_cuda = bool(torch.cuda.is_available())

            if use_cuda:
                try:
                    _ = torch.zeros(1).cuda()
                    self.device = "cuda"
                    logger.info(f"[OK] SAM3 using CUDA: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    logger.warning(f"CUDA detected but failed test: {e}. Using CPU for SAM3.")
                    self.device = "cpu"
            else:
                self.device = "cpu"

            if (cfg.checkpoint_path is None) and (not bool(cfg.load_from_hf)):
                logger.warning("SAM3 checkpoint not provided and load_from_hf is disabled.")
                return

            logger.info(
                "Loading SAM3 image model "
                f"(device={self.device}, load_from_hf={bool(cfg.load_from_hf)}, compile={bool(cfg.compile)})"
            )
            model = build_sam3_image_model(
                device=str(self.device),
                checkpoint_path=cfg.checkpoint_path,
                load_from_HF=bool(cfg.load_from_hf),
                compile=bool(cfg.compile),
                enable_segmentation=True,
            )
            self.sam3_model = model
            self.processor = Sam3Processor(
                model,
                resolution=int(cfg.resolution),
                device=str(self.device),
                confidence_threshold=float(cfg.text_confidence_threshold),
            )
            # Note: SAM3InteractiveImagePredictor is for SAM1/SAM2, not SAM3.
            # We use Sam3Processor for SAM3 (supports text prompts natively).
            self._enabled = True
            logger.info("[OK] SAM3 loaded successfully")

        except ImportError:
            logger.warning(
                "SAM3 not available. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam3.git\n"
                "And on Windows for speed:\n"
                "  pip install triton-windows\n"
            )
        except Exception as e:
            msg = str(e)
            if "gated repo" in msg.lower() or "401" in msg:
                logger.warning(
                    "SAM3 initialization failed (likely HuggingFace auth / gated checkpoint).\n"
                    "Fix options:\n"
                    "- Request access and authenticate: https://huggingface.co/facebook/sam3\n"
                    "- Run `huggingface-cli login` OR set HF token in `.env` (HF_TOKEN=...)\n"
                    "- Or download a checkpoint and pass it via --sam3-checkpoint with --sam3-no-hf\n"
                    f"Original error: {e}"
                )
            else:
                logger.warning(f"SAM3 initialization failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(
        self,
        image_path: str,
        bbox_prompts: Optional[List[List[float]]] = None,
        text_prompts: Optional[List[str]] = None,
        min_conf: float = 0.15,
    ) -> Tuple[List[Detection], int, int]:
        """Run SAM3 segmentation using either text prompts (preferred) or bounding box prompts.

        Args:
            image_path: Path to image file
            bbox_prompts: Optional list of [x1, y1, x2, y2] pixel boxes (fallback mode)
            text_prompts: Optional list of text prompts (preferred mode)
            min_conf: Confidence threshold to filter outputs

        Returns:
            detections, img_width, img_height
        """
        if not self.enabled:
            return [], 0, 0

        # Prefer text prompts if provided.
        if text_prompts:
            try:
                from PIL import Image

                image = Image.open(str(image_path)).convert("RGB")
                img_w, img_h = image.size

                if self.processor is None:
                    return [], img_w, img_h

                state = self.processor.set_image(image)

                dets: List[Detection] = []
                for class_id, prompt in enumerate(text_prompts):
                    if not isinstance(prompt, str) or not prompt.strip():
                        continue

                    state = self.processor.set_text_prompt(prompt=prompt, state=state)
                    masks = state.get("masks")
                    boxes = state.get("boxes")
                    scores = state.get("scores")
                    if masks is None or boxes is None or scores is None:
                        continue

                    try:
                        # torch tensors -> cpu numpy
                        masks_np = masks.detach().cpu().numpy()
                        boxes_np = boxes.detach().cpu().numpy()
                        scores_np = scores.detach().cpu().numpy()
                    except Exception:
                        continue

                    for m, b, s in zip(masks_np, boxes_np, scores_np):
                        score = float(s)
                        if score < float(min_conf):
                            continue

                        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
                        # Clamp to image bounds
                        x1 = max(0.0, min(x1, float(img_w - 1)))
                        y1 = max(0.0, min(y1, float(img_h - 1)))
                        x2 = max(0.0, min(x2, float(img_w - 1)))
                        y2 = max(0.0, min(y2, float(img_h - 1)))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        from .utils import mask_to_polygon, xyxy_to_yolo_norm, clamp01

                        poly = mask_to_polygon(m)
                        has_seg = bool(poly and len(poly) >= 6)
                        xc, yc, ww, hh = xyxy_to_yolo_norm(x1, y1, x2, y2, img_w, img_h)

                        dets.append(
                            {
                                "class_id": int(class_id),
                                "class_name": str(prompt),
                                "confidence": float(score),
                                "x_center": clamp01(xc),
                                "y_center": clamp01(yc),
                                "width": clamp01(ww),
                                "height": clamp01(hh),
                                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                "has_segmentation": bool(has_seg),
                                "segmentation": poly if has_seg else [],
                                # Not JSON-serializable; pipeline strips this before saving metadata.
                                "mask": m,
                            }
                        )

                logger.info(f"SAM3(text): {len(dets)} masks from {len(text_prompts)} prompts")
                return dets, int(img_w), int(img_h)
            except Exception as e:
                logger.warning(f"SAM3 text-prompt segmentation failed: {e}. Falling back to box prompts if provided.")

        # Fallback: box prompts
        if bbox_prompts:
            try:
                from PIL import Image

                image = Image.open(str(image_path)).convert("RGB")
                img_w, img_h = image.size

                if self.processor is None:
                    return [], img_w, img_h

                state = self.processor.set_image(image)
                # Set a dummy text prompt for geometric-only mode
                state = self.processor.set_text_prompt("visual", state=state)

                detections: List[Detection] = []

                # Process each bounding box prompt using add_geometric_prompt
                for bbox in bbox_prompts:
                    if len(bbox) < 4:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    x2 = max(0, min(x2, img_w - 1))
                    y2 = max(0, min(y2, img_h - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Convert xyxy to center_x, center_y, width, height (normalized [0,1])
                    cx = (x1 + x2) / 2.0 / float(img_w)
                    cy = (y1 + y2) / 2.0 / float(img_h)
                    w = (x2 - x1) / float(img_w)
                    h = (y2 - y1) / float(img_h)

                    try:
                        # add_geometric_prompt expects [cx, cy, w, h] normalized [0,1]
                        state = self.processor.add_geometric_prompt(
                            box=[cx, cy, w, h],
                            label=True,  # positive box
                            state=state,
                        )

                        masks = state.get("masks")
                        boxes = state.get("boxes")
                        scores = state.get("scores")
                        if masks is None or boxes is None or scores is None:
                            continue

                        try:
                            # torch tensors -> cpu numpy
                            masks_np = masks.detach().cpu().numpy()
                            boxes_np = boxes.detach().cpu().numpy()
                            scores_np = scores.detach().cpu().numpy()
                        except Exception:
                            continue

                        # Take the last (most recent) mask/box/score from the state
                        if len(masks_np) > 0:
                            m = masks_np[-1]
                            b = boxes_np[-1]
                            s = scores_np[-1]

                            score = float(s)
                            if score < float(min_conf):
                                continue

                            # b is already in xyxy format from processor
                            bx1, by1, bx2, by2 = [float(v) for v in b.tolist()]

                            from .utils import mask_to_polygon, xyxy_to_yolo_norm, clamp01

                            poly = mask_to_polygon(m)
                            has_seg = bool(poly and len(poly) >= 6)
                            xc, yc, ww, hh = xyxy_to_yolo_norm(bx1, by1, bx2, by2, img_w, img_h)

                            detections.append(
                                {
                                    "class_id": 0,
                                    "class_name": "object",
                                    "confidence": float(score),
                                    "x_center": clamp01(xc),
                                    "y_center": clamp01(yc),
                                    "width": clamp01(ww),
                                    "height": clamp01(hh),
                                    "bbox_xyxy": [float(bx1), float(by1), float(bx2), float(by2)],
                                    "has_segmentation": has_seg,
                                    "segmentation": poly if has_seg else [],
                                    "mask": m,
                                }
                            )
                    except Exception as e:
                        logger.warning(f"SAM3 geometric prompt failed for bbox {bbox}: {e}")
                        continue

                logger.info(f"SAM3: {len(detections)} masks generated from {len(bbox_prompts)} bbox prompts")
                return detections, img_w, img_h

            except Exception as e:
                logger.error(f"SAM3 segmentation error: {e}")
                return [], 0, 0

        return [], 0, 0
