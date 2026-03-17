"""Depth estimation pipeline using DepthAnything V2."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from depth_estimation.depth_anything_v2 import DepthAnythingV2Config, DepthAnythingV2Predictor

logger = logging.getLogger(__name__)

DepthResult = Dict[str, Any]


@dataclass
class DepthRunConfig:
    device: str = "auto"  # auto|cuda|cpu
    encoder: str = "vitb"  # vits|vitb|vitl
    checkpoint_path: Optional[str] = None
    save_color: bool = True
    save_raw_depth: bool = False
    cmap: str = "turbo"


class DepthEstimationPipeline:
    """Run depth estimation over a directory of images."""

    def __init__(self, cfg: DepthRunConfig):
        self.cfg = cfg
        predictor_cfg = DepthAnythingV2Config(
            encoder=cfg.encoder,
            checkpoint_path=cfg.checkpoint_path,
            device=cfg.device,
            cmap=cfg.cmap,
        )
        self.predictor = DepthAnythingV2Predictor(predictor_cfg)

    @property
    def enabled(self) -> bool:
        return bool(self.predictor and self.predictor.enabled)

    def process_image(self, image_path: str, output_dir: str) -> DepthResult:
        if not self.enabled:
            raise RuntimeError("DepthAnything V2 predictor is not available.")

        img_path = Path(image_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        depth = self.predictor.infer_image(img)

        color_bgr = None
        if self.cfg.save_color:
            color_bgr = self.predictor.colorize(depth)

        # Compute simple stats
        depth_min = float(np.min(depth))
        depth_max = float(np.max(depth))
        depth_mean = float(np.mean(depth))
        depth_std = float(np.std(depth))

        # Prepare output paths
        stem = img_path.stem
        color_path = Path(output_dir) / "visualizations" / f"{stem}_depth.jpg"
        raw_path = Path(output_dir) / "depth_raw" / f"{stem}.npy"
        meta_path = Path(output_dir) / "metadata" / f"{stem}.json"

        if self.cfg.save_color and color_bgr is not None:
            color_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(color_path), color_bgr)

        if self.cfg.save_raw_depth:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(raw_path), depth)

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "image": str(img_path),
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "encoder": self.cfg.encoder,
            "checkpoint_path": str(Path(self.cfg.checkpoint_path).absolute()) if self.cfg.checkpoint_path else None,
            "color_path": str(color_path) if self.cfg.save_color else None,
            "raw_depth_path": str(raw_path) if self.cfg.save_raw_depth else None,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {
            "image": str(img_path),
            "color_path": str(color_path) if self.cfg.save_color else None,
            "raw_depth_path": str(raw_path) if self.cfg.save_raw_depth else None,
            "meta_path": str(meta_path),
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_mean": depth_mean,
            "depth_std": depth_std,
        }

    def process_dir(self, images_dir: str, output_dir: str, max_images: Optional[int] = None) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DepthAnything V2 predictor is not available.")

        in_dir = Path(images_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts])
        if max_images is not None and int(max_images) > 0:
            images = images[: int(max_images)]

        from tqdm.auto import tqdm

        results: List[DepthResult] = []
        failures: List[Dict[str, str]] = []
        for p in tqdm(images, desc="Depth Estimation"):
            try:
                results.append(self.process_image(str(p), str(out_dir)))
            except Exception as e:
                failures.append({"image": str(p), "error": str(e)})

        summary = {
            "timestamp": datetime.now().isoformat(),
            "images_dir": str(in_dir),
            "output_dir": str(out_dir),
            "num_images": len(images),
            "processed": len(results),
            "failed": len(failures),
            "failures": failures[:50],
        }
        with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
