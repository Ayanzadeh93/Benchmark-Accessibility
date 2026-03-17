"""DepthAnything V2 predictor wrapper for depth estimation on images/videos."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import torch

# Headless-friendly backend
matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthAnythingV2Config:
    """Configuration for DepthAnything V2."""

    encoder: str = "vitb"  # vits | vitb | vitl
    checkpoint_path: Optional[str] = None
    device: str = "auto"  # auto|cuda|cpu
    cmap: str = "turbo"  # matplotlib colormap name


class DepthAnythingV2Predictor:
    """Thin wrapper around DepthAnything V2 for single-image depth estimation."""

    def __init__(self, cfg: DepthAnythingV2Config):
        self.cfg = cfg
        self.device: str = "cpu"
        self.model = None
        self.cmap = self._resolve_cmap(cfg.cmap)
        self._enabled = False

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            logger.warning(
                "DepthAnything V2 not available. Install with:\n"
                "  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            )
            return

        enc = cfg.encoder.lower().strip()
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        if enc not in model_configs:
            logger.error(f"Invalid encoder '{cfg.encoder}'. Choose from vits | vitb | vitl.")
            return

        # Resolve device
        requested = cfg.device.lower().strip()
        if requested == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested == "cpu":
            self.device = "cpu"
        else:  # auto
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve checkpoint
        default_ckpt = f"depth_anything_v2_{enc}.pth"
        ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else Path(default_ckpt)
        if not ckpt_path.exists():
            logger.error(
                "DepthAnything V2 checkpoint not found at %s\n"
                "Download from Hugging Face:\n"
                "  Small: https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth\n"
                "  Base:  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth\n"
                "  Large: https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth\n"
                "Save as %s or pass --depth-checkpoint",
                ckpt_path,
                default_ckpt,
            )
            return

        try:
            logger.info(
                "Loading DepthAnything V2 (encoder=%s, device=%s, checkpoint=%s)",
                enc,
                self.device,
                ckpt_path,
            )
            model = DepthAnythingV2(**model_configs[enc])
            state = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(state)
            model = model.to(self.device).eval()
            self.model = model
            self._enabled = True
            logger.info("[OK] DepthAnything V2 loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DepthAnything V2: {e}")
            self.model = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self.model is not None

    def infer_image(self, image: np.ndarray) -> np.ndarray:
        """Run depth estimation on a BGR image."""
        if not self.enabled:
            raise RuntimeError("DepthAnything V2 is not initialized.")
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array (BGR).")

        with torch.inference_mode():
            depth = self.model.infer_image(image)

        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()

        depth = depth.astype(np.float32)
        return depth

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        """Convert a depth map to a BGR colormap image."""
        if depth is None:
            raise ValueError("depth must not be None")
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        denom = depth_max - depth_min + 1e-8
        depth_norm = (depth - depth_min) / denom
        cmap_rgb = self.cmap(depth_norm)[:, :, :3]  # (H, W, 3) float [0,1]
        cmap_rgb = (cmap_rgb * 255).astype(np.uint8)
        return cv2.cvtColor(cmap_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _resolve_cmap(name: str):
        try:
            return colormaps.get_cmap(name)
        except Exception:
            return colormaps["turbo"]
