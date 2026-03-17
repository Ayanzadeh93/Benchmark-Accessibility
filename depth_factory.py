#!/usr/bin/env python3
"""Factory for creating depth estimators (DepthAnything V2)."""

from __future__ import annotations

import logging
from typing import Optional

from depth_estimation.depth_anything_v2 import DepthAnythingV2Config, DepthAnythingV2Predictor

logger = logging.getLogger(__name__)


class DepthFactory:
    """Factory for DepthAnything V2 predictors."""

    @staticmethod
    def create_predictor(
        encoder: str = "vitb",
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        cmap: str = "turbo",
    ) -> DepthAnythingV2Predictor:
        cfg = DepthAnythingV2Config(
            encoder=encoder,
            checkpoint_path=checkpoint_path,
            device=device,
            cmap=cmap,
        )
        predictor = DepthAnythingV2Predictor(cfg)
        if not predictor.enabled:
            raise RuntimeError("DepthAnything V2 predictor is not available (missing package or checkpoint).")
        return predictor

    @staticmethod
    def list_available_models():
        print("\nAvailable depth encoders: vits (small), vitb (base), vitl (large)\n")


def create_depth_predictor(
    encoder: str = "vitb",
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    cmap: str = "turbo",
) -> DepthAnythingV2Predictor:
    """Convenience shortcut."""
    return DepthFactory.create_predictor(
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        device=device,
        cmap=cmap,
    )


if __name__ == "__main__":
    DepthFactory.list_available_models()
