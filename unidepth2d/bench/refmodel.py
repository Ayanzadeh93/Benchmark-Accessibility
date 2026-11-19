"""Model wrappers for the benchmark harness. All expose the same `predict(rgb_hwc_uint8, gt_K)`
returning depth/points/rays/intrinsics/confidence as numpy at the ORIGINAL image resolution, so a
single eval driver can score the official UniDepthV2, our distilled students, or the V2++ wrapper.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch


def _check_full_res(hw, **maps) -> None:
    """Raise if any (…,H,W) map is not at the original image size ``hw``."""
    bad = {k: tuple(v.shape) for k, v in maps.items() if v is not None and tuple(v.shape[-2:]) != tuple(hw)}
    if bad:
        raise ValueError(f"UniDepth outputs not at input size {tuple(hw)}: {bad}; "
                         "rays/points/K would not match a resized depth")


class V2Model:
    """Official UniDepthV2 (frozen). resolution_level fixed here; the V2++ wrapper overrides it."""

    def __init__(self, checkpoint: str, device: str = "cuda", resolution_level: int = 5):
        from unidepth.models import UniDepthV2
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model = UniDepthV2.from_pretrained(checkpoint).to(self.device).eval()
        self.model.resolution_level = int(resolution_level)
        self.name = f"V2:{checkpoint.split('/')[-1]}@r{resolution_level}"
        self.checkpoint = checkpoint

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, gt_K: Optional[np.ndarray] = None,
                inject_camera: bool = False) -> dict:
        H, W = rgb.shape[:2]
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)  # (3,H,W) uint8
        camera = None
        if inject_camera and gt_K is not None:
            camera = torch.from_numpy(np.ascontiguousarray(gt_K)).float()
        out = self.model.infer(t.to(self.device), camera=camera)
        depth = out["depth"][0, 0].float().cpu().numpy()
        points = out["points"][0].float().cpu().numpy()          # (3,H,W)
        rays = out["rays"][0].float().cpu().numpy()              # (3,H,W)
        conf = out["confidence"][0, 0].float().cpu().numpy() if out.get("confidence") is not None else None
        K = out["intrinsics"][0].float().cpu().numpy()
        # UniDepthV2.infer returns every map at the input size and K rescaled to it. Resizing
        # depth alone would pair it with grid-size rays/points and a mismatched K (wrong rho_A and
        # F_A), so a size mismatch is an error, never a silent resize.
        _check_full_res((H, W), depth=depth, points=points, rays=rays, conf=conf)
        return {"depth_m": depth, "points": points, "rays": rays, "conf": conf, "K": K}


class StudentModel:
    """Our distilled student (efficientvit). No camera/rays output -> 2D metrics + K-from-GT for F_A."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from distill.student import build_student
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = ck.get("cfg")
        self.model = build_student(self.cfg).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        s = self.cfg["student"]
        self.th, self.tw = int(s["train_size"][0]), int(s["train_size"][1])
        self.use_intr = bool(s.get("use_intrinsics", False))
        import os
        self.name = f"student:{os.path.basename(os.path.dirname(ckpt_path))}"
        from distill.dataset import IMAGENET_MEAN, IMAGENET_STD
        self._mean, self._std = IMAGENET_MEAN, IMAGENET_STD
        self._default_K = torch.tensor([[0.8, 1.08, 0.5, 0.5]], dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, gt_K: Optional[np.ndarray] = None, inject_camera=False) -> dict:
        H, W = rgb.shape[:2]
        rz = cv2.resize(rgb, (self.tw, self.th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rz = (rz - self._mean) / self._std
        t = torch.from_numpy(np.ascontiguousarray(np.transpose(rz, (2, 0, 1))))[None].float().to(self.device)
        dm = self.model.infer_metres(t, self._default_K if self.use_intr else None)[0, 0].cpu().numpy()
        depth = cv2.resize(dm, (W, H), interpolation=cv2.INTER_LINEAR)
        return {"depth_m": depth, "points": None, "rays": None, "conf": None, "K": None}


def build_model(spec: str, device: str = "cuda", resolution_level: int = 5):
    if spec in ("vitl", "V2-L"):
        return V2Model("lpiccinelli/unidepth-v2-vitl14", device, resolution_level)
    if spec in ("vits", "V2-S"):
        return V2Model("lpiccinelli/unidepth-v2-vits14", device, resolution_level)
    if spec.startswith("v2:"):
        return V2Model(spec[3:], device, resolution_level)
    if spec.startswith("student:"):
        return StudentModel(spec.split(":", 1)[1], device)
    raise ValueError(f"unknown model spec {spec}")
