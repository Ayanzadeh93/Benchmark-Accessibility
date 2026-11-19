"""rho_A — camera-ray angular accuracy, matching the paper's protocol: AUC (to 15 degrees) of the
per-pixel angular error between predicted unit rays and the GT-intrinsics unit rays.

The paper reports rho_A as the area under the curve of the fraction of rays whose angular error is
below a threshold, integrated over thresholds up to 15 deg (normalized to [0,1]).
"""

from __future__ import annotations

import numpy as np
import torch


def rays_from_K(K: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Unit camera rays (3,H,W) from a pinhole K (3,3)."""
    v, u = torch.meshgrid(torch.arange(H, device=K.device, dtype=K.dtype),
                          torch.arange(W, device=K.device, dtype=K.dtype), indexing="ij")
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    z = torch.ones_like(x)
    r = torch.stack([x, y, z], dim=0)
    return r / r.norm(dim=0, keepdim=True).clamp_min(1e-8)


def rho_auc(pred_rays: torch.Tensor, gt_K: torch.Tensor, max_deg: float = 15.0,
            n: int = 30) -> float:
    """pred_rays (3,H,W) unit; gt_K (3,3). Returns AUC of P(angle<thr) over thr in [0,max_deg]."""
    _, H, W = pred_rays.shape
    gt = rays_from_K(gt_K.to(pred_rays.dtype), H, W)
    pr = pred_rays / pred_rays.norm(dim=0, keepdim=True).clamp_min(1e-8)
    cos = (pr * gt).sum(0).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.arccos(cos)).reshape(-1)              # per-pixel angular error (deg)
    thr = torch.linspace(0.0, max_deg, n, device=ang.device, dtype=ang.dtype)
    frac = (ang[None, :] < thr[:, None]).float().mean(dim=1)        # vectorized recall curve
    return float(torch.trapz(frac, thr) / max_deg)                 # normalized AUC in [0,1]


def rho_from_numpy(pred_rays: np.ndarray, gt_K: np.ndarray, device="cpu") -> float:
    return rho_auc(torch.from_numpy(np.ascontiguousarray(pred_rays)).float().to(device),
                   torch.from_numpy(np.ascontiguousarray(gt_K)).float().to(device))
