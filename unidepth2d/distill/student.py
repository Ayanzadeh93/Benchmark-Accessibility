"""Small real-time student that regresses METRIC depth from a single RGB image.

Design: a timm encoder (``features_only=True``) + a light DPT/FPN-style top-down decoder. The head
predicts LOG-depth so the model spans a wide metric range stably; ``infer_metres`` exponentiates and
clamps to [depth_min_m, depth_max_m]. Channels are read from the backbone's ``feature_info`` so the
decoder adapts to whatever backbone is configured (efficientvit_b0/b1, mobilenetv4, convnext...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


@dataclass
class StudentConfig:
    backbone: str = "efficientvit_b1.r224_in1k"
    pretrained_backbone: bool = True
    decoder_channels: int = 96
    depth_min_m: float = 0.2
    depth_max_m: float = 30.0
    use_intrinsics: bool = False       # FiLM-condition the decoder on camera intrinsics (fx/W,fy/H,cx/W,cy/H)
    uncertainty_head: bool = False     # 2nd output channel = log-sigma (heteroscedastic Laplace)
    ray_map: bool = False              # concat per-pixel unit camera rays (3ch) to the input (dense
                                       # camera conditioning, UniDepth-style, distilled to input side)
    bins: int = 0                      # >0: hybrid log-bin head (B logits -> softmax expectation)
    affine_head: bool = False          # N4 AnchDistill: light scene descriptor -> per-image global
                                       # (da,db) recalibrating log-depth (internalizes the SCMR fix)


class _ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthStudent(nn.Module):
    def __init__(self, cfg: StudentConfig = StudentConfig()):
        super().__init__()
        self.cfg = cfg
        in_chans = 6 if cfg.ray_map else 3
        self.encoder = timm.create_model(
            cfg.backbone,
            features_only=True,
            pretrained=cfg.pretrained_backbone,
            in_chans=in_chans,          # timm adapts pretrained stem weights for extra channels
        )
        chs: List[int] = self.encoder.feature_info.channels()  # coarse->fine order varies; timm is fine->coarse
        dc = cfg.decoder_channels

        # Lateral 1x1 projections for each pyramid level.
        self.laterals = nn.ModuleList([nn.Conv2d(c, dc, 1) for c in chs])
        # Smoothing 3x3 after each top-down merge.
        self.smooths = nn.ModuleList([_ConvBNAct(dc, dc, 3) for _ in chs])
        # Output layout: [B bin-logits when bins>0 | else 1 log-depth ch] + [1 log-sigma when unc].
        self.bins = int(cfg.bins)
        depth_ch = self.bins if self.bins > 0 else 1
        out_ch = depth_ch + (1 if cfg.uncertainty_head else 0)
        self.head = nn.Sequential(
            _ConvBNAct(dc, dc, 3),
            nn.Conv2d(dc, out_ch, 1),
        )
        mid_log = math.log(max(1e-3, (cfg.depth_min_m * cfg.depth_max_m) ** 0.5))
        nn.init.zeros_(self.head[-1].bias)
        if self.bins == 0:
            with torch.no_grad():
                self.head[-1].bias[0] = mid_log   # log-depth mu starts mid-range
        else:
            # log-spaced bin centres over [depth_min, depth_max]
            centres = torch.linspace(math.log(cfg.depth_min_m), math.log(cfg.depth_max_m), self.bins)
            self.register_buffer("bin_centres_log", centres)
        self.uncertainty_head = cfg.uncertainty_head
        self.ray_map = cfg.ray_map

        # N4 — AnchDistill: a global scene descriptor (avg-pooled decoder feature) -> a per-image
        # affine (da, db) that recalibrates the log-depth mu as logD' = (1+da)*logD + db. This bakes
        # the training-free SCMR recalibration into the student as a learned, image-conditioned
        # correction. Zero-initialized so it starts at identity and only *adds* a correction pathway.
        self.affine_head = cfg.affine_head
        if self.affine_head:
            self.affine = nn.Sequential(
                nn.Linear(dc, 32), nn.ReLU(inplace=True), nn.Linear(32, 2),
            )
            nn.init.zeros_(self.affine[-1].weight)
            nn.init.zeros_(self.affine[-1].bias)

        # Optional camera-intrinsics conditioning: a small MLP maps the 4 normalized intrinsics to
        # per-channel FiLM (scale gamma, shift beta) applied to the fused decoder feature. Intrinsics
        # encode field-of-view, which is what resolves metric-scale ambiguity across cameras.
        self.use_intrinsics = cfg.use_intrinsics
        if self.use_intrinsics:
            self.film = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 2 * dc),
            )
            nn.init.zeros_(self.film[-1].weight)     # start as identity modulation (gamma=0, beta=0)
            nn.init.zeros_(self.film[-1].bias)

        self.log_min = math.log(cfg.depth_min_m)
        self.log_max = math.log(cfg.depth_max_m)

    _DEFAULT_K = (0.8, 1.08, 0.5, 0.5)

    def _build_rays(self, B, H, W, intrinsics, device, dtype):
        """Per-pixel unit camera rays from normalized intrinsics (fx/W, fy/H, cx/W, cy/H)."""
        if intrinsics is None:
            intrinsics = torch.tensor([self._DEFAULT_K], device=device, dtype=dtype).expand(B, 4)
        fx, fy, cx, cy = (intrinsics[:, i].view(B, 1, 1) for i in range(4))
        u = ((torch.arange(W, device=device, dtype=dtype) + 0.5) / W).view(1, 1, W)
        v = ((torch.arange(H, device=device, dtype=dtype) + 0.5) / H).view(1, H, 1)
        x = (u - cx) / fx.clamp(min=1e-4)
        y = (v - cy) / fy.clamp(min=1e-4)
        z = torch.ones_like(x)
        r = torch.stack([x.expand(B, H, W), y.expand(B, H, W), z.expand(B, H, W)], dim=1)
        return r / r.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def forward(self, x: torch.Tensor, intrinsics: torch.Tensor = None) -> torch.Tensor:
        """x: (B,3,H,W) normalized RGB; intrinsics: (B,4) or None.

        Output layout along channels: [0]=log-depth; [1]=log-sigma (iff uncertainty_head);
        remaining = bin logits (iff bins>0). Callers that only want depth slice [:, :1]."""
        in_hw = x.shape[-2:]
        if self.ray_map:
            rays = self._build_rays(x.shape[0], in_hw[0], in_hw[1], intrinsics, x.device, x.dtype)
            x = torch.cat([x, rays], dim=1)
        feats = self.encoder(x)  # list, fine -> coarse (increasing stride)
        laterals = [l(f) for l, f in zip(self.laterals, feats)]

        # Top-down: start at the coarsest (last), upsample and merge toward the finest (first).
        x_td = self.smooths[-1](laterals[-1])
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(x_td, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            x_td = self.smooths[i](laterals[i] + up)

        if self.use_intrinsics and intrinsics is not None:
            gb = self.film(intrinsics)                       # (B, 2*dc)
            gamma, beta = gb.chunk(2, dim=1)
            x_td = x_td * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        # N4 affine: derive the per-image (da, db) from a global descriptor of the fused feature.
        # BOUNDED with tanh so the log-scale factor (1+da) stays strictly positive and the head can
        # only apply a modest recalibration — an unbounded affine diverges (da can cross -1, flipping
        # the whole depth map -> NaN). Ranges match the scale of the training-free SCMR fix
        # (a~0.97, b~-0.01): da in [-0.3,0.3] -> (1+da) in [0.7,1.3]; db in [-0.5,0.5] log-units.
        affine = None
        if self.affine_head:
            desc = x_td.mean(dim=(2, 3))                              # (B, dc) scene descriptor
            ab = self.affine(desc)
            da = 0.3 * torch.tanh(ab[:, 0:1])
            db = 0.5 * torch.tanh(ab[:, 1:2])
            affine = (da[:, :, None, None], db[:, :, None, None])      # (B,1,1,1) each

        out = self.head(x_td)
        if out.shape[-2:] != in_hw:
            out = F.interpolate(out, size=in_hw, mode="bilinear", align_corners=False)

        if self.bins > 0:
            # hybrid head: log-depth = softmax expectation over log-spaced bin centres.
            # raw head layout: [bins logits | logsigma?] -> public layout [mu, logsigma?, logits]
            logits = out[:, :self.bins]
            logsigma = out[:, self.bins:self.bins + 1] if self.uncertainty_head else None
            p = torch.softmax(logits, dim=1)
            centres = self.bin_centres_log.to(p.dtype).view(1, -1, 1, 1)
            mu = (p * centres).sum(dim=1, keepdim=True)
            if affine is not None:
                mu = mu * (1.0 + affine[0]) + affine[1]
            parts = [mu] + ([logsigma] if logsigma is not None else []) + [logits]
            return torch.cat(parts, dim=1)

        if affine is not None:
            mu = out[:, :1] * (1.0 + affine[0]) + affine[1]
            out = torch.cat([mu, out[:, 1:]], dim=1) if out.shape[1] > 1 else mu
        return out

    @torch.no_grad()
    def infer_metres(self, x: torch.Tensor, intrinsics: torch.Tensor = None) -> torch.Tensor:
        """Return clamped metric depth (B,1,H,W) in metres."""
        log_depth = self.forward(x, intrinsics)[:, :1].clamp(self.log_min, self.log_max)
        return torch.exp(log_depth)

    @torch.no_grad()
    def infer_with_uncertainty(self, x: torch.Tensor, intrinsics: torch.Tensor = None):
        """Return (depth_m (B,1,H,W), sigma_log (B,1,H,W) or None)."""
        out = self.forward(x, intrinsics)
        depth = torch.exp(out[:, :1].clamp(self.log_min, self.log_max))
        sigma = torch.exp(out[:, 1:2].clamp(-3.0, 2.0)) if out.shape[1] > 1 else None
        return depth, sigma


def build_student(cfg: dict) -> DepthStudent:
    s = cfg["student"]
    return DepthStudent(StudentConfig(
        backbone=s["backbone"],
        pretrained_backbone=bool(s.get("pretrained_backbone", True)),
        decoder_channels=int(s.get("decoder_channels", 96)),
        depth_min_m=float(s.get("depth_min_m", 0.2)),
        depth_max_m=float(s.get("depth_max_m", 30.0)),
        use_intrinsics=bool(s.get("use_intrinsics", False)),
        uncertainty_head=bool(s.get("uncertainty_head", False)),
        ray_map=bool(s.get("ray_map", False)),
        bins=int(s.get("bins", 0)),
        affine_head=bool(s.get("affine_head", False)),
    ))
