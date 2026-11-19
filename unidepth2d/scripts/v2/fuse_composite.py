"""Composite teacher labels (V2-L + MoGe-2-L), the L0 label-validation run and the A8 label variants.

Plan section 5.4, task 22, ladder run L0, amendment A8. Env: ``depth`` or ``depth_cmp`` (CPU only:
numpy, scipy, opencv, pyyaml; no torch, no GPU).

    python -m scripts.v2.fuse_composite fuse --v2l-dir outputs/depth_mm --moge-dir outputs/v2/labels/moge
    python -m scripts.v2.fuse_composite l0-list --roles configs/v2/roles.yaml --n 500
    python -m scripts.v2.label_unidepth --input outputs/v2/l0/l0_list_n500.csv --out outputs/v2/labels/l0_v2l
    <depth_cmp python> -m scripts.v2.label_moge --input outputs/v2/l0/l0_list_n500.csv --out outputs/v2/labels/l0_moge
    python -m scripts.v2.fuse_composite validate --list outputs/v2/l0/l0_list_n500.csv \
        --v2l-dir outputs/v2/labels/l0_v2l --moge-dir outputs/v2/labels/l0_moge [--register]
    (``--validate`` is accepted as an alias of the ``validate`` subcommand.)

COMPOSITE (per image, at the stored label resolution; both teachers on the same pixel grid)
  1. L_U = ln(V2-L), L_M = ln(MoGe-2-L). V2-L is valid where 0.2 <= d <= 30 m; MoGe-invalid
     pixels (PNG 0) are filled from L_U. V2-L-invalid pixels are filled with a validity-
     normalised Gaussian low-pass of the valid L_U so the spectral solve sees no artificial
     step; they stay invalid (0) in every output.
  2. Robust affine L_M' = a*L_M + b fitted to L_U by IRLS (8 rounds of weighted least squares,
     weights 1/max(|r|, 0.02), i.e. an L1 fit; pixels valid for both teachers only). If a falls
     outside [0.8, 1.25] it is clamped and b re-fitted as median(L_U - a*L_M).
  3. DCT-II solve of  min_L sum_pairs((L_i-L_j) - (M'_i-M'_j))^2 + lam*sum(L - L_U)^2 with
     Neumann boundary (4-neighbour graph Laplacian), exactly:
        L_C_hat = (lam*U_hat + mu*M_hat) / (lam + mu),
        mu(ky,kx) = (2 - 2cos(pi*ky/H)) + (2 - 2cos(pi*kx/W)),  lam = (pi/crossover)^2.
     mu = lam at the DCT frequency whose half-period is ``crossover`` px (period 2*crossover):
     coarser structure comes from V2-L, finer detail from MoGe. mu(0,0) = 0, so the DC
     coefficient is U_hat(0,0) and mean(L_C) == mean(L_U) exactly: the image's mean log depth
     (its metric scale) comes only from V2-L.
  4. Low-pass agreement mask: keep = V2-L valid & |G(L_M') - G(L_U)| <= ln(1.25), where G is a
     validity-normalised Gaussian with sigma 16 px.
  Resolution convention (my decision; plan defines both at short side 480): crossover 48 px and
  sigma 16 px scale linearly with the label's short side, so the split is the same fraction of
  the image at any label size and lands at exactly 48 / 16 px after the cover cache resizes the
  label to short side 480. At short side 480 (all D_SUN frames) the values are exactly the
  plan's lam = (pi/48)^2 and sigma = 16. --no-scale-with-resolution uses the raw values.

OUTPUTS of ``fuse`` (under --out, default outputs/v2/labels/composite)
  depth_mm/<id>.png    uint16 mm, exp(L_C) where V2-L is valid, 0 elsewhere (no agreement mask)
  agree_mask/<id>.png  uint8, 255 = low-pass agreement keep, 0 = masked
  meta/<id>.json       fit (a, b, clamped), effective lam/sigma, mask statistics, input hashes
  done.csv, run_<utc>.json
  --v2l-dir may be a V2 label dir (depth_mm/ + meta/, from label_unidepth.py) or the legacy
  flat cache outputs/depth_mm (<id>.png, no sidecars; the own-row V2-L labels of plan 5.2).
  Leak guard: when configs/v2/roles.yaml exists, ``fuse`` refuses ids in dev.d_sun,
  test.frozen_own or test.video_own (training labels never touch dev/test frames).
  A8 label variants (``load_composite_label``):
    "l1a" = composite target, valid = depth > 0                (no agreement mask)
    "l1b" = composite target, valid = depth > 0 & mask == 255  (low-pass agreement mask)

HOW THE L1b MASK DIFFERS FROM S8 (A8 asks for this to be explicit)
  S8 (distill/dataset.py:266-275, harmful: -2.8 NYU delta1) multiplied every pixel's loss by a
  continuous weight exp(-|ln d_teacher - ln d_assistant| / 0.2), computed per pixel at full
  resolution on the RAW, unaligned assistant. Any global scale gap between the two models
  therefore down-weighted the whole image, and sub-pixel edge misalignment down-weighted
  exactly the edge pixels. The L1b mask is (i) binary, (ii) computed after the affine
  alignment, so a global scale gap does not trigger it, (iii) computed on sigma-16 low-passed
  log depth, so edge misalignment does not trigger it, and (iv) applied to a composite target
  instead of reweighting the teacher. It removes only regions where the coarse layout of the
  two teachers disagrees by more than 25% (prototype E8: 2.7% of pixels masked, 89.0% of
  strong-edge gradient pairs kept). ``validate`` measures both side by side (mask_vs_s8).

L0 (plan 7.1; pre-registered gates, on D_SUN frames with sensor GT; camera_mode none)
  G1 composite delta1 >= V2-L delta1 - 0.2 points; G2 composite delta1_ssi >= V2-L - 0.1
  points; G3 composite boundary F1 >= V2-L + 0.005; G4 mean edge-pair survival under the
  mask >= 85%. Depth metrics come from benchmark.protocol.per_image_depth_metrics (per image,
  GT-side masking, unweighted mean). Both labels are scored on the SAME pixels: GT-valid
  (manifest dmin..dmax) and V2-L-valid. Boundary F1 = the corrected
  benchmark.protocol_v2.boundary_f1_si_v2 when importable (BF_CANDIDATES order), otherwise the
  legacy protocol.boundary_f1_si with a note; the legacy value is always reported alongside.
  Paired statistics use the D_SUN near-duplicate cluster bootstrap
  (protocol_v2.paired_bootstrap_clustered, clusters = roles dev.d_sun_clusters, others
  singletons); image-level when no roles file is available (flagged).
  A8: ``a8`` reports the L1a label (composite, all label pixels) against the L1b label
  (composite on mask-kept pixels): accuracy on the pixels each supervises, coverage, and the
  accuracy of the pixels the mask removes. L1a needs G1-G3; L1b additionally needs G4. The A8
  training decision (keep the mask only if L1b clears the margin on its own) is a ladder run,
  not part of this script.

Fairness: L0 runs only on development (D_SUN) frames, and its labels live in their own dirs
(outputs/v2/labels/l0_*); D_SUN never trains. The 500-frame sample is drawn with a fixed seed
(2027) from the frozen D_SUN ids in configs/v2/roles.yaml. Without roles.yaml ``l0-list``
refuses unless --provisional-sun is given; provisional and smoke results are flagged in the
JSON and are never registered. Registered L0 = one ``root`` record (run_id "L0", no parent,
no checkpoint) in benchmark/registry_v2.jsonl.

Tests: tests/test_composite.py (env depth or depth_cmp; CPU, offline, synthetic maps).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.fft import dctn, idctn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.v2.label_unidepth import (  # noqa: E402
    atomic_write_json, atomic_write_png, check_drives, depth_m_to_png16, done_ids, read_label_png,
    sha256_file,
)

LN125 = math.log(1.25)
LN110 = math.log(1.10)
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "v2", "labels", "composite")
DEFAULT_L0_DIR = os.path.join(REPO_ROOT, "outputs", "v2", "l0")
DEFAULT_ROLES = os.path.join(REPO_ROOT, "configs", "v2", "roles.yaml")
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "outputs", "data_gt", "manifest.csv")
LEGACY_V2L_DIR = os.path.join(REPO_ROOT, "outputs", "depth_mm")
LEGACY_V2L_CHECKPOINT = "lpiccinelli/unidepth-v2-vitl14"   # outputs/run_summary.json "checkpoint"
L0_SEED = 2027
L0_N = 500
L0_GATES = {"d1_drop_max_pts": 0.2, "d1_ssi_drop_max_pts": 0.1, "bf_gain_min": 0.005,
            "edge_pair_survival_min": 0.85}
S8_TAU = 0.2   # distill/dataset.py assistant_tau default
# (module, function) pairs for a corrected boundary F1 (plan task 1); first hit wins
BF_CANDIDATES = (("benchmark.protocol_v2", "boundary_f1_si_v2"),
                 ("benchmark.protocol", "boundary_f1_si_v2"),
                 ("benchmark.protocol", "boundary_f1_si_corrected"))
LABEL_VARIANTS = {"l1a": "composite target, no agreement mask (valid = depth > 0)",
                  "l1b": "composite target + low-pass agreement mask (valid = depth > 0 & mask)"}
FORBIDDEN_ROLE_KEYS = (("dev", "d_sun"), ("test", "frozen_own"), ("test", "video_own"))


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompositeConfig:
    crossover_px: float = 48.0        # lam = (pi / crossover)^2 at the reference short side
    sigma_px: float = 16.0            # agreement low-pass sigma at the reference short side
    ref_short_side: int = 480
    scale_with_resolution: bool = True
    agree_thr: float = LN125
    a_min: float = 0.8
    a_max: float = 1.25
    irls_iters: int = 8
    irls_eps: float = 0.02
    irls_max_points: int = 200_000
    dmin_m: float = 0.2
    dmax_m: float = 30.0
    min_overlap: float = 0.5          # fraction of pixels valid for both teachers needed to fuse
    edge_jump: float = LN110          # strong-edge neighbour log jump for edge-pair survival

    def effective(self, h: int, w: int) -> Tuple[float, float, float]:
        """(crossover_px, sigma_px, lam) for an h x w map."""
        s = (min(h, w) / float(self.ref_short_side)) if self.scale_with_resolution else 1.0
        cross = self.crossover_px * s
        return cross, self.sigma_px * s, lam_from_crossover(cross)


def lam_from_crossover(crossover_px: float) -> float:
    return (math.pi / float(crossover_px)) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# DCT-II solver
# ──────────────────────────────────────────────────────────────────────────────

def neumann_eigenvalues(h: int, w: int) -> np.ndarray:
    """Eigenvalues mu(ky,kx) of the 4-neighbour Neumann graph Laplacian, diagonalised by DCT-II."""
    ky = np.arange(h, dtype=np.float64)[:, None]
    kx = np.arange(w, dtype=np.float64)[None, :]
    return (2.0 - 2.0 * np.cos(np.pi * ky / h)) + (2.0 - 2.0 * np.cos(np.pi * kx / w))


def dct_composite(L_U: np.ndarray, L_M: np.ndarray, lam: float) -> np.ndarray:
    """Spectral blend L_C_hat = (lam*U_hat + mu*M_hat)/(lam + mu); the DC term is always U's.

    Exact minimiser of sum over neighbour pairs ((L_i-L_j) - (M_i-M_j))^2 + lam*sum (L-U)^2.
    lam = 0 is allowed (pure gradient transfer with V2-L's mean); lam = inf returns L_U.
    """
    L_U = np.asarray(L_U, dtype=np.float64)
    L_M = np.asarray(L_M, dtype=np.float64)
    if L_U.shape != L_M.shape or L_U.ndim != 2:
        raise ValueError(f"shape mismatch {L_U.shape} vs {L_M.shape}")
    if not (lam >= 0):
        raise ValueError(f"lam must be >= 0, got {lam}")
    if math.isinf(lam):
        return L_U.copy()
    U = dctn(L_U, type=2, norm="ortho")
    M = dctn(L_M, type=2, norm="ortho")
    mu = neumann_eigenvalues(*L_U.shape)
    den = lam + mu
    C = np.empty_like(U)
    nz = den > 0
    C[nz] = (lam * U[nz] + mu[nz] * M[nz]) / den[nz]
    C[0, 0] = U[0, 0]
    return idctn(C, type=2, norm="ortho")


def neumann_laplacian(L: np.ndarray) -> np.ndarray:
    """(Delta L)_i = sum over 4-neighbours j inside the image of (L_i - L_j). Independent of the DCT."""
    L = np.asarray(L, dtype=np.float64)
    out = np.zeros_like(L)
    dx = L[:, 1:] - L[:, :-1]
    out[:, :-1] -= dx
    out[:, 1:] += dx
    dy = L[1:, :] - L[:-1, :]
    out[:-1, :] -= dy
    out[1:, :] += dy
    return out


def normal_equation_residual(L_C: np.ndarray, L_U: np.ndarray, L_M: np.ndarray, lam: float) -> float:
    """max |(lam I + Delta) L_C - (lam L_U + Delta L_M)|: 0 for the exact minimiser."""
    r = lam * L_C + neumann_laplacian(L_C) - lam * np.asarray(L_U, np.float64) - neumann_laplacian(L_M)
    return float(np.abs(r).max())


def composite_energy(L: np.ndarray, L_U: np.ndarray, L_M: np.ndarray, lam: float) -> float:
    """The objective the solver minimises (for tests and diagnostics)."""
    L, L_U, L_M = (np.asarray(x, np.float64) for x in (L, L_U, L_M))
    gx = np.diff(L, axis=1) - np.diff(L_M, axis=1)
    gy = np.diff(L, axis=0) - np.diff(L_M, axis=0)
    return float((gx ** 2).sum() + (gy ** 2).sum() + lam * ((L - L_U) ** 2).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Robust affine alignment
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AffineFit:
    a: float
    b: float
    a_raw: float
    b_raw: float
    clamped: bool
    n: int
    ok: bool


def robust_affine_fit(x: np.ndarray, y: np.ndarray, a_min: float = 0.8, a_max: float = 1.25,
                      iters: int = 8, eps: float = 0.02, max_points: int = 200_000) -> AffineFit:
    """IRLS fit of y ~ a*x + b with weights 1/max(|r|, eps) (L1-like); a clamped to [a_min, a_max].

    When a is clamped, b is re-fitted as median(y - a*x). Fewer than 10 points: a = 1,
    b = median(y - x), ok False. A constant x (slope unidentifiable): a = clip(1), ok True.
    More than max_points: deterministic evenly spaced subsample.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(x.size)
    if n < 10:
        b = float(np.median(y - x)) if n else 0.0
        return AffineFit(1.0, b, float("nan"), float("nan"), False, n, False)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        x, y = x[idx], y[idx]
    if float(np.std(x)) < 1e-9:
        a = float(np.clip(1.0, a_min, a_max))
        return AffineFit(a, float(np.median(y - a * x)), float("nan"), float("nan"), True, n, True)
    w = np.ones_like(x)
    a = b = 0.0
    for it in range(int(iters)):
        # Closed-form weighted least squares for y ~ a*x + b (centred normal equations; exact).
        # Deliberately no np.linalg: MKL's lstsq aborts the process (OMP Error #15, duplicate
        # libiomp5md.dll) on Windows when torch is imported in the same interpreter.
        sw_ = float(w.sum())
        xm, ym = float((w * x).sum()) / sw_, float((w * y).sum()) / sw_
        dx = x - xm
        sxx = float((w * dx * dx).sum())
        if sxx <= 1e-300:                     # weights collapsed onto one x: keep last fit
            if it == 0:
                a, b = 1.0, float(np.median(y - x))
            break
        a = float((w * dx * (y - ym)).sum()) / sxx
        b = ym - a * xm
        r = y - a * x - b
        w = 1.0 / np.maximum(np.abs(r), eps)
    a_raw, b_raw = float(a), float(b)
    clamped = not (a_min <= a_raw <= a_max)
    if clamped:
        a = float(np.clip(a_raw, a_min, a_max))
        b = float(np.median(y - a * x))
    return AffineFit(float(a), float(b), a_raw, b_raw, bool(clamped), n, True)


# ──────────────────────────────────────────────────────────────────────────────
# Masks and edge statistics
# ──────────────────────────────────────────────────────────────────────────────

def lowpass(L: np.ndarray, w: Optional[np.ndarray], sigma: float) -> np.ndarray:
    """Validity-normalised Gaussian blur G(L*w)/G(w) (reflect-101 border). Plain blur when w is None."""
    L = np.asarray(L, dtype=np.float64)
    if w is None:
        return cv2.GaussianBlur(L, (0, 0), sigma)
    wf = np.asarray(w, dtype=np.float64)
    num = cv2.GaussianBlur(L * wf, (0, 0), sigma)
    den = cv2.GaussianBlur(wf, (0, 0), sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-12), L)


def agreement_mask(L_U: np.ndarray, L_Ma: np.ndarray, valid_U: np.ndarray, sigma: float,
                   thr: float = LN125) -> np.ndarray:
    """keep = valid_U & |G(L_Ma) - G(L_U)| <= thr (plan 5.4 step 4)."""
    gU = lowpass(L_U, valid_U, sigma)
    gM = lowpass(L_Ma, valid_U, sigma)
    return np.asarray(valid_U, bool) & (np.abs(gM - gU) <= thr)


def pixel_agreement_mask(L_C: np.ndarray, L_U: np.ndarray, valid_U: np.ndarray,
                         thr: float = LN125) -> np.ndarray:
    """The rejected per-pixel test |L_C - L_U| <= thr (E8 baseline; diagnostics only)."""
    return np.asarray(valid_U, bool) & (np.abs(np.asarray(L_C) - np.asarray(L_U)) <= thr)


def strong_edge_pairs(L: np.ndarray, jump: float = LN110):
    """Boolean maps of horizontal (H,W-1) and vertical (H-1,W) neighbour pairs with |dL| > jump."""
    L = np.asarray(L, dtype=np.float64)
    return np.abs(np.diff(L, axis=1)) > jump, np.abs(np.diff(L, axis=0)) > jump


def edge_pair_survival(L_C: np.ndarray, keep: np.ndarray, valid: Optional[np.ndarray] = None,
                       jump: float = LN110) -> float:
    """Fraction of strong-edge pairs of L_C (both pixels valid) whose two pixels are both kept.

    NaN when the image has no strong-edge pair (excluded from means).
    """
    px, py = strong_edge_pairs(L_C, jump)
    k = np.asarray(keep, bool)
    if valid is not None:
        v = np.asarray(valid, bool)
        px &= v[:, :-1] & v[:, 1:]
        py &= v[:-1, :] & v[1:, :]
    tot = int(px.sum() + py.sum())
    if tot == 0:
        return float("nan")
    surv = int((px & k[:, :-1] & k[:, 1:]).sum() + (py & k[:-1, :] & k[1:, :]).sum())
    return surv / tot


def strong_edge_pixels(L: np.ndarray, jump: float = LN110) -> np.ndarray:
    px, py = strong_edge_pairs(L, jump)
    e = np.zeros(np.asarray(L).shape, bool)
    e[:, :-1] |= px
    e[:, 1:] |= px
    e[:-1, :] |= py
    e[1:, :] |= py
    return e


def s8_weight(d_teacher: np.ndarray, d_assist: np.ndarray, tau: float = S8_TAU) -> np.ndarray:
    """The S8 per-pixel weight exp(-|ln d_T - ln d_A|/tau) (1 where either is invalid), for comparison."""
    d_teacher = np.asarray(d_teacher, np.float64)
    d_assist = np.asarray(d_assist, np.float64)
    both = (d_teacher > 1e-3) & (d_assist > 1e-3)
    w = np.ones(d_teacher.shape, np.float64)
    w[both] = np.exp(-np.abs(np.log(d_teacher[both]) - np.log(d_assist[both])) / tau)
    return w


# ──────────────────────────────────────────────────────────────────────────────
# One-image fusion
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FuseResult:
    L_C: np.ndarray
    L_U: np.ndarray            # filled V2-L log depth (fill only where valid_U is False)
    L_Ma: np.ndarray           # aligned, filled MoGe log depth
    valid_U: np.ndarray
    valid_M: np.ndarray
    keep: np.ndarray           # low-pass agreement mask (subset of valid_U)
    fit: AffineFit
    status: str
    params: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)

    def depth_composite(self) -> np.ndarray:
        """Composite metres, 0 where V2-L is invalid (the L1a label)."""
        return np.where(self.valid_U, np.exp(self.L_C), 0.0).astype(np.float32)


def fuse_one(depth_u: np.ndarray, depth_m: np.ndarray, cfg: CompositeConfig = CompositeConfig(),
             valid_u: Optional[np.ndarray] = None, valid_m: Optional[np.ndarray] = None,
             diagnostics: bool = True) -> FuseResult:
    """Fuse one V2-L depth map with one MoGe depth map (metres, same H x W).

    valid_u defaults to dmin <= d <= dmax; valid_m defaults to finite & > 1e-3. When fewer than
    100 pixels or less than cfg.min_overlap of the image are valid for both teachers, the
    result is V2-L unchanged with status "v2l_only_low_overlap" (keep = valid_U).
    """
    dU = np.asarray(depth_u, dtype=np.float64)
    dM = np.asarray(depth_m, dtype=np.float64)
    if dU.shape != dM.shape or dU.ndim != 2:
        raise ValueError(f"teacher maps must share one HxW grid, got {dU.shape} and {dM.shape}")
    h, w = dU.shape
    finU = np.isfinite(dU)
    rngU = finU & (dU >= cfg.dmin_m) & (dU <= cfg.dmax_m)
    vU = rngU if valid_u is None else (np.asarray(valid_u, bool) & rngU)
    finM = np.isfinite(dM) & (dM > 1e-3)
    vM = finM if valid_m is None else (np.asarray(valid_m, bool) & finM)
    cross, sigma, lam = cfg.effective(h, w)
    LU = np.log(np.clip(np.where(finU & (dU > 0), dU, cfg.dmin_m), cfg.dmin_m, cfg.dmax_m))
    if not vU.all() and vU.any():
        LU = np.where(vU, LU, lowpass(LU, vU, sigma))
    LM = np.where(vM, np.log(np.where(vM, dM, 1.0)), LU)
    ov = vU & vM
    params = {"crossover_px": cross, "sigma_px": sigma, "lam": lam, "h": h, "w": w}
    if ov.sum() < 100 or ov.mean() < cfg.min_overlap:
        fit = AffineFit(1.0, 0.0, float("nan"), float("nan"), False, int(ov.sum()), False)
        L_C, L_Ma, keep, status = LU.copy(), LU.copy(), vU.copy(), "v2l_only_low_overlap"
    else:
        fit = robust_affine_fit(LM[ov], LU[ov], cfg.a_min, cfg.a_max, cfg.irls_iters, cfg.irls_eps,
                                cfg.irls_max_points)
        L_Ma = np.where(vM, fit.a * LM + fit.b, LU)
        L_C = dct_composite(LU, L_Ma, lam)
        keep = agreement_mask(LU, L_Ma, vU, sigma, cfg.agree_thr)
        status = "ok"
    dC = np.exp(L_C)
    stats: Dict[str, float] = {
        "overlap_frac": float(ov.mean()), "valid_u_frac": float(vU.mean()), "valid_m_frac": float(vM.mean()),
        "masked_frac": float(1.0 - keep[vU].mean()) if vU.any() else float("nan"),
        "composite_out_of_range_frac": float(((dC < cfg.dmin_m) | (dC > cfg.dmax_m))[vU].mean())
        if vU.any() else float("nan"),
    }
    if diagnostics:
        pix = pixel_agreement_mask(L_C, LU, vU, cfg.agree_thr)
        stats.update(
            edge_pair_survival=edge_pair_survival(L_C, keep, vU, cfg.edge_jump),
            edge_pair_survival_pixel_test=edge_pair_survival(L_C, pix, vU, cfg.edge_jump),
            pixel_test_masked_frac=float(1.0 - pix[vU].mean()) if vU.any() else float("nan"),
            mean_abs_dlog_c_u=float(np.abs(L_C - LU)[vU].mean()) if vU.any() else float("nan"),
            mean_log_diff_c_u=float(L_C.mean() - LU.mean()),
        )
    return FuseResult(L_C, LU, L_Ma, vU, vM, keep, fit, status, params, stats)


def resize_to(depth_m: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize a 0-is-invalid depth map: bilinear on log depth of valid pixels, nearest validity."""
    h, w = hw
    if depth_m.shape[:2] == (h, w):
        return depth_m
    v = depth_m > 0
    med = float(np.median(depth_m[v])) if v.any() else 1.0
    L = np.log(np.where(v, depth_m, med)).astype(np.float32)
    Lr = cv2.resize(L, (w, h), interpolation=cv2.INTER_LINEAR)
    vr = cv2.resize(v.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return np.where(vr, np.exp(Lr), 0.0).astype(np.float32)


def load_composite_label(root: str, item_id: str, variant: str = "l1a") -> Tuple[np.ndarray, np.ndarray]:
    """(depth metres float32, valid bool) of a fused label. variant: "l1a" (no mask) or "l1b" (mask)."""
    if variant not in LABEL_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(LABEL_VARIANTS)}, got {variant!r}")
    d = read_label_png(os.path.join(root, "depth_mm", item_id + ".png"))
    valid = d > 0
    if variant == "l1b":
        m = cv2.imread(os.path.join(root, "agree_mask", item_id + ".png"), cv2.IMREAD_UNCHANGED)
        if m is None or m.shape != d.shape:
            raise FileNotFoundError(f"agreement mask missing or mis-sized for {item_id} in {root}")
        valid &= m >= 128
    return d, valid


# ──────────────────────────────────────────────────────────────────────────────
# Label directories (V2 layout or legacy flat cache)
# ──────────────────────────────────────────────────────────────────────────────

class LabelDir:
    """A teacher label directory: V2 layout (depth_mm/<id>.png + meta/<id>.json, done = sidecar ok)
    or the legacy flat cache (<id>.png only, e.g. outputs/depth_mm; every PNG counts as done)."""

    def __init__(self, root: str):
        self.root = root
        if os.path.isdir(os.path.join(root, "depth_mm")) or os.path.isdir(os.path.join(root, "meta")):
            self.layout = "v2"
        elif os.path.isdir(root):
            self.layout = "flat"
        else:
            check_drives([root])
            raise FileNotFoundError(f"label dir {root} does not exist")

    def ids(self) -> Set[str]:
        if self.layout == "v2":
            return done_ids(self.root)
        return {f[:-4] for f in os.listdir(self.root) if f.lower().endswith(".png")}

    def png(self, item_id: str) -> str:
        sub = "depth_mm" if self.layout == "v2" else ""
        return os.path.join(self.root, sub, item_id + ".png")

    def meta(self, item_id: str) -> Dict:
        if self.layout == "flat":
            return {"teacher": {"checkpoint": LEGACY_V2L_CHECKPOINT if os.path.normcase(
                os.path.abspath(self.root)) == os.path.normcase(os.path.abspath(LEGACY_V2L_DIR)) else None,
                "layout": "legacy flat cache (no sidecar)"},
                "png_sha256": None}
        p = os.path.join(self.root, "meta", item_id + ".json")
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}


# ──────────────────────────────────────────────────────────────────────────────
# Roles helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> Dict:
    import yaml
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=loader) or {}


def forbidden_training_ids(roles_path: str) -> Tuple[Optional[Set[str]], Optional[str]]:
    """Ids that must never get a training label (dev.d_sun, test.frozen_own, test.video_own).

    Returns (None, None) when the roles file does not exist yet.
    """
    if not os.path.exists(roles_path):
        return None, None
    roles = _load_yaml(roles_path)
    out: Set[str] = set()
    for a, b in FORBIDDEN_ROLE_KEYS:
        v = (roles.get(a) or {}).get(b) or []
        out |= {str(x) for x in v}
    return out, sha256_file(roles_path)


def _norm_key(k) -> str:
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _read_id_file(path: str) -> List[str]:
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return [str(x) for x in (obj if isinstance(obj, list) else obj.get("ids", []))]
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return [r["id"] for r in csv.DictReader(f)]
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def load_dsun(roles_path: str) -> Dict:
    """D_SUN ids and clusters from roles.yaml. Fails loudly if absent.

    Reads ``dev.d_sun`` (list of ids) and ``dev.d_sun_clusters`` (list of id lists; unlisted ids
    are singletons), as written by scripts/v2/build_roles.py. Returns {"ids", "clusters"
    (id -> cluster label), "key", "roles_sha256", "n_multi_clusters"}.
    """
    if not os.path.exists(roles_path):
        raise SystemExit(f"[L0] {roles_path} not found: D_SUN is defined only by the frozen roles file "
                         "(python -m scripts.v2.build_roles). Run that first, or pass --provisional-sun for "
                         "a flagged, unregistrable smoke run on raw local SUN rows.")
    roles = _load_yaml(roles_path)
    dev = roles.get("dev") or {}
    ids = dev.get("d_sun")
    if not ids:
        raise SystemExit(f"[L0] {roles_path} has no dev.d_sun id list (top-level keys {list(roles)[:20]})")
    ids = [str(x) for x in ids]
    clusters = {i: i for i in ids}
    multi = dev.get("d_sun_clusters") or []
    for members in multi:
        members = [str(m) for m in members]
        lab = "c:" + min(members)
        for m in members:
            if m in clusters:
                clusters[m] = lab
    return {"ids": ids, "clusters": clusters, "key": "dev.d_sun", "roles_sha256": sha256_file(roles_path),
            "n_multi_clusters": len(multi)}


def build_l0_list(ids: Sequence[str], manifest_path: str, n: int = L0_N, seed: int = L0_SEED) -> List[Dict]:
    """Seeded sample of n ids (sorted pool, numpy default_rng(seed)) joined with manifest rows."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        man = {r["id"]: r for r in csv.DictReader(f)}
    missing = [i for i in ids if i not in man]
    if missing:
        raise SystemExit(f"[L0] {len(missing)} D_SUN ids not in {manifest_path}, e.g. {missing[:5]}")
    pool = sorted(set(ids))
    if n > len(pool):
        raise SystemExit(f"[L0] asked for {n} frames but D_SUN has {len(pool)}")
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(pool), size=n, replace=False).tolist())
    rows = []
    for j in pick:
        r = man[pool[j]]
        if str(r.get("is_gt", "0")) not in ("1", "True", "true"):
            raise SystemExit(f"[L0] row {r['id']} has no sensor GT (is_gt={r.get('is_gt')})")
        rows.append({"id": r["id"], "img": os.path.normpath(r["img"]), "gt_depth": os.path.normpath(r["depth"]),
                     "dmin": r.get("dmin") or "0.2", "dmax": r.get("dmax") or "8.0", "source": r["source"]})
    return rows


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Batch fusion of training labels
# ──────────────────────────────────────────────────────────────────────────────

def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return None if not math.isfinite(xf) else round(xf, 6)
    return x


def _nanmean(v) -> float:
    v = np.asarray([np.nan if x is None else x for x in v], np.float64)
    return float(np.nanmean(v)) if v.size and np.isfinite(v).any() else float("nan")


def fuse_dir(v2l_dir: str, moge_dir: str, out_dir: str, cfg: CompositeConfig = CompositeConfig(),
             ids: Optional[Sequence[str]] = None, overwrite: bool = False, limit: int = 0,
             roles_path: Optional[str] = DEFAULT_ROLES, log: Callable[[str], None] = print) -> Dict:
    """Fuse every id labelled by both teachers into training composite labels; resumable.

    Refuses (SystemExit) ids that roles.yaml assigns to dev/test (leak guard); without a roles
    file the run proceeds and records roles_checked = False.
    """
    check_drives([v2l_dir, moge_dir, out_dir])
    src_u, src_m = LabelDir(v2l_dir), LabelDir(moge_dir)
    have_u, have_m = src_u.ids(), src_m.ids()
    want = sorted(have_m) if ids is None else list(ids)
    forbidden, roles_sha = forbidden_training_ids(roles_path) if roles_path else (None, None)
    if forbidden is not None:
        bad = sorted(set(want) & forbidden)
        if bad:
            raise SystemExit(f"[fuse] {len(bad)} ids are dev/test frames in {roles_path} (e.g. {bad[:5]}); "
                             "training composite labels are never made for them. L0 uses `validate`, which "
                             "fuses in memory.")
    else:
        log(f"[fuse] WARNING: {roles_path} not found; dev/test leak guard not applied (recorded)")
    both = [i for i in want if i in have_u and i in have_m]
    missing = {"v2l": sorted(set(want) - have_u)[:50], "moge": sorted(set(want) - have_m)[:50],
               "n_missing_v2l": len(set(want) - have_u), "n_missing_moge": len(set(want) - have_m)}
    if limit:
        both = both[:limit]
    for sub in ("depth_mm", "agree_mask", "meta"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    done = done_ids(out_dir)
    todo = [i for i in both if overwrite or i not in done]
    log(f"[fuse] {len(both)} ids with both labels ({src_u.layout} V2-L dir), {len(both) - len(todo)} done, "
        f"{len(todo)} to fuse; missing v2l={missing['n_missing_v2l']} moge={missing['n_missing_moge']}")
    done_csv = os.path.join(out_dir, "done.csv")
    new = not os.path.exists(done_csv)
    agg = {"masked_frac": [], "edge_pair_survival": [], "a": [], "clamped": [], "t": []}
    n_fallback = 0
    script_sha = sha256_file(__file__)[:16]
    with open(done_csv, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["id", "depth_png", "mask_png", "meta_json", "status", "utc"])
        for k, i in enumerate(todo):
            t0 = time.time()
            pu, pm = src_u.png(i), src_m.png(i)
            dU = read_label_png(pu)
            dM = read_label_png(pm)
            resized = dM.shape != dU.shape
            if resized:
                dM = resize_to(dM, dU.shape)
            r = fuse_one(dU, dM, cfg, valid_u=dU > 0, valid_m=dM > 0)
            dpng = os.path.join(out_dir, "depth_mm", i + ".png")
            mpng = os.path.join(out_dir, "agree_mask", i + ".png")
            d_sha = atomic_write_png(dpng, depth_m_to_png16(r.depth_composite(), clip_m=60.0))
            m_sha = atomic_write_png(mpng, np.where(r.keep, 255, 0).astype(np.uint8))
            mu, mm = src_u.meta(i), src_m.meta(i)
            meta = _json_safe({
                "schema": "v2-composite-label/1", "status": "ok", "fuse_status": r.status, "id": i,
                "depth_png": dpng, "depth_png_sha256": d_sha, "mask_png": mpng, "mask_png_sha256": m_sha,
                "inputs": {"v2l_png": pu, "moge_png": pm, "v2l_layout": src_u.layout,
                           "v2l_png_sha256": mu.get("png_sha256") or sha256_file(pu),
                           "moge_png_sha256": mm.get("png_sha256") or sha256_file(pm),
                           "v2l_teacher": (mu.get("teacher") or {}).get("checkpoint"),
                           "moge_teacher": (mm.get("teacher") or {}).get("checkpoint"),
                           "image_sha1": mm.get("image_sha1") or mu.get("image_sha1"),
                           "moge_resized_to_v2l": resized},
                "fit": asdict(r.fit), "params": r.params, "stats": r.stats, "config": asdict(cfg),
                "variants": LABEL_VARIANTS, "roles_checked": forbidden is not None, "roles_sha256": roles_sha,
                "script_sha256_16": script_sha, "utc": _utc(),
            })
            mp = os.path.join(out_dir, "meta", i + ".json")
            atomic_write_json(mp, meta)
            wr.writerow([i, dpng, mpng, mp, r.status, meta["utc"]])
            f.flush()
            n_fallback += int(r.status != "ok")
            agg["masked_frac"].append(r.stats["masked_frac"])
            agg["edge_pair_survival"].append(r.stats.get("edge_pair_survival", float("nan")))
            agg["a"].append(r.fit.a)
            agg["clamped"].append(float(r.fit.clamped))
            agg["t"].append(time.time() - t0)
            if k < 3 or (k + 1) % 200 == 0:
                log(f"[fuse] {k + 1}/{len(todo)} {i} a={r.fit.a:.3f} masked={r.stats['masked_frac']:.4f} "
                    f"edge_surv={r.stats.get('edge_pair_survival', float('nan')):.3f} {time.time() - t0:.2f}s")

    summary = _json_safe({
        "schema": "v2-composite-run/1", "utc": _utc(), "out_dir": out_dir, "v2l_dir": v2l_dir,
        "v2l_layout": src_u.layout, "moge_dir": moge_dir, "n_fused": len(todo),
        "n_already_done": len(both) - len(todo), "n_v2l_only_fallback": n_fallback, "missing": missing,
        "mean_masked_frac": _nanmean(agg["masked_frac"]),
        "mean_edge_pair_survival": _nanmean(agg["edge_pair_survival"]),
        "median_a": float(np.median(agg["a"])) if agg["a"] else float("nan"),
        "clamped_frac": _nanmean(agg["clamped"]), "s_per_image": _nanmean(agg["t"]), "config": asdict(cfg),
        "roles_checked": forbidden is not None, "roles_sha256": roles_sha, "script_sha256_16": script_sha,
    })
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write_json(os.path.join(out_dir, f"run_{stamp}.json"), summary)
    log(f"[fuse] done {len(todo)}; mean masked {summary['mean_masked_frac']}, "
        f"edge-pair survival {summary['mean_edge_pair_survival']}")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# L0: label validation against sensor GT
# ──────────────────────────────────────────────────────────────────────────────

LEGACY_BF_NOTE = ("No corrected boundary F1 importable; used the LEGACY protocol.boundary_f1_si, which drops "
                  "thresholds with no GT or no predicted edges instead of scoring them 0 and returns None when "
                  "every threshold is skipped (those images are excluded from the BF mean, counted in bf_n). "
                  "Re-run L0 once plan task 1 lands; the BF gate is provisional.")


def resolve_boundary_f1(name: Optional[str] = None):
    """Return (fn, qualified name, corrected, note). Prefers a corrected BF (BF_CANDIDATES)."""
    import importlib
    from benchmark import protocol
    if name:
        mod, _, fn_name = name.rpartition(".") if "." in name else ("benchmark.protocol", "", name)
        fn = getattr(importlib.import_module(mod), fn_name, None)
        if fn is None:
            raise SystemExit(f"{mod} has no function {fn_name!r}")
        corrected = fn_name != "boundary_f1_si"
        return fn, f"{mod}.{fn_name}", corrected, ("" if corrected else LEGACY_BF_NOTE)
    for mod, fn_name in BF_CANDIDATES:
        try:
            fn = getattr(importlib.import_module(mod), fn_name, None)
        except ImportError:
            fn = None
        if callable(fn):
            return fn, f"{mod}.{fn_name}", True, ""
    return protocol.boundary_f1_si, "benchmark.protocol.boundary_f1_si", False, LEGACY_BF_NOTE


def _metrics(gt, pred, valid, bf_fn, protocol, legacy_bf: bool = False) -> Dict[str, float]:
    m = protocol.per_image_depth_metrics(gt, pred, valid)
    if m is None:
        return {}
    keep = ("d1", "d1_ssi", "d1_si", "arel", "arel_ssi", "rmse", "silog")
    out = {k: m[k] for k in keep}
    bf = bf_fn(gt, pred, valid)
    out["bf"] = float("nan") if bf is None else float(bf)
    if legacy_bf:
        lb = protocol.boundary_f1_si(gt, pred, valid)
        out["bf_legacy"] = float("nan") if lb is None else float(lb)
    out["n_px"] = float(np.asarray(valid, bool).sum())
    return out


def _outlier_counts(gt, pred, mask) -> Tuple[int, int, int]:
    """(n pixels, n raw outliers max(g/p,p/g) >= 1.25, n outliers after per-image median scaling)."""
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return 0, 0, 0
    g = gt[m].astype(np.float64)
    p = np.clip(pred[m].astype(np.float64), 1e-6, None)
    r = np.maximum(g / p, p / g)
    s = np.median(g) / np.median(p)
    rs = np.maximum(g / (p * s), (p * s) / g)
    return n, int((r >= 1.25).sum()), int((rs >= 1.25).sum())


def _safe_bootstrap(a, b, clusters, n_boot) -> Dict:
    from benchmark.protocol_v2 import paired_bootstrap_clustered
    try:
        return paired_bootstrap_clustered(a, b, clusters=clusters, n_boot=n_boot, seed=0)
    except ValueError as e:
        return {"error": str(e)}


def run_l0(list_csv: str, v2l_dir: str, moge_dir: str, cfg: CompositeConfig = CompositeConfig(),
           bf_name: Optional[str] = None, limit: int = 0, n_boot: int = 10000,
           clusters: Optional[Dict[str, str]] = None, log: Callable[[str], None] = print) -> Dict:
    """Score V2-L and the composite labels (L1a, L1b) against sensor GT on the frames in list_csv.

    list_csv columns: id, gt_depth (uint16 mm PNG), dmin, dmax (from ``l0-list``). Every frame
    must have both teacher labels (no silent subset). clusters: id -> cluster label for the
    paired bootstrap (None = image-level resampling).
    """
    from benchmark import protocol
    bf_fn, bf_used, bf_corrected, bf_note = resolve_boundary_f1(bf_name)
    with open(list_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"[L0] empty list {list_csv}")
    check_drives([r["gt_depth"] for r in rows] + [v2l_dir, moge_dir])
    src_u, src_m = LabelDir(v2l_dir), LabelDir(moge_dir)
    have_u, have_m = src_u.ids(), src_m.ids()
    miss = [r["id"] for r in rows if r["id"] not in have_u or r["id"] not in have_m]
    if miss:
        raise SystemExit(f"[L0] {len(miss)} of {len(rows)} frames lack a V2-L or MoGe label, e.g. {miss[:5]}. "
                         "Label them first (label_unidepth / label_moge with --input on the L0 list); "
                         "L0 never scores a silent subset.")
    teachers = {"v2l": set(), "moge": set()}
    per_image, a_vals = [], []
    pix = {k: np.zeros(3, np.int64) for k in ("comp@kept", "comp@masked", "v2l@kept", "v2l@masked")}
    t0 = time.time()
    for k, r in enumerate(rows):
        i = r["id"]
        gt = read_label_png(r["gt_depth"])
        dmin, dmax = float(r.get("dmin") or 0.2), float(r.get("dmax") or 8.0)
        vg = (gt >= dmin) & (gt <= dmax)
        dU = resize_to(read_label_png(src_u.png(i)), gt.shape)
        dM = resize_to(read_label_png(src_m.png(i)), gt.shape)
        teachers["v2l"].add((src_u.meta(i).get("teacher") or {}).get("checkpoint"))
        teachers["moge"].add((src_m.meta(i).get("teacher") or {}).get("checkpoint"))
        res = fuse_one(dU, dM, cfg, valid_u=dU > 0, valid_m=dM > 0)
        dC = np.exp(res.L_C)
        keep = res.keep
        ev = vg & res.valid_U                     # pixels both labels define (paired)
        vk, vm = ev & keep, ev & ~keep
        row = {"id": i, "cluster": (clusters or {}).get(i, i), "fuse_status": res.status,
               "a": res.fit.a, "b": res.fit.b, "a_clamped": res.fit.clamped,
               "gt_valid_frac": float(vg.mean()),
               "gt_valid_excluded_v2l_invalid_frac": float((vg & ~res.valid_U).sum() / max(vg.sum(), 1)),
               "l1b_coverage_of_eval": float(vk.sum() / max(ev.sum(), 1)),
               **{k2: res.stats[k2] for k2 in ("masked_frac", "edge_pair_survival", "edge_pair_survival_pixel_test",
                                               "pixel_test_masked_frac", "composite_out_of_range_frac")}}
        row["v2l"] = _metrics(gt, dU, ev, bf_fn, protocol, legacy_bf=True)
        row["composite"] = _metrics(gt, dC, ev, bf_fn, protocol, legacy_bf=True)
        row["moge_aligned"] = _metrics(gt, np.exp(res.L_Ma), ev, bf_fn, protocol)
        row["moge_raw@moge_valid"] = _metrics(gt, np.where(dM > 0, dM, 1.0), vg & (dM > 0), bf_fn, protocol)
        row["v2l@keep"] = _metrics(gt, dU, vk, bf_fn, protocol)
        row["composite@keep"] = _metrics(gt, dC, vk, bf_fn, protocol)
        for key, pred, msk in (("comp@kept", dC, vk), ("comp@masked", dC, vm),
                               ("v2l@kept", dU, vk), ("v2l@masked", dU, vm)):
            pix[key] += np.asarray(_outlier_counts(gt, pred, msk), np.int64)
        w8 = s8_weight(dU, dM)
        edges = strong_edge_pixels(res.L_C, cfg.edge_jump) & res.valid_U
        vu = res.valid_U
        row["s8_mean_weight_all"] = float(w8[vu].mean()) if vu.any() else None
        row["s8_mean_weight_edges"] = float(w8[edges].mean()) if edges.any() else None
        row["mask_keep_frac_edges"] = float(keep[edges].mean()) if edges.any() else None
        row["mask_keep_frac_all"] = float(keep[vu].mean()) if vu.any() else None
        per_image.append(row)
        a_vals.append(res.fit.a)
        if k < 2 or (k + 1) % 50 == 0:
            log(f"[L0] {k + 1}/{len(rows)} {i} d1 v2l={row['v2l'].get('d1', float('nan')):.3f} "
                f"comp={row['composite'].get('d1', float('nan')):.3f} a={res.fit.a:.3f} "
                f"edge_surv={res.stats['edge_pair_survival']:.3f} ({time.time() - t0:.1f}s)")

    variants = ("v2l", "composite", "moge_aligned", "moge_raw@moge_valid", "v2l@keep", "composite@keep")
    agg = {}
    for v in variants:
        rs = [r[v] for r in per_image if r[v]]
        a = protocol.aggregate(rs) if rs else {}
        bfs = np.asarray([x.get("bf", np.nan) for x in rs], np.float64)
        a["bf_n"] = int(np.isfinite(bfs).sum())
        a["n"] = len(rs)
        agg[v] = a

    def col(v, m):
        return [r[v].get(m, np.nan) if r[v] else np.nan for r in per_image]

    cl = [r["cluster"] for r in per_image]
    paired = {}
    for base, cand in (("v2l", "composite"), ("v2l@keep", "composite@keep")):
        for m in ("d1", "d1_ssi", "bf", "arel"):
            paired[f"{cand}-{base}:{m}"] = _safe_bootstrap(col(base, m), col(cand, m),
                                                           cl if clusters else None, n_boot)
    paired["composite-v2l:bf_legacy"] = _safe_bootstrap(col("v2l", "bf_legacy"), col("composite", "bf_legacy"),
                                                        cl if clusters else None, n_boot)

    def paired_mean(base, cand, m):
        a_, b_ = np.asarray(col(base, m), np.float64), np.asarray(col(cand, m), np.float64)
        ok = np.isfinite(a_) & np.isfinite(b_)
        return (float(a_[ok].mean()), float(b_[ok].mean()), int(ok.sum())) if ok.any() else \
            (float("nan"), float("nan"), 0)

    def gates_for(base, cand):
        d1b, d1c = 100 * agg[base].get("d1", np.nan), 100 * agg[cand].get("d1", np.nan)
        sb, sc = 100 * agg[base].get("d1_ssi", np.nan), 100 * agg[cand].get("d1_ssi", np.nan)
        bfb, bfc, bfn = paired_mean(base, cand, "bf")
        return {
            "G1_d1": {"v2l": d1b, "composite": d1c, "rule": f"composite >= v2l - {L0_GATES['d1_drop_max_pts']} pts",
                      "pass": bool(d1c >= d1b - L0_GATES["d1_drop_max_pts"])},
            "G2_d1_ssi": {"v2l": sb, "composite": sc,
                          "rule": f"composite >= v2l - {L0_GATES['d1_ssi_drop_max_pts']} pts",
                          "pass": bool(sc >= sb - L0_GATES["d1_ssi_drop_max_pts"])},
            "G3_bf": {"v2l": bfb, "composite": bfc, "n_images": bfn,
                      "rule": f"composite >= v2l + {L0_GATES['bf_gain_min']} (images where BF is defined)",
                      "bf_function": bf_used, "bf_corrected": bf_corrected,
                      "pass": bool(bfc >= bfb + L0_GATES["bf_gain_min"])},
        }

    eps_mean = _nanmean([r["edge_pair_survival"] for r in per_image])
    gates_label = gates_for("v2l", "composite")
    g4 = {"value": eps_mean, "rule": f">= {L0_GATES['edge_pair_survival_min']}",
          "n_images": int(np.isfinite(np.asarray([r["edge_pair_survival"] for r in per_image], float)).sum()),
          "pass": bool(eps_mean >= L0_GATES["edge_pair_survival_min"])}
    pass_label = all(g["pass"] for g in gates_label.values())
    verdict = {
        "l0_pass": bool(pass_label and g4["pass"]),
        "label_gates_pass": bool(pass_label), "mask_gate_pass": bool(g4["pass"]),
        "l1a_eligible": bool(pass_label), "l1b_eligible": bool(pass_label and g4["pass"]),
        "note": "plan L0 = G1..G4 all; A8 splits L1 into L1a (no mask, needs G1..G3) and L1b (mask, also G4)",
    }

    def rate(c):
        n, o_raw, o_si = (int(x) for x in c)
        return {"n_px": n, "outlier_raw": (o_raw / n) if n else None, "outlier_median_scaled": (o_si / n) if n else None}

    a8 = {
        "l1a": {"label": LABEL_VARIANTS["l1a"], "d1": agg["composite"].get("d1"),
                "d1_ssi": agg["composite"].get("d1_ssi"), "bf": agg["composite"].get("bf"),
                "coverage_of_eval_px": 1.0},
        "l1b": {"label": LABEL_VARIANTS["l1b"], "d1": agg["composite@keep"].get("d1"),
                "d1_ssi": agg["composite@keep"].get("d1_ssi"), "bf": agg["composite@keep"].get("bf"),
                "coverage_of_eval_px": _nanmean([r["l1b_coverage_of_eval"] for r in per_image])},
        "l1b_label_gates_descriptive": gates_for("v2l@keep", "composite@keep"),
        "pixels_pooled": {k: rate(v) for k, v in pix.items()},
        "reading": ("pixels_pooled compares GT outlier rates (max ratio >= 1.25) of the pixels the mask keeps "
                    "vs removes; the mask only helps if removed pixels are clearly worse than kept ones"),
        "decision": "not made here: ladder runs L1a and L1b; keep the mask only if L1b clears the margin alone",
    }
    roles_note = None if clusters else "image-level bootstrap (no roles.yaml clusters): D_SUN near-duplicates not merged"
    result = _json_safe({
        "schema": "v2-l0/2", "run_id": "L0", "utc": _utc(), "n_images": len(per_image),
        "ids_sha256": ids_sha256([r["id"] for r in per_image]), "list_csv": list_csv,
        "list_sha256": sha256_file(list_csv), "v2l_dir": v2l_dir, "v2l_layout": src_u.layout, "moge_dir": moge_dir,
        "teachers": {k: sorted(x for x in v if x) for k, v in teachers.items()},
        "camera_mode": "none", "config": asdict(cfg), "gates_def": L0_GATES,
        "eval_pixels": "GT-valid (manifest dmin..dmax) & V2-L-valid; identical for both labels",
        "bf_function": bf_used, "bf_corrected": bf_corrected, "bf_note": bf_note,
        "bootstrap": {"resampling": "cluster" if clusters else "image", "n_boot": n_boot, "note": roles_note},
        "aggregate": agg, "paired_bootstrap": paired,
        "gates": {**gates_label, "G4_edge_pair_survival": g4},
        "a8": a8, "verdict": verdict,
        "mask_stats": {"mean_masked_frac": _nanmean([r["masked_frac"] for r in per_image]),
                       "mean_edge_pair_survival": eps_mean,
                       "mean_edge_pair_survival_pixel_test": _nanmean(
                           [r["edge_pair_survival_pixel_test"] for r in per_image]),
                       "mean_pixel_test_masked_frac": _nanmean([r["pixel_test_masked_frac"] for r in per_image]),
                       "median_a": float(np.median(a_vals)),
                       "a_clamped_frac": float(np.mean([r["a_clamped"] for r in per_image])),
                       "n_v2l_only_fallback": int(sum(r["fuse_status"] != "ok" for r in per_image))},
        "mask_vs_s8": {"s8_mean_weight_all": _nanmean([r["s8_mean_weight_all"] for r in per_image]),
                       "s8_mean_weight_edges": _nanmean([r["s8_mean_weight_edges"] for r in per_image]),
                       "mask_keep_frac_all": _nanmean([r["mask_keep_frac_all"] for r in per_image]),
                       "mask_keep_frac_edges": _nanmean([r["mask_keep_frac_edges"] for r in per_image]),
                       "explanation": "S8 = continuous exp(-|dlog|/0.2) on raw unaligned teachers per pixel; "
                                      "L1b = binary keep on sigma-16 low-passed, affine-aligned log depth"},
        "elapsed_s": round(time.time() - t0, 1), "script_sha256_16": sha256_file(__file__)[:16],
        "per_image": per_image,
    })
    return result


def register_l0(result: Dict, result_path: str, supersede: bool = False) -> Dict:
    """Append L0 to benchmark/registry_v2.jsonl as a ``root`` record (no parent, no checkpoint)."""
    try:
        from benchmark.provenance import code_tree_hash, config_hash, env_stamp, repo_relpath
        from benchmark.registry_v2 import append_record, latest_by_id, load_records
    except ImportError as e:
        raise SystemExit(f"[L0] --register given but the registry is unavailable ({e})")
    here = os.path.dirname(os.path.abspath(__file__))
    rec = {"type": "root", "run_id": "L0", "kind": "label-validation", "parent": None,
           "factor_changed": None, "dataset": "D_SUN", "split": "dev", "n_images": result["n_images"],
           "ids_sha256": result["ids_sha256"], "list_sha256": result["list_sha256"],
           "teachers": sorted({t for v in result["teachers"].values() for t in v}),
           "camera_mode": "none", "gates": result["gates"], "verdict": result["verdict"],
           "status": "accepted" if result["verdict"]["l0_pass"] else "rejected",
           "checkpoint_kind": None, "selection": "none",
           "results": [{"dataset": "D_SUN", "result_file": repo_relpath(result_path),
                        "result_sha256": sha256_file(result_path)}],
           "bf_function": result["bf_function"], "bf_corrected": result["bf_corrected"],
           "config_hash": config_hash(result["config"]),
           "code_tree_hash": code_tree_hash([os.path.join(here, n) for n in
                                             ("fuse_composite.py", "label_unidepth.py", "label_moge.py")]),
           "env": env_stamp(gpu=False)}
    if supersede:
        latest = latest_by_id(load_records())
        if "L0" in latest:
            rec["supersedes"] = latest["L0"]["record_hash"]
    return append_record(rec)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cfg_from_args(a) -> CompositeConfig:
    return CompositeConfig(crossover_px=a.crossover_px, sigma_px=a.sigma_px,
                           scale_with_resolution=not a.no_scale_with_resolution)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--validate" in argv:
        argv.remove("--validate")
        argv.insert(0, "validate")
    ap = argparse.ArgumentParser(description="V2 composite labels, L0 validation (plan 5.4, A8)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_cfg(p):
        p.add_argument("--crossover-px", type=float, default=48.0)
        p.add_argument("--sigma-px", type=float, default=16.0)
        p.add_argument("--no-scale-with-resolution", action="store_true")

    pf = sub.add_parser("fuse", help="fuse V2-L + MoGe labels into training composite labels")
    pf.add_argument("--v2l-dir", default=LEGACY_V2L_DIR,
                    help="V2 label dir or the legacy flat cache (default outputs/depth_mm)")
    pf.add_argument("--moge-dir", default=os.path.join(REPO_ROOT, "outputs", "v2", "labels", "moge"))
    pf.add_argument("--out", default=DEFAULT_OUT)
    pf.add_argument("--roles", default=DEFAULT_ROLES, help="roles.yaml for the dev/test leak guard")
    pf.add_argument("--ids-file", default=None, help="txt/csv/json list restricting the ids")
    pf.add_argument("--limit", type=int, default=0)
    pf.add_argument("--overwrite", action="store_true")
    add_cfg(pf)

    pl = sub.add_parser("l0-list", help="write the seeded L0 frame list from the D_SUN ids")
    pl.add_argument("--roles", default=DEFAULT_ROLES)
    pl.add_argument("--manifest", default=DEFAULT_MANIFEST)
    pl.add_argument("--n", type=int, default=L0_N)
    pl.add_argument("--seed", type=int, default=L0_SEED)
    pl.add_argument("--out", default=None)
    pl.add_argument("--provisional-sun", action="store_true",
                    help="roles.yaml missing: sample raw local SUN rows (flagged, never registrable)")

    pv = sub.add_parser("validate", help="L0 label validation against sensor GT")
    pv.add_argument("--list", required=True, help="CSV from l0-list")
    pv.add_argument("--v2l-dir", required=True)
    pv.add_argument("--moge-dir", required=True)
    pv.add_argument("--roles", default=DEFAULT_ROLES, help="for the D_SUN cluster bootstrap")
    pv.add_argument("--out", default=None)
    pv.add_argument("--limit", type=int, default=0, help="smoke runs only; recorded in the filename")
    pv.add_argument("--bf-fn", default=None, help="force a boundary-F1 function, e.g. benchmark.protocol.boundary_f1_si")
    pv.add_argument("--n-boot", type=int, default=10000)
    pv.add_argument("--register", action="store_true", help="append the L0 root record to registry_v2")
    pv.add_argument("--supersede", action="store_true", help="with --register: replace an existing L0 record")
    add_cfg(pv)
    a = ap.parse_args(argv)

    if a.cmd == "fuse":
        ids = _read_id_file(a.ids_file) if a.ids_file else None
        fuse_dir(a.v2l_dir, a.moge_dir, a.out, _cfg_from_args(a), ids=ids, overwrite=a.overwrite, limit=a.limit,
                 roles_path=a.roles)
        return 0

    if a.cmd == "l0-list":
        if os.path.exists(a.roles):
            ds = load_dsun(a.roles)
            ids, src, roles_sha, provisional = ds["ids"], f"{a.roles}:{ds['key']}", ds["roles_sha256"], False
        elif a.provisional_sun:
            with open(a.manifest, newline="", encoding="utf-8") as f:
                ids = [r["id"] for r in csv.DictReader(f) if r["source"] == "sun"]
            src, roles_sha, provisional = ("PROVISIONAL: raw local SUN rows (not D_SUN; NYU-derived and 8x-depth "
                                           "rows not removed)"), None, True
        else:
            load_dsun(a.roles)  # raises the explanatory SystemExit
            return 1
        rows = build_l0_list(ids, a.manifest, a.n, a.seed)
        out = a.out or os.path.join(DEFAULT_L0_DIR, f"l0_list_n{a.n}{'_provisional' if provisional else ''}.csv")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "img", "gt_depth", "dmin", "dmax", "source"])
            w.writeheader()
            w.writerows(rows)
        atomic_write_json(out + ".json", {"source": src, "roles_sha256": roles_sha, "provisional": provisional,
                                          "n": a.n, "seed": a.seed, "n_pool": len(set(ids)),
                                          "ids_sha256": ids_sha256([r["id"] for r in rows]), "utc": _utc()})
        print(f"[L0] wrote {len(rows)} frames to {out} (source: {src})")
        return 0

    # validate
    side = a.list + ".json"
    provisional, list_roles_sha = True, None
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            sj = json.load(f)
        provisional, list_roles_sha = bool(sj.get("provisional")), sj.get("roles_sha256")
    clusters, roles_sha = None, None
    if os.path.exists(a.roles):
        ds = load_dsun(a.roles)
        clusters, roles_sha = ds["clusters"], ds["roles_sha256"]
    res = run_l0(a.list, a.v2l_dir, a.moge_dir, _cfg_from_args(a), a.bf_fn, a.limit, a.n_boot, clusters=clusters)
    res["provisional"] = provisional
    res["roles_sha256"] = roles_sha
    res["roles_match_list"] = bool(roles_sha and roles_sha == list_roles_sha)
    res["smoke"] = bool(a.limit) or res["n_images"] < L0_N
    tag = ("_provisional" if provisional else "") + ("_smoke" if res["smoke"] else "")
    out = a.out or os.path.join(DEFAULT_L0_DIR, f"L0_n{res['n_images']}{tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    atomic_write_json(out, res)
    v, g = res["verdict"], res["gates"]
    print(f"[L0] n={res['n_images']} d1 v2l={g['G1_d1']['v2l']} comp={g['G1_d1']['composite']} | "
          f"d1_ssi v2l={g['G2_d1_ssi']['v2l']} comp={g['G2_d1_ssi']['composite']} | "
          f"BF({res['bf_function']}) v2l={g['G3_bf']['v2l']} comp={g['G3_bf']['composite']} | "
          f"edge-pair survival {g['G4_edge_pair_survival']['value']} -> L0 pass={v['l0_pass']} "
          f"(L1a eligible={v['l1a_eligible']}, L1b eligible={v['l1b_eligible']})")
    print(f"[L0] wrote {out}")
    if a.register:
        if provisional or res["smoke"] or not res["roles_match_list"]:
            raise SystemExit("[L0] refusing to register a provisional, smoke, or roles-mismatched L0 result")
        rec = register_l0(res, out, supersede=a.supersede)
        print(f"[L0] registry record {rec.get('record_hash', '')[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
