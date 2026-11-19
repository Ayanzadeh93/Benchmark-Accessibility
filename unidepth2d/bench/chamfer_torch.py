"""Pure-PyTorch chamfer / F1-AUC — a drop-in for the Pytorch3D-KNN path in the official
UniDepth `evaluation_depth.py` (Pytorch3D-KNN does not build on Blackwell/Windows sm_120).

Matches the official convention EXACTLY:
  * KNN returns SQUARED euclidean nearest-neighbour distances (like pytorch3d.ops.knn_points),
  * `chamfer_dist` = (sqrt(dist1)+sqrt(dist2))/2,
  * `f1_score` / `auc` compare the RAW (squared) NN distances against the given thresholds and
    integrate precision/recall exactly as the official code does.

Brute-force NN is chunked so a ~76.8k-point cloud (their eval downsamples to ~240x320) fits in VRAM.
Validated against a small closed-form reference in __main__ (run: python -m bench.chamfer_torch).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def _nn_sq_dists(a: torch.Tensor, b: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """For each point in a (Na,3), squared euclidean distance to its nearest point in b (Nb,3).
    Uses chunked torch.cdist (numerically stable — exact 0 on coincident points, unlike the
    |a|^2-2ab+|b|^2 expansion) so identical clouds give a true-zero chamfer."""
    out = a.new_empty(a.shape[0])
    for i in range(0, a.shape[0], chunk):
        d = torch.cdist(a[i : i + chunk], b)          # (c,Nb) euclidean
        out[i : i + chunk] = d.min(dim=1).values ** 2  # squared NN distance
    return out


def chamfer_components(pred: torch.Tensor, gt: torch.Tensor, chunk: int = 4096):
    """pred,gt: (N,3) point clouds. Returns (dist_pred2gt, dist_gt2pred) squared NN distances,
    mirroring pytorch3d knn (dist1 = x->y = pred->gt, dist2 = y->x = gt->pred)."""
    dist1 = _nn_sq_dists(pred, gt, chunk)     # pred -> gt
    dist2 = _nn_sq_dists(gt, pred, chunk)     # gt   -> pred
    return dist1, dist2


def chamfer_dist(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    d1, d2 = chamfer_components(pred, gt)
    return (torch.sqrt(d1) + torch.sqrt(d2)).mean() / 2.0


def f1_score(pred: torch.Tensor, gt: torch.Tensor, thresholds) -> torch.Tensor:
    """AUC over thresholds of F1(precision,recall) — precision from pred->gt NN, recall from gt->pred.
    Thresholds are compared against SQUARED distances, exactly like the official f1_score."""
    d1, d2 = chamfer_components(pred, gt)
    th = torch.as_tensor(thresholds, device=pred.device, dtype=pred.dtype)
    # vectorized threshold sweep (was a python loop over thresholds)
    precisions = (d1[None, :] < th[:, None]).float().mean(dim=1)
    recalls = (d2[None, :] < th[:, None]).float().mean(dim=1)
    f1 = 2 * precisions * recalls / (precisions + recalls)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return torch.trapz(f1) / len(th)


def auc(pred: torch.Tensor, gt: torch.Tensor, thresholds) -> torch.Tensor:
    """AUC of precision-recall (as the official `auc`): trapz(precisions, recalls)."""
    d1, d2 = chamfer_components(pred, gt)
    th = torch.as_tensor(thresholds, device=pred.device, dtype=pred.dtype)
    precisions = torch.stack([(d1 < t).float().mean() for t in th])
    recalls = torch.stack([(d2 < t).float().mean() for t in th])
    return torch.trapz(precisions, recalls)


if __name__ == "__main__":
    # Validation vs a tiny closed-form reference.
    torch.manual_seed(0)
    # Two identical clouds -> chamfer 0, F1 -> 1 for any positive threshold.
    p = torch.rand(500, 3)
    assert float(chamfer_dist(p, p)) < 1e-4, "identical clouds must have ~0 chamfer"
    # NOTE: the official f1_score uses `trapz(f1)/len(th)`; for a perfect all-ones F1 vector of
    # length N this yields (N-1)/N, NOT 1.0. We replicate that exactly (match their numbers, quirks
    # included) -> N=20 -> 0.95. This is the correctness check for the normalization.
    n = 20
    assert abs(float(f1_score(p, p, torch.linspace(1e-4, 1.0, n))) - (n - 1) / n) < 1e-4, \
        "F1-AUC of identical clouds must equal (N-1)/N, matching official trapz normalization"

    # Known offset on a WELL-SEPARATED grid (spacing 1.0 >> shift 0.1), so each point's nearest
    # neighbour stays its own shifted copy -> every NN sq-dist == d^2 and chamfer == d.
    gx, gy, gz = torch.meshgrid(torch.arange(5.), torch.arange(5.), torch.arange(5.), indexing="ij")
    grid = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    d = 0.1
    q = grid.clone(); q[:, 0] += d
    d1, d2 = chamfer_components(grid, q)
    assert torch.allclose(d1, torch.full_like(d1, d * d), atol=1e-5), "offset NN sq-dist wrong"
    cd = float(chamfer_dist(grid, q))
    assert abs(cd - d) < 1e-5, f"chamfer of a pure shift d must equal d, got {cd}"

    # Brute-force cross-check of NN against torch.cdist for a small case.
    a, b = torch.rand(64, 3), torch.rand(80, 3)
    ref = (torch.cdist(a, b) ** 2).min(dim=1).values
    got = _nn_sq_dists(a, b, chunk=17)
    assert torch.allclose(ref, got, atol=1e-5), "chunked NN != cdist reference"
    print("chamfer_torch validation PASSED (identical, pure-shift, cdist cross-check)")
