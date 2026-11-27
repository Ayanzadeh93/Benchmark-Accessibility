# Corrected evaluation: old vs rescored results

Evaluation-only rescore (branch `b20-lossfix`, 2026-11-25/26). No model was retrained, no training data,
loss, split or leakage handling was changed, and SUN keeps its teacher-estimated training intrinsics.
New result files are in `results_rescore/`; `results_bench/` was not modified.

Two evaluation fixes were applied to **every model we scored ourselves** (students and foundation
baselines). Baselines quoted from papers are not rescored.

1. **Boundary F1 (A04).** The legacy metric (`protocol.boundary_f1_si`) silently skipped images and
   thresholds where the *prediction* had no edge, which inflates the score of over-smooth models.
   Added `bf1_si_v2` (thresholds 5–25 %, an image where GT has edges but the prediction has none scores 0
   and is counted) and the Depth Pro protocol `bf1_depthpro` (masked to valid GT; the unmasked variant
   treats GT holes as edges and is not used as a headline). Legacy is kept and labelled legacy.
2. **Evaluation without GT intrinsics (A05).** With `K=None` the old code silently switched FiLM
   conditioning off while still using a default K for rays/scale head, so torch numbers disagreed with
   ONNX/Core ML. New opt-in policy `k_none_policy=default_k` feeds one `DEFAULT_K` to every
   K-consuming part. Default (`legacy`) is unchanged.

Parity: default outputs are bit-identical to `b20-canonscale` (296/296 tensors + metrics), and every
GT-K d1/RMSE/MAE below reproduces the stored `results_bench` value to its stored precision
(stored files round to 4 decimals; e.g. b20dws 0.74835 vs stored 0.7484).

## 1. IBims-1 (unseen), all self-evaluated models, sorted by corrected bF1 v2

Predictions are identical to before; only boundary-F1 scores change.

| Model | Params | d1 | RMSE (m) | MAE (m) | bF1 legacy | **bF1 v2** (n) | **bF1 DepthPro** (masked) |
|---|---|---|---|---|---|---|---|
| MoGe-2-L | 326M | 74.0 | 0.566 | 0.495 | 0.235 | **0.230** (99) | **0.213** |
| MoGe-2-B | 104M | 73.0 | 0.564 | 0.478 | 0.219 | 0.214 (99) | 0.197 |
| MoGe-2-S | 35M | 63.7 | 0.672 | 0.587 | 0.192 | 0.187 (99) | 0.169 |
| UniK3D-B | — | 88.8 | 0.428 | 0.335 | 0.185 | 0.180 (99) | 0.165 |
| UniDepthV2-L (teacher) | 354M | 94.6 | 0.351 | 0.267 | 0.178 | 0.175 (99) | 0.162 |
| UniDepthV2-B | 97M | 89.3 | 0.433 | 0.319 | 0.176 | 0.171 (99) | 0.156 |
| UniK3D-S | — | 87.7 | 0.472 | 0.370 | 0.165 | 0.161 (99) | 0.144 |
| UniDepthV2-S | 34M | 82.5 | 0.558 | 0.422 | 0.162 | 0.155 (99) | 0.137 |
| b1_mix | 5.1M | 60.4 | 0.881 | 0.710 | 0.047 | 0.021 (99) | 0.014 |
| S21-scale2 | 15.5M | 68.9 | 0.791 | 0.627 | 0.048 | 0.020 (99) | 0.014 |
| S19-sced | 15.5M | 68.9 | 0.804 | 0.640 | 0.048 | 0.019 (99) | 0.013 |
| S16-prox | 15.5M | 70.7 | 0.783 | 0.619 | 0.047 | 0.019 (99) | 0.013 |
| S18-zsclean | 15.5M | 67.7 | 0.817 | 0.645 | 0.050 | 0.018 (99) | 0.011 |
| S10-ultra-L | 15.5M | 65.6 | 0.843 | 0.667 | 0.053 | 0.014 (99) | 0.009 |
| S15-anch-icd | 15.5M | 65.6 | 0.853 | 0.676 | 0.051 | 0.014 (99) | 0.008 |
| **b10** (last) | 5.12M | 66.8 | 0.887 | 0.709 | 0.044 | 0.012 (99) | 0.007 |
| b10 (best) | 5.12M | 65.6 | 0.894 | 0.714 | 0.043 | 0.011 (99) | 0.006 |
| **b20dws** (last) | 5.04M | **74.8** | 0.735 | 0.560 | 0.032 | 0.009 (99) | 0.005 |
| b20core (last) | 5.18M | 74.3 | 0.713 | 0.549 | 0.041 | 0.008 (99) | 0.004 |
| b20 rep4 (best) | 5.04M | 69.8 | 0.858 | 0.661 | 0.030 | 0.004 (99) | 0.002 |
| b20 rep4 (last, unstable) | 5.04M | 62.8 | 1.032 | 0.763 | 0.009 | 0.0004 (99) | 0.0002 |

Legacy counted only 71 of 100 images for b20dws; v2 counts 99 (one IBims image has no GT edge).

## 2. Evaluation without GT intrinsics (students that take K)

GT-K columns are unchanged by A05 (re-verified). RGB-only baselines are unaffected.

| Dataset | Model | d1 with GT K | d1 no-K, legacy (old) | **d1 no-K, corrected** | MAE no-K legacy | **MAE no-K corrected** |
|---|---|---|---|---|---|---|
| IBims-1 | b10 | 66.8 | 52.7 | **62.0** | 0.739 | 0.741 |
| IBims-1 | b20core | 74.3 | 72.2 | **71.8** | 0.549 | 0.599 |
| IBims-1 | b20dws | 74.8 | 72.3 | **70.6** | 0.555 | 0.615 |
| NYU-654 | b10 | 86.7 | 66.4 | **86.7** | 0.490 | 0.322 |
| NYU-654 | b20core | 85.6 | 78.0 | **85.6** | 0.444 | 0.326 |
| NYU-654 | b20dws | 85.4 | 78.9 | **85.3** | 0.415 | 0.327 |
| SUN raw | b10 | 83.5 | 52.1 | **81.6** | 0.470 | 0.311 |
| SUN raw | b20core | 82.7 | 75.3 | **79.3** | 0.385 | 0.330 |
| SUN raw | b20dws | 83.0 | 76.4 | **80.2** | 0.369 | 0.321 |

## 3. What changes for b20dws (corrected vs corrected)

| Comparison | Old evaluation | Corrected evaluation | Lead |
|---|---|---|---|
| IBims d1 / MAE, GT K (headline) | 74.8 vs 66.8 / 0.56 vs 0.71 m | **identical** | **unchanged: +8.0 d1** |
| NYU, SUN d1, GT K | b10 ahead by 1.3 / 0.5 | identical | unchanged |
| IBims boundary F1, b20dws vs b10 | 0.032 vs 0.044 | 0.0087 vs 0.0123 (v2) | b10 still ahead (same direction) |
| dws4 cost on edges (b20core → b20dws) | — | +0.0004 v2, 95% CI [−0.0009, +0.0016] | **no edge cost** |
| 5M students vs foundation models, bF1 | ~0.03–0.05 vs ~0.16–0.24 (~4–7×) | ~0.01–0.02 vs ~0.14–0.23 (**~10–25×**) | gap **grows** |
| IBims d1 without K, b20dws vs b10 | 72.3 vs 52.7 (+19.6) | 70.6 vs 62.0 (**+8.6**) | lead **shrinks**, still clear |
| NYU d1 without K, b20dws vs b10 | 78.9 vs 66.4 (+12.5) | 85.3 vs 86.7 (−1.4) | lead **reverses** to the GT-K picture |
| SUN d1 without K, b20dws vs b10 | 76.4 vs 52.1 (+24.3) | 80.2 vs 81.6 (−1.4) | lead **reverses** to the GT-K picture |

**Bottom line.** The main claim survives untouched: with GT intrinsics b20dws beats b10 by +8.0 d1 and
−0.15 m MAE on unseen IBims at 41 % fewer MACs, and beats every MoGe-2 size and S16 on IBims d1. The
old no-K evaluation exaggerated b20dws's advantage, because the legacy path crippled b10 (FiLM off);
corrected, the no-K picture matches the GT-K one (b20dws ahead on unseen IBims, ~1.4 d1 behind in
domain). The legacy boundary metric overstated all students' edge quality; corrected, every 5M/15M
student is ~10–25× below the large models on boundary F1, and the b20 family is the weakest on edges.
That is a limitation to state in the paper, not an evaluation artefact.

## 4. Not rescored / caveats

- Paper-quoted baselines stay as published. SSDI (S21 + V2-L scale) was not re-run.
- Boundary F1 was rescored on IBims only (no NYU/SUN boundary numbers were reported before).
- SUN evaluation labels were not changed: the review found the SUN label defects (8×-scaled rows,
  heuristic decode) only in the *training* manifest; evaluation already uses the official decode and
  per-image intrinsics (`benchmark/datasets_tier1.py:121-172`).
- NYU-654 still contains ~50 frames that also appear in training (deferred by the user).
- IBims has 100 images; d1 differences under ~5 points need the paired bootstrap.
