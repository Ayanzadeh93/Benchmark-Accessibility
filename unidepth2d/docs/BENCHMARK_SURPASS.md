# Surpassing UniDepthV2 on its own zero-shot benchmark

*All numbers are computed by our harness (`bench/`) using UniDepthV2's **exact** metric definitions
(ported 1:1 from `unidepth/utils/evaluation_depth.py`): δ1/AbsRel/τ with the paper's `ssi`
(least-squares scale+shift) and `si` (median) rescalings, `SILog`, and `F_A` = AUC of chamfer-F1 on
back-projected point clouds — computed with a **validated pure-PyTorch chamfer** (Pytorch3D-KNN does
not build on Blackwell/sm_120; our port is validated ≤1e-3 and reproduces the official trapz
normalization exactly). ρ_A = AUC-to-15° of predicted-ray angular error vs GT-intrinsics rays.
"Vanilla V2" = the official frozen checkpoint at `resolution_level 5`. No scale alignment is applied
to the raw metrics.*

## Result — training-free V2++ surpasses vanilla UniDepthV2 (NYU-v2, exact metrics)

**T1 ablation (V2-L, NYU-250), each row adds one novelty so contributions are isolated:**

| config | AbsRel↓ | F_A↑ | SILog↓ | δ1_ssi↑ | AbsRel_ssi↓ | ρ_A↑ |
|---|---|---|---|---|---|---|
| vanilla V2-L | 0.1032 | 0.8398 | 8.019 | 0.9624 | 0.0469 | 0.9202 |
| N1 IO-TTA (oracle fusion) | 0.1047 | 0.8391 | 7.967 | 0.9624 | 0.0467 | 0.9202 |
| N1 mean-TTA (baseline) | 0.1045 | 0.8393 | 7.924 | 0.9626 | 0.0465 | 0.9202 |
| **N2 SCMR (recalibration)** | **0.0826** | **0.8654** | 8.035 | 0.9625 | 0.0469 | 0.9202 |
| N1+N2 | 0.0830 | 0.8644 | 7.985 | 0.9625 | 0.0467 | 0.9202 |
| N1+N2+N3 camera-search | 0.0830 | 0.8644 | 7.985 | 0.9625 | 0.0467 | **0.9208** |

**Headline:** a **training-free** wrapper reduces UniDepthV2-Large's zero-shot **AbsRel by ~20 %**
(0.1032 → 0.0826) and raises its **3D F-score F_A by +3 %** (0.840 → 0.865) — F_A being one of the
paper's three headline metrics — with **no retraining and 2 parameters**.

**The surpass holds for V2-Small too** (NYU-300, SCMR `a=0.974, b=−0.011` fit on SUN):

| model | AbsRel↓ | RMSE↓ | F_A↑ | δ1_ssi↑ |
|---|---|---|---|---|
| vanilla V2-S | 0.140 | 0.470 | 0.788 | 0.9415 |
| **V2-S + SCMR** | **0.120 (−14 %)** | **0.422 (−10 %)** | **0.824 (+4.5 %)** | 0.9420 |

Consistent across both model sizes: SCMR lifts absolute accuracy (AbsRel/RMSE) and the 3D F-score,
flat on the scale-invariant δ1_ssi.

## What drives it, and what doesn't (honest attribution)

- **N2 — Self-Calibrated Metric Recalibration is the entire win.** A label-efficient log-affine
  `logD' = a·logD + b` (Huber-fit, `a=0.996, b=−0.042`) **fit on SUN-RGBD and applied zero-shot to
  NYU** cuts AbsRel 20 % and lifts F_A 3 %. It corrects the systematic near-field **over-estimation**
  we diagnosed earlier (V2 over-predicts close depth); because F_A is a scale-sensitive 3D metric,
  fixing the scale directly improves the point-cloud match. The coefficients transfer across indoor
  domains (SUN→NYU), so this is a genuine zero-shot-transfer method, not a per-set tune.
- **N1 — IO-TTA multi-shape fusion is near-null here, and there is a clean reason.** UniDepthV2 is
  *already shape-invariant by design* (the paper's own contribution + geometric-invariance loss), so
  fusing predictions across resolution levels of an already-invariant model adds almost nothing
  (AbsRel/F_A flat; SILog marginally better). The oracle vs mean-TTA gap is negligible on this
  benchmark. Honest negative result — and an insight: their invariance training leaves little for
  test-time multi-shape ensembling to recover.
- **N3 — consistency camera-prompt search barely moves ρ_A** (+0.0006). The paper already calls ρ_A
  "stagnant"; a training-free focal sweep is not enough to move it materially.

## What we surpass, and what we do NOT (no overselling)

| metric | scale-sensitive? | vanilla → V2++ | verdict |
|---|---|---|---|
| AbsRel (absolute) | yes | 0.103 → 0.083 | **surpass (−20 %)** |
| F_A (3D, headline) | yes | 0.840 → 0.865 | **surpass (+3 %)** |
| RMSE | yes | ↓ | surpass |
| SILog | mostly | ~flat/↓ | marginal |
| **δ1_ssi (paper PRIMARY)** | **no (scale-invariant)** | 0.962 → 0.962 | **flat — not surpassed** |
| AbsRel_ssi | no | ~flat | flat |
| ρ_A | camera | ~flat | flat |

The scale-**invariant** metrics (δ1_ssi, AbsRel_ssi) are, by construction, immune to a scale
correction — SCMR removes exactly the error those metrics already normalize away — and δ1_ssi is
near-saturated (0.962). So on the paper's *primary* δ1_ssi we are flat, not better; our surpass is on
the **absolute-metric axis (AbsRel/RMSE) and the 3D axis (F_A)**, which is where a metric-depth method
for real deployment actually matters. Moving δ1_ssi requires improving relative structure, which is a
*training* target (T2/T3), not a test-time scale fix.

## Result — T2: a 15.5 M student Pareto-beats V2-Small (34 M) in-domain, exact metrics

Our distilled students (`S10_ultra_L` 15.5 M, `S5_crop` 5.1 M — 2D depth only, no ray head) vs the official
**V2-Small (34 M)** on the exact benchmark metrics.

**NYU-300 (in-domain for the student):**

| model | params | AbsRel↓ | RMSE↓ | F_A↑ | δ1 (raw)↑ | d_auc↑ | δ1_ssi↑ | SILog↓ |
|---|---|---|---|---|---|---|---|---|
| vanilla V2-S | 34 M | 0.140 | 0.470 | 0.788 | 0.852 | 0.884 | **0.9415** | **10.31** |
| V2-S + SCMR (T1) | 34 M | 0.120 | 0.422 | 0.824 | — | — | 0.9420 | — |
| **student S10** | **15.5 M** | **0.116 (−17 %)** | **0.430 (−8 %)** | **0.817 (+3.7 %)** | **0.863** | **0.895** | 0.9368 | 11.81 |
| student S5 | 5.1 M | 0.123 (−12 %) | 0.432 (−8 %) | 0.809 (+2.7 %) | 0.855 | 0.891 | 0.9305 | 12.25 |

At **2.2× fewer parameters**, the 15.5 M student **beats V2-S on every absolute and 3D metric**
(AbsRel −17 %, RMSE −8 %, F_A +3.7 %, d_auc +1.2 %, raw δ1 +1.4 %) — and even **edges out the
SCMR-recalibrated V2-S** on AbsRel (0.116 vs 0.120) while nearly matching its F_A. A clear Pareto point:
less than half the compute, higher absolute + 3D accuracy, on the paper's own metrics.

**SUN-500 (out-of-domain for the student):**

| model | params | AbsRel↓ | RMSE↓ | F_A↑ | δ1_ssi↑ | SILog↓ |
|---|---|---|---|---|---|---|
| **vanilla V2-S** | 34 M | **0.098** | **0.288** | **0.810** | **0.983** | **7.43** |
| student S10 | 15.5 M | 0.105 | 0.341 | 0.802 | 0.960 | 10.20 |
| student S5 | 5.1 M | 0.116 | 0.356 | 0.785 | 0.950 | 10.93 |

**Honest boundary (the OOD gap):** on SUN — which the student saw far less of than the foundation model —
**V2-S wins across the board.** The foundation model's massive, diverse pre-training generalizes better
zero-shot than our narrowly-distilled student. So the T2 Pareto claim holds **in-domain (NYU)** but **not
OOD (SUN)**. Two consistent sub-findings mirror T1: (a) the student trails V2-S on the scale-invariant
**δ1_ssi** on both sets (−0.5 pt NYU, −2.3 pt SUN) — small models / scale fixes cannot beat the foundation
model on relative *structure*; (b) the student is worse on **SILog** everywhere. These are exactly the two
axes the heavy-T2 work (**N4 AnchDistill** to bake in calibrated scale, **N5 ICD** to transfer V2's
cross-camera invariance, plus **multi-domain teacher pseudo-labels** to close the OOD gap) and **T3 LoRA**
are designed to move — the current quick-T2 result *motivates* them rather than replacing them.

## Method (N2 SCMR), reproduce

Fit `(a,b)` by robust (Huber) regression of `log GT ~= a·log(pred)+b` over pooled pixels of a
calibration set, apply the same `(a,b)` to the test set. `bench/v2pp.py` (`fit_affine_pixels` + the
`affine` toggle); `bench/ablate_t1.py` runs the full isolation matrix.

```bash
# flagship + ablation (V2-L, NYU, SCMR calibrated on SUN)
conda run -n depth python -m bench.ablate_t1 --dataset nyu --variant l --limit 250 \
    --calib-dataset sun --calib-limit 100
# exact-metric eval of any model
conda run -n depth python -m bench.eval --dataset nyu --model vitl --model vits --limit 654
```

## Status of the campaign
- **Phase 0 harness**: complete + validated (exact metrics, pure-torch chamfer, NYU/SUN loaders; F_A
  point-cap = 20k for O(N²) speed, identical for all models so comparisons are fair).
- **Phase 1 (T1, training-free)**: **done — V2-L++ and V2-S++ surpass vanilla on AbsRel/RMSE/F_A**, driven
  by SCMR; N1/N3 ablated as near-null (with reasons).
- **Phase 2 (T2, quick part)**: **done — 15.5 M student Pareto-beats V2-S (34 M) in-domain (NYU)** on
  absolute + 3D metrics at 2.2× fewer params; loses OOD (SUN) and on δ1_ssi/SILog. Heavy part (N4
  AnchDistill, N5 ICD, multi-domain teacher labels) is the next training job — aimed squarely at the OOD +
  structure gap this quick result exposed.
- **Phase 3 (T3 LoRA N6)**: next training job — targets δ1_ssi / structure and OOD uncertainty
  calibration, which neither the training-free path nor the current student can move.
