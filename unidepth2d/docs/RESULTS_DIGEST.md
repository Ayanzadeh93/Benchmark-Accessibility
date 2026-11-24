# Results digest — first full paper-grade evaluation campaign (2026-07-18/19)

Everything below is measured by the v2 harness (official splits, no pred clip, native
`infer()`, res-level 9 for officials) — auto-assembled tables in PAPER_TABLES.md; raw JSON
in results_bench/ and results_ocmr/. Claim labels C1–C4 = PAPER_PLAN abstract checklist.

## C1 — OCMR reveals what dense metrics hide ✅ (evidenced)
- V2-L on SUN: dense δ1 **95.7** (saturated) but proxF1@1m only **0.669** on 57,677 objects.
- Object ranking ≠ dense ranking: S10-ultra-L beats official V2-S on NYU dense
  (δ1 87.8 vs 87.4, ARel 11.0 vs 13.1) yet trails it on OCMR-SUN objAbsRel (.147 vs .134).
- The benchmark discriminates: box5 objAbsRel spread .105 (V2-L) → .121 (V2-B) → .134
  (V2-S) → .147 (S10) → .173 (b1); proxF1@1m spread 0.67 → 0.31.

## C2 — correctable metric-scale drift + training-free fix ✅ (headline result)
- **Family-wide drift measured**: V2-L/B/S all over-range objects by bias_log +0.021…+0.026;
  distilled students sit at ≈0 (−0.007) — metric distillation is self-calibrating.
- **Ground-lock removes it** (V2-L, SUN, 3,165/4,379 images floor-locked): dense ARel
  7.80→**7.05** (−9.6% rel — success bar ≥5% MET), bias_log +0.024→**−0.003**,
  proxF1@1m 66.9→**69.2**, objAbsRel 10.45→10.10. (δ1 dips 95.7→95.0 — scale correction
  trades threshold-margin pixels; report both.)
- **SMA (size priors) is a negative result indoors**: V2-L ARel 7.8→17.0. Cause: occluded/
  truncated boxes shrink h_px → d_geo over-estimates. Kept as an honest ablation; it
  motivates the ground plane as the reliable indoor anchor.
- Students (already calibrated) gain nothing from ground-lock (ARel 12.6→14.0; the scale
  estimate adds variance where there is no drift to remove) except proxF1@1m 37.6→41.2.
  Framing: anchoring fixes foundation-model drift; distillation already yields calibrated
  students.

## C3 — efficient students ✅ on NYU, honest gaps elsewhere
- Official protocol, NYU-654: **S10-ultra-L (15.5M) beats official V2-S (34M)**:
  δ1 87.80 vs 87.42, ARel 11.00 vs 13.08. S15 ≈ S10 (affine+ICD adds ~nothing measurable).
- SUN full alltest: V2-S 89.5/11.2 > S10 85.5/12.6 (students trail).
- IBims (unseen cameras): students collapse (δ1 65.6 vs V2-S 82.5) → the OOD gap motivates
  the Phase-4 zero-shot-clean training and is reported as scope, not hidden.
- KITTI outdoor: S10 collapses entirely (bias_log −0.50; indoor-trained, 30 m clamp) —
  documented scope boundary.

## OCMR-KITTI (first outdoor object-ranging rows; 30,537 objects, z_front GT)
- V2-L .119 (center) / .144 (box5); V2-S .216/.252 with **+14% bias**; proxF1@5m 0.45 (V2-L
  center) — far from solved outdoors too.
- **Protocol finding**: center BEATS box5 outdoors (thin objects, background-hitting
  corners) while indoors maskmed < box5 < center — aggregation reliability is
  scene-dependent (paper §3.4 ablation now has a real finding).

## C4 — uncertainty (partial)
Teacher AUSE/nAUSE measured on all slices (SUN .0235/.797, NYU .0328/.754, IBims
.0118/.223) — zero-shot nAUSE ≈0.75–0.80 on SUN/NYU echoes the paper's Tab. VII collapse
(published zero-shot V2-L .032/.645). Remaining: S6-unc student σ-head eval + object-level
conformal intervals (cal/test split wiring).

## Gate verdict (full-n, raw-δ1 semantics)
IBims V2-L δ1 PASSES (+0.09). Everything else misses ±0.5 via the model-independent
+0.7–1.0pt ARel offset (data-packaging/ckpt-drift; PROTOCOL_NOTES §2c) → measured
baselines everywhere; published rows only [paper]-tagged with the footnote.

## Infrastructure facts settled this campaign
1. **V1-L cannot run** in this env: needs `xformers.components` (removed in modern
   xformers; legacy builds incompatible with Blackwell/torch-2.11). Measured V1 rows
   dropped everywhere; published V1 rows may be cited [paper]-tagged.
2. **xformers side-effect incident**: installing xformers 0.0.35 for V1 silently broke ALL
   V2 model inference (different attention path) → uninstalled; V2 verified working again.
   Env rule: never add xformers to `depth`.
3. **anchloss data finding**: the own corpus's objects are 99.5% class-less (`class_name`
   null; `class_id` is a per-image open-vocab index, not a global vocabulary) → the
   anchor-consistency loss has ~70 usable objects and S17-anchloss training is DEFERRED.
   Paths to revive: (a) gold set (classes captured at annotation), (b) a YOLO-relabel pass
   over the 7.9k images + distance resampling from the cached depth PNGs.
4. Timing on RTX 5070 laptop: NYU-654 two officials 13 min; SUN-4396 three students 69 min;
   anchor_lock full SUN ~90 min/model; OCMR-KITTI 7,481 frames: V2-L 24 min, S10 5 min.

## Retry completions (02:12, all green)
- **V2-B dense rows complete**: NYU 92.17/10.16, IBims 89.32/11.73 (published 89.7/11.1 —
  the same +0.6 ARel family offset), SUN 92.54/9.67. Dense table now spans V2-S/B/L +
  3 students on all three sets.
- **Same-rig efficiency (RTX 5070 Ti laptop, fp32, VGA input, peak-VRAM)**:
  V2-L 141ms/7.1fps/2.40GiB · V2-B 72ms/13.9fps/1.18GiB · V2-S 40ms/25.1fps/0.63GiB ·
  S10 17.8ms/56.3fps/0.20GiB · S15 17.9ms/56.0fps/0.20GiB · b1 13.9ms/71.7fps/0.09GiB.
  → S10 is 2.2× faster and 3× lighter than official V2-S while beating it on NYU (C3+C5).
- **Res-marginalization (A3-lite) = honest NULL on IBims/V2-L**: precision-fusion of
  levels {3,6,9} matches but never beats the best single level (ARel .0859 vs .0848,
  bF1 −0.007) at 2.4× compute; precision ≈ uniform weighting. Consistent with the paper's
  own shape-invariance claim (their Fig. 4) — reported as a consistency/negative finding;
  anchoring carries the training-free contribution.

## Phase 4 — DONE (2026-07-19 02:12 → 09:55, 9/9 steps exit=0)
scripts/run_phase4_trainings.ps1; log in outputs/eval_queue.log. Both students: 40 ep,
efficientvit_b2 15.5M, recipe s4,s5,ray,range (+prox for S16).

| model | NYU δ1/ARel | SUN δ1/ARel | IBims δ1/ARel | OCMR-SUN box5 objAbsRel / proxF1@1m / bias_log |
|---|---|---|---|---|
| S10-ultra-L (ref) | 87.80/.110 | 85.47/.126 | 65.63/.184 | .1465 / .376 / −.007 |
| **S16-prox** | **88.39/.108** | **86.02/.125** | **70.68/.176** | **.1458** / .374 / **−.001** |
| S18-zsclean | 87.51/.113 | 81.69/.145 | 67.70/.177 | .1668 / .300 / +.013 |

- **S16-prox = best plain student**, better than S10 on every dense set (largest on IBims,
  +5.1 δ1); OCMR-SUN essentially tied with S10. Bias ≈0 → C2 "distillation self-calibrates"
  still holds.
- **S18-zsclean (own:0.5 + nyu:0.5, no SUN)**: SUN δ1 drops 85.5→81.7 (≈ b1 baseline) and IBims
  does not improve → the S-series SUN numbers are partly IN-DOMAIN (default mix own .4 /
  nyu .4 / sun .2). Must be stated wherever S-students sit next to zero-shot V2 rows.
- **Scale-gap decomposition** (docs/SCALE_GAP.md): students' IBims δ1_ssi ≈ .93 (V2-S .975)
  vs raw δ1 .66–.71 → the OOD loss is almost entirely per-image global SCALE, not structure;
  one scalar closes 82–86% of it.

## After Phase 4 (2026-07-20 → 07-22)
**S19-sced / S21-scale2 probes** — warm-started from S16 best.pt, +15 ep at lr 1e-4, adding
per-image scale losses (`scale,sced` / `scale2`). Caveat: compared to S16 they also get 15
extra epochs, so gains are confounded.

| model | NYU δ1/ARel | SUN δ1/ARel | IBims δ1/ARel | OCMR-SUN box5 objAbsRel |
|---|---|---|---|---|
| S16-prox (start point) | 88.39/.108 | 86.02/.125 | 70.68/.176 | .1458 |
| S19-sced | 88.54/.109 | 85.91/.127 | 68.94/.187 | — |
| S21-scale2 | 88.79/.108 | 86.09/.126 | 68.89/.184 | .1466 |

→ small NYU gain, flat SUN/OCMR, **IBims gets worse** (−1.8 δ1). In-domain scale losses do
not transfer the scale latent OOD — consistent with the SCALE_GAP diagnosis.

**SSDI hybrid (Scale-Structure Decoupled Inference)** — `benchmark/run_bench.py:HybridSSDI`:
S21 student depth rescaled per image by ONE scalar = median log-ratio to a V2-L pass at its
lowest res level (r0), over the 0.2–8 m mutual range; no GT. (ssdi_s21full.log, H1–H6 exit=0.)

| model | NYU δ1/ARel | SUN δ1/ARel | IBims δ1/ARel | OCMR-SUN box5 objAbsRel / proxF1@1m |
|---|---|---|---|---|
| S21-scale2 alone | 88.79/.108 | 86.09/.126 | 68.89/.184 | .1466 / .368 |
| **SSDI S21 + V2-L r0** | **90.12/.102** | **89.75/.110** | **82.75/.130** | **.1353 / .554** |
| V2-S r9 (official) | 87.42/.131 | 89.52/.112 | 82.53/.142 | .1344 / .521 |

→ the hybrid matches/edges official V2-S on all three dense sets and OCMR-SUN, recovering
+13.9 δ1 on IBims from one scalar. Latency NOT yet measured on the same rig (worst case =
student + V2-L-r0; the ~1 Hz donor-refresh amortization is a design claim, unverified).

**S21-scale2-full (from scratch, 40 ep) — INCOMPLETE, do not use.** Started 07-20 09:21
(died) and again 07-21 04:15 (s21full_only.log); last log line is epoch 5, it1050/3000, no
END line and no train_summary.json. `checkpoints/S21_scale2_full/best.pt` is an early
checkpoint (val AbsRel .140 vs ≈.086 for finished students). No evals exist for it.

**P-series (fair zero-shot-clean ladder, scripts/run_p_series.ps1) — only step C0 ran.**
C0 consensus-scale build finished 07-22 (outputs/data_gt/own_consensus_scale.json, n=7,931;
teacher scale offset delta mean −.021, std .208, |delta|>.05 on 69.9% of images; σ median
.030). The .err.log holds only import warnings. Rungs P0–P5 (own-only training, own-val
selection) have NOT been trained — no P* checkpoints exist.

## 2025 zero-shot baselines on IBims-1 (measured 2026-11-16)
Run in our harness, same protocol as every other row (official-mirror metrics, no pred
clip, native `infer()`, resolution level 9, no GT camera). Env `depth_cmp` (clone of
`depth` + moge 3.0.0, utils3d_moge, unik3d 0.1, installed `--no-deps`; torch 2.11+cu128
untouched). Wrapper: `benchmark/run_bench.py:ExternalModel` (`--model moge2-s|b|l`,
`unik3d-s|b|l`). Log: outputs/ibims_2025_models.log; JSON: results_bench/ibims__*.json.

| model | params (counted) | δ1 | δ1_ssi | ARel | RMSE | bF1 |
|---|---|---|---|---|---|---|
| UniK3D ViT-B | 117.2M | 88.84 | 98.14 | .1220 | .428 | .185 |
| UniK3D ViT-S | 35.4M | 87.73 | 97.70 | .1243 | .472 | .165 |
| (ours) SSDI S21+V2-L r0 | 15.5M + donor | 82.75 | 93.00 | .1301 | .597 | .048 |
| MoGe-2 ViT-L | 330.9M | 73.96 | 98.87 | .1455 | .566 | .235 |
| MoGe-2 ViT-B | 104.7M | 72.98 | 98.50 | .1454 | .564 | .219 |
| (ours) S16-prox | 15.5M | 70.68 | 92.77 | .1756 | .783 | .047 |
| MoGe-2 ViT-S | 35.1M | 63.70 | 97.78 | .1744 | .672 | .192 |
| (ours) b1-mix | 5.1M | 60.40 | 91.37 | .2100 | .881 | .047 |

- **Harness validated**: UniK3D-S 87.7/.124 vs its paper's 87.7/.130 (Tab. 20).
- **MoGe-2-L under-reproduces**: 74.0/.146 vs 83.0/.136 in the DepthMaster paper (Tab. A1).
  Unexplained — candidates: their input resolution/token budget, or our `apply_mask=False`.
  Resolve before citing MoGe-2 rows.
- **Claims**: SSDI beats MoGe-2 S/B/L on δ1 AND ARel. S16 beats MoGe-2-S on δ1 (+7.0) at
  0.44× params, ARel tied (.176 vs .174). b1 (5.1M) trails MoGe-2-S by 3.3 δ1.
  UniK3D-S/B beat every student and SSDI.
- **Structure vs scale**: every 2025 model has δ1_ssi 97.7–98.9 vs students' 91–93, so the
  baselines also win on relative structure, not only scale. MoGe-2's raw-δ1 deficit is all
  scale (δ1_ssi 98.9) — the same failure mode as our students.
- Our students' boundary F1 (~.047) is ~4× below every foundation model (.165–.235).

## 8-bit compression of the current students (measured 2026-11-16 to 11-18)

**8-bit weights only** (simulated, `edge/quant_sim.py`, spec `q:<bits>:<ckpt>`): Conv/Linear weights
per-channel on an integer grid; activations, biases and BatchNorm stay float. IBims-1, n=100. Size =
quantized weights at N bits + remaining parameters at 4 bytes. Paired 95% CI from
`benchmark/compare_paired.py` (10,000 image-level bootstrap resamples).

| model | bits | size MB | d1 % | AbsRel | d1 change vs fp32 (95% CI) |
|---|---|---|---|---|---|
| b1 (5.1M) | 32 | 20.48 | 60.40 | .2100 | reference |
| b1 | 8 | 5.19 | 61.66 | .2099 | +1.25 [+0.14, +2.43] |
| b1 | 6 | 3.92 | 52.64 | .2340 | -7.76 [-11.42, -4.20] |
| b1 | 4 | 2.65 | 35.34 | .4179 | not paired |
| S16 (15.5M) | 32 | 61.90 | 70.68 | .1756 | reference |
| S16 | 8 | 15.63 | 70.17 | .1769 | -0.50 [-1.27, +0.25] |
| S16 | 6 | 11.77 | 67.44 | .1918 | -3.24 [-4.96, -1.51] |
| S16 | 4 | 7.91 | 22.93 | .4144 | not paired |

- 8-bit weights cut storage about 4x with no meaningful accuracy loss. 6 bits and below hurt.
- The b1 +1.25 gain is a quantization side effect, not an improvement to rely on: its d1_ssi fell
  0.28 points (CI [-0.55, -0.04]).

**Full int8 (weights and activations) FAILS on these students.** ONNX Runtime static QDQ, per-channel,
128 `own` calibration images, last conv kept float (`edge/ort_int8.py`). Agreement with the fp32 parent
on 64 held-out `own` images (`edge/int8_agreement.py`; agreement only, not benchmark accuracy):

| model | fp32 MB | int8 MB | mean abs log-depth change | pixels within 1.25x of fp32 | CPU ms fp32 / int8 (4 threads) |
|---|---|---|---|---|---|
| b1 | 20.68 | 6.03 | 0.73 | 21.5% | 65.1 / 72.6 |
| S16 | 63.00 | 16.92 | 1.04 | 10.2% | 131.6 / 95.7 |

Diagnosis on b1 (32 calibration, 32 held-out images): percentile calibration does not help (0.99 vs
1.01). Keeping the EfficientViT attention blocks in float cuts the error to 0.26 (60.5% within 1.25x,
10.93 MB), so attention is a main cause but not the only one. Consequence for V2: choose a
quantization-friendly backbone rather than EfficientViT, or plan on quantization-aware training.

IBims-1 accuracy of the exported files (measured 2026-11-19, ONNX Runtime CPU, GT intrinsics as for the PyTorch rows,
n=100, paired 95% CI against the PyTorch model):

| model | file | MB | d1 % | AbsRel | d1_ssi % | d1 change (95% CI) |
|---|---|---|---|---|---|---|
| b1 | PyTorch fp32 | 20.48 | 60.40 | .2100 | 91.37 | reference |
| b1 | ONNX fp32 | 20.68 | 60.41 | .2101 | 91.37 | +0.01 [-0.02, +0.04] |
| b1 | ONNX full int8 | 6.03 | 24.02 | .4316 | 54.04 | -36.38 [-42.83, -29.88] |
| S16 | PyTorch fp32 | 61.90 | 70.68 | .1756 | 92.77 | reference |
| S16 | ONNX fp32 | 63.00 | 70.66 | .1757 | 92.77 | -0.02 [-0.04, +0.00] |
| S16 | ONNX full int8 | 16.92 | 16.38 | .4800 | 53.34 | -54.29 [-61.06, -47.08] |

The ONNX export is exact. Full int8 destroys both models, structure as well as scale (d1_ssi falls about 38 points).
The shippable 8-bit form of the current students is therefore 8-bit WEIGHTS with float activations.

## Next
1. Finish S21-scale2-full or drop it from the tables.
2. Run the P-series ladder (P0–P5) → the only students that are zero-shot on NYU/SUN.
3. Same-rig latency for SSDI (per-frame donor and 1 Hz donor).
4. Still open from C4: S6-unc σ-head eval + object-level conformal intervals.
