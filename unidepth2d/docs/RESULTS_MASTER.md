# Results master — everything measured so far (UniDepth baselines + all students + B20)

Consolidated 2026-11-24 from `docs/B20_RESULTS.md`, `docs/PAPER_TABLES.md`, `docs/RESULTS_DIGEST.md`,
`docs/EXPERIMENTS.md`, `docs/SCALE_GAP.md`, and the raw JSON/CSV in `results_bench/`, `results_ocmr/`,
`outputs/edge/`. Nothing here is new measurement; where a number comes from a specific doc it is tagged.
Weights: private HF repo `Ayanzadeh93/depthv2` (b10, b20core, b20dws, S16_prox). Code/results: branch
`b20-canonscale`.

**Protocol (v2 harness):** official splits, no prediction clip, native `infer()`, UniDepth/MoGe/UniK3D at
res-level 9 with no GT intrinsics; students use GT intrinsics. δ1 in % (↑), AbsRel/RMSE/MAE ↓. SUN-RGBD is partly
in-domain for the students (SUN is in the training mix). Never compare absolute values across the
sun/kitti/gold slices.

---

## 1. Trained checkpoints (own-val AbsRel at best epoch)

Own-val is the student's *own* validation split, NOT the benchmark; use §2–§4 for comparisons.

| Run | Best ep | Val AbsRel | Size | Note |
|---|---|---|---|---|
| b1_mix | 34 | 0.0921 | 20 MB | B1 5.1M, plain multi-source |
| b2_mix | 38 | 0.0789 | 59 MB | B2 15.5M, plain |
| S4_edge | 34 | 0.0942 | 20 MB | |
| S5_crop | 39 | 0.1014 | 20 MB | a TAINTED concurrent-run copy exists (0.1032) |
| S6_unc | 38 | 0.0982 | 20 MB | TAINTED copy 0.1006 |
| S7_task | 39 | 0.1105 | 20 MB | TAINTED copy 0.1087 |
| S8_assist | 38 | 0.1994 | 20 MB | TAINTED copy 0.1996; clearly worse |
| S9_ultra | 31 | 0.1054 | 20 MB | |
| S10_ultra_L | 38 | 0.0900 | 59 MB | |
| S11_ray / S12_bins / S13_norm / S14_range | 39 | 0.0947 / 0.0920 / 0.1013 / 0.0950 | 20 MB | S12 is an architecture change, not a loss ablation |
| S15_anch_icd / S15_scratch | 0 / 33 | 0.0927 / 0.0927 | 59 MB | |
| **S16_prox** | 39 | 0.0876 | 59 MB | main B2 student; ONNX/int8 export and OCMR conformal |
| S18_zsclean | 39 | 0.0940 | 59 MB | |
| S19_sced | 10 | 0.0879 | 59 MB | |
| S21_scale2 | 14 | 0.0862 | 59 MB | |
| S21_scale2_full | 7 | 0.1218 | 59 MB | **INCOMPLETE** (died at epoch 5 of 40); do not use |
| intrinsics | 28 | 0.1397 | 20 MB | |
| pilot, pilot_s4…s8 | 1 | 0.239–0.335 | 20 MB | 1-epoch pilots only |
| b10 / b20core / b20dws / b20 | — | see §3 | 20 MB | B20 family (B1 backbone) |

**TAINTED_concurrent_run** checkpoints were trained while another job shared the GPU/data; they are kept only for
reference and are not used in any table.

---

## 2. UniDepth and other foundation baselines vs students (dense, official protocol)

### IBims-1 (unseen cameras, n=100)
| Model | Params (M) | δ1 | AbsRel | RMSE (m) | MAE (m) | δ1 scale-aligned |
|---|---|---|---|---|---|---|
| UniDepthV2-L (teacher) | 354 | 94.6 | 0.085 | 0.351 | 0.267 | 98.7 |
| UniDepthV2-B | 97 | 89.3 | 0.117 | 0.433 | 0.319 | 97.5 |
| UniDepthV2-S | 34 | 82.5 | 0.142 | 0.558 | 0.422 | 97.5 |
| UniK3D-B / -S | — | 88.8 / 87.7 | 0.122 / 0.124 | 0.428 / 0.472 | 0.335 / 0.370 | 98.1 / 97.7 |
| MoGe-2 L / B / S | 326 / 104 / 35 | 74.0 / 73.0 / 63.7 | 0.145 / 0.145 / 0.174 | 0.566 / 0.564 / 0.672 | 0.495 / 0.478 / 0.587 | 98.9 / 98.5 / 97.8 |
| SSDI (S21 + V2-L r0 scale) | — | 82.7 | 0.130 | 0.597 | 0.436 | 93.0 |
| S16-prox (B2) | 15.5 | 70.7 | 0.176 | 0.783 | 0.619 | 92.8 |
| S21-scale2 (B2) | 15.5 | 68.9 | 0.184 | 0.791 | 0.627 | 93.0 |
| S10-ultra-L (B2) | 15.5 | 65.6 | 0.184 | 0.843 | 0.667 | 92.7 |
| b1_mix (B1, old) | 5.1 | 60.4 | 0.210 | 0.881 | 0.710 | 91.4 |
| b10 (B1) | 5.12 | 66.8 | 0.209 | 0.887 | 0.709 | 91.4 |
| b20core | 5.18 | 74.3 | 0.179 | 0.713 | 0.549 | 91.4 |
| **b20dws** | 5.04 | **74.8** | 0.179 | 0.735 | 0.560 | 91.2 |
| b20 (+rep4, unstable) | 5.04 | 62.8 | 0.244 | 1.032 | 0.763 | 85.0 |

### NYU-654
| Model | δ1 | AbsRel | RMSE (m) | MAE (m) |
|---|---|---|---|---|
| UniDepthV2-L | 94.3 | 0.086 | 0.357 | 0.235 |
| UniDepthV2-B | 92.2 | 0.102 | 0.393 | 0.272 |
| UniDepthV2-S | 87.4 | 0.131 | 0.449 | 0.338 |
| SSDI (S21 + V2-L r0 scale) | 90.1 | 0.102 | 0.394 | 0.274 |
| S16-prox | 88.4 | 0.108 | 0.415 | 0.295 |
| S21-scale2 | 88.8 | 0.108 | 0.412 | 0.292 |
| S10-ultra-L | 87.8 | 0.110 | 0.430 | 0.307 |
| b1_mix | 86.1 | 0.119 | 0.435 | 0.317 |
| b10 | 86.7 | 0.116 | 0.446 | 0.322 |
| b20core | 85.6 | 0.120 | 0.448 | 0.323 |
| b20dws | 85.4 | 0.122 | 0.449 | 0.325 |
| b20 (+rep4, unstable) | 69.3 | 0.198 | 0.642 | 0.493 |

### SUN-RGBD raw (n=4396; partly in-domain for students)
| Model | δ1 | AbsRel | RMSE (m) |
|---|---|---|---|
| UniDepthV2-L | 95.7 | 0.078 | 0.242 |
| UniDepthV2-B | 92.5 | 0.097 | 0.281 |
| UniDepthV2-S | 89.5 | 0.112 | 0.316 |
| SSDI | 89.8 | 0.110 | 0.328 |
| S16-prox / S21-scale2 / S10-ultra-L | 86.0 / 86.1 / 85.5 | 0.125 / 0.126 / 0.125 | 0.363 / 0.361 / 0.369 |
| b1_mix | 81.9 | 0.153 | 0.409 |
| b10 / b20core / b20dws | 83.5 / 82.7 / 83.0 | 0.134 / 0.140 / 0.138 | 0.395 / 0.405 / 0.400 |
| b20 (+rep4) | 68.3 | 0.205 | 0.557 |

Other students on NYU/SUN (PAPER_TABLES): S15 ≈ S10; S18_zsclean is worse on SUN (δ1 81.7, AbsRel 0.145);
S19_sced ≈ S16 (NYU 88.5, SUN 85.9).

**Reading.** On NYU the 15.5M S10-ultra-L (δ1 87.8, AbsRel 0.110) beats the official 34M UniDepthV2-S
(87.4 / 0.131). On SUN full, V2-S is ahead (89.5 vs 85.5). On unseen IBims all students trail the V2 family,
but b20dws (74.8) is above every MoGe-2 size and above the 3× larger S16 (70.7).

### Resolution marginalisation (V2-L, IBims)
L3 δ1 93.8 / AbsRel 0.0867 (82 ms); L6 93.6 / 0.0881 (111 ms); **L9 94.6 / 0.0848 (138 ms)**; fusing levels
(uniform or precision-weighted) gives 94.5 / 0.0859 at 331 ms, so fusion costs more and does not beat L9.

---

## 3. B20 family

b10 = S16 recipe on B1. b20core = b10 + dsh (decoupled scale head) + cstm + fov + ownar + trim.
b20dws = b20core + depthwise-separable stride-4 decoder. b20 = b20core + multi-branch rep4.

### Speed
GPU (RTX 5070 Ti laptop, fp32, 640×480, batch 1): S16 19.0 ms / 52.6 FPS; b1_mix 15.1 / 66.4; b10 12.5 / 80.0;
b20core 10.4 / 95.8; b20dws 10.6 / 94.3; b20 10.9 / 91.7.

CPU (ONNX Runtime 1.20, fp32, median of 60):
| Model | GMAC | 4 threads (ms) | 1 thread (ms) |
|---|---|---|---|
| b1_mix / b10 / b20core | 6.84 | 43.1 / 43.4 / 43.1 | 109.9 / 113.0 / 114.7 |
| **b20dws** | **4.04 (−41%)** | **33.8 (−22%)** | **76.5 (−32%)** |

Batch-1 GPU times are launch-bound for 5M models and do not reflect the MAC reduction. ONNX parity
max|Δlog D| ≤ 2.3e-5. Core ML / Neural Engine latency and fp16 parity are **not measured** yet.

### Paired bootstrap, b20dws vs b10 (last vs last, 10k resamples; mean [95% CI])
| Dataset | δ1 | AbsRel | RMSE | MAE |
|---|---|---|---|---|
| IBims | **+8.0 [+3.1, +12.9]** | −.030 (n.s.) | **−0.153 m** | **−0.149 m** |
| NYU-654 | −1.3 [−2.0, −0.5] | +.006 | n.s. | n.s. |
| SUN raw | −0.4 [−0.8, −0.0] | +.004 | +.006 m | — |

### Verdict
- **b20dws** is the recommended general model: 5.04M params, −41% MACs, up to −32% CPU latency, +8.0 δ1 and
  −15 cm MAE on unseen IBims, at −1.3 δ1 on NYU and −0.4 on SUN.
- **b10** is the best 5M model in-domain (NYU/SUN, OCMR objAbsRel .155).
- **b20 with rep4 is not usable**: the scale-head loss diverged after ~epoch 10. rep4 is therefore *not proven*
  in isolation (the spec intended it as a final rung on b20core). `dws4` replaces it at identical deploy cost.

---

## 4. OCMR object-range (box5 unless stated; objAbsRel ↓, obj δ1 ↑)

### OCMR-SUN (57,677 objects)
| Model | objAbsRel | obj δ1 | MAE (m) | RMSE (m) | bias_log | proxF1@1m |
|---|---|---|---|---|---|---|
| UniDepthV2-L | 0.105 | 91.3 | 0.254 | 0.491 | +0.024 | 0.669 |
| UniDepthV2-B | 0.121 | 87.9 | 0.291 | 0.519 | +0.026 | 0.594 |
| UniDepthV2-S | 0.134 | 84.8 | 0.325 | 0.547 | +0.026 | 0.521 |
| SSDI | 0.135 | 84.2 | 0.334 | 0.537 | +0.030 | 0.554 |
| S16-prox | 0.146 | 80.8 | 0.362 | 0.571 | −0.001 | 0.374 |
| S21-scale2 | 0.147 | 81.0 | 0.362 | 0.571 | +0.002 | 0.368 |
| S10-ultra-L | 0.146 | 80.3 | 0.367 | 0.573 | −0.007 | 0.376 |
| S18_zsclean | 0.167 | 76.4 | 0.411 | — | +0.013 | 0.300 |
| b1_mix | 0.173 | 77.0 | 0.405 | 0.629 | +0.011 | 0.312 |
| b10 | 0.155 | 78.1 | 0.392 | 0.614 | −0.012 | — |
| b20core | 0.162 | 77.3 | 0.404 | 0.639 | +0.001 | — |
| b20dws | 0.160 | 77.6 | 0.399 | 0.626 | +0.002 | — |

Aggregation protocol ablation (V2-L, SUN): maskmed objAbsRel 0.089 < box5 0.105 < center 0.112 indoors.

### OCMR-KITTI (30,537 objects, outdoor)
| Model | box5 objAbsRel | center objAbsRel | bias_log (box5) | proxF1@5m (center) |
|---|---|---|---|---|
| V2-L | 0.144 | 0.119 | +0.019 | 0.449 |
| V2-S | 0.252 | 0.216 | +0.141 | 0.311 |
| S10-ultra-L | 0.382 | 0.389 | −0.502 | 0.200 |

Center beats box5 outdoors (thin objects, background-hitting corners), the opposite of indoors. The students
are indoor-trained and collapse outdoors (30 m clamp); this is documented scope, not a bug.

### Training-free anchoring (SUN, 4,379 images)
| Model | variant | dense AbsRel | objAbsRel | bias_log | proxF1@1m |
|---|---|---|---|---|---|
| V2-L | base | 0.078 | 0.1045 | +0.024 | 0.669 |
| V2-L | **ground-lock** | **0.0705** | 0.101 | **−0.003** | **0.692** |
| V2-L | size-prior (sma) | 0.170 | 0.209 | +0.102 | 0.599 |
| S10-ultra-L | base | 0.1255 | 0.1465 | −0.007 | 0.376 |
| S10-ultra-L | ground-lock | 0.140 | 0.161 | −0.020 | 0.412 |

Ground-lock removes the V2 family's +2.1…+2.6% range over-estimate (dense AbsRel −9.6% relative). The size-prior
variant is a negative result indoors (occluded boxes shrink h_px). Calibrated students gain nothing from it
except proxF1@1m.

### Conformal intervals (S10-ultra-L, box5, α=0.1, calibration split crc32(image)%5==0)
Coverage 0.899 against the 0.90 target, relative width 0.63. Proximity alert at 1 m, ε=0.05: test FNR 0.040 vs
naive FNR 0.743, at FPR 0.263 and precision 0.106; at 2 m, ε=0.05: FNR 0.061, precision 0.755.
Raw files: `results_ocmr/conformal__*.json`, `conformal_summary.json`.

---

## 5. Scale-gap decomposition (SCALE_GAP.md)

On IBims the students' δ1 headroom from a per-image **scalar** is 0.18–0.27, and a single scalar closes 82–86%
of the full scale+shift gap (b1_mix 0.86). So most of the IBims error is a global metric-scale error, not
structure: this is the motivation for the B20 decoupled scale head. UniDepthV2-L's scalar gap is only 0.035.

---

## 6. Edge / quantisation (outputs/edge, results_bench `+w4/+w6/+w8/+onnx-*`)

- ONNX size: b1_mix 20.7 MB fp32 → 6.0 MB int8; S16_prox 63.0 MB → 16.9 MB.
- **`int8_agreement.json` (plain int8 vs fp32, 64 images, 4 threads) is poor**: mean |Δlog D| 0.73 (b1_mix) and
  1.04 (S16_prox), with only 21% / 10% of pixels within δ1 of the fp32 output. CPU time did not improve for
  b1_mix (65 → 73 ms) and improved for S16_prox (132 → 96 ms). This is the naive full-int8 export.
- The head-in-fp variants (`onnx-int8-headfp`) and the weight-only w8/w6/w4 runs were evaluated against GT in
  `results_bench/ibims__student-*+w*.json` and `*+onnx-*__pi.json`. **I have not tabulated those here**; the
  JSON schema needs a short script to extract, so treat the edge-accuracy story as incomplete.
- Core ML export script exists (`edge/export_coreml.py`) but has not been run.

---

## 7. Findings and caveats worth keeping straight

1. **Leakage correction (EXPERIMENTS §6).** Earlier "own-domain" numbers for S2-mix/S3-mix-L were measured on
   training images; the conclusion that auxiliary losses hurt was an artifact. On real GT the recipes are neutral
   and only capacity (B1→B2) helps. S6/S7 predate a loss-weighting fix, so their null result is a weak test.
2. **Gate status (RESULTS_DIGEST).** Reproduction of the published V2-L numbers passes only on IBims δ1; the rest
   sit +0.7–1.0pt AbsRel off, model-independent, so all baselines here are *measured* rather than quoted. V1-L
   cannot run in this environment (xformers incompatibility).
3. **Open gaps:** IBims gap to the teacher (74.8 vs 94.6); rep4 unproven; b20 edge export, int8 accuracy and Core
   ML unmeasured; b20 student OCMR rows are slightly worse than b10 (0.160 vs 0.155).
4. **Audit status:** see `AUDIT_STATUS_2026-07-13.md` for the full list of verified and open evaluation issues.

## 8. Where the raw data lives

| What | Path |
|---|---|
| Dense benchmark JSON (88 files) | `results_bench/` |
| OCMR per-object CSV, summaries, conformal, anchoring | `results_ocmr/` |
| Auto-generated tables | `docs/PAPER_TABLES.md`, `docs/SCALE_GAP.md`, `docs/ALL_RESULTS.html` |
| B20 comparison | `docs/B20_RESULTS.md`, spec `docs/B20_SPEC.md`, research `docs/B20_RESEARCH.md` |
| Earlier campaign | `docs/RESULTS_DIGEST.md`, `docs/EXPERIMENTS.md`, `docs/BENCHMARK_SURPASS.md` |
| Weights | HF `Ayanzadeh93/depthv2` (private); local `outputs/checkpoints/` |
