# b10 / b20 results — full comparison

Official v2 harness (no pred clip, per-image mean). δ1 in % (↑); AbsRel, RMSE (m), MAE (m) ↓. Students use GT intrinsics; foundation models at res-level 9, no GT-K. SUN is partly IN-DOMAIN for students (SUN is in the training mix). '—' = not measured.

## IBims-1 (unseen)

| Model | Params (M) | δ1 ↑ | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 scale-aligned ↑ |
|---|---|---|---|---|---|---|
| UniDepthV2-L (teacher) | 354.0 | 94.6 | 0.085 | 0.351 | 0.267 | 98.7 |
| UniDepthV2-B | 97.0 | 89.3 | 0.117 | 0.433 | 0.319 | 97.5 |
| UniDepthV2-S | 34.0 | 82.5 | 0.142 | 0.558 | 0.422 | 97.5 |
| MoGe-2-L | 326.0 | 74.0 | 0.145 | 0.566 | 0.495 | 98.9 |
| MoGe-2-B | 104.0 | 73.0 | 0.145 | 0.564 | 0.478 | 98.5 |
| MoGe-2-S | 35.0 | 63.7 | 0.174 | 0.672 | 0.587 | 97.8 |
| UniK3D-B | — | 88.8 | 0.122 | 0.428 | 0.335 | 98.1 |
| UniK3D-S | — | 87.7 | 0.124 | 0.472 | 0.370 | 97.7 |
| SSDI (S21 + V2-L r0 scale) | — | 82.7 | 0.130 | 0.597 | 0.436 | 93.0 |
| S16-prox (B2) | 15.5 | 70.7 | 0.176 | 0.783 | 0.619 | 92.8 |
| S21-scale2 (B2) | 15.5 | 68.9 | 0.184 | 0.791 | 0.627 | 93.0 |
| S10-ultra-L (B2) | 15.5 | 65.6 | 0.184 | 0.843 | 0.667 | 92.7 |
| b1_mix (B1, old) | 5.1 | 60.4 | 0.210 | 0.881 | 0.710 | 91.4 |
| **b10** (B1, last) | 5.12 | 66.8 | 0.209 | 0.887 | 0.709 | 91.4 |
| b10 (B1, best own-val) | 5.12 | 65.6 | 0.207 | 0.894 | 0.714 | 91.4 |
| **b20core** (dsh+cstm+fov+ownar+trim) | 5.18 | 74.3 | 0.179 | 0.713 | 0.549 | 91.4 |
| **b20dws** (b20core + dws4, -41% MACs, last) | 5.04 | 74.8 | 0.179 | 0.735 | 0.560 | 91.2 |
| b20dws (best gt-val) | 5.04 | 74.8 | 0.179 | 0.735 | 0.560 | 91.2 |
| b20 (+rep4, unstable run, last) | 5.04 | 62.8 | 0.244 | 1.032 | 0.763 | 85.0 |
| b20 (+rep4, best gt-val) | 5.04 | 69.8 | 0.223 | 0.858 | 0.661 | 89.1 |

## NYU-654

| Model | Params (M) | δ1 ↑ | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 scale-aligned ↑ |
|---|---|---|---|---|---|---|
| UniDepthV2-L (teacher) | 354.0 | 94.3 | 0.086 | 0.357 | 0.235 | 95.9 |
| UniDepthV2-B | 97.0 | 92.2 | 0.102 | 0.393 | 0.272 | 95.6 |
| UniDepthV2-S | 34.0 | 87.4 | 0.131 | 0.449 | 0.338 | 94.9 |
| SSDI (S21 + V2-L r0 scale) | — | 90.1 | 0.102 | 0.394 | 0.274 | 94.3 |
| S16-prox (B2) | 15.5 | 88.4 | 0.108 | 0.415 | 0.295 | 94.2 |
| S21-scale2 (B2) | 15.5 | 88.8 | 0.108 | 0.412 | 0.292 | 94.3 |
| S10-ultra-L (B2) | 15.5 | 87.8 | 0.110 | 0.430 | 0.307 | 93.8 |
| b1_mix (B1, old) | 5.1 | 86.1 | 0.119 | 0.435 | 0.317 | 92.6 |
| **b10** (B1, last) | 5.12 | 86.7 | 0.116 | 0.446 | 0.322 | 93.0 |
| b10 (B1, best own-val) | 5.12 | 86.4 | 0.117 | 0.452 | 0.326 | 92.7 |
| **b20core** (dsh+cstm+fov+ownar+trim) | 5.18 | 85.6 | 0.120 | 0.448 | 0.323 | 92.7 |
| **b20dws** (b20core + dws4, -41% MACs, last) | 5.04 | 85.4 | 0.122 | 0.449 | 0.325 | 92.8 |
| b20dws (best gt-val) | 5.04 | 85.4 | 0.122 | 0.449 | 0.325 | 92.8 |
| b20 (+rep4, unstable run, last) | 5.04 | 69.3 | 0.198 | 0.642 | 0.493 | 88.3 |
| b20 (+rep4, best gt-val) | 5.04 | 82.6 | 0.136 | 0.489 | 0.359 | 90.5 |

## SUN-RGBD raw

| Model | Params (M) | δ1 ↑ | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 scale-aligned ↑ |
|---|---|---|---|---|---|---|
| UniDepthV2-L (teacher) | 354.0 | 95.7 | 0.078 | 0.242 | — | 98.1 |
| UniDepthV2-B | 97.0 | 92.5 | 0.097 | 0.281 | — | 97.8 |
| UniDepthV2-S | 34.0 | 89.5 | 0.112 | 0.316 | — | 97.2 |
| SSDI (S21 + V2-L r0 scale) | — | 89.8 | 0.110 | 0.328 | — | 94.6 |
| S16-prox (B2) | 15.5 | 86.0 | 0.125 | 0.363 | — | 94.6 |
| S21-scale2 (B2) | 15.5 | 86.1 | 0.126 | 0.361 | — | 94.6 |
| S10-ultra-L (B2) | 15.5 | 85.5 | 0.125 | 0.369 | — | 94.4 |
| b1_mix (B1, old) | 5.1 | 81.9 | 0.153 | 0.409 | — | 93.0 |
| **b10** (B1, last) | 5.12 | 83.5 | 0.134 | 0.395 | — | 93.6 |
| **b20core** (dsh+cstm+fov+ownar+trim) | 5.18 | 82.7 | 0.140 | 0.405 | — | 93.1 |
| **b20dws** (b20core + dws4, -41% MACs, last) | 5.04 | 83.0 | 0.138 | 0.400 | 0.298 | 93.1 |
| b20 (+rep4, unstable run, last) | 5.04 | 68.3 | 0.205 | 0.557 | — | 89.3 |

## OCMR-SUN object distance (57,677 objects, box5)

| Model | objAbsRel ↓ | obj δ1 ↑ | MAE (m) ↓ | RMSE (m) ↓ | bias_log |
|---|---|---|---|---|---|
| UniDepthV2-L (teacher) | 0.105 | 91.3 | 0.254 | 0.491 | +0.024 |
| UniDepthV2-B | 0.121 | 87.9 | 0.291 | 0.519 | +0.026 |
| UniDepthV2-S | 0.134 | 84.8 | 0.325 | 0.547 | +0.026 |
| SSDI (S21 + V2-L r0 scale) | 0.135 | 84.2 | 0.334 | 0.537 | +0.030 |
| S16-prox (B2) | 0.146 | 80.8 | 0.362 | 0.571 | -0.001 |
| S21-scale2 (B2) | 0.147 | 81.0 | 0.362 | 0.571 | +0.002 |
| S10-ultra-L (B2) | 0.146 | 80.3 | 0.367 | 0.573 | -0.007 |
| b1_mix (B1, old) | 0.173 | 77.0 | 0.405 | 0.629 | +0.011 |
| **b10** (B1, last) | 0.155 | 78.1 | 0.392 | 0.614 | -0.012 |
| **b20core** (dsh+cstm+fov+ownar+trim) | 0.162 | 77.3 | 0.404 | 0.639 | +0.001 |
| **b20dws** (b20core + dws4, -41% MACs, last) | 0.160 | 77.6 | 0.399 | 0.626 | +0.002 |
| b20 (+rep4, unstable run, last) | 0.219 | 62.7 | 0.553 | 0.827 | -0.008 |

## Speed (RTX 5070 Ti laptop, fp32, 640x480, batch 1)

| Model | Params (M) | ms ↓ | FPS ↑ | peak VRAM (GiB) |
|---|---|---|---|---|
| S16-prox (B2) | 15.48 | 19.0 | 52.6 | 0.2 |
| b1_mix (B1, old) | 5.12 | 15.1 | 66.4 | 0.09 |
| **b10** (B1, last) | 5.12 | 12.5 | 80.0 | 0.1 |
| **b20core** (dsh+cstm+fov+ownar+trim) | 5.18 | 10.4 | 95.8 | 0.1 |
| **b20dws** (b20core + dws4, -41% MACs, last) | 5.04 | 10.6 | 94.3 | 0.1 |
| b20 (+rep4, unstable run, last) | 5.04 | 10.9 | 91.7 | 0.1 |

## CPU latency (ONNX Runtime 1.20, fp32, 640x480, batch 1, this laptop's CPU; median of 60 runs)

| Model | GMAC | 4 threads (ms) ↓ | 1 thread (ms) ↓ |
|---|---|---|---|
| b1_mix (old) | 6.84 | 43.1 | 109.9 |
| **b10** | 6.84 | 43.4 | 113.0 |
| b20core | 6.84 | 43.1 | 114.7 |
| **b20dws** | **4.04** | **33.8 (−22%)** | **76.5 (−32%)** |

Batch-1 GPU timings (above) are launch-bound for 5M models and do not reflect the MAC reduction; CPU (and
Core ML / ANE, to be measured on the Mac) are compute-bound. ONNX export parity max|Δlog D| ≤ 2.3e-5.

## Paired bootstrap, b20dws vs b10 (last.pt vs last.pt, 10k resamples)

| Dataset | δ1 | AbsRel | RMSE | MAE |
|---|---|---|---|---|
| IBims-1 (unseen) | **+8.0 [+3.1, +12.9]** | −.030 [−.077, +.001] (n.s.) | **−0.153 m [−.250, −.072]** | **−0.149 m [−.245, −.068]** |
| NYU-654 | −1.3 [−2.0, −0.5] | +.006 [+.003, +.008] | n.s. | n.s. |
| SUN raw (partly in-domain) | −0.4 [−0.8, −0.0] | +.004 [+.002, +.006] | +.006 m [+.002, +.009] | — |

## Verdict

- **b20dws** (CanonScale + depthwise-separable stride-4 decoder): 5.04 M params, 4.04 GMAC (−41%), −22…−32% CPU
  latency, **+8.0 δ1 / −15 cm MAE on unseen IBims**, at a cost of −1.3 δ1 on NYU and −0.4 on SUN (MAE/RMSE on NYU
  unchanged). Beats the 3x larger S16 and all MoGe-2 sizes on IBims δ1. Recommended general-purpose / new-environment model.
- **b10** (S16 recipe on B1): best 5 M model in-domain (NYU/SUN, OCMR object distance .155); +6.4 IBims δ1 over the old b1_mix.
- **b20 with multi-branch rep4** trained unstably (scale-head loss diverged after ~epoch 10) and is not recommended;
  `dws4` replaces it with identical deploy cost.
- Not yet measured: Core ML / Neural Engine latency and fp16 parity (run `edge/export_coreml.py` on the Mac).
