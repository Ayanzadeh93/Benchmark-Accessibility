---
tags: [depth-estimation, knowledge-distillation, unidepth]
---
# depthv2 — UniDepthV2 distilled student checkpoints

Students distilled from UniDepthV2 (ViT-L/B/S teachers). Each folder has `best.pt`
(lowest val AbsRel), `last.pt`, and `run_info.json` (full training config).

Checkpoint format: `torch.load(path, weights_only=False)` -> dict with
`model` (state_dict), `cfg`, `epoch`, `val_absrel`. Load with `distill/` code from the
unidepth2d repo.

Full b10/b20 comparison vs UniDepthV2, MoGe-2, UniK3D: see `B20_RESULTS.md`.

| Run | Best epoch | Val AbsRel | Size | Note |
|---|---|---|---|---|
| `b10` | 26 | 0.1493 | 20 MB | B1 5.1M, S16 recipe (s4,s5,ray,range,prox). Best 5M in-domain. IBims d1 66.8, NYU 86.7. ONNX in b10/fp32.onnx. |
| `b1_mix` | 34 | 0.0921 | 20 MB |  |
| `b20` | 11 | 0.1381 | 19 MB | b20core + multi-branch rep4. Trained UNSTABLY (scale loss diverged) - do not use; superseded by b20dws. |
| `b20core` | 32 | 0.1099 | 20 MB | b10 + dsh,cstm,fov,ownar,trim (CanonScale). IBims d1 74.3, NYU 85.6. Same MACs as b10. |
| `b20dws` | 39 | 0.1123 | 19 MB | RECOMMENDED. b20core + dws4 depthwise-separable stride-4 decoder: 5.04M, 4.04 GMAC (-41%), IBims d1 74.8 / MAE 0.56 m, NYU 85.4, CPU -22..-32% latency. ONNX in b20dws/fp32.onnx. |
| `b2_mix` | 38 | 0.0789 | 59 MB |  |
| `intrinsics` | 28 | 0.1397 | 20 MB |  |
| `pilot` | 1 | 0.2389 | 20 MB |  |
| `pilot_s4` | 1 | 0.2519 | 20 MB |  |
| `pilot_s5` | 1 | 0.2561 | 20 MB |  |
| `pilot_s6` | 1 | 0.3346 | 20 MB |  |
| `pilot_s7` | 1 | 0.2659 | 20 MB |  |
| `pilot_s8` | 1 | 0.2861 | 20 MB |  |
| `S10_ultra_L` | 38 | 0.0900 | 59 MB |  |
| `S11_ray` | 39 | 0.0947 | 20 MB |  |
| `S12_bins` | 39 | 0.0920 | 20 MB |  |
| `S13_norm` | 39 | 0.1013 | 20 MB |  |
| `S14_range` | 39 | 0.0950 | 20 MB |  |
| `S15_anch_icd` | 0 | 0.0927 | 59 MB |  |
| `S15_scratch` | 33 | 0.0927 | 59 MB |  |
| `S16_prox` | 39 | 0.0876 | 59 MB | Main student used for ONNX/int8 edge export and OCMR conformal results. |
| `S18_zsclean` | 39 | 0.0940 | 59 MB |  |
| `S19_sced` | 10 | 0.0879 | 59 MB |  |
| `S21_scale2` | 14 | 0.0862 | 59 MB |  |
| `S21_scale2_full` | 7 | 0.1218 | 59 MB | INCOMPLETE (died at epoch 5 of 40) — early checkpoint, do not use. |
| `S4_edge` | 34 | 0.0942 | 20 MB |  |
| `S5_crop` | 39 | 0.1014 | 20 MB |  |
| `S5_crop_TAINTED_concurrent_run` | 38 | 0.1032 | 20 MB |  |
| `S6_unc` | 38 | 0.0982 | 20 MB |  |
| `S6_unc_TAINTED_concurrent_run` | 37 | 0.1006 | 20 MB |  |
| `S7_task` | 39 | 0.1105 | 20 MB |  |
| `S7_task_TAINTED_concurrent_run` | 35 | 0.1087 | 20 MB |  |
| `S8_assist` | 38 | 0.1994 | 20 MB |  |
| `S8_assist_TAINTED_concurrent_run` | 34 | 0.1996 | 20 MB |  |
| `S9_ultra` | 31 | 0.1054 | 20 MB |  |
