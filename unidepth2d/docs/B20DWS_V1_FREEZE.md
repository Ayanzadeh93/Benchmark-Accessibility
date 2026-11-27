# b20dws v1 — frozen reference (in-domain NYU/SUN version)

Branch: `b20dws-v1-indomain`. This branch is the **first trained version of b20dws** and must not
change. New work (zero-shot-clean training, recipe changes) goes on `b20dws-v2-improved`.

## What v1 is

- 5.04M-parameter metric depth student, EfficientViT-B1 backbone, distilled from UniDepthV2-L.
- Trained 2026-11-23 (40 epochs × 3000 batches, ~3 GPU-hours), train=27820 / val=861.
- Command (from `scripts/run_b20dws.ps1`):
  `python -m distill.train --config config.yaml --data gt --use-intrinsics --backbone efficientvit_b1.r224_in1k --recipe s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt --ckpt-subdir b20dws --name b20dws`
- Training mix: own 0.4 (≈7.5k indoor-video frames, UniDepthV2-L pseudo-labels), NYU-v2 0.4, SUN RGB-D 0.2.
- No random seed is set in `distill/train.py`.

## Which results are in-domain

| Set | Status for v1 |
|---|---|
| IBims-1, DIODE val/indoors, ETH3D indoor | zero-shot |
| NYU-v2 (654 filled) | **in-domain** (NYU train in the mix; 50 NYU-test ids also reached training via own rows; use the clean 602 split, `scripts/nyu_clean_rescore.py`) |
| SUN RGB-D raw | **in-domain** (SUN in the mix; SUN K is teacher-estimated) |

## Headline numbers (last.pt, GT intrinsics, our harness)

δ1 / d1_si in %, RMSE in m. Full tables: `docs/ALL_RESULTS.md`, per-run JSONs in `results_bench/`.

| Set | δ1 | AbsRel | RMSE | d1_si |
|---|---|---|---|---|
| IBims-1 | 74.8 | 0.179 | 0.735 | 86.5 |
| NYU-v2 filled (in-domain) | 85.4 | 0.122 | 0.449 | 89.6 |
| SUN RGB-D raw (in-domain) | 83.0 | 0.138 | 0.400 | 90.5 |
| DIODE indoors | 36.9 | 0.330 | 1.755 | 83.9 |
| ETH3D indoor | 56.6 | 0.234 | 1.359 | 88.2 |

Latency at 640×480: 14.9 ms GPU; CPU ONNX 78.2 / 33.4 / 23.1 ms at 1 / 4 / 8 threads.

Run-to-run noise (rep1 = same recipe retrained, 2026-11-27): raw δ1 moved +1.4 (IBims),
+1.3 (NYU), −4.4 (DIODE), +2.3 (ETH3D); d1_si moved ≤0.7. Report v1 as mean ± std over
b20dws / rep1 / rep2 once the ablation queue (`scripts/run_ablation_b20.sh`) finishes.

## Checkpoints (not in git: `outputs/` is ignored)

Read-only copy: `outputs/checkpoints/FROZEN_b20dws_v1/` (best.pt, last.pt, train_summary.json).
Published rows use **last.pt**. SHA-256:

| File | SHA-256 |
|---|---|
| b20dws/last.pt (= b20dws_last/best.pt, FROZEN last.pt) | `d5e9c7b6f2e4687037490c541cd0e11f8637f37df58ab0937b8b80b6dd9bfbf6` |
| b20dws/best.pt (= b20dws_best/best.pt, FROZEN best.pt) | `982448bd360638ce57218346e2660a8773bc60934ff7cb29c83b81c02cb63711` |
| b20dws_ft2_all/last.pt | `f07f89635fb1973346ab75da108babae45987be5b664a444197b8e8c6a48c067` |
| b20dws_ft2_ctrl/last.pt | `fae58ee1d5823a3a672a9391c1b0d73efa45c1d524334815cc8c6f76fcfb4e80` |
| b20dws_ft2_rest/last.pt | `a7cef0e045b6e0436b6a3384b2da7c9339012c40f6e937cf5d1dfb441c1bfbc8` |
| b20dws_ft_ctrl/last.pt | `de57fe35f50e51593f2b42d8bf20d76582f91d1cd791a52a5ad26570fef28477` |
| b20dws_ft_hyper/last.pt | `956fb8fc057e21d267914b31dde69a14ab63e8dc4b67817f3e0cf1c959ebae9e` |
| b20core/last.pt | `a07c1ec6dbe5e408021a879753c073a87e36c68bbe97aac6da659285f9159410` |
| b10/last.pt | `b2dab00ec23e947ac693ae88ad1d8c500eeef2948557b3fa7d8a41bb35f29622` |

Verify: `sha256sum outputs/checkpoints/FROZEN_b20dws_v1/*.pt`.

## Related results kept with v1

`results_bench/` (all baselines + students), `results_finetune/` (ARKit/Hypersim fine-tunes —
flagged, not adopted: they lower raw DIODE/ETH3D δ1), `results_tta/`, `results_bf1v2/`
(boundary F1 v2), `results_rescore/` (K=None / A05 rescore), docs `B20_*.md`, `NYU_CLEAN_RESCORE.md`,
`ALL_RESULTS.md`. Ablation results (`results_ablation/`) are added when the queue finishes.
