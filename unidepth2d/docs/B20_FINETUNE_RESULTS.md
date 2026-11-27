# b20dws fine-tune comparison

Same protocol for every row (IBims default K policy, NYU filled, SUN raw).

| Model | IBims d1 ^ | IBims RMSE v | IBims AbsRel v | IBims bF1 ^ | NYU d1 ^ | NYU RMSE v | SUN d1 ^ | SUN RMSE v |
|---|---|---|---|---|---|---|---|---|
| Original b20dws | 0.7484 | 0.7347 | 0.1792 | 0.0318 | 0.8540 | 0.4486 | 0.8302 | 0.4004 |
| Control ft (lr 2e-5, orig data) | 0.7448 | 0.7379 | 0.1807 | 0.0321 | 0.8540 | 0.4504 | 0.8308 | 0.4004 |
| All-data ft (lr 2e-5, +ARKit +Hypersim) | 0.7517 | 0.7062 | 0.1728 | 0.0342 | 0.8495 | 0.4601 | 0.8271 | 0.4022 |
| [dropped] ctrl ft lr 1e-4 | 0.7391 | 0.7492 | 0.1803 | 0.0317 | 0.8501 | 0.4610 | 0.8297 | 0.4045 |
| [dropped] Hypersim-only ft lr 1e-4 | 0.7379 | 0.7692 | 0.1870 | 0.0386 | 0.8414 | 0.4717 | 0.8234 | 0.4071 |

## Verdict (all runs: init = original b20dws, lr 2e-5, 10 epochs; control = original data only)

Paired bootstrap over the 100 IBims images (95% CI of the per-image gain, positive = better):

| Comparison | RMSE (m) | AbsRel | d1 |
|---|---|---|---|
| Control vs original | -0.003 [-0.013, +0.006] (no change) | -0.0015 [-0.005, +0.002] | -0.004 [-0.013, +0.005] |
| All-data vs original | +0.029 [+0.005, +0.054] | +0.006 [-0.004, +0.019] | +0.003 [-0.021, +0.027] |
| All-data vs control | +0.032 [+0.007, +0.057] (64/100 images better) | +0.008 [-0.002, +0.019] | +0.007 [-0.016, +0.029] |

* Fine-tuning alone does nothing: the control run is statistically identical to the original on IBims, NYU and SUN.
* Adding ARKitScenes + Hypersim lowers IBims (zero-shot) RMSE by ~4% (0.735 -> 0.706); only RMSE is
  significant, AbsRel and d1 point the same way but their CIs include zero.
* FLAG: the all-data model is slightly worse on in-domain sets (NYU d1 0.854 -> 0.850, RMSE 0.449 -> 0.460;
  SUN d1 0.830 -> 0.827). Likely cause (untested): the NYU/SUN share of the mix drops from 0.4/0.2 to 0.26/0.13.
* Dropped: lr 1e-4 runs (control and Hypersim-only) - worse everywhere.

## Post-hoc inference options (no retraining) - results_tta/, scripts/run_tta_eval.sh

Options: flip (mirror, cx -> 1-cx, log-mean), ms (0.8x/1x/1.2x of 480x640, log-mean), sz576x768 (single larger input).
Columns: raw = metric, no GT used; si = median-ratio scale (GT used); lsl = log-least-squares scale (GT used,
the alignment Ultralytics states for YOLO26-Depth). Aligned columns are diagnostics, never metric accuracy.

| Variant | IBims raw d1 | IBims RMSE | IBims si d1 | IBims lsl d1 | NYU raw d1 | NYU RMSE | NYU si d1 | NYU lsl d1 |
|---|---|---|---|---|---|---|---|---|
| b20dws | 0.748 | 0.735 | 0.865 | 0.879 | 0.854 | 0.449 | 0.896 | 0.907 |
| b20dws +flip | 0.756 | 0.731 | 0.868 | 0.881 | 0.856 | 0.446 | 0.898 | 0.909 |
| b20dws +ms | 0.751 | 0.739 | 0.864 | 0.878 | 0.853 | 0.453 | 0.896 | 0.906 |
| b20dws +ms+flip | 0.751 | 0.737 | 0.866 | 0.881 | 0.854 | 0.452 | 0.897 | 0.908 |
| b20dws 576x768 | 0.697 | 0.741 | 0.862 | 0.874 | 0.776 | 0.505 | 0.885 | 0.896 |
| all-data | 0.752 | 0.706 | 0.867 | 0.880 | 0.850 | 0.460 | 0.893 | 0.904 |
| all-data +flip | 0.756 | 0.701 | 0.869 | 0.882 | 0.852 | 0.457 | 0.895 | 0.906 |
| all-data +ms+flip | 0.759 | 0.709 | 0.868 | 0.882 | 0.849 | 0.465 | 0.894 | 0.905 |
| all-data 576x768 | 0.709 | 0.717 | 0.864 | 0.875 | 0.793 | 0.489 | 0.879 | 0.891 |

YOLO26n-depth (published, TTA + log-LS): IBims 0.923, NYU 0.882; NYU median-scale, no TTA: 0.783.
Flip = small consistent gain; multi-scale = none; larger input at inference hurts (scale drift; says nothing
about training at that size). NYU is partly in-domain for b20dws and zero-shot for YOLO26n.

## UniK3D / MoGe-2 on NYU-654 (filled) and SUN-raw (n=4396), same protocol (+r9), scripts/run_baselines_nyu_sun.sh

| Model | NYU d1 | NYU AbsRel | NYU RMSE | NYU si d1 | SUN d1 | SUN AbsRel | SUN RMSE | SUN si d1 |
|---|---|---|---|---|---|---|---|---|
| UniK3D-S | 0.885 | 0.123 | 0.438 | 0.944 | 0.897 | 0.112 | 0.317 | 0.964 |
| UniK3D-B | 0.914 | 0.107 | 0.410 | 0.957 | 0.928 | 0.097 | 0.285 | 0.974 |
| MoGe-2 S | 0.850 | 0.123 | 0.470 | 0.942 | 0.800 | 0.136 | 0.381 | 0.963 |
| MoGe-2 B | 0.900 | 0.109 | 0.428 | 0.956 | 0.833 | 0.125 | 0.351 | 0.972 |
| MoGe-2 L | 0.941 | 0.087 | 0.366 | 0.963 | 0.928 | 0.094 | 0.281 | 0.978 |
| b20dws (5.04M) | 0.854 | 0.122 | 0.449 | 0.896 | 0.830 | 0.138 | 0.400 | 0.905 |
| b20dws all-data | 0.850 | 0.123 | 0.460 | 0.893 | 0.827 | 0.138 | 0.402 | 0.905 |

NYU and SUN are partly in-domain for b20dws; MoGe-2 training lineage is unverified (docs/V2_PLAN.md 9.1).
