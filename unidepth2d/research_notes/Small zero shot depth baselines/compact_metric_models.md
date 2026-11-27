# Compact (about 2–15M params) zero-shot / generalizable METRIC monocular depth models for indoor use

Scope note: researched 2026-11-27. Relative-only distillations and NYU-only supervised nets are out of scope and appear only as context or as baselines inside the papers covered. Shorthand used below: "raw metric" = no per-image scale/shift fitting; "aligned" = per-image median/LSQ scale (and/or shift) fitting before metrics; "ZS" = dataset not in training; "ID" = in-domain (trained or fine-tuned on that dataset).

## Q1. Which compact metric-depth models exist, and what are their exact specs?

### Takeaway
Only three families have checkpoints that are actually in the 2–15M range and produce metric output. (1) **DepthART-S/B** (6.0M / 11.4M encoder+decoder; the camera-aware metric-S is 8.0M) is the only one with a proper raw-metric zero-shot indoor table (iBims-1, SUN RGB-D, DIODE-indoor), but its metric head is fine-tuned on NYU and needs intrinsics. (2) **YOLO26n/s-depth** (6.3M / 13.2M) is trained on a 2.19M-image mix. Its zero-shot numbers are only published *after log-LSQ scale alignment + TTA*, so they are not raw-metric numbers. (3) **LightlyTrain DAv3-metric tiny/tiny-plus** (6.2M / 7.9M, DINOv3 ViT-T) needs intrinsics and reports NYU only, raw and aligned. Every "small" variant of the foundation metric models lies above 15M and counts only as a near-miss: UniDepthV2-S 34.18M, UniK3D-S 35.4M, Metric3Dv2-S 37.5M, DAv2-metric-Small 24.8M, LightlyTrain DAv3-metric-small 24.7M, X-Lens about 40M.

### Cited Findings

**DepthART: "Scaling Foundation Monocular Depth to Tiny Models" (arXiv 2607.17099, Jul 2026, accepted ACM Multimedia 2026; Feng Xue et al.)**
- Variants and params: DepthART-S 6.0M (TinyViM-S), DepthART-B 11.4M (TinyViM-B), DepthART-L 32.6M (TinyViM-L). Architecture is a TinyViM (Mamba-style) encoder plus a DPT decoder. — [arXiv HTML](https://arxiv.org/html/2607.17099)
- The project page gives **8.0M params for Metric-S** (camera adapters + scale head added) against 6.0M for Relative-S. — [Project page](https://xuefeng-cvr.github.io/DepthART)
- Teacher is Depth Anything v2-Large. Stage 1 distils on a 1.7M "bias-resistant" subset (BRDS) of a 44M corpus drawn from 11 sources: ApolloScapeExtra, BDD100K, Cityscapes, DIML-indoor, GoogleLandmark, Holopix50K, HRWSI, ImageNet21K, LSUN, Objects365, ReDWeb. — [arXiv HTML](https://arxiv.org/html/2607.17099)
- Metric stage (CamFT) freezes the distilled encoder. It trains camera adapters, a multi-query scale head and the DPT decoder on **one metric dataset (NYUD v2 or KITTI)** with d_max = 10 m (NYU) or 80 m. — [arXiv HTML](https://arxiv.org/html/2607.17099)
- The model **requires known intrinsics** (fx, fy, cx, cy). They are turned into a per-pixel ray field, Fourier-encoded, and injected by cross-attention adapters at strides 4/8/16/32. Output is d_metric = d_max · (s ⊙ d_rel). The authors state that CamFT assumes known intrinsics. — [arXiv HTML](https://arxiv.org/html/2607.17099)
- FLOPs and latency, RTX A6000, FP32, batch size 1:

  | Model | 224² FLOPs | 224² model-only | 448² FLOPs | 448² model-only |
  |---|---|---|---|---|
  | S | 1.749 G | 2.88 ms | 6.996 G | 4.09 ms |
  | B | 2.348 G | 3.36 ms | 9.394 G | 4.91 ms |
  | L | 6.178 G | 5.23 ms | 24.711 G | 8.07 ms |

  For reference, DAv2-S at 518² takes 13.06 ms and 41.3 GFLOPs. — [arXiv HTML](https://arxiv.org/html/2607.17099)
- Jetson Orin NX, FP32, model-only latency: S 9.80 ms at 224² and 29.48 ms at 448²; B 11.59 ms and 35.35 ms. The project page reports S at 224 with TensorRT FP16 at 0.92 ms on an A6000 and 247.8 FPS on Orin NX. — [arXiv HTML](https://arxiv.org/html/2607.17099); [Project page](https://xuefeng-cvr.github.io/DepthART)
- Code is at github.com/xuefeng-cvr/DepthART and weights at huggingface.co/Fengxue93/DepthART. The HF repo is **Apache-2.0** and has `metric/`, `relative/` and `onnx/` folders; the visible files are MobileNetV4-S relative checkpoints. A September 2026 update added MobileNetV4-S/M/M-slim-SPF encoders. The project page names a "DepthART-Metric-L (fine-tuned on NYUD v2 only)". — [HF repo](https://huggingface.co/Fengxue93/DepthART/tree/main); [Project page](https://xuefeng-cvr.github.io/DepthART)
- A Chinese write-up says the repo offers relative, camera-aware metric, ONNX and TensorRT paths. This is secondary. — [Sina](https://finance.sina.cn/stock/jdts/2026-10-07/detail-iniuksym4764240.d.html)
- Do not confuse it with arXiv 2409.15010 (IJCAI 2025), an unrelated autoregressive "DepthART". — [arXiv 2409.15010](https://arxiv.org/pdf/2409.15010)

**Ultralytics YOLO26-Depth (docs/blog, 2026; no paper)**
- Model specs at 768 px:

  | Model | Params | FLOPs | CPU ONNX | T4 TensorRT fp16 |
  |---|---|---|---|---|
  | YOLO26n-depth | **6.3M** | 46.9 B | 272.0 ms | 2.7 ms |
  | YOLO26s-depth | **13.2M** | 68.0 B | 393.7 ms | 3.8 ms |
  | YOLO26m-depth | 23.3M | 130.4 B | n/a | n/a |
  | YOLO26l-depth | 27.7M | 157.0 B | n/a | n/a |
  | YOLO26x-depth | 57.0M | 301.7 B | n/a | n/a |

  The comparison point is DAv2-Small at 24.8M and 346.9 B FLOPs, 20.99 ms on T4. — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth)
- Training data is a multi-dataset indoor and outdoor mix of about 2.19M images. The datasets are not named on the page, except that KITTI Eigen training drives are in the mix. — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth); [iBims-1 dataset page](https://docs.ultralytics.com/datasets/depth/ibims-1)
- Output: the head predicts unbounded depth of about 0.02–150 m and separates shape from absolute scale. Meters come from a two-parameter transform. The released checkpoints carry **one global scale calibration** fit on the pretraining validation mix. **No intrinsics input**. DA3METRIC-LARGE was used to label some data, using a focal length. — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth)
- The licence is not stated on the depth pages. Ultralytics is generally AGPL-3.0, but I did not verify that here. Weights are `yolo26n-depth.pt` and so on. — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth)
- A tutorial calls it "metric monocular depth". This is secondary. — [LearnOpenCV](https://learnopencv.com/yolo26-depth-estimation-metric-monocular-depth-and-what-you-can-build-with-it/)

**LightlyTrain distilled Depth Anything V3 (LightlyTrain 0.16–0.17, 2026)**
- Metric checkpoints:

  | Checkpoint | Backbone | Params |
  |---|---|---|
  | `dinov3/dav3-metric-tiny` | DINOv3 ViT-T/16 | **6.2M** |
  | `dinov3/dav3-metric-tiny-plus` | DINOv3 ViT-T+/16 | **7.9M** |
  | `dinov2/dav3-metric-small` | ViT-S/14 | 24.7M |
  | `dinov2/dav3-metric-large` | ViT-L/14 | 334.2M |

  The large checkpoint is an official DAv3 port, not a distillation. Tiny, tiny-plus and small are distilled from the large model. Relative tiny/tiny-plus have the same sizes; relative-tiny runs at 5.27 ms on T4 FP16 TensorRT at 576². — [Lightly docs](https://docs.lightly.ai/train/stable/depth_estimation.html)
- Metric DAv3 models **require camera intrinsics**; an FOV-based approximation is acceptable. The weights licence is not stated. Load with `lightly_train.load_model(...)`, which downloads the weights automatically. — [Lightly docs](https://docs.lightly.ai/train/stable/depth_estimation.html); [LightlyTrain 0.17.0 release](https://lightly.ai/product-updates/lightlytrain-0-17-0-ltdetrv2-for-instance-segmentation-and-lingbot-vision-backbone-support)
- Training data for the distillation is not stated.
- The same docs list DAv2 `dav2-metric-small-hypersim` (indoor, ≤20 m, Apache-2.0, no intrinsics needed). — [Lightly docs](https://docs.lightly.ai/train/stable/depth_estimation.html)

**Near-miss "small" variants of foundation metric models (above 15M)**
- **UniDepthV2-Small** (arXiv 2502.20110), ViT-S, DINOv2: **34.18M params**, 23.0 ms, 0.29 TFLOPs, 0.66 GiB (Table VIII). It is trained on 23 datasets (about 16M images) that include ScanNet, ScanNet++, ARKitScenes, Taskonomy, HM3D and Matterport3D. NYU and KITTI are not in training. — [UniDepthV2 arXiv](https://arxiv.org/html/2502.20110v2)
  - Intrinsics are optional: the model predicts the camera or accepts GT K. Weights are `lpiccinelli/unidepth-v2-vits14`, **CC BY-NC 4.0**. — [UniDepth GitHub](https://github.com/lpiccinelli-eth/UniDepth)
- **UniK3D-Small** (CVPR 2025, arXiv 2503.16591), ViT-S: the HF safetensors show **35.4M params** (`lpiccinelli/unik3d-vits`). Intrinsics or rays are optional, and the licence is **CC BY-NC 4.0**. — [HF unik3d-vits](https://huggingface.co/lpiccinelli/unik3d-vits); [UniK3D GitHub](https://github.com/lpiccinelli-eth/UniK3D)
- **Metric3Dv2 ViT-S** (v2-S, DINOv2-reg ViT-S, RAFT 4 iterations): **37.5M params** per DepthART Table 1. X-Lens lists it at 0.03B.
  - Code is BSD-2-Clause, and ONNX for v2-S is on HF. Intrinsics are optional: without them it falls back to nine default focal settings.
  - Sources: [DepthART](https://arxiv.org/html/2607.17099); [X-Lens](https://arxiv.org/html/2607.12993); [Metric3D GitHub](https://github.com/YvanYin/Metric3D)
- **Depth Anything V2 metric Small**: 24.8M params (as cited by Ultralytics, DepthART and ZipDepth; the paper says "~25M"). The paper's in-domain metric fine-tune of ViT-S gives NYU δ1 0.961, AbsRel 0.073, RMSE 0.261 and KITTI δ1 0.973, AbsRel 0.053, RMSE 2.235 (Table 4, ID). The paper reports no zero-shot metric numbers for ViT-S. — [DAv2 arXiv](https://arxiv.org/html/2406.09414v2); [Ultralytics docs](https://docs.ultralytics.com/tasks/depth)
- **X-Lens** (arXiv 2607.12993, 2026) has about 0.04B params, a compact DINOv2 backbone, and a DPT head plus a scale MLP.
  - It **requires GT intrinsics and extrinsics**.
  - It evaluates only on KITTI360, OmniScene, two-view ETH3D/ScanNet++ and OmniOcc, with no standard indoor single-image benchmarks. It runs at 41 FPS on an H100.
  - Code is at github.com/zhouhengamerica/XLens. The weights licence is not stated. — [X-Lens arXiv](https://arxiv.org/html/2607.12993)

**Out of scope but surfaced (context only)**
- **ZipDepth** (arXiv 2607.08771, 2026) has 6.1M params at inference and 3.0 GMACs at 384². It is **relative** (affine-invariant inverse depth) and distilled from DAv2-L, so it is not metric. — [ZipDepth arXiv](https://arxiv.org/html/2607.08771v1)
- **"The Midas Touch for Metric Depth"** (CVPR 2026, arXiv 2605.11578) distils DAv2 into EfficientViT-B0/B1 and TinyViT students. It recovers metric scale from **sparse 3D seeds**, so it is not pure monocular. The text gives no exact student param counts, only that RMSE "remains stable until the parameter count drops below ∼20M". — [MTD arXiv](https://arxiv.org/html/2605.11578v1)

### Inferences
- In the 5–10M priority band, the runnable candidates with metric output are DepthART-S (metric about 8.0M), YOLO26n-depth (6.3M), and LightlyTrain DAv3-metric-tiny/tiny-plus (6.2M / 7.9M). YOLO26s-depth (13.2M) and DepthART-B (11.4M) fill the 10–15M band.
- DepthART's metric head is NYU-fine-tuned. Treat it as "generalizable metric with one indoor fine-tune", not as zero-shot on NYU. On iBims, SUN and DIODE it is zero-shot.
- YOLO26 ships one global scale and no camera conditioning. Its raw-metric accuracy across cameras with different focal lengths is likely to be much worse than its aligned numbers suggest; this is untested.

### Gaps
- **Not found:** ZoeDepth small variants. No official ZoeDepth checkpoint below BEiT-L or Swin2 turned up.
- **Not found:** PatchFusion-lite.
- **Not found:** any MobileNet or EfficientViT metric student distilled from UniDepth, Metric3D or Depth Pro, other than DepthART's MobileNetV4 update. That update is DAv2-L-distilled, and its metric checkpoints are not documented.
- **Not resolved:** the exact contents of DepthART's HF `metric/` folder (NYU vs KITTI heads, S vs B vs L).
- **Not resolved:** the YOLO26 weights licence and the dataset list.
- **Not resolved:** LightlyTrain's training data and weights licence.
- **Not found:** Metric3Dv2 ViT-S zero-shot numbers. The Metric3D README reports only ViT-L and giant2.

## Q2. Which 2025–2026 papers cover "lightweight / efficient zero-shot metric depth" or "camera-aware lightweight depth"?

### Takeaway
The directly relevant 2026 items are DepthART (camera-aware tiny metric), YOLO26-Depth (vendor docs, not a paper), the LightlyTrain DAv3 distillations (vendor), X-Lens (about 40M, camera-aware), ZipDepth (relative only) and Midas Touch (3D-seed metric). TR2M (2025) is a further candidate that was not verified.

### Cited Findings
- **DepthART** (2607.17099) uses camera-conditioned fine-tuning on a frozen distilled encoder. The abstract reports a zero-shot affine δ1 of 0.964 for DepthART-S on NYU. — [arXiv abs](https://arxiv.org/abs/2607.17099)
- **X-Lens** (2607.12993) bills itself as a "lightweight 0.04B-parameter" real-time metric depth model for heterogeneous cameras. — [X-Lens](https://arxiv.org/html/2607.12993)
- **TR2M** (arXiv 2506.13387, 2025) transfers relative to metric depth with language descriptions using "a single lightweight network". Its param count, intrinsics use and benchmarks were not verified. — [TR2M](https://arxiv.org/pdf/2506.13387)
- **Zero-shot metric depth via visual-inertial rescaling** (arXiv 2509.08159, 2025) needs an IMU, so it is not monocular-only. It runs at 15 Hz on board an aerial platform. — [arXiv 2509.08159](https://arxiv.org/abs/2509.08159)
- **UniDAC** (arXiv 2603.27105) is a universal any-camera metric model at 0.36B params per X-Lens's table, so it is not compact. — [X-Lens](https://arxiv.org/html/2607.12993); [UniDAC](https://arxiv.org/pdf/2603.27105)
- **MD2E** (CVPR 2026) does metric depth without camera parameters using edge cues. It is not described as lightweight. — [MD2E](https://openaccess.thecvf.com/content/CVPR2026/papers/Ning_MD2E_Modeling_Depth-to-Edge_Cues_for_Monocular_Metric_Depth_Estimation_CVPR_2026_paper.pdf)

### Inferences
- The field appears to have two kinds of compact model. Camera-aware compact metric models (DepthART, LightlyTrain DAv3-metric) need K. The intrinsics-free compact model (YOLO26) uses a global scale and is evaluated aligned. No published 5–10M model yet matches UniDepthV2-S's intrinsics-free zero-shot raw-metric protocol on iBims and SUN.

### Gaps
- TR2M's size and numbers were not checked.
- No IROS, ICRA or WACV 2025–26 compact metric paper surfaced in the searches. Coverage of those venues is incomplete.

## Q3. Which report zero-shot indoor numbers (IBims-1, NYU, SUN RGB-D, DIODE, ETH3D), and what are they?

### Takeaway
DepthART is the only compact model with **raw metric** zero-shot indoor numbers. For its NYU-fine-tuned metric head, DepthART-S scores iBims δ1 0.713 / RMSE 0.693, SUN δ1 0.809 / RMSE 0.379, and DIODE-indoor δ1 0.292 / RMSE 1.687. YOLO26 reports only **aligned + TTA** δ1 (iBims: n 0.923, s 0.899). LightlyTrain reports only NYU (tiny: raw δ1 0.818, aligned 0.915). The near-miss UniDepthV2-S (34M) has raw-metric ZS numbers: SUN δ1 90.8, iBims δ1 86.6, ETH3D δ1 64.6.

### Cited Findings

**DepthART, Table 2: raw metric, no rescaling, δ1 / RMSE (m)** — [arXiv HTML](https://arxiv.org/html/2607.17099)

Metric head fine-tuned on NYU v2, so NYU is ID and the other three are ZS:

| Method | NYU (ID) | iBims-1 (ZS) | SUN RGB-D (ZS) | DIODE-Indoor (ZS) |
|---|---|---|---|---|
| DepthART-S | 0.924 / 0.340 | 0.713 / 0.693 | 0.809 / 0.379 | 0.292 / 1.687 |
| DepthART-B | 0.942 / 0.307 | 0.732 / 0.672 | 0.831 / 0.342 | 0.287 / 2.018 |
| DepthART-L | 0.946 / 0.295 | 0.747 / 0.646 | 0.847 / 0.326 | 0.298 / 2.012 |
| Depth Anything v1-L | 0.984 / 0.206 | 0.639 / 0.965 | 0.647 / 0.585 | 0.291 / 2.504 |
| FastDepth | 0.771 / 0.604 | 0.429 / 1.607 | 0.560 / 0.736 | 0.275 / 2.778 |
| DANet | 0.831 / 0.488 | 0.55 / 1.11 | 0.616 / 0.623 | 0.221 / 2.909 |
| GuideDepth | 0.840 / 0.478 | 0.560 / 1.014 | 0.603 / 0.657 | 0.271 / 2.766 |

Metric head fine-tuned on KITTI, so KITTI is ID and the other three are ZS:

| Method | KITTI (ID) | DDAD (ZS) | ETH3D-Outdoor (ZS) | DIODE-Outdoor (ZS) |
|---|---|---|---|---|
| DepthART-S | 0.943 / 3.12 | 0.811 / 7.140 | 0.228 / 6.518 | 0.216 / 9.862 |
| DepthART-B | 0.951 / 2.948 | 0.847 / 6.475 | 0.227 / 6.831 | 0.231 / 9.474 |
| DepthART-L | 0.956 / 2.784 | 0.867 / 6.340 | 0.258 / 5.794 | 0.249 / 8.823 |

- No AbsRel values were extracted for Table 2.
- UniDepth, Metric3D and Depth Pro are **not** in DepthART's metric table.
- The affine-aligned zero-shot results are in Table 1 (NYU, KITTI, ETH3D, DIODE-Full, DDAD). For DepthART-S on NYU: AbsRel 0.059, δ1 0.964. The project page gives DepthART-L at 0.971 on NYU. — [arXiv HTML](https://arxiv.org/html/2607.17099); [Project page](https://xuefeng-cvr.github.io/DepthART)

**YOLO26-Depth: aligned (multi-scale + flip TTA, log-least-squares scale), 768 px**
- NYU Eigen (654 images). Its status is unclear: the docs list NYU among the zero-shot sets but do not state whether it was in the 2.19M mix. — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth); [iBims page](https://docs.ultralytics.com/datasets/depth/ibims-1)

  | Model | δ1 | AbsRel | RMSE |
  |---|---|---|---|
  | n | 0.882 | 0.109 | 0.414 |
  | s | 0.896 | 0.104 | 0.399 |
  | m | 0.921 | 0.089 | 0.364 |
  | l | 0.930 | 0.083 | 0.351 |
  | x | 0.933 | 0.080 | 0.344 |

- NYU with no TTA and median scale-only alignment (still aligned, not raw), δ1: n 0.783, s 0.793, m 0.840, l 0.853, x 0.860. — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth)
  - Discrepancy: the Russian docs copy gives n 0.785 and s 0.786. — [Ultralytics RU docs](https://docs.ultralytics.com/ru/tasks/depth)
- iBims-1 (explicitly ZS, aligned + TTA), δ1: **n 0.923, s 0.899**, m 0.953, l 0.952, x 0.961. Note that s scores below n as published. — [Ultralytics iBims-1 page](https://docs.ultralytics.com/datasets/depth/ibims-1)
- Cross-range table, YOLO26x only (aligned δ1): NYU 0.933, iBims 0.961, ETH3D 0.959, Make3D 0.302, KITTI 0.942. KITTI is **not** zero-shot because its training drives are in the mix. — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth); [iBims page](https://docs.ultralytics.com/datasets/depth/ibims-1)
- No raw-metric numbers, no SUN RGB-D release numbers and no DIODE numbers are published. SUN RGB-D appears only in a fine-tuning comparison (δ1 0.78). — [Ultralytics docs](https://docs.ultralytics.com/tasks/depth)

**LightlyTrain DAv3-metric: NYU v2 (654 images, Eigen crop, 0.1–10 m), NYU not used in training** — [Lightly docs](https://docs.lightly.ai/train/stable/depth_estimation.html)

"Raw δ1" is metric with intrinsics; "aligned δ1" is after alignment.

| Checkpoint | Raw δ1 | Aligned δ1 | AbsRel | RMSE |
|---|---|---|---|---|
| tiny (6.2M) | 0.818 | 0.915 | 0.131 | 0.506 |
| tiny-plus (7.9M) | 0.846 | 0.923 | 0.123 | 0.457 |
| small (24.7M) | 0.912 | 0.938 | 0.099 | 0.377 |
| large (334.2M) | 0.950 | 0.948 | 0.078 | 0.339 |

- No iBims, SUN, DIODE or ETH3D numbers are published.

**UniDepthV2-Small (34.18M, near-miss): raw metric, zero-shot, δ1 / ARel (%)** — [UniDepthV2 arXiv](https://arxiv.org/html/2502.20110v2)

| Benchmark | δ1 | ARel |
|---|---|---|
| SUN-RGBD | 90.8 | 10.5 |
| IBims-1 | 86.6 | 13.5 |
| TUM-RGBD | 69.0 | 23.6 |
| ETH3D | 64.6 | 113.2 (as printed) |
| DDAD | 83.3 | 16.4 |
| NuScenes | 82.1 | 18.3 |

- F_A is SUN 74.2 and iBims 62.4.
- NYU and KITTI are reported for the Large model only, and only after fine-tuning.
- UniDepthV2-S is not evaluated on DIODE-indoor.

**Other compact or relative context (aligned, not metric)**
- ZipDepth (6.1M) zero-shot, AbsRel / δ1 (%): NYU 8.4 / 93.3, KITTI 12.3 / 86.4, ETH3D 10.0 / 92.2, ScanNet 8.8 / 92.1, DIODE 22.6 / 73.9.
- In the same table, DAv2-Small (24.8M): NYU 6.1 / 96.2, ETH3D 6.3 / 96.7, DIODE 21.4 / 75.9. — [ZipDepth arXiv](https://arxiv.org/html/2607.08771v1)

### Inferences
- A δ1 near 0.29 on DIODE-indoor for every DepthART variant, and for DAv1-L, suggests a protocol or scale mismatch (DIODE ranges up to about 50 m against d_max = 10 m) rather than a model property. Treat those cells cautiously.
- YOLO26's s < n ordering on iBims, and the gap between its TTA+LSQ and median-only numbers on NYU (0.882 against 0.783 for n), mean its published numbers cannot be compared with raw-metric tables. It would have to be re-evaluated in our harness.

### Gaps
- No raw-metric zero-shot numbers exist for YOLO26, the LightlyTrain tiny models, UniK3D-S or Metric3Dv2-S on iBims, SUN or DIODE.
- DepthART Table 2 AbsRel values were not extracted.
- The full UniK3D-S per-dataset tables were not retrieved: the CVPR supplemental returned 403. The README gives only F1 over metric 3D point clouds for UniK3D-Small: SmallFoV 61.3, SmallFoV+Distort 48.4, LargeFoV 55.5, Panoramic 72.5. — [UniK3D GitHub](https://github.com/lpiccinelli-eth/UniK3D)

## Q4. Which have code and weights runnable in our harness?

### Takeaway
All the in-range candidates have downloadable weights. DepthART (Apache-2.0 on HF; needs K; metric folder contents unverified) is the best fit for a raw-metric, camera-aware baseline. The LightlyTrain DAv3-metric tiny/tiny-plus models (pip `lightly-train`, auto-download, needs K, licence unstated) are the easiest to load. YOLO26n/s-depth (`ultralytics` package; global-scale metric; licence unverified) is the easiest intrinsics-free option. The near-miss foundation smalls (UniDepthV2-S, UniK3D-S) are CC BY-NC 4.0; Metric3Dv2-S code is BSD-2-Clause, weights licence unstated.

### Cited Findings

| Model | Params | Weights | Licence | Intrinsics |
|---|---|---|---|---|
| DepthART | S 6.0M (metric about 8.0M), B 11.4M | [HF Fengxue93/DepthART](https://huggingface.co/Fengxue93/DepthART/tree/main); code [github.com/xuefeng-cvr/DepthART](https://xuefeng-cvr.github.io/DepthART); ONNX folder present | Apache-2.0 (HF) | Required for metric |
| YOLO26-depth | n 6.3M, s 13.2M | `yolo26n-depth.pt`, `yolo26s-depth.pt` via the Ultralytics package ([docs](https://docs.ultralytics.com/tasks/depth)) | Not stated on page | Not used; global scale |
| LightlyTrain DAv3-metric | tiny 6.2M, tiny-plus 7.9M | `lightly_train.load_model("dinov3/dav3-metric-tiny")` ([docs](https://docs.lightly.ai/train/stable/depth_estimation.html)) | Not stated | Required (FOV approximation OK) |
| UniDepthV2-S | 34.18M | `lpiccinelli/unidepth-v2-vits14` ([GitHub](https://github.com/lpiccinelli-eth/UniDepth)) | CC BY-NC 4.0 | Optional |
| UniK3D-S | 35.4M | `lpiccinelli/unik3d-vits` ([HF](https://huggingface.co/lpiccinelli/unik3d-vits)) | CC BY-NC 4.0 | Optional |
| Metric3Dv2-S | 37.5M | PyTorch Hub `metric3d_vit_small` plus ONNX on HF ([GitHub](https://github.com/YvanYin/Metric3D)) | Code BSD-2-Clause; weights not stated | Optional (default focals) |
| DAv2-metric-small-hypersim | 24.8M | Hosted by LightlyTrain ([Lightly docs](https://docs.lightly.ai/train/stable/depth_estimation.html)) | Apache-2.0 | Not used; indoor ≤20 m |
| X-Lens | about 40M | Code at github.com/zhouhengamerica/XLens ([arXiv](https://arxiv.org/html/2607.12993)); weights not stated | Not stated | Requires GT intrinsics and extrinsics |

### Inferences
- For an apples-to-apples raw-metric comparison with our UniDepth-distilled students, a sensible order is the following. Run DepthART-S/B metric with GT K. Run LightlyTrain DAv3-metric-tiny/tiny-plus with GT K. Run YOLO26n/s-depth raw, using its shipped global scale. Then report both raw and median-aligned results, since YOLO26 publishes only aligned numbers.
- If only an NYU-fine-tuned DepthART metric head is released, its NYU results are in-domain and should be marked as such. If the HF `metric/` folder contains only Metric-L or MobileNetV4 heads, the S/B metric baselines may need re-training via their CamFT recipe.

### Gaps
- The DepthART metric checkpoint inventory is unknown: the HF folder listing was not expanded, and the GitHub page returned 404 under the Fengxue93 account.
- The YOLO26 and LightlyTrain weights licences are unverified. Ultralytics is typically AGPL-3.0 or Enterprise, but this was not confirmed on the depth pages.
- Whether the YOLO26 global scale was fit on data that overlaps iBims or NYU is undisclosed.
