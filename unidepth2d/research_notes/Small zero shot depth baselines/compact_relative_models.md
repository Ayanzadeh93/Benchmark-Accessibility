# Compact (2-15M, priority 5-10M) zero-shot relative / affine-invariant depth models from foundation-model families

Scope note: notes compiled 2026-11-27 from 18 tool calls (searches plus primary-source fetches). Metric-only compact models and NYU-only supervised lightweight nets are mentioned only in passing. All numbers are AbsRel (lower is better) / delta1 (higher is better) unless stated otherwise. "SSI-LSQ" means per-image least-squares scale-and-shift alignment.

## Q1. Which small members of foundation families exist, and how big are they exactly?

### Takeaway
No official foundation-family member falls in the 2-15M range. The smallest official checkpoints are MiDaS v2.1 Small (16.6-21.3M depending on source), Depth Anything V1/V2 Small (24.7-24.8M), Distill-Any-Depth Small (24.8M), Video Depth Anything Small (28.4M), MiDaS v3.1 Swin2-T (42M) and LeViT (51M), and DA3-Small (0.08B). The 2-15M range is filled only by 2026 distillation papers (see Q2).

### Cited Findings
**MiDaS (Intel ISL). Output: relative inverse depth (disparity), affine-invariant. Licence: MIT.**
- MiDaS v3.1 model zoo (arXiv 2307.14460). The official README table lists params and RTX 3090 FPS at training resolution: v2.1 Small-256 has 21M params and runs at 90 FPS; v3.1 LeViT-224 has 51M and runs at 73 FPS; v3.1 Swin2-T-256 has 42M and runs at 64 FPS; v3.1 Next-ViT-L-384 has 72M; v3.1 Swin2-B-384 has 102M; v2.1 Large-384 has 105M; v3.1 Swin2-L-384 has 213M; v3.1 BEiT-L-512 has 345M. The repo is MIT-licensed and was archived (read-only) on 25 Aug 2025 — [MiDaS GitHub](https://github.com/isl-org/MiDaS); [arXiv 2307.14460](https://arxiv.org/abs/2307.14460)
- MiDaS README zero-shot errors. Columns are DIW WHDR, ETH3D AbsRel, Sintel AbsRel, and TUM / KITTI / NYUv2 as the percentage of pixels with delta > 1.25 (lower is better). An asterisk means the model was trained on that dataset, so the figure is not zero-shot. Rows:
  - v2.1 Small-256: 0.1344 / 0.1344 / 0.3370 / 14.53 / 29.27 / 13.43, Imp. -76% vs DPT-L.
  - v3.1 Swin2-T-256: 0.1211 / 0.1106 / 0.2868 / 13.43 / 10.13* / 5.55*, Imp. -11%.
  - v3.1 LeViT-224: 0.1314 / 0.1206 / 0.3148 / 18.21 / 15.27* / 8.64*, Imp. -40%.
  - Source: [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- Other MiDaS deployment details. The README links iOS and Android sample apps for the v2.1 lightweight model and an OpenVINO v2.1-small-256 model. It states that the TF/ONNX code supports only v2.1, and it gives no model file sizes — [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- Conflicting MiDaS-small parameter counts. Qualcomm AI Hub's "Midas-V2" card uses the MiDaS_small checkpoint at 256x256 and lists 16.6M params, 63.2 MB float and 16.9 MB w8a8 — [Qualcomm AI Hub MiDaS](https://aihub.qualcomm.com/mobile/models/midas). The MiDaS README says 21M — [MiDaS GitHub](https://github.com/isl-org/MiDaS). ZipDepth's table says 21.3M — [ZipDepth arXiv 2607.08771](https://arxiv.org/html/2607.08771v1).
- MiDaS v3.1 LeViT (51M) and Swin2-T (42M) as evaluated by DepthART (SSI-LSQ):
  - LeViT: NYU 0.090/0.915*, KITTI 0.127/0.830*, ETH3D 0.129/0.875, DIODE-Full 0.240/0.698, DDAD 0.127/0.836.
  - Swin2-T: NYU 0.075/0.942*, KITTI 0.103/0.883*, ETH3D 0.119/0.886, DIODE-Full 0.229/0.718, DDAD 0.117/0.848.
  - An asterisk means in-domain. DepthART reports A6000 native latencies of 3.42 ms (LeViT) and 7.31 ms (Swin2-T).
  - Source: [DepthART arXiv 2607.17099](https://arxiv.org/html/2607.17099)
- MiDaS-Small (21.3M) as evaluated by ZipDepth (SSI-LSQ): NYUv2 11.5/86.8, KITTI 25.9/54.3, ETH3D 13.4/83.7, ScanNet 10.8/87.9, DIODE 24.2/71.1. Values are in percent — [ZipDepth arXiv](https://arxiv.org/html/2607.08771v1)

**Depth Anything V1 / V2 Small (ViT-S + DPT). Output: relative disparity, affine-invariant.**
- V1 Small is 24.8M (PyTorch, 99.2 MB fp32) per Apple's Core ML release — [apple/coreml-depth-anything-small](https://huggingface.co/apple/coreml-depth-anything-small/blob/refs%2Fpr%2F5/README.md)
- V2 Small is listed as 24.8M on Apple's Core ML page — [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small). Other listings: Luxonis 24.710M — [Luxonis model zoo](https://models.luxonis.com/luxonis/depth-anything-v2/aim_DJHwf6Qjh12Rb8f962aw7x?backTo=%2F); Qualcomm AI Hub 24.7M (518x518, 94.3 MB float, MIT per the card) — [Qualcomm AI Hub DA-V2](https://aihub.qualcomm.com/mobile/models/depth_anything_v2). The exact figure is about 24.7-24.8M; the difference is rounding or counting convention.
- DA-V2-S as evaluated by DepthART (SSI-LSQ): NYU 0.053/0.973, KITTI 0.078/0.936, ETH3D 0.142/0.851, DIODE-Full 0.228/0.751, DDAD 0.085/0.921, A6000 latency 13.06 ms — [DepthART](https://arxiv.org/html/2607.17099)
- DA-V2-S as evaluated by ZipDepth (SSI-LSQ, %): NYUv2 6.1/96.2, KITTI 8.4/93.4, ETH3D 6.3/96.7, ScanNet 5.2/97.2, DIODE 21.4/75.9 — [ZipDepth](https://arxiv.org/html/2607.08771v1). The two papers' DA-V2-S numbers conflict: ETH3D is 0.142 in DepthART vs 0.063 in ZipDepth, and NYU is 0.053 vs 0.061. Their evaluation pipelines (resolution, depth vs disparity alignment, crops) evidently differ, so cross-paper comparison is unsafe.
- DA-V2-S on NYUv2 as evaluated by Sensors 2026 (SSI-LSQ, Eigen crop, 0.1-10 m): 0.1366/0.8293. The original DA V1 ViT-S/14 scored 0.1668/0.7858 under the same protocol. These are far worse than the numbers above, which again shows protocol sensitivity — [Huang & Chen, Sensors 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13807952/)

**Other family members (all about 25M or more):**
- Distill-Any-Depth Small (arXiv 2502.19204; CVPR 2026 Findings version exists) is a "Dav2-small" architecture with 24.8M params. The HF repo is xingyang1/Distill-Any-Depth (small/model.safetensors, 99.2 MB), with an HF-format repo xingyang1/Distill-Any-Depth-Small-hf. The HF repo is tagged Apache-2.0. Teachers include GenPercept and DepthAnythingV2 — [HF repo](https://huggingface.co/xingyang1/Distill-Any-Depth/tree/main/small); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/He_Distill_Any_Depth_Distillation_Creates_a_Stronger_Monocular_Depth_Estimator_CVPRF_2026_paper.pdf)
- Video Depth Anything Small is relative, 28.4M, Apache-2.0. Base is 113.1M and Large is 381.8M, both CC-BY-NC-4.0. Metric-VDA-S is also 28.4M, with no licence stated. Small latency on an A100 for a 32x518x518 clip is 9.1 ms FP32 / 7.5 ms FP16 — [Video-Depth-Anything GitHub](https://github.com/DepthAnything/Video-Depth-Anything)
- Depth Anything 3. DA3-SMALL is 0.08B (relative depth + pose, Apache-2.0) and DA3-BASE is 0.12B (Apache-2.0). DA3MONO-LARGE (0.35B) is the monocular relative model; no small Mono variant is listed — [Depth-Anything-3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- AnyDepth ViT-S has 26.5M params. DepthART's table (SSI-LSQ) gives NYU 0.082/0.932, KITTI 0.102/0.883, ETH3D 0.084/0.935, DIODE-Full 0.247/0.714 — [DepthART](https://arxiv.org/html/2607.17099)
- Metric3D v2 ViT-S has 37.5M params. It is a metric model, but DepthART also evaluates it affine-aligned: NYU 0.056/0.965, KITTI 0.064/0.950, ETH3D 0.062/0.955, DIODE-Full 0.247/0.789 — [DepthART](https://arxiv.org/html/2607.17099)

### Inferences
- In the 2-15M range there is nothing "official" from the MiDaS, Depth Anything, Marigold, Lotus, DepthPro or Distill-Any-Depth families. The nearest official near-misses are MiDaS v2.1 Small (16.6-21M) and DA-V1/V2-S (24.8M).
- Diffusion-based relative models (Marigold, Lotus, PrimeDepth, DepthFM) are all SD-based and in the hundreds of millions. ZipDepth lists Marigold at 899.2M. I found no "light" variant of these below 100M.

### Gaps
- Depth Anything AC (arXiv 2507.01634): I did not verify its released checkpoint size. I believe it is ViT-S-based (about 24.8M) but did not confirm it.
- DepthPro: I found no "lite" or mobile variant. Only the full model appears, including a CoreML conversion (coreml-projects/DepthPro-coreml), and its parameter count was not verified in this session.
- Apple mobile depth: Apple publishes CoreML conversions of DA V1/V2 Small (24.8M). I found no Apple-native compact relative depth model.
- I did not find Lotus, Marigold-light or PrimeDepth compact variants. Absence was not exhaustively verified.
- The official DA-V1/V2 paper tables for ViT-S were not fetched directly. The DepthART-reproduced V2-S numbers (NYU 0.053/0.973, KITTI 0.078/0.936, ETH3D 0.142/0.851) appear to match the DA-V2 paper from recollection, which is unverified.

## Q2. Are there community or paper distillations of Depth Anything / DINOv2 depth into 2-15M students, with released weights?

### Takeaway
Yes. There are two strong 2026 papers in the priority 5-10M band. ZipDepth (6.1M, ECCV 2026, MIT, weights in the repo) is a RepVGG-style CNN distilled from DA-V2-L. DepthART-S/B (6.0M / 11.4M, TinyViM + DPT, arXiv July 2026) is distilled from DA-V2-L. Its weights appear to be on HF (Fengxue93/DepthART, Apache-2.0, ONNX), but the model card is empty. Other students found are either larger (17.7M, 24.5M) or not evaluated on standard benchmarks.

### Cited Findings
**ZipDepth (Tosi, Bartolomei, Poggi, Mattoccia; Univ. Bologna). arXiv 2607.08771, ECCV 2026.**
- Size: 6.79M training params including reparameterizable branches, 6.14M at inference after fusion (quoted as 6.1M). Compute is 3.0 GMACs — [arXiv](https://arxiv.org/html/2607.08771v1)
- Architecture: a four-stage RepVGG-style encoder (split stem, strip-pooling attention, SE, Global Context, SPPF) with an FPN-style decoder. It uses no pretrained backbone. Output is a full-resolution affine-invariant inverse depth — [arXiv](https://arxiv.org/html/2607.08771v1)
- Teacher and data: DA-V2-Large offline pseudo-labels on about 14.1M images from 17 datasets, trained in 3 days on 2x RTX 3090. The training loss is SSI with median/MAD normalisation — [arXiv](https://arxiv.org/html/2607.08771v1)
- Runtime: on a Jetson Orin NX at 15 W, 34.4 FPS and about 397 mJ/frame (PyTorch fused FP32), and up to 77 FPS (TensorRT). On an RTX 3090 with TensorRT FP16, up to 1317 FPS — [arXiv](https://arxiv.org/html/2607.08771v1). The repo gives 384x384 RTX 3090 latency as 3.9 ms (eager FP32) and 0.8 ms (TRT FP16) — [GitHub](https://github.com/fabiotosi92/ZipDepth)
- Zero-shot results (SSI-LSQ, %), as NYUv2 / KITTI / ETH3D / ScanNet / DIODE: ZipDepth 8.4/93.3, 12.3/86.4, 10.0/92.2, 8.8/92.1, 22.6/73.9. IBims-1, SUN RGB-D and Sintel are not reported — [arXiv](https://arxiv.org/html/2607.08771v1)
- ZipDepth also retrained compact baselines (marked †) on its own distillation set. These rows are effectively additional compact distilled relative models:
  - Lite-Mono† (8.3M): 8.7/92.7, 10.9/89.5, 8.7/93.6, 9.6/90.2, 22.4/72.9.
  - GuideDepth† (5.8M): 9.2/91.8, 14.9/80.6, 10.2/91.9, 9.6/90.5, 22.7/72.4.
  - FastDepth† (4.0M): 10.3/89.8, 16.8/74.3, 12.3/87.2, 11.0/87.6, 23.4/71.0.
  - PyDNet† (1.9M): 11.4/87.1, 12.9/84.9, 10.4/90.6, 11.7/85.8, 23.4/70.8.
  - Source: [arXiv](https://arxiv.org/html/2607.08771v1)
- Weights: `checkpoints/zipdepth_base.pth` (GPU) and `zipdepth_base_npu.pth` (NPU/mobile/CPU) are in the repo. The licence is MIT. The repo has infer, export (ONNX/TorchScript), benchmark, eval and train scripts. There is an iPhone demo on the project page. Phone latencies are only in the supplementary material — [GitHub](https://github.com/fabiotosi92/ZipDepth)

**DepthART: Scaling Foundation Monocular Depth to Tiny Models (Feng Xue, ..., Anlong Ming, Nicu Sebe; BUPT / Univ. Trento). arXiv 2607.17099, July 2026.**
- Venue is unconfirmed; the paper still carries a template placeholder — [arXiv HTML](https://arxiv.org/html/2607.17099)
- Sizes: DepthART-S 6.0M, -B 11.4M, -L 32.6M. The backbone is TinyViM (a Mamba-style hybrid) with a DPT decoder — [arXiv HTML](https://arxiv.org/html/2607.17099)
- Training: Stage 1 distils affine-invariant relative depth from DA-V2-Large, using 11 unlabeled sources with bias-resistant sampling that cuts about 44M images to a 1.7M pool. Stage 2 (CamFT) freezes the encoder and fine-tunes metric depth on NYU or KITTI — [arXiv HTML](https://arxiv.org/html/2607.17099)
- FLOPs at 224² (S/B/L): 1.749 / 2.348 / 6.178 G. At 448²: 6.996 / 9.394 / 24.711 G — [arXiv HTML](https://arxiv.org/html/2607.17099)
- Latency on an A6000 (FP32, 448²): 4.09 / 4.91 / 8.07 ms. On a Jetson Orin NX: about 102/86/55 FPS at 224² and 34/28/16 FPS at 448². DepthART-S exceeds 15 FPS on a Jetson Nano at 224² — [arXiv HTML](https://arxiv.org/html/2607.17099)
- Zero-shot affine-invariant results (Table 1, SSI-LSQ), as NYU / KITTI / ETH3D / DIODE-Full / DDAD:
  - DepthART-S: 0.059/0.964, 0.082/0.930, 0.084/0.950, 0.214/0.745, 0.095/0.900.
  - DepthART-B: 0.057/0.969, 0.088/0.929, 0.092/0.954, 0.214/0.746, 0.094/0.906.
  - DepthART-L: 0.053/0.971, 0.079/0.933, 0.091/0.958, 0.208/0.754, 0.095/0.910.
  - ScanNet and Sintel are not evaluated. IBims-1 and SUN RGB-D appear only in the metric table.
  - Source: [arXiv HTML](https://arxiv.org/html/2607.17099)
- Metric results after CamFT (no rescaling; delta1 / RMSE), as NYU / iBims-1 / SUN RGB-D / DIODE-Indoor:
  - DepthART-S: 0.924/0.340, 0.713/0.693, 0.809/0.379, 0.292/1.687.
  - DepthART-B: 0.942/0.307, 0.732/0.672, 0.831/0.342, 0.287/2.018.
  - DepthART-L: 0.946/0.295, 0.747/0.646, 0.847/0.326, 0.298/2.012.
  - These are metric, out of scope, but relevant for IBims-1 and SUN RGB-D context.
  - Source: [arXiv HTML](https://arxiv.org/html/2607.17099)
- Weights: the paper says code is on GitHub, with a project page at xuefeng-cvr.github.io/DepthART. An HF repo, Fengxue93/DepthART, exists with an Apache-2.0 tag and the ONNX library tag, but its model card is empty — [HF Fengxue93/DepthART](https://huggingface.co/Fengxue93/DepthART). A third-party TypeGPU conversion cites that repo as the origin of the weights — [modulsx/depthart-typegpu](https://huggingface.co/modulsx/depthart-typegpu). A Chinese press write-up reports a September 2026 update that adds MobileNetV4-based variants, which is unverified against a primary source — [Sina write-up](https://finance.sina.cn/stock/jdts/2026-10-07/detail-iniuksym4764240.d.html)

**Other students (outside 5-10M or no standard benchmarks):**
- Student8 (Huang & Chen, Sensors 2026, 26(22):7121). This is a structure-pruned DA ViT-S/14 that keeps 8 of 12 blocks, with 17.684M params and relative inverse depth output. It was fine-tuned on NYUv2, so it is not zero-shot on NYU: NYU 0.1350/0.8320. Zero-shot KITTI is 0.2543/0.5297 under SSI-LSQ. On a Jetson Orin Nano TRT FP16 it runs in 4.16 ms (240 FPS). Code is available only on request — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13807952/)
- DANIEL student (Mizlish & Ben-Moshe, arXiv 2611.32718, Nov 2026). This is a ResNet-34 U-Net with no attention, 24.5M params, outputting relative disparity distilled from DA-V2-L. It runs at 27.1 FPS on a Hailo-8L with a Raspberry Pi 5. It is evaluated only on the authors' own simulated and real robot data, with no NYU/KITTI/ETH3D. Code and models are stated to be at github.com/itaim18/DANIEL, with no licence stated — [arXiv HTML](https://arxiv.org/html/2611.32718)
- Jackupjcup/Lightweight_mono_depth_estimation is a GitHub project distilling DA3 into a roughly 52M MobileNetV2 student. It is metric and out of scope — [GitHub](https://github.com/Jackupjcup/Lightweight_mono_depth_estimation)
- Fast-FoundationStereo uses MobileNetV2/EdgeNeXt students, but for stereo, so it is not relevant — [alphaxiv](https://www.alphaxiv.org/abs/2512.11130)

### Inferences
- ZipDepth (6.1M) and DepthART-S (6.0M) are the only two found models that sit exactly in the 5-10M priority band and report multi-dataset zero-shot affine-invariant numbers. DepthART-B (11.4M) is in the broader 2-15M band.
- On their own papers' protocols, DepthART-S (NYU 0.059, ETH3D 0.084) looks stronger than ZipDepth (NYU 0.084, ETH3D 0.100). However, DA-V2-S reference numbers differ between the papers (ETH3D 0.142 vs 0.063), so the two cannot be compared without re-evaluating both under one protocol.
- ZipDepth's retrained Lite-Mono†, GuideDepth† and FastDepth† rows are useful as "architecture-controlled" compact baselines. It is unclear whether those retrained weights are released.

### Gaps
- I found no evidence of community "TinyDepth", "MiniDepth", "LiteDepthAnything" or "DepthAnything-Lite" checkpoints with EfficientViT, MobileViT, RepViT or FastViT backbones and benchmark numbers. The searches surfaced none, but they were not exhaustive.
- The DepthART GitHub repo URL and the HF file list were not confirmed, and the MobileNetV4 variants and their sizes are unverified.
- Qualcomm AI Hub and Hailo zoos list only MiDaS-small and DA-V2-S as relative depth models, plus SC-DepthV3 and FastDepth in Hailo. I found no sub-15M distilled DA entry there.

## Q3. Which report zero-shot numbers on IBims-1, NYU, SUN RGB-D, DIODE and ETH3D?

### Takeaway
For compact relative models, zero-shot affine-invariant results exist for NYU, KITTI, ETH3D and DIODE (ZipDepth, DepthART, MiDaS-small via the third-party tables), and for ScanNet (ZipDepth). No compact relative model reports zero-shot affine-invariant IBims-1 or SUN RGB-D. Those two appear only in DepthART's metric-after-fine-tuning table.

### Cited Findings
- ZipDepth evaluates NYUv2, KITTI, ETH3D, ScanNet and DIODE, all with SSI-LSQ. It has no IBims-1, SUN RGB-D or Sintel — [ZipDepth](https://arxiv.org/html/2607.08771v1)
- DepthART evaluates affine-invariant NYU, KITTI, ETH3D, DIODE-Full and DDAD with SSI-LSQ. It reports IBims-1, SUN RGB-D, DIODE-Indoor and ETH3D-outdoor only as metric results without rescaling — [DepthART](https://arxiv.org/html/2607.17099)
- The MiDaS README reports DIW, ETH3D, Sintel, TUM, KITTI and NYU, but NYU and KITTI are not zero-shot for the v3.1 models — [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- Sensors 2026 Student8 reports NYU (fine-tuned) and zero-shot KITTI only — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13807952/)
- The alignment protocol varies. ZipDepth, DepthART and Sensors (main results) all use per-image least-squares scale-and-shift. Sensors uses median scaling only for ablations and warns that the two protocols are not comparable — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13807952/). The papers' text does not settle whether the fitting is done in depth or disparity space, but the DA-V2-S reference rows diverge widely: NYU AbsRel is 0.053 in DepthART, 0.061 in ZipDepth and 0.137 in Sensors.

### Inferences
- To get IBims-1, SUN RGB-D and DIODE-Indoor zero-shot affine-invariant numbers for compact baselines, we would have to run them ourselves.
- We should re-run DA-V2-S alongside the compact models in our own pipeline, because published DA-V2-S numbers vary by up to about 2.5x in AbsRel across papers.

### Gaps
- None of the found works report affine-invariant IBims-1 or SUN RGB-D for any model under 25M.
- The DIODE split differs across papers. DepthART uses DIODE-Full, and ZipDepth's "DIODE" is presumably also Full given the AbsRel of about 0.21-0.24, but this is unconfirmed.

## Q4. Which can realistically be run zero-shot by us (weights and inference code available)?

### Takeaway
The ready-to-run sub-25M options are ZipDepth (6.1M, MIT, .pth plus inference scripts in the repo) and MiDaS v2.1 Small (about 21M, MIT, torch hub, ONNX and OpenVINO). DepthART-S/B (6.0M/11.4M) can probably be run via HF Fengxue93/DepthART (Apache-2.0, ONNX), but this needs verification. For the about-25M reference baselines, DA-V2-S, Distill-Any-Depth-S and VDA-S (Apache-2.0) are all straightforward to run.

### Cited Findings
- ZipDepth: checkpoints are in the repo under `checkpoints/` with `scripts/infer.py`, `scripts/export.py` and evaluation scripts, MIT licence — [GitHub](https://github.com/fabiotosi92/ZipDepth)
- DepthART: HF Fengxue93/DepthART has an Apache-2.0 tag, an ONNX library tag and an empty card. A derived TypeGPU conversion exists — [HF](https://huggingface.co/Fengxue93/DepthART); [modulsx/depthart-typegpu](https://huggingface.co/modulsx/depthart-typegpu)
- MiDaS v2.1 Small: MIT licence, with an archived repo that still includes mobile, ONNX and OpenVINO paths — [MiDaS GitHub](https://github.com/isl-org/MiDaS). Qualcomm AI Hub hosts float and w8a8 exports — [Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models/midas)
- DA-V2-S has CoreML (Apple), Qualcomm AI Hub, Luxonis and ONNX community exports — [Apple HF](https://huggingface.co/apple/coreml-depth-anything-v2-small); [Qualcomm](https://aihub.qualcomm.com/mobile/models/depth_anything_v2)
- Distill-Any-Depth-Small: Apache-2.0 tag on the HF repo, with a Transformers-pipeline-compatible "-hf" repo — [HF](https://huggingface.co/xingyang1/Distill-Any-Depth/tree/main/small)
- Video-Depth-Anything-Small: Apache-2.0 (Base and Large are NC) — [GitHub](https://github.com/DepthAnything/Video-Depth-Anything)
- DA3-Small: Apache-2.0, 0.08B, multi-view relative depth and pose — [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- The DANIEL student (24.5M) is stated to be released at github.com/itaim18/DANIEL with no licence, and it is domain-specific (robot navigation) — [arXiv](https://arxiv.org/html/2611.32718)
- Student8 (17.7M) has code available only on request — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13807952/)

### Inferences
- A realistic compact-baseline table for our paper would include:
  - ZipDepth (6.1M)
  - DepthART-S (6.0M) and DepthART-B (11.4M), if the HF weights load
  - MiDaS v2.1 Small (about 21M)
  - DA-V2-S and Distill-Any-Depth-S as near-miss (24.8M) references
  - optionally MiDaS v3.1 Swin2-T (42M) and LeViT (51M) as larger references
- All of these output relative inverse depth (disparity), so they need consistent disparity-space vs depth-space alignment in our evaluator.

### Gaps
- The DepthART HF file contents, preprocessing (input size and normalisation) and output convention were not verified.
- ZipDepth's on-phone latencies are only in its supplementary material, which was not fetched.
