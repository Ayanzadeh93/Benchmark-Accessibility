# Edge / on-device monocular depth releases and small-model zero-shot benchmarks (2025-2026)

Scope: edge depth model zoos, 2025-2026 efficient-depth papers, and benchmark tables that include ~2-15M models zero-shot on indoor benchmarks (IBims-1, NYU-v2, SUN RGB-D, DIODE-indoor, ETH3D). Reference point: a 5.04M-parameter metric student. Researched 2026-11-27. All numbers are copied from the cited pages. Where two sources disagree, both are given.

## Q1. Which edge depth model zoos exist, and which small models do they include (sizes, weights, licence, latency)?

### Takeaway
The vendor zoos mostly carry the same few models. In the 2-15M class there are only two kinds. The first is old NYU-supervised CNNs: Hailo fast_depth at 1.35M and scdepthv3 at 14.8M. The second is the new Ultralytics YOLO26-depth family: n is 6.3M and s is 13.2M, both metric. Qualcomm, Apple, NVIDIA and LightlyTrain only ship models of about 16.6M and up: MiDaS-small at 16.6M, then DAv2-S / DA3-S at about 24.7-24.8M, and larger. I found no MediaPipe depth model.

### Cited Findings
**Qualcomm AI Hub**
- Depth-Anything-V2 entry: checkpoint DepthAnything_V2_Small, 24.7M params, 94.3 MB float, 518x518 input, licence MIT. It is listed for Snapdragon X Elite / X Plus / X2 Elite compute devices, and the mobile page says "currently not supported on any Mobile chipset" — [AI Hub DAv2 compute](https://aihub.qualcomm.com/compute/models/depth_anything_v2); [AI Hub DAv2 mobile](https://aihub.qualcomm.com/mobile/models/depth_anything_v2)
- Depth-Anything-V3 entry: checkpoint da3-small, 24.7M params, 94.3 MB float, 518x518, Apache-2.0. It is also not supported on mobile chipsets. Other depth models in the zoo are Midas-V2, Depth-Anything (V1) and Depth-Anything-V2 — [AI Hub DA3](https://aihub.qualcomm.com/mobile/models/depth_anything_v3)
- Parameter conflict: the official DA3-Small Hugging Face card says ~0.08B params (Apache-2.0), but AI Hub says 24.7M. The official card is more authoritative. AI Hub may be counting only the monocular-depth path. This is unresolved — [HF depth-anything/da3-small](https://huggingface.co/depth-anything/da3-small); [AI Hub DA3](https://aihub.qualcomm.com/mobile/models/depth_anything_v3)
- Midas-V2 (MiDaS_small checkpoint): 16.6M params, 63.2 MB float, 16.9 MB w8a8, 256x256 input. A quantized variant also exists — [AI Hub Midas](https://aihub.qualcomm.com/compute/models/midas); [HF qualcomm/Midas-V2-Quantized](https://huggingface.co/qualcomm/Midas-V2-Quantized)
- Parameter conflict: ZipDepth's table lists MiDaS-Small at 21.3M params / 4.6 GMACs, versus 16.6M on AI Hub — [ZipDepth](https://arxiv.org/html/2607.08771v1)
- Not retrieved: per-device latency numbers on the AI Hub pages (dynamic content).

**Apple Core ML (Hugging Face `apple/`)**
- coreml-depth-anything-v2-small has 24.8M params. The F32 package is 99.2 MB and F16 is 49.8 MB; abs-rel vs. the PyTorch model is 0.0072 (F32) and 0.0089 (F16). F16 inference takes 31.10 ms on iPhone 12 Pro Max (iOS 18) and 33.90 ms on iPhone 15 Pro Max (iOS 17.4), mainly on the Neural Engine. INT8 and palettized (F16P8) variants are also in the repo — [HF apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small)
- The older apple/coreml-depth-anything-small (V1) is reported at roughly 99-160 ms on iPhone 14/15 — [HF apple coreml-depth-anything-small README](https://huggingface.co/apple/coreml-depth-anything-small/raw/refs%2Fpr%2F5/README.md)
- I found no small Apple Core ML model under 15M. Depth Pro was not checked (it is far above the size class).

**Google (MediaPipe / LiteRT)**
- litert-community/MiDaS-small is MiDaS v2.1 small converted to LiteRT/TFLite. It outputs inverse depth and is about 33 MB in fp16 — [HF litert-community/MiDaS-small](https://huggingface.co/litert-community/MiDaS-small)
- litert-community also hosts Depth-Anything-3-Small — [HF litert-community/Depth-Anything-3-Small](https://huggingface.co/litert-community/Depth-Anything-3-Small)
- I found no MediaPipe depth-estimation task or model.

**NVIDIA TAO / NGC**
- NvDepthAnythingV2 (NGC, released 07/25/2025) is a relative-depth model with a ViT-Large backbone (~3.6x10^8 params) under the NVIDIA Open Model License, "ready for commercial use". It was trained on 6.02M images, mostly NVIDIA internal plus CREStereo synthetic. Zero-shot NYUv2 (654 images), scale-shift-invariant: AbsRel 0.047, δ1 0.979. FP16 at 3x518x914 with batch 32 runs at 3.71 FPS on Jetson AGX Orin and 3.34 FPS on AGX Thor — [NGC NvDepthAnythingV2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/nvdepthanythingv2)
- TAO also documents MetricDepthAnything, a DINOv2-ViT-based metric model. No small variant was found — [TAO Mono Depth docs](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/depth_estimation/monocular_depth_estimation.html)
- Neither NVIDIA model is in the 2-15M class.

**Hailo Model Zoo (Hailo-8; listed under "Nyu"; metric = RMSE)**
- fast_depth: 1.35M params, 0.74 GOPs, 224x224 input, float RMSE 0.60 / hardware RMSE 0.61, 2519 FPS at batch 1. Source is dwofk/fast-depth.
- scdepthv3: 14.8M params, 10.7 GOPs, 256x320 input, RMSE 0.48 / 0.48, 929 FPS at batch 1. Source is JiawangBian/sc_depth_pl.
- Both rows: Hailo-8 test setup with DFC v2.20.0 — [Hailo MZ HAILO8 depth](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8/HAILO8_depth_estimation.rst)
- The HAILO15H page lists the same two models (fast_depth 2287 FPS, scdepthv3 736 FPS) — [Hailo MZ HAILO15H depth](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO15H/HAILO15H_depth_estimation.rst)
- A community report gives scdepthv3 at about 30 FPS on Raspberry Pi 5 + Hailo-8 — [Hailo community](https://community.hailo.ai/t/depth-estimation-rpi5-hailo-kit/1758)
- Both are NYU-trained (supervised or self-supervised) models, not zero-shot models.

**Ultralytics YOLO26-depth (metric). This is the most relevant zoo entry for a 5.04M metric student.**

| Model | Input | Params (M) | FLOPs (B) | NYU δ1 (TTA, log-LS align) | NYU AbsRel | NYU RMSE (m) | CPU ONNX fp32 (ms) | T4 TRT fp16 (ms) |
|---|---|---|---|---|---|---|---|---|
| YOLO26n-depth | 768 | 6.3 | 46.9 | 0.882 | 0.109 | 0.414 | 272.0 | 2.7 |
| YOLO26s-depth | 768 | 13.2 | 68.0 | 0.896 | 0.104 | 0.399 | 393.7 | 3.8 |
| YOLO26m-depth | 768 | 23.3 | 130.4 | 0.921 | 0.089 | 0.364 | 621.5 | 6.0 |
| YOLO26l-depth | 768 | 27.7 | 157.0 | 0.930 | 0.083 | 0.351 | 821.9 | 7.7 |
| YOLO26x-depth | 768 | 57.0 | 301.7 | 0.933 | 0.080 | 0.344 | 1240.9 | 13.6 |

Source for the table and the notes below: [Ultralytics depth task docs](https://docs.ultralytics.com/tasks/depth)

- Protocol caveat: the headline δ1 uses multi-scale + flip TTA with log-least-squares alignment. Single-scale, no-TTA δ1 with median alignment is 0.783 (n), 0.793 (s), 0.840 (m), 0.853 (l), 0.860 (x).
- The models were trained on a multi-dataset indoor+outdoor mix of about 2.19M images. NYU is therefore probably not strictly zero-shot (see Inferences).
- The head outputs exp(logit), roughly 0.02-150 m, with a global log-affine calibration baked into the released checkpoints.
- Cross-dataset results are reported only for YOLO26x (scale-aligned δ1): NYU 0.933, iBims-1 0.961, ETH3D 0.959, Make3D 0.302, KITTI 0.942; mean 0.819.
- For speed, DAv2-S (24.8M, 346.9 GFLOPs at about 768 px) runs at 20.99 ms on T4 TRT. Ultralytics explicitly says it did not compare accuracy against DAv2 under a shared protocol.
- The docs do not state a licence or Jetson latency.
- Weights auto-download from the v8.4.0 assets release. Roboflow lists n/s/m-depth-768 checkpoints — [Roboflow inference-models](https://inference-models.roboflow.com/models/yolo26-depth-estimation/)
- Ultralytics also hosts iBims-1 and ETH3D dataset pages for this task — [Ultralytics iBims-1](https://docs.ultralytics.com/datasets/depth/ibims-1); [Ultralytics ETH3D](https://docs.ultralytics.com/datasets/depth/eth3d)

**LightlyTrain**
- LightlyTrain supports depth inference with ports of Depth Anything V2 and V3, relative and metric, including a small DAv3 variant. It ships no own sub-15M model — [LightlyTrain depth docs](https://docs.lightly.ai/train/stable/depth_estimation.html)

### Inferences
- In the vendor zoos, the only deployable models in the 5-15M class are YOLO26n/s-depth and Hailo scdepthv3/fast_depth.
- YOLO26n-depth (6.3M, metric, weights public) is the closest commercial-zoo peer to a 5.04M metric student. Its NYU numbers are not zero-shot comparable, because NYU-like indoor data is probably in the 2.19M mix and the headline numbers use TTA plus log-LS alignment.
- Qualcomm and Apple ship no sub-15M depth model. Their smallest transformer entries (about 24.7-24.8M) are about 5x the size of a 5.04M student.

### Gaps
- Not found in the docs: the YOLO26-depth licence (Ultralytics usually uses AGPL-3.0 / Enterprise, but this was not verified for depth), the composition of the 2.19M training mix, per-size iBims/ETH3D numbers, and Jetson/phone latency.
- AI Hub per-device latencies were not retrieved because the pages are dynamic.
- OpenMMLab "mmdepth" was not checked. I found no evidence that an official OpenMMLab depth zoo with sub-15M zero-shot models exists.

## Q2. Which 2025-2026 papers propose efficient depth, and what sizes and zero-shot numbers do they report?

### Takeaway
Two 2026 papers sit squarely in the 5-12M zero-shot class, and both distill from DAv2-L:
- **DepthART** (S 6.0M, B 11.4M, L 32.6M). It reports both affine-invariant zero-shot results and a metric stage with zero-shot iBims-1, SUN RGB-D and DIODE-indoor numbers.
- **ZipDepth** (6.1M at inference). It is relative-only and reports Jetson Orin, iPhone and CPU latency.

The other efficient-depth papers are either about 25M+ (EfficientDepth on MiT-B5, AnyDepth-S at 26.5M, DAv2-S-based challenge entries) or do not report parameter counts.

### Cited Findings
**DepthART: Scaling Foundation Monocular Depth to Tiny Models (arXiv 2607.17099)**
- Authors: Xue et al., Univ. Trento / BUPT, with N. Sebe.
- Architecture and size: TinyViM-S/B/L encoder + DPT decoder, at 6.0M / 11.4M / 32.6M params.
- Training: teacher DAv2-L. 11 unlabeled/relative sources, with BRDS sampling cutting a 44M corpus to a 1.7M pool.
- Stage 1 outputs affine-invariant depth. Evaluation uses a per-image least-squares scale and shift.
- Stage 2 (CamFT) fine-tunes for metric depth on NYUv2 (d_max = 10) or KITTI (80). Metric results are reported without re-scaling.
- Code is said to be on GitHub; project page xuefeng-cvr.github.io/DepthART. The weights licence is not stated.
- Source for all DepthART items: [DepthART arXiv HTML](https://arxiv.org/html/2607.17099)
- Latency (model-only / end-to-end), strict FP32:
  - DepthART-S: 2.88 / 3.72 ms at 224² on an RTX A6000; 9.80 / 12.43 ms at 224² and 29.48 / 40.40 ms at 448² on Orin NX 8GB (MAXN).
  - DepthART-S exceeds 15 FPS on Jetson Nano at 224².
  - For comparison, native DAv2-S/B/L takes 13.06 / 33.04 / 102.46 ms on the A6000.

**ZipDepth: Bringing Lightweight Zero-Shot Monocular Depth Anywhere, on Any Device (arXiv 2607.08771)**
- Authors: Tosi, Bartolomei, Poggi, Mattoccia (Univ. Bologna).
- Size and compute: RepVGG-style reparameterizable CNN encoder + light FPN, 6.79M params when training and 6.14M at inference (6.1M), 3.0 GMACs.
- Training: teacher DAv2-L pseudo-labels on about 14.1M images from 17 datasets. SSI + gradient loss with median/MAD normalization.
- Output and evaluation: affine-invariant inverse depth, evaluated with least-squares alignment.
- Source for all ZipDepth items: [ZipDepth arXiv HTML](https://arxiv.org/html/2607.08771v1)
- Latency at 384x384 (Table 6):

| Device | Setting | FPS |
|---|---|---|
| Jetson Orin NX 15 W | PyTorch | 34 |
| Jetson Orin NX 15 W | TRT FP16 | 77 |
| Jetson Orin NX 50 W | PyTorch | 118 |
| Jetson Orin NX 50 W | TRT FP16 | 196 |
| iPhone 12 A14 | CoreML ANE | 240 |
| iPhone 12 A14 | CoreML ANE, NPU path | 375 |
| iPad Pro M4 | CoreML ANE | 360 |
| iPad Pro M4 | CoreML ANE, NPU path | 715 |
| Intel i7-11800H | ONNX Runtime | 45 |
| Poco X3 | TFLite GPU | 16 |
| RTX 3090 | TRT | 1317 |

- Weights and licence are not stated in the paper. Project page: zipdepth.github.io.

**EfficientDepth (arXiv 2509.22527, Leia Inc. / Texas A&M)**
- Architecture: MiT-B5 encoder + UNet decoder + bimodal density head.
- Training: teacher DA V1-L pseudo-labels; about 8M images (2M labelled + 6M pseudo-labelled) from 23 datasets.
- Speed: 0.055 s per image on an A40, versus 0.048 s for DAv2-S.
- The paper gives no parameter count. MiT-B5 alone is about 82M (from general knowledge, not from the paper), so EfficientDepth is not a sub-15M model.
- Zero-shot results, AbsRel / 100·(1−δ1):
  - EfficientDepth: NYU 0.029 / 1.789, ETH3D 0.0729 / 19.92.
  - DAv2-S: NYU 0.035 / 2.698, ETH3D 0.0767 / 20.90.
- The paper uses the MiDaS evaluation scripts, but its alignment is not spelled out. Its ETH3D "1−δ1" values are not on the same scale as other papers.
- Weights and licence are not stated.
- Source for all EfficientDepth items: [EfficientDepth arXiv HTML](https://arxiv.org/html/2509.22527v1)

**AnyDepth (arXiv 2601.02760)**
- Architecture: frozen DINOv3 encoder + a light "SDT" decoder. Total params are 26.5M (S) / 95.5M (B) / 313.4M (L); the decoder alone is 5.51M for S.
- Trained on 369K synthetic samples.
- Zero-shot results, AbsRel / δ1 ×100, for AnyDepth-S: NYU 8.2 / 93.2, KITTI 10.2 / 88.3, ETH3D 8.4 / 93.5, ScanNet 8.0 / 93.6, DIODE 24.7 / 71.4.
- Latency at 256² on Jetson Orin Nano: 213 ms with SDT, versus 306 ms with DPT.
- Code: github.com/AIGeeksGroup/AnyDepth. Paper licence is CC BY-NC-SA 4.0; the weights licence is not stated.
- Error in this paper: its Table 2 lists DAv2-S at 71.8M params, which contradicts the usual 24.8M.
- Source for all AnyDepth items: [AnyDepth arXiv HTML](https://arxiv.org/html/2601.02760)

**Distill Any Depth (arXiv 2502.19204)**
- Multi-teacher distillation with a diffusion-based assistant teacher. It claims SOTA zero-shot results on NYUv2, ScanNet, KITTI, DIODE and ETH3D — [Distill Any Depth](https://arxiv.org/abs/2502.19204)
- Its small variants are ViT-S-based (about 25M), not sub-15M. The per-size rows were not retrieved.

**2025 IEEE Low-Power Computer Vision Challenge (LPCVC), Track 3: monocular relative depth (arXiv 2604.19054)**
- Device and latency limit: Snapdragon 8 Elite QRD via Qualcomm AI Hub, ≤34 ms.
- Input/output: fixed 640x480 input; output is relative depth normalized to [0,1].
- Metric: point-cloud F-score (unprojection with intrinsics; threshold τ).
- Baseline is DAv2-based: F-score 62.37 at 24.1 ms.
- Results: winner "Sailor Moon" (U. Minnesota) 83.14 at 30.4 ms; 2nd 81.65 at 24.7 ms; 3rd "Circle" 78.83 at 30.3 ms.
- The winning entry (named "EfficientDepth", not the Leia paper) is DAv2 ViT-S plus graph fusion and resolution tuning. At 434x322 it reaches F-score 83.8 at 30.2 ms.
- The paper gives no parameter counts. The winning entry is about 24.8M because it is DAv2-S-based.
- Source for all LPCVC items: [LPCVC 2025 winners report](https://arxiv.org/pdf/2604.19054)

**Other leads (snippet-level only, not verified)**
- LMDepth: a Mamba-based lightweight MDE (arXiv 2505.00980). I did not confirm zero-shot numbers or params — [LMDepth](https://arxiv.org/pdf/2505.00980)
- OptiGeo (arXiv 2608.29881): about 30M params. It evaluates zero-shot on NYUv2, ETH3D, iBims-1, Sintel and DIODE — [OptiGeo](https://arxiv.org/pdf/2608.29881)
- Fourth Monocular Depth Estimation Challenge (CVPR 2025): accuracy-focused, with most leading methods affine-invariant. It is not an efficiency track — [MDEC CVPR2025](https://personalpages.surrey.ac.uk/r.bowden/publications/2025/MDEC_CVPR2025pp.pdf)

### Inferences
- DepthART-S (6.0M) and ZipDepth (6.1M) are the 2026 state of the art for sub-15M zero-shot relative depth.
- On their shared benchmarks, DepthART-S scores higher than ZipDepth (NYU AbsRel 0.059 vs 0.084; δ1 0.964 vs 0.933), but the two use different evaluation codebases, so treat this as indicative.
- Both are DAv2-L distillations. This mirrors a distill-from-foundation-model student recipe.

### Gaps
- I could not verify that "DepthART", "FlashDepth", "Prompt Depth Anything", "Depth Anything at Any Condition", "MobileDepth" or "TinyDepth" have sub-15M variants. FlashDepth, Prompt-DA and DA-AC are built on DAv2 ViT-S or larger (general knowledge, not verified this session). "DepthART" turned out to be the 2026 tiny-model paper above.
- No 2025-2026 Mobile AI (MAI) depth-challenge report was found. The MAI depth challenges I found date from 2021-2022.

## Q3. Are there benchmark tables with sub-15M models zero-shot on IBims / ETH3D / DIODE / NYU / SUN, and with what alignment protocol?

### Takeaway
Yes. Two 2026 tables give the most complete comparisons:
- **ZipDepth Table 2:** affine-invariant LS alignment on NYU, KITTI, ETH3D, ScanNet and DIODE-full. It includes retrained sub-10M baselines (Lite-Mono 8.3M, GuideDepth 5.8M, FastDepth 4.0M, PyDNet 1.9M) and MiDaS-Small.
- **DepthART Tables 1 and 2:** affine-invariant zero-shot results, plus metric results after NYU fine-tuning with zero-shot iBims-1, SUN RGB-D and DIODE-indoor. This metric table is the closest published analogue to a sub-15M metric student's evaluation.

### Cited Findings
**ZipDepth Table 2 (zero-shot, affine-invariant LS-aligned inverse depth; AbsRel ×100 / δ1 ×100)**

† = retrained on ZipDepth's 14.1M multi-domain DAv2-L pseudo-labels. FPS and mJ are on Orin NX 15 W.

| Model | Params (M) | GMACs | FPS | mJ/frame | NYU | KITTI | ETH3D | ScanNet | DIODE |
|---|---|---|---|---|---|---|---|---|---|
| ZipDepth | 6.1 | 3.0 | 34.4 | 396.6 | 8.4 / 93.3 | 12.3 / 86.4 | 10.0 / 92.2 | 8.8 / 92.1 | 22.6 / 73.9 |
| Lite-Mono† | 8.3 | 13.5 | 7.3 | 1883.5 | 8.7 / 92.7 | 10.9 / 89.5 | 8.7 / 93.6 | 9.6 / 90.2 | 22.4 / 72.9 |
| GuideDepth† | 5.8 | 5.1 | 16.6 | 860.0 | 9.2 / 91.8 | 14.9 / 80.6 | 10.2 / 91.9 | 9.6 / 90.5 | 22.7 / 72.4 |
| FastDepth† | 4.0 | 2.2 | 28.9 | 483.2 | 10.3 / 89.8 | 16.8 / 74.3 | 12.3 / 87.2 | 11.0 / 87.6 | 23.4 / 71.0 |
| PyDNet† | 1.9 | 5.5 | 34.8 | 398.0 | 11.4 / 87.1 | 12.9 / 84.9 | 10.4 / 90.6 | 11.7 / 85.8 | 23.4 / 70.8 |
| MiDaS-Small | 21.3 | 4.6 | 20.1 | 683.8 | 11.5 / 86.8 | 25.9 / 54.3 | 13.4 / 83.7 | 10.8 / 87.9 | 24.2 / 71.1 |
| DA-V2-Small | 24.8 | 57.8 | 2.2 | 6740.3 | 6.1 / 96.2 | 8.4 / 93.4 | 6.3 / 96.7 | 5.2 / 97.2 | 21.4 / 75.9 |
| DA-V2-Large (teacher) | 335 | 652.7 | 0.3 | 54457 | 5.1 / 97.1 | 7.7 / 94.9 | 5.1 / 97.8 | 4.2 / 98.0 | 21.0 / 76.7 |

- The DIODE column is the full set: 325 indoor + 446 outdoor images. No DIODE-indoor split was reported.
- The table also includes DPT-Hybrid, DPT-Large, Omnidata-v2 and Marigold.
- Fig. 2 plots energy vs. accuracy (mJ/frame vs. mean AbsRel), not parameters vs. accuracy.
- Source for this block: [ZipDepth](https://arxiv.org/html/2607.08771v1)

**DepthART Table 1 (zero-shot affine-invariant, per-image LS scale+shift on depth; AbsRel / δ1)**

\* = in-domain for MiDaS.

| Method | Params | NYU | KITTI | ETH3D | DIODE-Full | DDAD |
|---|---|---|---|---|---|---|
| DepthART-S | 6.0M | 0.059 / 0.964 | 0.082 / 0.930 | 0.084 / 0.950 | 0.214 / 0.745 | 0.095 / 0.900 |
| DepthART-B | 11.4M | 0.057 / 0.969 | 0.088 / 0.929 | 0.092 / 0.954 | 0.214 / 0.746 | 0.094 / 0.906 |
| DepthART-L | 32.6M | 0.053 / 0.971 | 0.079 / 0.933 | 0.091 / 0.958 | 0.208 / 0.754 | 0.095 / 0.910 |
| MiDaS v3.1 LeViT | 51M | 0.090* / 0.915* | 0.127* / 0.830* | 0.129 / 0.875 | 0.240 / 0.698 | 0.127 / 0.836 |
| MiDaS v3.1 Swin2-T | 42M | 0.075* / 0.942* | 0.103* / 0.883* | 0.119 / 0.886 | 0.229 / 0.718 | 0.117 / 0.848 |
| Metric3D v2 ViT-S | 37.5M | 0.056 / 0.965 | 0.064 / 0.950 | 0.062 / 0.955 | 0.247 / 0.789 | – |
| DAv2 ViT-S | 24.8M | 0.053 / 0.973 | 0.078 / 0.936 | 0.142 / 0.851 | 0.228 / 0.751 | 0.085 / 0.921 |
| AnyDepth ViT-S | 26.5M | 0.082 / 0.932 | 0.102 / 0.883 | 0.084 / 0.935 | 0.247 / 0.714 | – |

Source: [DepthART](https://arxiv.org/html/2607.17099)

**DepthART Table 2 (metric, no re-scaling; δ1 / RMSE)**
- Fine-tuning: NYU and iBims-1 results come from the NYU fine-tune; KITTI results come from the KITTI fine-tune.
- Column assignment for SUN RGB-D, DIODE-indoor, DDAD and the outdoor columns comes from the fetched table layout. The extracted header was ambiguous, so verify against the PDF.

| Method | NYU (in-domain) | iBims-1 (zero-shot) | SUN RGB-D (zero-shot) | DIODE-Indoor (zero-shot) |
|---|---|---|---|---|
| DepthART-S (6.0M) | 0.924 / 0.340 | 0.713 / 0.693 | 0.809 / 0.379 | 0.292 / 1.687 |
| DepthART-B (11.4M) | 0.942 / 0.307 | 0.732 / 0.672 | 0.831 / 0.342 | 0.287 / 2.018 |
| DepthART-L (32.6M) | 0.946 / 0.295 | 0.747 / 0.646 | 0.847 / 0.326 | 0.298 / 2.012 |
| Depth Anything v1-L | 0.984 / 0.206 | 0.639 / 0.965 | 0.647 / 0.585 | 0.291 / 2.504 |
| FastDepth | 0.771 / 0.604 | 0.429 / 1.607 | 0.560 / 0.736 | 0.275 / 2.778 |
| DANet | 0.831 / 0.488 | 0.55 / 1.11 | 0.616 / 0.623 | 0.221 / 2.909 |
| GuideDepth | 0.840 / 0.478 | 0.560 / 1.014 | 0.603 / 0.657 | 0.271 / 2.766 |

- Ablation: full fine-tuning drops iBims-1 to δ1 0.572 / RMSE 0.844. The frozen CamFT setup gets 0.713 / 0.693.
- Source for this block: [DepthART](https://arxiv.org/html/2607.17099)

**Ultralytics YOLO26x-depth (57M)**
- Scale-aligned δ1: iBims-1 0.961, ETH3D 0.959, NYU 0.933.
- Smaller sizes are not reported on these benchmarks.
- Source: [Ultralytics](https://docs.ultralytics.com/tasks/depth)

**Protocol conflicts on DAv2-S (important when reusing numbers)**
- ETH3D (AbsRel / δ1):
  - DepthART, AnyDepth and the DAv2 paper's own setting give 0.142 / 0.851 — [DepthART](https://arxiv.org/html/2607.17099); [AnyDepth](https://arxiv.org/html/2601.02760)
  - ZipDepth's own LS-aligned re-evaluation gives 0.063 / 0.967 — [ZipDepth](https://arxiv.org/html/2607.08771v1)
- DIODE (AbsRel / δ1):
  - AnyDepth copies 7.3 / 94.2. This is presumably the DAv2 paper's DIODE number, on a different subset or protocol.
  - ZipDepth (DIODE-full) gives 21.4 / 75.9.
  - DepthART (DIODE-full) gives 0.228 / 0.751.

### Inferences
- For a 5.04M metric student, the most like-for-like published comparison is DepthART-S Table 2: metric, NYU fine-tuned, zero-shot iBims-1 δ1 0.713 / RMSE 0.693, SUN RGB-D δ1 0.809, DIODE-indoor δ1 0.292, no re-scaling.
- For relative (LS-aligned) zero-shot, ZipDepth Table 2 and DepthART Table 1 give 2-12M reference points.
- DAv2-S ETH3D and DIODE numbers must be recomputed under one protocol. Cross-paper copies differ by about 2x in AbsRel.

### Gaps
- No found paper tabulates sub-15M models zero-shot on all of IBims-1, NYU, SUN RGB-D, DIODE-indoor and ETH3D-indoor together with median or LS alignment.
- DepthART's metric table uses ETH3D-outdoor and has an ambiguous header.
- No robustness benchmark paper (e.g., RoboDepth-style) including sub-15M zero-shot models was found in 2025-2026.

## Q4. Are there explicit accuracy-vs-parameters Pareto plots or tables with sub-15M depth models, and which models are directly comparable to a 5.04M metric student?

### Takeaway
I found no explicit params-vs-accuracy Pareto figure. The closest substitutes are:
- ZipDepth's energy-vs-mean-AbsRel bubble plot on Orin NX.
- The ZipDepth and DepthART tables, which list params next to accuracy.
- The YOLO26-depth size ladder (6.3-57M).

The directly comparable peers (same size class, zero-shot capable, weights public or claimed) are DepthART-S/B (6.0M / 11.4M; metric and relative), ZipDepth (6.1M; relative only) and YOLO26n/s-depth (6.3M / 13.2M; metric).

### Cited Findings
- ZipDepth Fig. 2 plots energy per frame vs. mean AbsRel over 5 datasets on Orin NX 15 W, with bubble size = FPS. There is no params-vs-accuracy plot — [ZipDepth](https://arxiv.org/html/2607.08771v1)
- ZipDepth's data-scale ablation: NYU AbsRel / δ1 goes from 12.2 / 84.9 with 1% of the 14.1M images to 8.4 / 93.3 with 100% — [ZipDepth](https://arxiv.org/html/2607.08771v1)
- The DepthART S/B/L ladder gives an accuracy-vs-size curve. NYU AbsRel is 0.059 / 0.057 / 0.053 and iBims-1 metric δ1 is 0.713 / 0.732 / 0.747 at 6.0 / 11.4 / 32.6M — [DepthART](https://arxiv.org/html/2607.17099)
- The YOLO26-depth size ladder: NYU δ1 0.882 → 0.933 from 6.3M to 57M (TTA, log-LS) — [Ultralytics](https://docs.ultralytics.com/tasks/depth)

Comparability of each candidate:

| Candidate | Params | Output | Zero-shot indoor numbers | Weights | Fit vs 5.04M metric student |
|---|---|---|---|---|---|
| DepthART-S | 6.0M | metric (after NYU fine-tune) and relative | iBims-1, SUN RGB-D, DIODE-indoor (metric); NYU, ETH3D, DIODE (relative) | code claimed on GitHub; weights not confirmed | best match |
| DepthART-B | 11.4M | same as DepthART-S | same as DepthART-S | same as DepthART-S | good match |
| ZipDepth | 6.1M | relative only | NYU, ETH3D, ScanNet, DIODE-full (relative) | not stated | compare on relative-protocol numbers only |
| YOLO26n-depth | 6.3M | metric | NYU only (likely not zero-shot) | auto-downloaded | good match for size and output type |
| YOLO26s-depth | 13.2M | metric | NYU only (likely not zero-shot) | auto-downloaded | good match for size and output type |
| GuideDepth† / FastDepth† / Lite-Mono† / PyDNet† | 5.8 / 4.0 / 8.3 / 1.9M | relative (ZipDepth retrains) | as in ZipDepth Table 2 | original code is public; the retrained weights are not | useful size-class reference |
| Hailo fast_depth / scdepthv3 | 1.35M / 14.8M | NYU-trained | none (NYU RMSE only) | Hailo Model Zoo | not zero-shot models |

Sources: [DepthART](https://arxiv.org/html/2607.17099); [ZipDepth](https://arxiv.org/html/2607.08771v1); [Ultralytics](https://docs.ultralytics.com/tasks/depth); [Hailo MZ](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8/HAILO8_depth_estimation.rst)

- DAv2-S, DA3-S, MiDaS-small, AnyDepth-S and Metric3D v2 ViT-S are not in the sub-15M class (24.8M, ~24.7M to 0.08B, 16.6-21.3M, 26.5M and 37.5M respectively). Use them only as "next size up" anchors — [AI Hub](https://aihub.qualcomm.com/mobile/models/depth_anything_v3); [DepthART](https://arxiv.org/html/2607.17099)

### Inferences
- A 5.04M metric student fits a Pareto table with these points: DepthART-S (6.0M), ZipDepth (6.1M), YOLO26n (6.3M), GuideDepth (5.8M), FastDepth (4.0M) and DepthART-B (11.4M), anchored by DAv2-S (24.8M).
- To beat DepthART-S on its own terms, the student would need iBims-1 metric δ1 > 0.713 / RMSE < 0.693 and SUN RGB-D δ1 > 0.809 after NYU-only metric training without re-scaling, assuming the column mapping is confirmed.

### Gaps
- The weight availability of DepthART and ZipDepth is unverified. Their GitHub and project pages were not opened.
- The YOLO26-depth licence and the per-size zero-shot iBims/ETH3D numbers are not published in the docs.
- A properly comparable Pareto plot would require re-running these models under one protocol (e.g., median alignment or no alignment for metric). No public source does this for sub-15M models.
