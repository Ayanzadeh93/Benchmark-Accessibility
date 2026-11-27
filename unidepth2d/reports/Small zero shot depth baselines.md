# DepthART-S emerges as b20dws's closest rival

Only one published model in the 5–10M band reports **raw metric, zero-shot indoor numbers** on the benchmarks we use: **DepthART-S** (6.0M relative, about 8.0M with its metric camera head). On iBims-1 it reaches δ1 0.713 / RMSE 0.693, on SUN RGB-D 0.809 / 0.379, and on DIODE-indoor 0.292 / 1.687. That metric head is fine-tuned on NYU and requires ground-truth intrinsics ([DepthART](https://arxiv.org/html/2607.17099)). Our 5.04M b20dws student, measured in our own harness, beats DepthART-S on iBims-1 δ1 (74.8 vs 71.3), roughly matches the 32.6M DepthART-L (74.7), and has a worse RMSE (0.735 vs 0.693). On SUN RGB-D it ties DepthART-B (83.0 vs 83.1), though SUN is partly in-domain for b20dws. Three other in-band candidates have released weights but publish thin or aligned-only evidence. **LightlyTrain DAv3-metric-tiny/tiny-plus** (6.2M / 7.9M, needs intrinsics) reports only NYU. **YOLO26n/s-depth** (6.3M / 13.2M, no intrinsics) reports only TTA plus log-least-squares-aligned numbers. **ZipDepth** (6.1M, MIT) is relative-only and reports affine-aligned NYU, ETH3D and DIODE-full. Older NYU-supervised compact nets (GuideDepth 5.8M, DANet 8.2M, FastDepth 3.9M) collapse to δ1 of about 0.43–0.56 on iBims-1, which makes them a floor, not a rival. Every foundation-family "small" model is above 15M: UniDepthV2-S at 34.18M is the key teacher-family anchor, with raw zero-shot iBims δ1 86.6 and SUN δ1 90.8 ([UniDepthV2](https://arxiv.org/html/2502.20110v2)). Most published numbers use incompatible alignment, crops and subsets, so our paper should re-run about eight models in our harness rather than copy any cell. DepthART-S/B metric, LightlyTrain tiny/tiny-plus and UniDepthV2-S come first.

**Labelling convention used throughout.** Numbers tagged *[ours]* come from our harness (raw metric unless stated). Every other number is *published* and carries a citation. In our harness, NYU means our 602-frame "clean" subset of the 654-image Eigen test set, and SUN RGB-D means 4,396 frames. Published NYU numbers use 654 frames unless noted.

## Three model families fill the 2–15M metric band, and none is intrinsics-free and raw-metric validated

The search turned up three kinds of compact model that output metric depth. Each fits a different evaluation regime. **DepthART** distils Depth Anything V2-Large into a TinyViM (Mamba-style) encoder plus a DPT decoder, at **6.0M (S), 11.4M (B) and 32.6M (L)**. Its distillation set is a 1.7M-image "bias-resistant" subset of a 44M corpus drawn from 11 sources ([DepthART](https://arxiv.org/html/2607.17099)). Metric depth comes from a second stage, CamFT. CamFT freezes the encoder and trains camera adapters, a scale head and the decoder on **NYU only** (d_max 10 m) or KITTI only (80 m). It turns **known intrinsics** into a Fourier-encoded ray field ([DepthART](https://arxiv.org/html/2607.17099)), which brings the metric-S checkpoint to **8.0M** ([project page](https://xuefeng-cvr.github.io/DepthART)). Weights sit on Hugging Face under **Apache-2.0** in `metric/`, `relative/` and `onnx/` folders. The visible files, however, are MobileNetV4-S relative checkpoints from a September 2026 update, so it is unverified whether TinyViM S/B *metric* heads are actually released ([HF Fengxue93/DepthART](https://huggingface.co/Fengxue93/DepthART/tree/main)).

**LightlyTrain's distilled Depth Anything V3 metric models** cover the same band: `dav3-metric-tiny` (DINOv3 ViT-T, **6.2M**) and `dav3-metric-tiny-plus` (ViT-T+, **7.9M**). Both are distilled from the official DAv3-metric-Large and load with `lightly_train.load_model(...)`. They **require intrinsics**, though an FOV approximation is accepted. Neither the training data nor the weight licence is stated ([Lightly docs](https://docs.lightly.ai/train/stable/depth_estimation.html)).

**Ultralytics YOLO26-depth** is the only intrinsics-free option in band: **n 6.3M** (46.9 GFLOPs at 768 px, 2.7 ms on T4 TensorRT FP16) and **s 13.2M**. It was trained on an undisclosed mix of about 2.19M indoor and outdoor images, which includes KITTI Eigen drives. Its released checkpoints carry **one global scale calibration** and take no camera input ([Ultralytics](https://docs.ultralytics.com/tasks/depth)). No licence appears on the depth pages. Ultralytics' usual AGPL-3.0 licence is an unverified assumption, and it would matter for any product use.

The nearest structural analogue to b20dws (an EfficientViT student distilled from a metric foundation model) is **"The Midas Touch for Metric Depth"**. It distils DAv2 into EfficientViT-B0/B1 and TinyViT students, but it recovers scale from **sparse 3D seeds**, so it is not pure monocular and is not a valid baseline ([MTD](https://arxiv.org/html/2605.11578v1)). The search found **no published MobileNet or EfficientViT student distilled from UniDepth, Metric3D or Depth Pro**. That absence, though not exhaustively verified, means b20dws's recipe appears to be unoccupied in the literature.

Below 15M, the remaining metric models are NYU-supervised CNNs. Only **GuideDepth (5.8M; GuideDepth-S 5.7M)** and **DANet (8.2M, EfficientNet-B0)** fall in the 5–10M priority band ([LMDepth table](https://arxiv.org/html/2505.00980v1); [DANet](https://ar5iv.labs.arxiv.org/html/2203.04538)). In the 2–4M near-miss tier are **FastDepth (3.9M)**, **METER-S (3.29M, MIT, weights claimed)** ([METER](https://arxiv.org/html/2403.08368v1)) and **LMDepth (2.9M, no code)** ([LMDepth](https://arxiv.org/html/2505.00980v1)). All of these bake NYU's camera into their output scale, so their cross-dataset metric numbers mix generalization failure with focal-length mismatch. GuideDepth has confirmed Google Drive checkpoints at NYU half and full resolution, with no licence stated ([GuidedDecoding](https://github.com/mic-rud/GuidedDecoding)). Checkpoints for DANet and FastDepth were not confirmed. One parameter conflict is worth noting: Hailo's zoo lists its fast_depth at **1.35M**, against 3.9M in the papers ([Hailo MZ](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8/HAILO8_depth_estimation.rst)).

Relative/affine models in band are rarer. **ZipDepth** is a RepVGG-style CNN distilled from DAv2-L on about 14.1M images, with **6.14M params at inference**, **3.0 GMACs** and an **MIT** licence. Its checkpoints and inference, export and eval scripts are in the repo ([ZipDepth arXiv](https://arxiv.org/html/2607.08771v1); [GitHub](https://github.com/fabiotosi92/ZipDepth)). DepthART's Stage-1 relative heads are the other option. No official MiDaS, Depth Anything, Marigold, Lotus or Depth Pro checkpoint falls below 15M.

| Model | Params | Output | Intrinsics | Weights / licence | Band |
|---|---|---|---|---|---|
| DepthART-S / B (metric) | 8.0M / ~11.4M+ | metric (NYU-ft) + relative | required (GT K) | HF, Apache-2.0; S/B metric files unverified | 5–10 / 10–15 |
| LightlyTrain DAv3-metric tiny / tiny+ | 6.2M / 7.9M | metric | required (FOV ok) | pip auto-download; licence unstated | 5–10 |
| YOLO26n / s-depth | 6.3M / 13.2M | metric, global scale | none | Ultralytics package; licence unstated | 5–10 / 10–15 |
| ZipDepth | 6.1M | affine inverse depth | none | repo, MIT | 5–10 |
| GuideDepth / -S | 5.8M / 5.7M | metric (NYU-supervised) | baked-in NYU | Google Drive; no licence | 5–10 |
| DANet | 8.2M | metric (NYU-supervised) | baked-in NYU | code; weights unverified | 5–10 |
| FastDepth, METER-S, LMDepth | 3.9M, 3.3M, 2.9M | metric (NYU-supervised) | baked-in NYU | unverified / MIT / none | 2–5 near-miss |
| MiDaS v2.1 Small | 16.6–21.3M | affine inverse depth | none | MIT, archived repo | >15 |
| DAv2-S, Distill-Any-Depth-S | 24.8M | affine inverse depth | none | Apache-2.0 tags | >15 |
| DAv2-metric-small-hypersim | 24.8M | metric, indoor ≤20 m | none | Apache-2.0 (LightlyTrain) | >15 |
| UniDepthV2-S / UniK3D-S | 34.18M / 35.4M | metric | optional | CC BY-NC 4.0 | >15 |
| Metric3Dv2 ViT-S | 37.5M | metric | optional (default focals) | code BSD-2; weights licence unstated | >15 |

Sources for the table: [DepthART](https://arxiv.org/html/2607.17099), [Lightly](https://docs.lightly.ai/train/stable/depth_estimation.html), [Ultralytics](https://docs.ultralytics.com/tasks/depth), [ZipDepth GitHub](https://github.com/fabiotosi92/ZipDepth), [GuidedDecoding](https://github.com/mic-rud/GuidedDecoding), [MiDaS](https://github.com/isl-org/MiDaS), [Distill-Any-Depth HF](https://huggingface.co/xingyang1/Distill-Any-Depth/tree/main/small), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [UniK3D HF](https://huggingface.co/lpiccinelli/unik3d-vits), [Metric3D](https://github.com/YvanYin/Metric3D). The MiDaS parameter count conflicts between sources: 16.6M on [Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models/midas) and 21M in the README.

## Published zero-shot numbers split into three incompatible protocols

The published evidence falls into three protocols that cannot share a column. The first is **raw metric** (no per-image fitting). The second is **scale-only alignment** (median, or log-least-squares). The third is **affine scale-and-shift least squares (SSI-LSQ)**, usually fitted in disparity. Only the first is comparable to b20dws's headline numbers. The second and third are comparable to b20dws's median-scaled and log-LS-scaled numbers *only approximately*, because SSI-LSQ also removes a shift and is the most forgiving protocol.

### Raw metric: DepthART's Table 2 is the only compact source for iBims, SUN and DIODE-indoor

DepthART's metric table uses the NYU-fine-tuned head with **no rescaling** and GT intrinsics. NYU is in-domain; iBims-1, SUN RGB-D and DIODE-indoor are zero-shot ([DepthART](https://arxiv.org/html/2607.17099)). The same table re-evaluates three NYU-supervised baselines.

| Model (published, δ1 / RMSE m) | NYU (ID) | iBims-1 | SUN RGB-D | DIODE-indoor |
|---|---|---|---|---|
| DepthART-S (6.0M; metric 8.0M) | 0.924 / 0.340 | 0.713 / 0.693 | 0.809 / 0.379 | 0.292 / 1.687 |
| DepthART-B (11.4M) | 0.942 / 0.307 | 0.732 / 0.672 | 0.831 / 0.342 | 0.287 / 2.018 |
| DepthART-L (32.6M) | 0.946 / 0.295 | 0.747 / 0.646 | 0.847 / 0.326 | 0.298 / 2.012 |
| GuideDepth (5.8M) | 0.840 / 0.478 | 0.560 / 1.014 | 0.603 / 0.657 | 0.271 / 2.766 |
| DANet (8.2M) | 0.831 / 0.488 | 0.55 / 1.11 | 0.616 / 0.623 | 0.221 / 2.909 |
| FastDepth (3.9M) | 0.771 / 0.604 | 0.429 / 1.607 | 0.560 / 0.736 | 0.275 / 2.778 |

Source: [DepthART Table 2](https://arxiv.org/html/2607.17099). AbsRel was not extracted.

Three cautions apply to this table. First, the HTML extraction flagged the **column pairing as ambiguous**, so the SUN and DIODE-indoor assignments should be checked against the PDF. Second, every model, including Depth Anything v1-L, sits at δ1 of about 0.22–0.30 on DIODE-indoor. That pattern points to a protocol or range-cap artefact rather than a model property. Third, DepthART and DANet share authors, and DANet's iBims numbers match DANet's own paper exactly. DANet's own paper gives δ1 0.55 / REL 0.26 / RMS 1.11 for DANet and δ1 0.30 / REL 0.38 for FastDepth ([DANet](https://ar5iv.labs.arxiv.org/html/2203.04538)). DepthART's FastDepth row is far kinder (0.429), so the two papers' protocols evidently differ. Hu et al. offer an independent SUN RGB-D check: FastDepth δ1 0.404 / REL 0.376 / RMSE 0.662, and a 1.7M KD student at 0.430 / 0.338 / 0.577 ([Hu et al.](https://ar5iv.labs.arxiv.org/html/2105.06143)). Against DepthART's 0.560 for FastDepth, the same model's SUN δ1 differs by 16 points between papers. Neither paper is confirmed to remove NYU test images from SUN RGB-D, whereas UniDepth does ([UniDepth](https://arxiv.org/pdf/2403.18913)).

The other raw-metric compact numbers are NYU-only. **LightlyTrain DAv3-metric** reports raw δ1 with intrinsics on NYU (654 Eigen, 0.1–10 m, NYU not in training): **tiny 0.818, tiny-plus 0.846**, small (24.7M) 0.912. The corresponding aligned δ1 values are 0.915, 0.923 and 0.938, with AbsRel 0.131 / 0.123 / 0.099 and RMSE 0.506 / 0.457 / 0.377. The page does not specify which protocol the AbsRel and RMSE columns use ([Lightly docs](https://docs.lightly.ai/train/stable/depth_estimation.html)). The above-band anchor **UniDepthV2-S (34.18M)** has the cleanest protocol of all: raw metric, intrinsics-free, NYU and KITTI excluded from training. It scores **SUN δ1 90.8 / ARel 10.5, iBims δ1 86.6 / ARel 13.5 and ETH3D δ1 64.6** ([UniDepthV2](https://arxiv.org/html/2502.20110v2)). It reports neither NYU nor DIODE-indoor.

### Scale-aligned: YOLO26's numbers inflate under TTA and least squares

YOLO26 publishes only aligned numbers. With multi-scale plus flip TTA and log-least-squares scale, NYU (654) scores **n 0.882 δ1 / 0.109 AbsRel / 0.414 RMSE** and **s 0.896 / 0.104 / 0.399**. Without TTA and with median scaling, NYU δ1 falls to **n 0.783, s 0.793** ([Ultralytics](https://docs.ultralytics.com/tasks/depth)); the Russian docs copy gives 0.785 and 0.786 ([RU docs](https://docs.ultralytics.com/ru/tasks/depth)). iBims-1 is published only with TTA plus alignment: **n 0.923, s 0.899**. The s variant scoring below n is an unexplained inversion ([Ultralytics iBims-1](https://docs.ultralytics.com/datasets/depth/ibims-1)). ETH3D is published only for the 57M x variant (0.959). No raw-metric, SUN RGB-D or DIODE numbers exist for any size. Ultralytics does not state whether NYU was in the 2.19M training mix, so its "zero-shot" NYU label is unconfirmed.

### Affine SSI-LSQ: ZipDepth and DepthART cover NYU, ETH3D and DIODE-full

For relative models, ZipDepth's Table 2 and DepthART's Table 1 give zero-shot SSI-LSQ results ([ZipDepth](https://arxiv.org/html/2607.08771v1); [DepthART](https://arxiv.org/html/2607.17099)). Both use DIODE-full (indoor plus outdoor), not DIODE-indoor. **No compact relative model reports affine iBims-1 or SUN RGB-D.**

| Model (published, AbsRel / δ1) | Params | NYU | ETH3D | DIODE-full | Source |
|---|---|---|---|---|---|
| DepthART-S | 6.0M | 0.059 / 0.964 | 0.084 / 0.950 | 0.214 / 0.745 | DepthART T1 |
| DepthART-B | 11.4M | 0.057 / 0.969 | 0.092 / 0.954 | 0.214 / 0.746 | DepthART T1 |
| ZipDepth | 6.1M | 0.084 / 0.933 | 0.100 / 0.922 | 0.226 / 0.739 | ZipDepth T2 |
| Lite-Mono† (retrained) | 8.3M | 0.087 / 0.927 | 0.087 / 0.936 | 0.224 / 0.729 | ZipDepth T2 |
| GuideDepth† (retrained) | 5.8M | 0.092 / 0.918 | 0.102 / 0.919 | 0.227 / 0.724 | ZipDepth T2 |
| FastDepth† (retrained) | 4.0M | 0.103 / 0.898 | 0.123 / 0.872 | 0.234 / 0.710 | ZipDepth T2 |
| MiDaS v2.1 Small | 21.3M | 0.115 / 0.868 | 0.134 / 0.837 | 0.242 / 0.711 | ZipDepth T2 |
| DAv2-S | 24.8M | 0.061 / 0.962 | 0.063 / 0.967 | 0.214 / 0.759 | ZipDepth T2 |
| DAv2-S | 24.8M | 0.053 / 0.973 | 0.142 / 0.851 | 0.228 / 0.751 | DepthART T1 |
| Metric3Dv2-S (aligned) | 37.5M | 0.056 / 0.965 | 0.062 / 0.955 | 0.247 / 0.789 | DepthART T1 |

† = retrained on ZipDepth's 14.1M DAv2-L pseudo-labels; release of these weights is unconfirmed.

The two DAv2-S rows show why this table cannot be used as-is: **DAv2-S's ETH3D AbsRel is 0.142 in DepthART and 0.063 in ZipDepth**, a 2.3x gap for the same checkpoint. A third paper reports NYU AbsRel 0.137 for DAv2-S under SSI-LSQ ([Sensors 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13807952/)). DepthART-S therefore "beats" ZipDepth only inside each paper's own pipeline. For example, the 0.950 vs 0.922 gap on ETH3D falls well within the cross-paper noise seen on DAv2-S.

## b20dws already matches DepthART-L on iBims raw metric and ties YOLO26n once aligned

Set against these published figures, our own measurements place b20dws favourably, but with caveats that a reviewer will raise. The table pairs each *[ours]* number with the closest published protocol.

| Dataset | b20dws 5.04M [ours] | YOLO26n 6.36M [ours] | Closest published compact peer | Above-band anchor (published) |
|---|---|---|---|---|
| iBims-1 raw | δ1 74.8 / AbsRel 0.179 / RMSE 0.735 | δ1 56.7 / AbsRel 0.254 | DepthART-S 71.3 / RMSE 0.693; -B 73.2 / 0.672; -L 74.7 / 0.646 | UniDepthV2-S δ1 86.6 / ARel 0.135 |
| iBims-1 aligned | median 86.5; log-LS 87.9 | log-LS 87.9; flip TTA 88.7 | YOLO26n 92.3 (multi-scale + flip TTA, log-LS) | none |
| NYU raw | δ1 85.0 / AbsRel 0.123 (602 clean, partly ID) | none | LightlyTrain tiny 81.8, tiny+ 84.6 (654, with K); DepthART-S 92.4 (NYU-ft, ID) | LightlyTrain small 91.2 |
| NYU aligned | median 89.5 | median 82.0 | YOLO26n 78.3 (median, no TTA); LightlyTrain tiny 91.5 (alignment unspecified) | none |
| SUN RGB-D raw | δ1 83.0 / AbsRel 0.138 / RMSE 0.400 (4,396, partly ID) | none | DepthART-S 80.9 / 0.379; -B 83.1 / 0.342 | UniDepthV2-S 90.8 / ARel 0.105 |
| DIODE-indoor raw | δ1 36.9 / AbsRel 0.330 (325) | none | DepthART-S 29.2 / RMSE 1.687 | none |
| DIODE-indoor aligned | median 83.9 | none | none (DIODE-full SSI only: DepthART-S 74.5, ZipDepth 73.9) | none |
| ETH3D | pending | none | DepthART-S 95.0, ZipDepth 92.2 (SSI-LSQ) | UniDepthV2-S raw 64.6 |

Published-number sources: [DepthART](https://arxiv.org/html/2607.17099), [Lightly](https://docs.lightly.ai/train/stable/depth_estimation.html), [Ultralytics](https://docs.ultralytics.com/tasks/depth), [Ultralytics iBims-1](https://docs.ultralytics.com/datasets/depth/ibims-1), [ZipDepth](https://arxiv.org/html/2607.08771v1), [UniDepthV2](https://arxiv.org/html/2502.20110v2).

**On iBims-1, the strongest zero-shot claim, b20dws leads its size class on δ1 but not on RMSE.** It scores δ1 3.5 points above DepthART-S and matches DepthART-L, a model 6.5x larger, yet its RMSE (0.735 m) is 0.04–0.09 m worse than all three DepthART sizes. This pattern fits a model whose typical scale is better but whose far-range errors are larger, and our harness can test that by stratifying errors by depth. There is a hidden assumption in comparing the two. DepthART is given **GT intrinsics** and is NYU-fine-tuned. If b20dws runs without intrinsics, the comparison is tilted against b20dws, and the δ1 lead is stronger evidence than it looks. If b20dws receives GT K, the comparison is like-for-like. The paper must state which. The 12-point δ1 gap to UniDepthV2-S (86.6), the smallest member of the teacher family, is the honest measure of what distillation to 5M costs.

**NYU and SUN RGB-D are the weakest comparisons** because b20dws saw these domains in training. On NYU, b20dws's raw 85.0 sits between LightlyTrain tiny-plus (84.6) and DepthART-S (92.4), but none of these numbers is cleanly zero-shot. LightlyTrain excludes NYU but uses intrinsics and 654 frames. DepthART is NYU-fine-tuned. b20dws is partly in-domain and evaluated on 602 frames. On SUN, b20dws ties DepthART-B on δ1 and trails on RMSE (0.400 vs 0.342), with both sides potentially inflated by unfiltered NYU overlap.

**DIODE-indoor shows the gap between raw and aligned results most clearly.** Every metric model, b20dws included, scores raw δ1 below 0.37. Median scaling lifts b20dws to 83.9, so the failure is almost entirely global scale. DIODE's long-range laser ground truth against a 10 m-trained scale prior explains this better than a shape failure does.

**YOLO26n shows how far protocol alone moves the numbers.** In our harness its iBims log-LS δ1 is 87.9, or 88.7 with flip TTA, against a published 92.3 with multi-scale plus flip TTA: a **3.6–4.4 point** gap in the published number's favour. On NYU, our median-scaled 82.0 *exceeds* the published median-scaled 78.3 by 3.7 points. That gap most likely comes from our 602-frame clean subset and different crop. The same checkpoint therefore swings about ±4 δ1 points across harnesses, in both directions. That is larger than the b20dws vs DepthART-S iBims gap. Any cross-paper cell in our tables should carry a "published, protocol differs" marker. Where a claim matters, the baseline must be re-run.

## Run eight baselines, led by DepthART and LightlyTrain with GT intrinsics

The priority order below follows three criteria: how directly a model challenges b20dws's metric zero-shot claim at equal size, whether its weights load, and whether it fills a protocol hole that published numbers leave open.

| Priority | Model | Params | Run modes in our harness | Why | Main risk |
|---|---|---|---|---|---|
| 1 | DepthART-S metric (+ relative head) | 8.0M (6.0M rel.) | raw with GT K; median; log-LS; SSI-LSQ (relative head) | Only same-size raw-metric peer with iBims/SUN/DIODE numbers; lets us verify its Table 2 under our crops | S/B TinyViM metric files may not be on HF; might need CamFT re-training |
| 1 | DepthART-B metric | ~11.4M+ | same as S | 10–15M point; ties b20dws on SUN | same |
| 1 | LightlyTrain DAv3-metric tiny / tiny-plus | 6.2M / 7.9M | raw with GT K and with FOV approximation; median; log-LS | Same-size metric students of a foundation teacher, which is the closest recipe to b20dws; no iBims/SUN/DIODE/ETH3D published | Licence and training data unstated (possible overlap with eval sets) |
| 1 | UniDepthV2-S | 34.18M | raw without K and with GT K; median | Smallest teacher-family member; quantifies the distillation gap; adds DIODE-indoor and NYU, which it never reported | CC BY-NC 4.0 (fine for research eval) |
| 2 | ZipDepth | 6.1M | SSI-LSQ in disparity; median; log-LS on the depth inverse | Best released same-size relative model (MIT); gives affine iBims/SUN/DIODE-indoor numbers nobody has published | Disparity-space vs depth-space alignment choice changes results |
| 2 | YOLO26s-depth (and finish YOLO26n on SUN, DIODE, ETH3D) | 13.2M (6.3M) | raw with shipped global scale; median; log-LS ± TTA | Only intrinsics-free compact metric family; raw numbers are unpublished | Licence unverified (likely AGPL); training mix undisclosed |
| 2 | GuideDepth / GuideDepth-S | 5.8M / 5.7M | raw; median | In-band NYU-supervised floor with confirmed weights; checks DepthART's baseline rows | No licence stated; NYU camera baked in |
| 3 | DAv2-S, MiDaS v2.1 Small, DAv2-metric-small-hypersim, Metric3Dv2-S | 24.8M, 16.6–21.3M, 24.8M, 37.5M | SSI-LSQ for relative; raw + median for metric | Next-size-up anchors; DAv2-S re-run resolves the 2.3x cross-paper ETH3D conflict | Compute only |

Sources for licences and weights: [DepthART HF](https://huggingface.co/Fengxue93/DepthART/tree/main), [Lightly](https://docs.lightly.ai/train/stable/depth_estimation.html), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [ZipDepth GitHub](https://github.com/fabiotosi92/ZipDepth), [Ultralytics](https://docs.ultralytics.com/tasks/depth), [GuidedDecoding](https://github.com/mic-rud/GuidedDecoding), [MiDaS](https://github.com/isl-org/MiDaS), [Metric3D](https://github.com/YvanYin/Metric3D).

Several candidates should be skipped or deprioritized. **X-Lens** (about 40M) needs GT intrinsics *and* extrinsics and has no standard indoor single-image benchmarks ([X-Lens](https://arxiv.org/html/2607.12993)). **LMDepth** has no code. **Student8** (17.7M) provides code only on request, and **DANIEL** (24.5M) is robot-domain-specific ([Sensors 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13807952/); [DANIEL](https://arxiv.org/html/2611.32718)). **The Midas Touch** depends on sparse 3D seeds ([MTD](https://arxiv.org/html/2605.11578v1)). DANet, FastDepth and METER-S are optional extras. METER in particular failed to reproduce in DepthART's hands ([DepthART](https://arxiv.org/html/2607.17099)).

The harness changes matter as much as the model list. Report four columns per dataset: raw metric, median scale, log-LS scale, and SSI-LSQ (for relative models only, fitted in disparity space and stated as such). Run every camera-aware model both with GT K and with its intrinsics-free fallback. Remove NYU test images from SUN RGB-D, as UniDepth does ([UniDepth](https://arxiv.org/pdf/2403.18913)). Publish frame counts and depth caps for every set. Pick and state an ETH3D split: published compact relative numbers use ETH3D without an indoor-only filter, and DepthART's metric table uses ETH3D-outdoor only, so neither matches an indoor-only evaluation. Finally, stratify iBims and DIODE errors by depth range to explain b20dws's δ1-good / RMSE-worse profile.

## Conclusion

The literature has a hole that b20dws fits. No published 5–10M model is both **intrinsics-free and validated raw-metric zero-shot indoors**. DepthART needs GT intrinsics and an NYU fine-tune, LightlyTrain's tiny models need intrinsics and report only NYU, and YOLO26 publishes only aligned numbers. A single harness table of all three, plus ZipDepth and UniDepthV2-S, under raw, median, log-LS and SSI protocols would be the first like-for-like compact metric benchmark on iBims-1, SUN RGB-D, DIODE-indoor and ETH3D. That table is a contribution in its own right.

The YOLO26n evidence also sets a practical bar. Our harness and the vendor's disagree by about ±4 δ1 points on the same checkpoint, so b20dws's 3.5-point iBims lead over DepthART-S counts only once DepthART-S is re-run in our pipeline. Until then, DIODE-indoor is the clearest place for b20dws to differentiate. There, every metric model's raw δ1 collapses for scale reasons while b20dws's median-scaled 83.9 shows its shape is intact. The highest-leverage next step is to confirm the contents of DepthART's HF `metric/` folder, because b20dws's headline comparison depends on that one baseline.
