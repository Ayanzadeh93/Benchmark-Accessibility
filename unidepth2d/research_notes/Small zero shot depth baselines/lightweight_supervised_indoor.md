# Lightweight supervised indoor monocular depth networks (~1-15M params) with cross-dataset / zero-shot evaluation

Scope: supervised (mostly NYU-v2-trained) compact networks. Foundation-model distillations and zero-shot metric foundation models are excluded except where a paper of that kind is the *only* source of zero-shot numbers for the lightweight supervised baselines (DepthART, used here only as a benchmark of FastDepth/DANet/GuideDepth). Research done 2026-11-27, 15 tool calls. Any value not traced to a fetched source is listed under Gaps.

## Q1. Candidate models: paper, params, FLOPs/latency, training set, metric output, weights, NYU-v2 numbers

### Takeaway
Six NYU-trained lightweight supervised models have verified parameter counts and NYU numbers: FastDepth (3.9M), Joint-depth/Nekrasov (2.99-3.1M), Hu et al. KD student (1.7M), METER (0.7/1.4/3.3M), GuideDepth (5.7/5.8M), DANet (8.2M), plus LMDepth (2.3/2.9M, 2025). Only DANet (8.2M) and GuideDepth (5.8M) fall in the 5-10M priority band. All of them output metric depth trained in metres on NYU (RMSE is reported in m); none is affine-invariant.

### Cited Findings

**Dense table: NYU-v2 (Eigen test split, 654 images, metric, no alignment unless noted)**

| Model (paper, venue, arXiv) | Params (exact, source) | Compute / speed / device | Train set | NYU REL | NYU RMSE (m) | NYU δ1 | Code / weights |
|---|---|---|---|---|---|---|---|
| FastDepth (Wofk et al.) | 3.9M | 0.75 GFLOPs @224x224-ish (DANet table) / 0.75 GFLOPs @240x320 and 4.00 @480x640 (LMDepth table); 7 ms on GTX 1080Ti (Hu et al.); 174 FPS in LMDepth's Table IV | NYU-v2 | 0.162 (DANet table); 0.135 @240x320, 0.142 @480x640 (LMDepth table) | 0.591 / 0.580 / 0.579 | 0.778 / 0.771 / 0.772 / 0.775 | see Gaps |
| Joint-depth (Nekrasov et al., "Real-time joint semantic segmentation and depth estimation using asymmetric annotations") | 2.99M (DANet table) / 3.1M (Hu et al.) / 3.0M (LMDepth) | 6.49 GFLOPs @480x640; 1.38 GFLOPs @240x320; 21 ms on GTX 1080Ti | NYU-v2 (+segmentation labels) | 0.149 | 0.565 | 0.790 | see Gaps |
| Hu et al. KD student, "Boosting Light-Weight Depth Estimation Via Knowledge Distillation", KSEM 2023, arXiv 2105.06143 | 1.7M (1.6M MobileNetV2 enc + 0.3M dec; teacher ResNet-34 21.9M) | FLOPs not reported; ~11 ms on GTX 1080Ti | NYU-v2 (~50k pairs) + 204k ScanNet images (unlabeled, or labeled in the U′ variant) | 0.131 (X∪U); 0.121 (X∪U′); 0.138 in DANet's table | 0.482; 0.461; 0.499 | 0.837; 0.855; 0.818 | github.com/JunjH/Boosting-Light-Weight-Depth-Estimation (weights not mentioned in paper) |
| METER-S / XS / XXS (Papa, Russo, Amerini), IEEE TCSVT 33(10):5882-5893, 2023, DOI 10.1109/TCSVT.2023.3260310; arXiv 2403.08368 | 3.29M / 1.45M / 0.71M | 0.975 / 0.579 / 0.186 GMACs on NYU (256x192 input); 16.3/18.3/25.8 FPS Jetson TX1, 12.0/15.6/23.2 FPS Jetson Nano | NYU-v2 (downsampled to 256x192); separate KITTI models | 0.134 / 0.154 / 0.174 | 0.471 / 0.522 / 0.580 | 0.831 / 0.793 / 0.744 | github.com/lorenzopapa5/METER, MIT licence; paper says code + pretrained weights public |
| GuideDepth / GuideDepth-S (Rudolph et al., "Lightweight Monocular Depth Estimation through Guided Decoding"), ICRA 2022, arXiv 2203.04206 | 5.8M / 5.7M (from LMDepth's table; not verified in GuideDepth paper itself) | 2.63 / 1.52 GFLOPs @240x320; 10.47 / 6.03 GFLOPs @480x640 (LMDepth text says 5.72 for GuideDepth full-res: inconsistency); 170 FPS in LMDepth Table IV; evaluated on Jetson Nano and Xavier NX (TensorRT 7.1.3) | NYU-v2 (DenseDepth subset); separate KITTI models | 0.138 @240x320; 0.129 @480x640 | 0.501; 0.449 | 0.823; 0.844 (GuideDepth-S: 0.812 / 0.830) | github.com/mic-rud/GuidedDecoding, Google Drive weights for NYU half/full res (both variants) and KITTI (GuideDepth only); licence not stated |
| DANet (Sheng, Xue, Chang, Liang, Ming, BUPT), "Monocular Depth Distribution Alignment with Low Computation", ICRA 2022, arXiv 2203.04538 | 8.2M (EfficientNet-B0 backbone) | 1.5 GFLOPs @228x304; no latency reported in own paper; 12 FPS in LMDepth Table IV | NYU-v2 | 0.135 | 0.488 | 0.831 | github.com/YiLiM1/DANet (code; weights not mentioned in paper) |
| LMDepth / LMDepth-S (Long, Zhou), arXiv 2505.00980 (May 2025, preprint) | 2.9M / 2.3M | 0.72 / 0.59 GFLOPs @240x320; 2.77 / 2.10 @480x640; Jetson AGX Xavier TensorRT INT8: 122 FPS low-res, 43 FPS 640x480 | NYU-v2; KITTI | 0.134 / 0.140 @240x320; 0.123 / 0.135 @480x640 | 0.471 / 0.476; 0.438 / 0.463 | 0.833 / 0.825; 0.854 / 0.833 | no code link found |

Sources for the table:
- LMDepth NYU Table I (FastDepth, JointDepth, TuMDE 4.6M, YinMDE 2.7M, DaNet, GuideDepth(-S), LMDepth(-S)), FPS, Jetson INT8 numbers — [LMDepth arXiv HTML](https://arxiv.org/html/2505.00980v1)
- DANet Table I (Wofk/FastDepth 3.9M 0.75G, Nekrasov 2.99M 6.49G, Yin 2.7M 15.6G, Hu 1.7M, DANet 8.2M 1.5G) — [DANet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.04538)
- Hu et al. student params, ScanNet auxiliary data, NYU tables and timings — [Hu et al. ar5iv](https://ar5iv.labs.arxiv.org/html/2105.06143); KSEM 2023 publication — [Unpaywall/Springer DOI 10.1007/978-3-031-40283-8_3](https://unpaywall.org/10.1007%2F978-3-031-40283-8_3)
- METER params, MACs, Jetson FPS, NYU/KITTI tables, weights claim — [METER arXiv HTML](https://arxiv.org/html/2403.08368v1); TCSVT venue/DOI — [IRIS Sapienza](https://iris.uniroma1.it/handle/11573/1675431); MIT licence — [METER GitHub](https://github.com/lorenzopapa5/METER)
- GuideDepth weights, ICRA 2022, Jetson toolchain — [GuidedDecoding GitHub](https://github.com/mic-rud/GuidedDecoding); arXiv id — [arXiv 2203.04206](https://arxiv.org/pdf/2203.04206)
- METER's own table lists other tiny baselines: MobileNetV2+FBNet 2.6M (NYU RMSE 0.564, δ1 0.790), SPEED 2.6M (RMSE 0.566, REL 0.158, δ1 0.783), CReaM (RMSE 0.687, REL 0.190, δ1 0.704) — [METER arXiv HTML](https://arxiv.org/html/2403.08368v1)
- DANet's re-implementations on EfficientNet-B0: Hu et al. (2019 SENet-style decoder) 5.3M / 14G, REL 0.142, RMS 0.505, δ1 0.814; Chen et al. 12M / 8.22G, REL 0.135, RMS 0.514, δ1 0.828; Yin et al. 4.6M / 18G, REL 0.145, RMS 0.567, δ1 0.771 — [DANet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.04538)

**Other candidates checked**
- MobileXNet (Dong, Garratt, Anavatti, Abbass, arXiv 2111.12334, 2021): evaluated on NYU-v2, KITTI, Make3D, Unreal; ~7-9 ms inference; best NYU row approx. RMSE 0.537, REL 0.146, δ1 0.799 (column reading uncertain). Parameter count 24.95M comes from a third-party table (EGD-Net), so it sits **outside** the 1-15M band — [MobileXNet ar5iv](https://ar5iv.labs.arxiv.org/html/2111.12334); [EGD-Net arXiv 2209.14829](https://arxiv.org/pdf/2209.14829)
- EGD-Net (Dong et al., arXiv 2209.14829, 2022): edge-guided lightweight network, about 96 FPS on GTX 1080, NYU-v2 only, no cross-dataset evaluation, no code link; claims >11x fewer params than MobileXNet, which implies roughly <2.3M (my arithmetic, not a stated number) — [arXiv abs](https://arxiv.org/abs/2209.14829); [PDF](https://arxiv.org/pdf/2209.14829)
- TuMDE (4.6M, NYU REL 0.162, RMS 0.536, δ1 0.772) and YinMDE (2.7M, REL 0.135, δ1 0.813) appear only as rows in LMDepth's table — [LMDepth](https://arxiv.org/html/2505.00980v1)
- SimMDE ("A simple monocular depth estimation network for balancing complexity and accuracy", PMC 2025): 30.9M params (outside range), reports iBims-1 and SUN RGB-D zero-shot tables; values not extracted — [PMC11997066](https://pmc.ncbi.nlm.nih.gov/articles/PMC11997066/)

### Inferences
- In the 5-10M band, only GuideDepth (5.7-5.8M) and DANet (8.2M) are verified supervised NYU models. METER-S (3.3M), FastDepth (3.9M), LMDepth (2.9M) and Joint-depth (3.0M) are the 2-4M tier.
- NYU numbers for the same model vary between papers (FastDepth REL 0.135 to 0.162), because of input resolution (224 / 228x304 / 240x320 / 480x640), crop, and re-implementation. Any comparison should re-run the official checkpoints under one protocol.
- All of these models predict metric depth with NYU's camera intrinsics baked in. Cross-dataset metric numbers therefore mix generalization with focal-length/scale mismatch. iBims-1 (DSLR) and DIODE (laser, longer ranges) are especially affected.

### Gaps
- FastDepth's primary paper (Wofk et al., ICRA 2019, arXiv 1903.03273) and repo (github.com/dwofk/fast-depth) were not fetched in this session. The licence, weight availability and exact param count (3.9M is consistent across three secondary tables) are unverified from the primary source.
- Joint-depth (Nekrasov et al., ICRA 2019): weights/repo (believed github.com/DrSleep/multi-task-refinenet) not verified.
- GuideDepth's exact parameter counts are from LMDepth's table, not from the GuideDepth paper. The GuideDepth repo states no licence. Jetson latency numbers were not extracted from the GuideDepth paper.
- DANet and Hu et al.: whether trained weights are in the repos was not verified.
- Not verified this session (no fetch): LiteDepth (MAI 2022 challenge, trained on the MAI/ZED dataset, not NYU), RT-MonoDepth, Lite-Mono, SwiftDepth, MiniNet. All of these are believed to be self-supervised/KITTI-centric or non-NYU and so outside "supervised NYU indoor". No indoor zero-shot numbers were found for them. "DepthFormer-lite", "PixelFormer-lite", "EfficientDepth" and "LightDepth" as NYU-supervised 1-15M models were not found in any fetched source and may not exist under those names.
- MAI 2021/2022/2023 depth challenge reports were not fetched. They use a challenge-specific (outdoor, ZED stereo) dataset, so they are unlikely to give NYU-to-indoor zero-shot numbers.

## Q2. Which report zero-shot results on SUN RGB-D, iBims-1, DIODE, ETH3D, ScanNet, TUM? Numbers?

### Takeaway
Three sources give zero-shot indoor numbers for lightweight supervised NYU models:
- **DANet**: iBims-1, for DANet, FastDepth and Joint-depth.
- **Hu et al.**: SUN RGB-D and TUM, for FastDepth, Joint-depth and the 1.7M KD student.
- **DepthART** (ACM MM 2026): a later benchmark that evaluates FastDepth, DANet and GuideDepth metrically, with no alignment, on iBims-1, SUN RGB-D and DIODE-Indoor (and outdoor sets).

No source evaluates these models on ETH3D-indoor or ScanNet-test. METER, GuideDepth's own paper and LMDepth report no public cross-dataset sets.

### Cited Findings

**iBims-1, zero-shot from NYU, metric (no fine-tuning; DANet Table, no DBE/planarity reported)**

| Model | Params | REL | RMS | log10 | δ1 | δ2 | δ3 |
|---|---|---|---|---|---|---|---|
| FastDepth (Wofk) | 3.9M | 0.38 | 1.76 | 0.21 | 0.30 | 0.56 | 0.74 |
| Joint-depth (Nekrasov) | 2.99M | 0.52 | 1.57 | 0.16 | 0.33 | 0.66 | 0.87 |
| DANet | 8.2M | 0.26 | 1.11 | 0.11 | 0.55 | 0.86 | 0.94 |
| Yin et al. (heavy ref.) | – | 0.24 | 1.06 | 0.11 | 0.54 | 0.84 | 0.94 |
| AdaBins (Bhat, heavy ref.) | – | 0.21 | 0.91 | 0.10 | 0.55 | 0.86 | 0.95 |
| Laina et al. (heavy ref.) | 60.6M | 0.23 | 1.20 | 0.12 | 0.50 | 0.78 | 0.91 |

Source: [DANet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.04538). The paper states it evaluates generalization "by using the model trained on NYUDv2 dataset".

**SUN RGB-D and TUM, zero-shot from NYU, metric (Hu et al., Tables 4-5)**

| Model | Params | SUN RMSE | SUN REL | SUN δ1 | TUM δ1 avg (fr1/360, desk, desk2, rpy, xyz) |
|---|---|---|---|---|---|
| FastDepth | 3.9M | 0.662 | 0.376 | 0.404 | 0.369 (0.548/0.308/0.358/0.333/0.287) |
| Joint-depth | 3.1M | 0.634 | 0.338 | 0.454 | 0.494 (0.512/0.410/0.441/0.552/0.583) |
| Hu KD student (NYU + unlabeled ScanNet) | 1.7M | 0.577 | 0.338 | 0.430 | 0.530 (0.615/0.442/0.498/0.611/0.486) |
| Hu KD student (NYU + labeled ScanNet) | 1.7M | 0.531 | 0.306 | 0.446 | 0.781 (0.854/0.695/0.772/0.679/0.905) |

Source: [Hu et al. ar5iv](https://ar5iv.labs.arxiv.org/html/2105.06143). The TUM protocol follows DeepFactors (5 sequences). The authors note that the labeled-ScanNet variant is not a fair comparison.

**DepthART benchmark: metric depth, no rescaling, NYU-trained, zero-shot (δ1↑ / RMSE↓)**
Paper: "DepthART: Scaling Foundation Monocular Depth to Tiny Models", Xue, Chen, Zhao, Zhong, Ming, ..., Sebe, arXiv 2607.17099 (19 Jul 2026), accepted at ACM MM 2026 — [arXiv abs](https://arxiv.org/abs/2607.17099)

| Model | Params | NYU (in-domain) | iBims-1 | SUN RGB-D | DIODE-Indoor |
|---|---|---|---|---|---|
| FastDepth | 3.9M* | 0.771 / 0.604 | 0.429 / 1.607 | 0.560 / 0.736 | 0.275 / 2.778 |
| DANet | 8.2M* | 0.831 / 0.488 | 0.55 / 1.11 | 0.616 / 0.623 | 0.221 / 2.909 |
| GuideDepth | 5.8M* | 0.840 / 0.478 | 0.560 / 1.014 | 0.603 / 0.657 | 0.271 / 2.766 |
| (ref) DepthART-S, distilled, out of scope | 6.0M | 0.924 / 0.340 | 0.713 / 0.693 | 0.809 / 0.379 | 0.292 / 1.687 |
| (ref) Depth Anything v1-L, NYU-ft | heavy | 0.984 / 0.206 | 0.639 / 0.965 | 0.647 / 0.585 | 0.291 / 2.504 |

*Params are not given in DepthART; filled in from the LMDepth/DANet tables. The same table gives KITTI-trained GuideDepth outdoor numbers (DDAD 0.197/21.31, ETH3D-outdoor 0.171/9.539, DIODE-outdoor 0.158/14.25) and KITTI δ1/RMSE for FastDepth 0.825/4.33, DANet 0.920/3.52, GuideDepth 0.882/4.10. METER was excluded because only partial code was public and reproduced results were "much worse than reported". The extraction flagged that the column pairing of this table is approximate. Source: [DepthART HTML](https://arxiv.org/html/2607.17099)

**DepthART affine-invariant table (per-image least-squares scale+shift):** covers NYU, KITTI, ETH3D, DIODE-Full and DDAD, but only for foundation/distilled models (MiDaS v3.1 LeViT 51M / Swin2-T 42M, Depth Anything v2-S 24.8M, Metric3D v2 ViT-S 37.5M, DepthART). No lightweight supervised NYU models are in it — [DepthART HTML](https://arxiv.org/html/2607.17099)

**Models with no public cross-dataset sets in their own papers**
- METER: NYU + KITTI only; one qualitative Kinect V2 scene — [METER](https://arxiv.org/html/2403.08368v1)
- LMDepth: zero-shot only on a self-collected indoor set. iBims-1 is mentioned in the contributions, but no iBims numbers appear — [LMDepth](https://arxiv.org/html/2505.00980v1)
- EGD-Net: NYU only — [EGD-Net](https://arxiv.org/abs/2209.14829)
- Evaluation caveat: SUN RGB-D's validation split contains the NYU test images, which UniDepth removes to avoid overlap. Neither Hu et al. nor DepthART was confirmed to do the same — [UniDepth arXiv 2403.18913](https://arxiv.org/pdf/2403.18913)

### Inferences
- The two sources roughly agree on iBims-1: DANet δ1 0.55 in both DANet and DepthART, and RMSE 1.11 in both. That suggests DepthART reused DANet's protocol, or the same group's code: DepthART and DANet share authors Feng Xue and Anlong Ming. That is a possible source of favourable setup for DANet.
- The sources disagree on FastDepth (iBims δ1 0.30 vs 0.429; SUN δ1 0.404 in Hu et al. vs 0.560 in DepthART). Protocols clearly differ (crop, depth cap, resolution, possibly SUN sensor subset or NYU-overlap filtering). Numbers from different papers should not be mixed in one table without a note.
- Metric zero-shot δ1 for NYU-trained 3-8M models is about 0.30-0.56 on iBims-1, 0.40-0.62 on SUN RGB-D and 0.22-0.28 on DIODE-Indoor. This is far below their in-domain 0.77-0.84, so these models make weak but honest "small supervised" baselines.
- No scale/shift-aligned (affine-invariant) zero-shot numbers exist for any of these lightweight supervised models. Running them ourselves with median or LSQ alignment would be new.

### Gaps
- No ETH3D-indoor, ScanNet-test or DIODE-indoor numbers for METER, LMDepth, Hu et al. or Joint-depth were found.
- DepthART's exact evaluation details for the baselines (official weights or retrained, crop, max depth, SUN NYU-overlap filtering) were not confirmed. The abstract page does not say, and the HTML extraction flagged the table column pairing as approximate. Verify against the PDF (it was too large to fetch, >10MB).
- SimMDE's (30.9M) iBims/SUN values were not extracted.

## Q3. Which have public weights we could run on IBims/NYU/SUN/DIODE/ETH3D?

### Takeaway
Confirmed public NYU weights:
- **GuideDepth / GuideDepth-S**: Google Drive, NYU half- and full-resolution checkpoints.
- **METER**: the paper states pretrained weights are released; MIT repo.

Code repos with unconfirmed checkpoints: DANet (YiLiM1/DANet) and Hu et al. (JunjH/Boosting-Light-Weight-Depth-Estimation). FastDepth and Joint-depth are widely believed to have official checkpoints, but this was not verified here. DepthART reported that METER's public code was partial and could not reproduce the paper.

### Cited Findings
- GuideDepth: six Google Drive checkpoints (NYU 240x320 and 480x640 for GuideDepth and GuideDepth-S; KITTI 192x640 and 384x1280 for GuideDepth); no licence stated — [GuidedDecoding GitHub](https://github.com/mic-rud/GuidedDecoding)
- METER: "Code and corresponding pre-trained weights are made publicly available"; repo is MIT-licensed with a `models/` folder (contents not listed; 7 stars, 19 commits) — [METER arXiv](https://arxiv.org/html/2403.08368v1); [METER GitHub](https://github.com/lorenzopapa5/METER)
- METER was excluded from DepthART because only partial code was available and reproduced results were much worse than reported — [DepthART HTML](https://arxiv.org/html/2607.17099)
- DANet official PyTorch code: github.com/YiLiM1/DANet — [DANet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.04538); [GitHub](https://github.com/yilim1/danet)
- Hu et al. code: github.com/JunjH/Boosting-Light-Weight-Depth-Estimation; weights not mentioned — [Hu et al. ar5iv](https://ar5iv.labs.arxiv.org/html/2105.06143); [PapersWithCode](https://paperswithcode.com/paper/boosting-light-weight-depth-estimation-via)
- LMDepth: no code link in the paper — [LMDepth](https://arxiv.org/html/2505.00980v1)
- DepthART code: github.com/xuefeng-cvr/DepthART (out of scope as a model, but its eval scripts may carry the FastDepth/DANet/GuideDepth benchmark protocol) — [arXiv abs](https://arxiv.org/abs/2607.17099)

### Inferences
- The most practical runnable set for an NYU-to-{iBims, SUN, DIODE, ETH3D} benchmark of 1-15M supervised baselines is: GuideDepth (5.8M), GuideDepth-S (5.7M), FastDepth (3.9M), DANet (8.2M, if a checkpoint exists or can be retrained cheaply on NYU) and METER-S (3.3M, with the reproducibility caveat).
- DepthART already ran FastDepth, DANet and GuideDepth, so those checkpoints were obtainable in 2026.

### Gaps
- The FastDepth checkpoint and licence (github.com/dwofk/fast-depth) were not verified.
- Whether DANet's and Hu et al.'s repos contain trained checkpoints was not verified.
- The licences for GuideDepth, DANet and Hu et al. code were not found.

## Q4. Surveys (2023-2026) tabulating params vs accuracy and zero-shot generalization for lightweight MDE

### Takeaway
No dedicated 2023-2026 survey of lightweight supervised MDE that tabulates params against zero-shot accuracy was found. The best substitutes are the comparison tables inside DANet (NYU + iBims), Hu et al. (NYU + SUN + TUM), LMDepth (NYU params/GFLOPs/FPS) and DepthART (NYU + iBims + SUN + DIODE metric zero-shot for FastDepth/DANet/GuideDepth).

### Cited Findings
- "Survey on Monocular Metric Depth Estimation" (Zhang, arXiv 2501.11841, 2025; also MDPI Computers 14(11):502) covers the field. Its treatment of efficiency is a brief accuracy/efficiency trade-off discussion, not a params-vs-zero-shot table — [arXiv 2501.11841](https://arxiv.org/html/2501.11841v3); [MDPI](https://www.mdpi.com/2073-431X/14/11/502)
- "Monocular Depth Estimation from a Single Image: Progress and Opportunities" (arXiv 2609.01172, 2026) has dataset tables (e.g., SUN RGB-D 10,335 samples; iBims-1 DSLR + laser scanner). Whether it has a lightweight params table was not determined — [arXiv 2609.01172](https://arxiv.org/pdf/2609.01172)
- Other 2025-2026 lightweight papers surfaced but are out of scope (distillation/zero-shot foundation, or not checked): ZipDepth (arXiv 2607.08771), DepthART (2607.17099), BoRe-Depth (self-supervised, embedded, arXiv 2511.04388) — [ZipDepth](https://arxiv.org/pdf/2607.08771); [BoRe-Depth](https://arxiv.org/html/2511.04388v1)

### Inferences
- A table of 1-15M supervised NYU models evaluated under one aligned protocol (affine and metric) on iBims-1, SUN RGB-D (with NYU overlap removed), DIODE-indoor and ETH3D-indoor does not appear to exist in the literature. It would be a gap a paper could fill.

### Gaps
- MAI 2021/2022/2023 depth challenge reports were not reviewed. They benchmark mobile latency, but on a non-NYU dataset.
- No efficiency-focused survey from Sensors or TCSVT was found within the tool budget. A targeted search ("efficient monocular depth survey Sensors 2024") may still turn one up.
