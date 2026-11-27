# Knowledge-distillation losses for dense depth regression into compact students (2021-2026), applied to b20dws

Scope note: docs/B20_RESEARCH.md already covers Distill Any Depth (headline numbers), DAv2 top-10% trimming, DAv1 strong perturbation, DisDepth, TIE-KD, Beyer "patient and consistent", FKD offline crops, PatchRefiner/ACDepth ranking, Busbridge capacity gap, Hu et al. 1.7M student, EMA. These notes add the exact formulas and new evidence (normalization equations, SKD pair-wise numbers for a 2.7M depth student, DIST relational KD, Depth Anything 3's teacher-to-GT alignment, uncertainty-weighting evidence). They turn this into loss proposals aimed at the measured bottleneck. b20dws scale-aligned IBims d1 is 91.2 against raw 74.8, so about 16 d1 points sit in the per-image global scale s_hat.

## Q1. Cross-context / normalization distillation (Distill Any Depth) and DAv2-style pseudo-label distillation: exact losses, numbers, fit to b20dws

### Takeaway
For pixel-level distillation, how the student and teacher maps are normalized matters as much as the loss itself. Global median/MAD normalization is worst for same-crop distillation (shared-context). "None" and "hybrid" (normalizing per depth-range segment) are best. b20dws's current choice (log space, scale-only median alignment s*) sits between "global" and "none". Hybrid normalization is a cheap, offline-compatible change aimed at the structure terms.

### Cited Findings
- Distill Any Depth global normalization (Eq. 1-2). d~_i = (d_i - med(d)) / ((1/M) Σ_j |d_j - med(d)|) is applied to both student and teacher, and L_Dis = (1/M) Σ_i |d~s_i - d~t_i|. — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- Hybrid normalization (Eq. 3-4). The depth range is split into S ∈ {1,2,4} segments. Each pixel i belongs to a set of contexts U_i, one per granularity. The loss is L^i = (1/|U_i|) Σ_{u∈U_i} |N_u(d^s_i) - N_u(d^t_i)|, where N_u is median/MAD normalization computed inside context u. "Local" uses only S=4; "none" is plain L1. — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- Normalization ablation, AbsRel on ETH3D / DIODE. Shared-context: global 0.064/0.259, none 0.057/0.239, local 0.070/0.245, hybrid 0.057/0.238. Local-global: global 0.065/0.239, none 0.273/0.300 (collapses), local 0.076/0.244, hybrid 0.064/0.238. — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- Shared-context loss L_sc = L_Dis(d^s_local, d^t_local): student and teacher see the same crop. Local-global loss L_lg = (1/N) Σ_n L_Dis(Crop(d^s_global), d^t_local_n): the teacher is run on N overlapping crops and the student on the full image. Total: L = L_sc + 0.5 L_lg + 1.0 L_feat + 2.0 L_grad. The gradient and feature terms follow DAv2, with no formula given. Crops are 1:1, sampled from 644 px up to the short side and resized to 560. N is not specified. — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- Component ablation, ETH3D / DIODE AbsRel: none 0.075/0.270; +LG 0.064/0.238; +SC 0.058/0.237; both 0.056/0.232. The paper's quoted "-13.3%" for LG-only DIODE does not match its own numbers (0.270 to 0.238 is about -11.9%). — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- Small students: DA-L to DA-S, DIODE 0.290 to 0.262 and ETH3D 0.110 to 0.098; MiDaS-L to MiDaS-S, 0.303 to 0.272 and 0.150 to 0.120. All evaluation is scale-and-shift aligned, so metric depth is not addressed. Online vs offline teacher is not stated. — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- Depth Anything 3 (ICLR 2026). The synthetic-trained teacher's relative depth is aligned to sparse or noisy real metric depth by RANSAC least squares, (ŝ, t̂) = argmin_{s>0,t} Σ_p m_p (s·D~_p + t - D_p)², with the inlier threshold set to the MAD of residuals from their median. The aligned teacher map becomes the label. The metric model uses teacher labels scale/shift-aligned to GT, keeps the original GT with probability 0.2, and uses canonical focal f_c = 300. — [Depth Anything 3](https://arxiv.org/html/2511.10647v1)
- DA3 metric model with vs without teacher labels (delta1 / AbsRel as extracted). NYU 0.966/0.073 vs 0.969/0.066; KITTI 0.947/0.086 vs 0.965/0.067; SUN-RGBD 0.973/0.104 vs 0.975/0.099; DIODE 0.824/0.132 vs 0.816/0.134. The paper says teacher supervision mainly improves sharpness and fine detail, not the aggregate metrics. Caveat: column direction comes from an automated extraction and should be checked against Table 11. — [Depth Anything 3](https://arxiv.org/html/2511.10647v1)
- DA3 multi-view ablation: removing teacher supervision drops HiRoom Auc3/F1 from 39.2/47.0 to 11.2/16.0 and ScanNet++ from 30.3/51.1 to 26.2/47.7. It improves DTU (Auc3 45.8 to 52.5). — [Depth Anything 3](https://arxiv.org/html/2511.10647v1)

### Inferences
- b20dws aligns in log space with a single median shift (s*). That is equivalent to scale-only alignment in linear depth, with no MAD rescaling, so it is closest to DAD's "none"/"global-median" family. The SILog and gradient terms are already approximately scale-invariant. The new, untested lever is hybrid normalization of the structure term. Split each image's teacher log-depth into 1/2/4 quantile bands and apply median-centred L1 inside each band. This targets near/far structure that one global alignment underweights, and could replace the hand-tuned proximity-weighted SILog (near band <2 m, gain 2) with a principled, scale-free equivalent. Train-only, offline, zero inference cost.
- Do not use plain un-normalized L1 for any cropped-student vs full-image-teacher comparison. DAD's local-global "none" collapses (0.273). This is the likely failure mode of the earlier s5/sced crop variant.
- "GT-anchored teacher structure" is the DA3 idea adapted to b20dws. On NYU and SUN rows, use the cached V2-L log-depth shifted to GT scale as the pixel structure target: target = logD_T + median(logD_GT - logD_T) over valid GT pixels. Keep the raw GT median for the L_s target. Kinect/structured-light GT is holey and noisy at edges, while the teacher map is dense and sharp, so this gives dense, sharp supervision on 60% of the mix with GT-correct scale. DA3 keeps raw GT 20% of the time, which is a safe default. This is a zero-inference-cost change and partly novel in a metric-distillation setting, inferred from DA3's stated motivation. Per DA3 Table 11, expect sharpness and boundary gains more than AbsRel gains, so measure edge F1 and the OCMR object metrics too.

### Gaps
- DAD does not ablate the number of crops N, the patch size, or the gradient/feature formulas. Its effect on metric (unaligned) students is unknown.
- I did not verify the column orientation of DA3 Table 11 or the DA3 loss weights for the metric model (not given in the extracted text).

## Q2. Feature / relation / affinity distillation and light students (Lite-Mono class, 1-10M): what transfers, numbers

### Takeaway
For compact depth students, pixel-wise output KD on already-labeled images gives roughly nothing. Structured (pair-wise affinity) KD on features and relational, correlation-based KD give small but real gains (about 2-5% rel). They need online teacher features, unless the relation is defined on the cached output depth. For b20dws's offline setting, relational losses on outputs, including across images in a batch, are the transferable part.

### Cited Findings
- Structured KD (Liu et al., TPAMI 2020), pair-wise loss: ℓ_pa = β/(W'H'α) Σ_i Σ_{j∈α} (a^s_ij - a^t_ij)², with a_ij = f_iᵀf_j / (‖f_i‖‖f_j‖) on β-pixel average-pooled feature nodes connected to their α nearest nodes. Holistic loss: a WGAN-GP discriminator on (output, image), weighted 0.1. The pair-wise term is applied to intermediate features, not outputs. — [Structured KD for Dense Prediction](https://ar5iv.labs.arxiv.org/html/1903.04197)
- SKD depth setting: VNL MobileNetV2 student (2.7M) from a ResNeXt-101 teacher (86.2M) on NYU. Small-set ablation rel: baseline 0.181; +pixel-wise 0.183 (worse); +pair-wise 0.175; +PA+holistic 0.173; +unlabeled data 0.160. Full set: student 0.135/rms 0.576/d1 0.813 goes to 0.130/0.544/0.838 with distillation (teacher 0.108/0.416/0.875). The authors drop pixel-wise KD for depth because it "hardly improves the accuracy". — [Structured KD for Dense Prediction](https://ar5iv.labs.arxiv.org/html/1903.04197)
- DIST (NeurIPS 2022) replaces exact KL matching with a Pearson-correlation relation loss, between teacher and student predictions per instance (inter-class) and across instances in a batch per class (intra-class). It is designed for "a stronger teacher", where exact matching is too hard for the student. It beats CIRKD by 1.58% mIoU on PSPNet-R18 segmentation. ImageNet R34 to R18: KD 71.21 vs DIST 72.07. — [DIST](https://arxiv.org/abs/2205.10536v3); [DIST NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2022/file/da669dfd3c36c93905a17ddba01eef06-Paper-Conference.pdf)
- Lite-Mono (3.1M, CNN+Transformer) has been shrunk to about 310K params with knowledge distillation used to recover accuracy (nano-drone setting). — [Channel-Aware Distillation Transformer for Nano Drones](https://arxiv.org/pdf/2303.10386)
- NimbleD adds pseudo-label supervision from a large vision model to a lightweight self-supervised depth framework. Numbers were not extracted because the PDF was unreadable through the fetch tool. — [NimbleD](https://www.arxiv.org/pdf/2408.14177)
- Lightweight radar-camera depth (LiRCDepth) reports a 6.6% MAE improvement on nuScenes from multi-level distillation plus uncertainty guidance. XD-RCDepth adds depth-distribution distillation. These are sensor-fusion students, not monocular. — [LiRCDepth](https://arxiv.org/abs/2412.16380); [XD-RCDepth](https://arxiv.org/abs/2510.13565)
- Data-free dense depth distillation with OOD simulated images reports about 0.05 m (NYU) and 0.08 m (ScanNet) average RMSE improvement over baseline KD. — [Dense Depth Distillation with OOD Simulated Images](https://arxiv.org/html/2208.12464v3)

### Inferences
- The SKD finding (pixel KD +0.002 rel worse; pair-wise -0.006; unlabeled data -0.013) matches the B1 recipe ablations sitting near .092. For 1-10M depth students, data volume beats the choice of distillation loss. Loss changes should be judged against a 1-3% expected gain on in-domain AbsRel.
- Offline analogue of pair-wise and DIST relations: compute affinities on cached teacher OUTPUT log-depth rather than features. Within an image, that is close to existing gradient and SSI terms, so it adds little. Across images, it is new for b20dws: a batch-level relational scale loss (DIST "intra-class" across instances) on the scale head. For example: L_rel = 1 - Pearson({ŝ_b + c_b}, {s*_b + c_b}) over a batch, or a pairwise L_pair = mean_{i<j} SmoothL1((ŝ_i - ŝ_j) - (s*_i - s*_j)). This teaches the ordering of scene scale between images, which is exactly what is wrong on unseen IBims. The batch is 4 x 3 accumulation, so pairs need a small memory queue of (ŝ_EMA, s*) from recent steps (MoCo-style). That keeps the cost train-only. Novelty claim (inferred, not found in the literature searched): relational KD has not been applied to the global metric-scale scalar of a decoupled depth model.
- Feature affinity KD from UniDepthV2-L would need cached ViT-L features (large storage) or an online teacher, so it is incompatible with the offline cache constraint. Low priority, consistent with DisDepth's finding that naive ViT-to-CNN feature KD needs adaptation.

### Gaps
- I found no 2023-2026 study of relation or affinity KD on a 1-10M monocular METRIC depth student with numbers. Lite-Mono+DAv2-teacher distillation numbers (NimbleD) were not retrieved.
- The DIST loss formula (β(1-ρ_inter) + γ(1-ρ_intra), β=γ=1) is from my knowledge of the paper and was not re-read in this session.

## Q3. Multi-teacher distillation (L/B/S teachers, ensemble agreement) and teacher-uncertainty weighting of pseudo labels

### Takeaway
Averaging teachers' dense maps is harmful. Per-sample probabilistic teacher selection works. Continuous per-pixel confidence weighting of depth losses has repeatedly hurt (Depth Pro 2; the project's own s6). Uncertainty is safest used (a) as a hard trimming mask or (b) on the single global-scale scalar, where it does not discount hard pixels.

### Cited Findings
- Multi-teacher, ETH3D / DIODE AbsRel: DAv2 alone 0.131/0.262; GenPercept alone 0.096/0.226; pixel-averaged by agreement 0.228/0.371 (much worse than either); per-iteration selection at 7:3 0.054/0.258. On DIODE, selection beats DAv2 but not GenPercept alone. — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- Uncertainty-weighted pseudo-label distillation is common in self-supervised depth. SUB-Depth uses Kendall-style learned uncertainty to down-weight the distillation loss where the teacher is noisy. FSNet models log-depth as Laplacian around the teacher pseudo-label with per-pixel predicted uncertainty. URCDC-Depth uses pixel-wise uncertainty to rectify loss weights for noisy pseudo-labels. — [SUB-Depth BMVC 2022](https://bmvc2022.mpi-inf.mpg.de/0007.pdf); [FSNet](https://arxiv.org/pdf/2304.10719); [URCDC-Depth](https://arxiv.org/pdf/2302.08149v2.pdf)
- A 2026 all-weather self-supervised study uses multiple teachers with an uncertainty-aware distillation loss, so the student "learns selectively" from teachers that are not always reliable. Numbers not extracted. — [Boosting Robustness for All-Weather SSDE](https://www.alphaxiv.org/abs/2607.21526)
- Contradicting evidence for supervised metric depth: Depth Pro 2's controlled study finds confidence-weighted losses harmful (boundary F1 -6.8). The project's own s6 NLL run was worse (val .098 vs .092). — [docs/B20_RESEARCH.md, Depth Pro 2 entry, arXiv:2611.26603](https://arxiv.org/abs/2611.26603)
- DA3 handles teacher unreliability by robust (RANSAC) alignment to real measurements rather than by weighting. — [Depth Anything 3](https://arxiv.org/html/2511.10647v1)

### Inferences
- For b20dws pixel terms: implement multi-teacher as per-sample selection among cached V2-L/B/S, for example L 0.7 / B 0.2 / S 0.1, never averaging. Use teacher disagreement only to extend the existing trim. For example, drop pixels where |logD_L - logD_B| after scale alignment exceeds the image's 90th percentile, capped at 10% so that hard but correct pixels are not discounted wholesale.
- The most promising new use of the three caches is a teacher-ensemble scale posterior for the scale head. Per own image, compute s*_L, s*_B, s*_S, the median log-ratio of each teacher to U(r_c)+c. The target is a distribution: a Gaussian mixture or soft label over K log-scale bins, built from the three teacher scales with weights 0.7/0.2/0.1 and a width set by their spread. Train the scale head with a distributional loss: TIE-KD-style KL to soft bins, or a Gaussian NLL on (μ, log σ) of the scalar. GT rows get a narrow target. Rationale: (i) teacher scale errors on unseen scenes are what the student imports, and the ensemble spread is a free per-image reliability signal for that scalar; (ii) it applies uncertainty only to ONE scalar per image, avoiding the Depth Pro 2 / s6 failure on pixels; (iii) at inference only μ is used, so cost is zero. Novelty (inferred): soft-label, multi-teacher distillation of a decoupled metric scale. No prior work combining these was found in the searches.
- A capacity-gap alternative: if V2-B/S scales are closer to what a 5M model can predict from GAP features, mixing them reduces target variance. Test L-only vs mixture.

### Gaps
- No numbers for the per-image L/B/S scale spread on the own corpus. This should be measured first: if the spread is tiny, the posterior idea reduces to the current L_s.
- No published ablation found on uncertainty weighting of a global scale scalar in decoupled depth models (MoGe-2 uses plain L2 on log scale with stop-gradient).

## Q4. Consistent teaching under augmentation, function matching, capacity gap / assistant teachers

### Takeaway
A fixed precomputed teacher is the "fixed teacher" failure mode for classification. For depth it is largely avoidable, because geometric augmentations can be applied identically to the image and the cached map. Exceptions are crop-dependent scale cues, which need teacher-on-crop caches. Teacher assistants address capacity gaps, but no dense-depth evidence was found. Of these ideas, a self-consistency loss on the scale head under FoV augmentation needs no teacher and is cheap to implement.

### Cited Findings
- TAKD: student performance drops when the teacher-student gap is large. An intermediate teacher assistant mitigates this (classification). — [TAKD](https://arxiv.org/pdf/1902.03393); multiple-assistant extension [DGKD](https://arxiv.org/pdf/2009.08825)
- 2024-2025 capacity-gap work (automatic multi-step distillation; dual-forward-path teacher) remains classification or large-model focused. No dense regression evidence was found. — [AMD 2407.04208](https://arxiv.org/html/2407.04208v1); [Dual-Forward Path Teacher KD](https://arxiv.org/pdf/2506.18244)
- Fixed teacher vs consistent teaching vs function matching (ImageNet R50, 9600 epochs): 76.95 / 81.76 / 82.31. Fixed teachers overfit after about 600 epochs. — [docs/B20_RESEARCH.md, Beyer et al., arXiv:2106.05237](https://arxiv.org/abs/2106.05237)
- UniDepthV2 geometric-invariance consistency in output space: delta1 54.5 to 55.3. — [docs/B20_RESEARCH.md, arXiv:2502.20110](https://arxiv.org/abs/2502.20110)

### Inferences
- Function matching (mixup inputs with a fresh teacher output) is impossible offline. CutMix with pasted cached depth is its offline proxy (already in B20_RESEARCH).
- Scale-equivariance consistency, a train-only loss with no teacher: take two FoV-augmented views (crop/uncrop, with intrinsics updated) of the same image and their predicted log-depth on the overlapping region. Metric depth there is physically identical, so L_cons = |median_overlap(logD_1) - median_overlap(logD_2)|, with a stop-gradient on the less-cropped view. This trains the scale head to be invariant to FoV and context, which is the failure seen on unseen IBims. It costs a second forward pass only on a fraction p of batches. Expect a small in-domain effect and a possible unseen-scale gain. Novelty is moderate (UniDepthV2 does dense geometric consistency; applying it to the decoupled scale scalar alone is new as far as these searches show).
- Assistant teacher for b20dws: V2-B or V2-S cached maps act as natural assistants. A two-stage curriculum (V2-S targets for epochs 1-10, then V2-L) is a cheap TAKD analogue, but the evidence is classification-only, so it should be a low-priority ablation.

### Gaps
- No dense-depth evidence on teacher assistants or the capacity gap for 1-10M students. Busbridge (LM) and TAKD (classification) are indirect.

## Q5. Which methods work OFFLINE with cached teacher depth, expected gains for 1-10M students, and a proposed b20dws loss + ablation plan

### Takeaway
Fully offline-compatible: output-space normalization choices (hybrid), trimming, per-sample teacher selection, GT-anchored teacher structure, teacher-ensemble scale posterior, relational scale loss, and scale-consistency under augmentation. Offline-compatible only with an extra one-time teacher pass: teacher-on-crop (shared-context or local-global) caches. Not offline: feature or affinity KD, and function matching. Published small-student gains are about 2-10% AbsRel, or +2.5 to +3.5 d1. Since 16 of b20dws's missing IBims d1 points are in scale, loss work on the scale head has the largest ceiling.

### Cited Findings
- 2.7M student: d1 0.813 to 0.838 (+2.5), rel 0.135 to 0.130 with structured KD. — [SKD](https://ar5iv.labs.arxiv.org/html/1903.04197)
- 1.7M student: d1 0.802 (GT) to 0.808 (KD on the same images) to 0.837 (KD plus unlabeled in-domain data). — [Boosting Light-Weight Depth via KD](https://ar5iv.labs.arxiv.org/html/2105.06143)
- Small students with cross-context KD: -9.6% to -20% AbsRel (affine-aligned). — [Distill Any Depth v2](https://arxiv.org/html/2502.19204v2)
- 17.6M student with teacher-output soft-bin KD: AbsRel 0.0663 to 0.0654. — [TIE-KD](https://ar5iv.labs.arxiv.org/html/2402.14340)
- Scale as classification over unnormalized depth-range bins with range-domain prediction (SM4Depth, ACM MM 2024) gives strong zero-shot metric results from 150K RGB-D pairs. This supports distributional, not purely regressive, scale prediction. — [SM4Depth](https://arxiv.org/html/2403.08556v2)

### Inferences
Proposed loss (all train-only, zero inference cost), written in the b20dws notation:

L = L_struct + λ_s L_scale + λ_r L_relscale + λ_c L_scalecons

- L_struct: current SILog + gradient matching + EG-SSI on p_al, with three changes. (a) Hybrid-normalized L1 over depth bands {1,2,4} replaces the proximity-weighted SILog (A1). (b) NYU/SUN rows use GT-anchored V2-L structure as the dense target, with raw GT kept 20% of the time (A2). (c) Per-sample teacher selection L/B/S at 0.7/0.2/0.1 on own rows (A3).
- L_scale: a distributional teacher-ensemble scale posterior. Soft log-scale bins (K≈64 over log(0.2)..log(30) of median depth, Gaussian σ = max(σ_min, spread of s*_L/B/S)), trained with KL, plus SmoothL1 on the expectation to the selected target (A4). GT rows use σ_min.
- L_relscale: pairwise relational scale loss over the batch plus a queue of 256 (s*, ŝ_EMA) entries (A5), λ_r ~0.25.
- L_scalecons: FoV-view scale consistency on the overlap with stop-gradient, p = 0.25 of batches (A6), λ_c ~0.1.

Ablation ladder: one change per run on top of b20dws, 40 epochs. Report IBims raw d1, bias_log, scale-aligned d1, NYU d1 and OCMR objAbsRel.
- A0: b20dws (74.8 / 85.4).
- A1: hybrid norm. Expected to act on structure, visible in scale-aligned d1 and in-domain.
- A2: GT-anchored teacher. Structure and sharpness gain on NYU/SUN; also check edge F1.
- A3: teacher selection. Risk: B/S have worse scale, so if raw scale drops, apply selection only to L_struct.
- A4: scale posterior. Main target is IBims raw d1.
- A5: relational scale.
- A6: scale consistency.
- A7: best combination.
- A8: A7 plus teacher-on-crop caches (shared-context L_sc on 2-4 cached V2-L crops per own image), which needs one extra offline teacher pass.

Measure first: the L/B/S scale spread and how well it predicts |s*_L - s*_GT| on NYU/SUN, where V2 predictions exist for GT images. That spread is the evidence the novelty of A4 depends on.

Paper novelty angle: "Distilling metric scale, not pixels". The pieces are decoupled scale-head KD with a multi-teacher scale posterior (A4), relational cross-image scale KD (A5) and scale self-consistency (A6). Each targets the measured 16-point gap between scale-aligned and raw d1, and A1-A3 protect in-domain structure. These claims are inferences from the evidence above, not results.

### Gaps
- No published 1-10M METRIC-depth student ablates scale-head distillation losses, so the expected gains for A4-A6 are unknown and must be measured.
- I did not retrieve numbers for NimbleD, the 2026 all-weather multi-teacher paper, or XD-RCDepth's distribution KD.
