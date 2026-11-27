# Prior-art check for candidate novel loss ideas for a decoupled-scale tiny depth student (b20dws)

Scope note: I ran about 18 searches and fetches (Nov 2026). "No prior art found" means none turned up in these searches; it is not an exhaustive survey. Items marked [project notes] come from docs/B20_RESEARCH.md, where they were gathered earlier. I did not re-verify them here. Everything else was fetched or searched for this note.

Context recap: b20dws gets IBims d1 74.8 and scale-aligned d1 91.2, so roughly 16 d1 points are lost to the per-image global scale s_hat. Pixel losses act on p_al = U(r_c) + c + s*, with s* under stop-grad. The scale head is trained only by SmoothL1(s_hat - s*).

---

## (a) Scale loss weighted by teacher reliability (multi-teacher V2-L/B/S, or multi-view/crop agreement)

### Takeaway
Multi-teacher distillation and per-pixel uncertainty-weighted distillation are both published. I found no work that weights a per-image global SCALE target by how well several teachers agree on scale. That idea looks unpublished. One caution: per-pixel confidence weighting has been reported as harmful for metric depth.

### Cited Findings
- Distill Any Depth: averaging two teachers' pseudo-labels was much worse than either teacher alone (AbsRel 0.228/0.371 vs 0.131/0.262 and 0.096/0.226). Probabilistic per-sample teacher selection (7:3) gave 0.054/0.258. It is per-pixel/global-normalized distillation, with no scale-level reliability. — [Distill Any Depth, arXiv 2502.19204](https://arxiv.org/abs/2502.19204) [project notes for the numbers]
- SUB-Depth weights the self-distillation regression loss by learned uncertainty, which down-weights it where the teacher is noisy. This is per-pixel and single-teacher. — [SUB-Depth, arXiv 2111.09692](https://arxiv.org/html/2111.09692.pdf)
- ADU-Depth uses per-pixel uncertainty to guide distillation from a stereo teacher. — [ADU-Depth, arXiv 2309.14744](https://arxiv.org/pdf/2309.14744)
- Mutual distillation between two branches with depth and uncertainty: uncertainty weighting was best in its ablations. — [Semi-supervised mutual distillation, arXiv 2203.09737](https://arxiv.org/html/2203.09737v1)
- An event-camera method (2026) weights each pixel by the ratio of teacher to student confidence, so the student does not copy unreliable teacher predictions. — [SFE-VGGT, arXiv 2609.36929](https://arxiv.org/pdf/2609.36929)
- Adaptive Co-Teaching (ECCV 2022) uses a cost-aware co-teaching loss over an ensemble of unsupervised depth teachers. — [ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610089.pdf)
- Depth Pro 2: confidence-weighted losses lowered boundary F1 by 6.8 points in a controlled ViT-L study. — [Depth Pro 2, arXiv 2611.26603](https://arxiv.org/abs/2611.26603) [project notes]
- Selective distillation that excludes unreliable pseudo-labels appears in self-supervised teacher-student work. — [arXiv 2304.12685](https://arxiv.org/pdf/2304.12685); [WaterMono, arXiv 2406.13344](https://arxiv.org/pdf/2406.13344)

### Inferences
- This idea is new because it moves reliability from per-pixel to per-image global scale. The negative Depth Pro 2 result concerns per-pixel weighting, which removes hard pixels. Per-image scale weighting only reweights one scalar target, so that result does not transfer directly. It is still a risk to ablate.
- A concrete loss using only offline caches: for each 'own' image, compute s*_T = median(logD_T - U(r_c) - c) for T in {L, B, S}. Treat the target as heteroscedastic with sigma_i^2 = sigma_0^2 + Var_T(s*_T). The loss is L_s = (s_hat - s*_L)^2 / (2 sigma_i^2) + 0.5 log sigma_i^2, or the robust equivalent SmoothL1 / sigma_i. Use fixed sigma_0 on GT rows (NYU/SUN). Because sigma_i comes from data and is not learned, this avoids the "learn to ignore hard samples" failure of learned NLL (s6 hurt earlier, val .098 vs .092).
- Distill Any Depth's result argues for sampling one teacher per image for the pixel loss rather than averaging. Teacher disagreement should only set the weight of the scale target.
- A crop-agreement version (the student's own s_hat on two crops) needs no teacher and overlaps with idea (b).

### Gaps
- I found no paper reporting teacher-disagreement statistics for global scale specifically, so I could not check how strongly Var_T(s*_T) correlates with V2-L's actual scale error on GT data. This can be measured on the NYU/SUN rows where V2-L/B/S caches and GT both exist.

---

## (b) Camera-conditional / FoV-equivariant scale-consistency loss (the scale head must shift exactly by log zoom under crop/uncrop)

### Takeaway
Output- and feature-level invariance under geometric augmentation is published (UniDepth/UniDepthV2). Focal-aware crop augmentation is also published (FS-Depth, DMD, Metric3D CSTM). I found no loss that enforces an exact analytic equivariance on a decoupled scalar scale head. That idea looks unpublished and is the strongest novelty candidate here.

### Cited Findings
- UniDepth samples two geometric augmentations per image and enforces consistency between camera-conditioned output tensors. The loss must be applied after camera conditioning. — [UniDepth CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Piccinelli_UniDepth_Universal_Monocular_Metric_Depth_Estimation_CVPR_2024_paper.pdf)
- UniDepth ablation: removing the consistency loss cost about 4.2 OOD d1 (57.06 to 52.89). Removing the camera-branch detach cost 10.5. — [arXiv 2403.18913](https://arxiv.org/abs/2403.18913) [project notes]
- UniDepthV2 moved the consistency loss to output space: d1 54.5 to 55.3. — [UniDepthV2, arXiv 2502.20110](https://arxiv.org/abs/2502.20110) [project notes]
- FS-Depth: a center crop of size km x kn from a focal-f image is equivalent to focal f/k. Scaling coordinates by k and depth by 1/k keeps focal fixed. — [FS-Depth, arXiv 2307.14624](https://arxiv.org/pdf/2307.14624)
- Detaching and Boosting uses camera-zoom augmentation that changes focal together with the image so depth is unchanged (self-supervised). — [arXiv 2210.03952](https://arxiv.org/pdf/2210.03952)
- DMD: FoV augmentation plus conditioning moved iBims-1 AbsRel from 0.180 to 0.130. NYU in-domain went slightly worse, 0.074 to 0.076. — [DMD, arXiv 2312.13252](https://arxiv.org/abs/2312.13252) [project notes]
- Metric3D CSTM-label: NYU AbsRel was 0.584 without it vs 0.212 with it. CamConvs gave 0.333. — [Metric3D, arXiv 2307.10984](https://arxiv.org/pdf/2307.10984.pdf) [project notes]
- A 2026 MDPI Electronics paper models crop, resize and padding as an affine intrinsics change. It separates resize-induced focal errors from crop-induced principal-point shifts. A consistency loss there lowered mismatched AbsRel from 0.141 to 0.114. This is from the search snippet only; the page returned 403, so I did not verify it. — [MDPI Electronics 15(10):2180](https://www.mdpi.com/2079-9292/15/10/2180)
- Depth Pro 2: training on square crops only cost 46.6 metric d1 on Waymo, so FoV and aspect diversity is needed to learn scale. — [arXiv 2611.26603](https://arxiv.org/abs/2611.26603) [project notes]
- DepthART uses only mild intrinsics perturbation (rescale [0.9, 1.1], crop [0.8, 1.0] of area) during metric fine-tuning, with no consistency loss. — [DepthART, arXiv 2607.17099](https://arxiv.org/html/2607.17099)

### Inferences
- Exact constraint for this parameterization, log D = U(r_c) + s_hat + c with c = log(f_px / 518.9):
  - Pure resize by k (zoom, no crop). f_px becomes k f_px, so c shifts by exactly log k, and metric depth is unchanged. The constraint is s_hat(resize_k x, kK) = s_hat(x, K), up to a structure-mean term that is about 0 because r_c is mean-centred. In words, c must absorb the zoom exactly.
  - Crop (f_px unchanged, smaller FoV). True log-depth is unchanged per pixel, but the mean of U(r_c) is re-centred over the crop. The target is s_hat(crop) = sg( median_crop[ logD_full_student - U(r_c,crop) - c ] ), using the student's own full-image metric prediction, or s*_crop from GT/teacher when available.
  - Uncrop or pad (FoV larger). Use the same expression with the roles swapped, and only over the original pixels.
- Proposed L_eq = SmoothL1( s_hat(T x) - sg( s_hat(x) + Delta_T ) ), where Delta_T is the analytic offset above, computed from the full view's own structure. Apply it symmetrically, with stop-grad on the less-cropped view (teacher view). The loss needs no labels, so it can use extra unlabeled indoor images, which the ZipDepth/DAv2 data-scale evidence says matters most [project notes].
- Why it can raise unseen-scene scale accuracy: the IBims error is almost entirely s_hat, and L_s only supervises s_hat on about 6.8k + NYU + SUN images. L_eq adds a dense set of self-consistency constraints on exactly the failing scalar. It also penalizes "scale from FoV shortcut" solutions, in which the head reads the image extent rather than scene content. That shortcut may cause cross-dataset scale bias, but this is an inference I have not tested.
- Ablation axis: crop-only vs zoom-only vs both, and stop-grad direction (full to crop vs symmetric).

### Gaps
- I found no published ablation of equivariance on a scalar scale head, so I have no evidence for the size of the gain. The UniDepth consistency (+4 OOD d1) and DMD (iBims AbsRel -28%) results are the closest proxies, and both are for large models.

---

## (c) Proximity / near-field weighted SILog for assistive navigation

### Takeaway
Distance-balanced and inverse-depth losses that implicitly emphasize near depth are published (Depth Pro, Metric Anything). A recent robotics paper documents dangerous near-range over-estimation by metric foundation models. I found no near-band-weighted SILog for assistive navigation, but the novelty is incremental. An asymmetric, safety-aware near-field term (penalizing over-estimation more) is more defensible.

### Cited Findings
- "From Depth to Safe Depth" (2026): frozen Depth Anything V2 metric systematically over-estimates distance to nearby obstacles. 59.8% of near-range pixels were over-estimated, and 63% of 120 Nav2 episodes ended in collision. A 0.58M correction head trained with a beta-weighted Gaussian NLL on log-depth correction cut near-range over-estimation to 12.0%, AbsRel from 0.262 to 0.134, and collisions from 63% to 3%. The safe bound is d_safe = d e^(mu - k sigma), with k from conformal calibration. — [arXiv 2611.13769](https://arxiv.org/abs/2611.13769)
- Metric Anything student loss: distance-balanced inverse-depth loss with Dlog = 1 - ln(x)/ln(C), C = 400 chosen by ablation. — [Metric Anything, arXiv 2601.22054](https://arxiv.org/html/2601.22054)
- Depth Pro 2 uses a trimmed, focal-normalized inverse-depth MAE as its metric loss. Inverse depth up-weights near pixels. — [arXiv 2611.26603](https://arxiv.org/abs/2611.26603) [project notes]
- A Frontiers 2025 paper on obstacle distance estimation adjusts loss weights by obstacle-region size. — [Frontiers in Robotics and AI 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1560342/full)

### Inferences
- The current proximity SILog (near < 2 m, gain 2, w 0.5) acts only on structure, because it uses p_al with s*. Near-field error after alignment is a structure error. In metric terms, near-field over-estimation comes mostly from s_hat. So a near-field term that targets safety must also touch the scale path: for example, an asymmetric near-band hinge on the metric prediction log D = U + s_hat + c with stop-grad on U, giving gradient to s_hat only: L_near = mean_{d_gt < 2 m} [ alpha * relu(logD - logD_gt) + relu(logD_gt - logD) ], alpha ~ 2-3.
- Present it as a task-aware, asymmetric term, motivated by the Safe Depth over-estimation evidence. Report both standard d1 and a near-band over-estimation rate, so the trade-off is visible.

### Gaps
- I found no published ablation of near-band weighting on standard metric benchmarks, so its effect on d1 is unknown.

---

## (d) Joint scale-structure consistency: align structure with the PREDICTED scale late in training (scheduled switch s* to s_hat)

### Takeaway
MoGe-2's decoupled, stop-grad design is published, and the entangled alternative was worse and unstable. I found no published scheduled switch from oracle-aligned to predicted-scale-aligned pixel losses. In this parameterization, the switch has limited leverage.

### Cited Findings
- MoGe-2 Table 4 (ViT-B, 10 datasets, Rel/d1): entangled SI-log 17.9/68.6, decoupled CLS-MLP scale 16.5/72.8. The entangled variant was reported unstable. Relative geometry also improved under decoupling (d1 89.0 to 91.0). — [MoGe-2, arXiv 2507.02546](https://arxiv.org/abs/2507.02546) [project notes]
- UniDepth: letting depth-loss gradients flow into the camera branch (no detach) cost 10.5 OOD d1. — [arXiv 2403.18913](https://arxiv.org/abs/2403.18913) [project notes]

### Inferences
- Key structural observation: r_c is mean-centred, and gradient matching and the SSI/edge terms are shift-invariant in log space. Replacing s* with sg(s_hat) in p_al only adds the per-image constant (s_hat - s*) to every residual. That changes only the non-invariant parts: the (1 - lambda) = 0.15 mean term of SILog, proximity-band membership, and which pixels the top-10% trim removes. So the switch mostly changes the trim mask and the near-band weights under scale error. Do not expect a large gain from it.
- If the gradient is NOT stopped through s_hat, it becomes MoGe-2's entangled setting, which the evidence says is worse. Recommendation: at most a short late-phase ablation (last 5-10 epochs), with sg(s_hat) and the trim mask still computed with s*. It is low priority.
- A more useful joint term is median-vs-mean consistency. s* uses a median while the U(r_c) centring uses a mean. Aligning both on the same robust statistic removes a systematic offset the scale head must otherwise learn. This is cheap to check, but it is my inference and has not been tested.

### Gaps
- No literature on scheduled oracle-to-predicted alignment in depth turned up, so I have no evidence either way.

---

## (e) Scale-head distillation from teacher global tokens or a teacher scale distribution

### Takeaway
Bin/distribution-level distillation exists per pixel (TIE-KD, CKD), and scale/class auxiliaries exist (ScaleDepth scene CE, TR2M scale-oriented contrast). I found no work that distills a distribution over a global log-scale scalar, least of all one built from several teachers. Teacher global tokens are not available from offline PNG caches. A multi-teacher soft label over log-scale bins is feasible and looks unpublished.

### Cited Findings
- TIE-KD: Gaussian soft labels over 257 depth bins (sigma 0.8), KL plus SSIM. KITTI AbsRel 0.0663 to 0.0654. KL alone 0.0718, depth loss alone 0.0696. — [TIE-KD, arXiv 2402.14340](https://arxiv.org/pdf/2402.14340) [project notes]
- Cross-interaction KD (Dec 2025): distills depth-bin probabilities from a foundation model into a lightweight student. Metric bin centers are guided by GT, and distillation is at the probability level to avoid output-level scale sensitivity. — [arXiv 2512.08700](https://arxiv.org/html/2512.08700v1)
- TR2M: scale-oriented contrastive loss L_soc (20 linear depth classes, weight 0.1). iBims-1 d1 went from 0.708 to 0.736 and AbsRel from 0.160 to 0.154 when L_soc was added. On NeWCRFs, iBims AbsRel went from 0.223 to 0.210. About 19M trainable parameters plus frozen ViT-L/CLIP. — [TR2M, arXiv 2506.13387](https://arxiv.org/html/2506.13387v1)
- ScaleDepth: scale decomposition plus a 28-class scene CE (w 0.01). NYU-trained iBims d1 0.788 vs ZoeDepth 0.668. — [arXiv 2407.08187](https://arxiv.org/abs/2407.08187) [project notes]
- DepthART: a multi-query scale head (Q = 8, bounded sigmoid, MLP 128 to 128 to 1) trained only through SiLog plus edge, with no separate scale loss. iBims d1 was 0.534 with the scale head alone on a frozen encoder and 0.713 with camera adapters plus the scale head. — [DepthART, arXiv 2607.17099](https://arxiv.org/html/2607.17099)
- UniDAC predicts a spatially varying coarse scale map, upsampled using relative depth (the DGSE module), instead of a single scalar. Loss details were not in the abstract. — [UniDAC, arXiv 2603.27105](https://arxiv.org/abs/2603.27105)

### Inferences
- Proposed: add a K-bin (e.g. 48) log-scale classification head next to the regressor, train-only, or used as an expectation. The soft target is a mixture of Gaussians centred on s*_L, s*_B, s*_S, with weights such as 0.6/0.25/0.15 and width sigma_i from (a). Use a single Gaussian on GT rows. The loss is CE(p_hat, q) plus the existing SmoothL1. This combines (a) and (e) into one "multi-teacher scale-posterior distillation" contribution, at zero inference cost if the bin head is dropped.
- TR2M's L_soc gain on iBims (+2.8 d1) supports adding a scale-discriminative auxiliary on the features that feed the scale head. It is cheap, but adds a hyperparameter.

### Gaps
- Online teacher global tokens (V2-L camera/scale tokens) would need re-running the teacher. I found no evidence on whether token-level scale KD beats output-level scale KD.

---

## (f) Quantization-aware or distribution-matching losses for tiny depth students

### Takeaway
Depth-specific PTQ is published (QuartDepth, SPA-Q). QAT with distillation exists for detection and restoration, but I found no QAT-with-distillation loss for monocular depth. For Apple fp16 plus 8-bit weight-only deployment, which is already lossless, this is not an accuracy lever and is a weak fit for this paper.

### Cited Findings
- QuartDepth (CVPR 2025): PTQ handling depth-decoder activation outliers. Metric3D ViT-S NYU W4A8 AbsRel/d1: 0.103/0.899 vs FP 0.087/0.938. — [QuartDepth](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_QuartDepth_Post-Training_Quantization_for_Real-Time_Depth_Estimation_on_the_Edge_CVPR_2025_paper.pdf)
- QD-BEV combines QAT with view-guided distillation, for 3D detection. — [arXiv 2308.10515](https://arxiv.org/pdf/2308.10515)
- QDR (QAT plus KD for image restoration) reports a tug-of-war between reconstruction and distillation losses. — [arXiv 2603.09624](https://arxiv.org/pdf/2603.09624)
- An INT8 QAT U-Net depth model on a Kria KV260 FPGA (LASCAS 2026); no distillation loss was seen. — [LASCAS 2026](https://orbi.umons.ac.be/bitstream/20.500.12907/55714/1/LASCAS2026_paper_26.pdf)
- SPA-Q (Nov 2026) calibrates PTQ against a SILog loss between FP and quantized depth. — [arXiv 2611.04491](https://arxiv.org/abs/2611.04491) [project notes]

### Inferences
- Exclude this from the accuracy-focused loss paper, or keep it as a deployment appendix.

### Gaps
- No QAT-plus-KD depth paper turned up.

---

## (g) Object-size-prior or detection-box metric anchor losses

### Takeaway
Published, but mostly as test-time adaptation (Metric from Human), self-supervised category height priors (Casser et al. 2019), focal estimation from object priors (fCOP), or monocular 3D detection geometry. I found no training-time loss that anchors a decoupled global scale head with detector-box size priors in a supervised or distillation student. The idea is plausible but noisy.

### Cited Findings
- Metric from Human (NeurIPS 2024) paints humans into the image, estimates their size with HMR, and propagates metric scale to the scene at test time. It is annotation-free, motivated by MMDE "scene dependency". — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bd19ca8039547b339a6a37bd4df24405-Abstract.html)
- Casser et al. 2019 use a learnable per-category height prior. Approximate depth comes from segment pixel height and focal length, as a per-object scale loss normalized by mean depth to avoid trivial shrinkage. — [arXiv 1906.05717](https://arxiv.org/pdf/1906.05717)
- fCOP estimates focal length from category-level object priors. — [arXiv 2409.19641](https://arxiv.org/pdf/2409.19641)
- Monocular 3D detection: depth from 2D box height and 3D height (MonoDGP, CVPR 2025), but box height often misrepresents the true central height. — [MonoDGP](https://openaccess.thecvf.com/content/CVPR2025/papers/Pu_MonoDGP_Monocular_3D_Object_Detection_with_Decoupled-Query_and_Geometry-Error_Priors_CVPR_2025_paper.pdf). Box-height depth recovers the front surface, not the object center. — [arXiv 2611.11607](https://arxiv.org/html/2611.11607)

### Inferences
- For indoor assistive scenes (doors, chairs, people), a category-height anchor could give an extra weak scale constraint on unlabeled images. It would need a detector run offline plus per-category height priors. Given the noise in box height documented above, use only a robust (e.g. Cauchy) low-weight term on s_hat. It is lower priority than (b) and (a+e).

### Gaps
- I found no indoor numbers on how accurate category-height priors are as scale anchors.

---

## 2025-2026 tiny / edge metric-depth models (<10M params) and their losses

### Takeaway
Very few sub-10M students are truly metric. DepthART-S (6.0M) is the closest published peer. b20dws (5.04M, iBims d1 74.8, RMSE .735) already beats it on d1 (71.3) but not on RMSE (.693). ZipDepth (6.1M) is relative-only.

### Cited Findings
- DepthART-S has 6.0M params (TinyViM-S plus DPT) with a DAv2-L teacher.
  - Stage-1 loss: trimmed (top-10%) scale/shift-aligned MSE plus 0.5 x L1-gradient edge loss.
  - Metric fine-tuning loss: SiLog plus lambda x multi-scale edge, with d_metric = d_max (s ⊙ d_rel), d_max = 10 for NYU.
  - Results: iBims d1/RMSE 0.713/0.693 (S), 0.732/0.672 (B, 11.4M), 0.747/0.646 (L, 32.6M). NYU in-domain d1 0.924.
  - Neither multi-teacher, scale distillation nor a consistency loss is used.
  — [DepthART, arXiv 2607.17099](https://arxiv.org/html/2607.17099)
- ZipDepth: 6.1M, SSI-MAE plus 2x multi-scale gradient, relative only. The authors list metric extension as future work. — [ZipDepth, arXiv 2607.08771](https://arxiv.org/abs/2607.08771v1)
- Depth Pro 2-S (24M family) reaches iBims d1 78.2 with GT focal. — [arXiv 2611.26603](https://arxiv.org/abs/2611.26603) [project notes]
- TR2M: ViT-S-scale trainable decoders (about 19M) but large frozen encoders. iBims d1 0.736. — [TR2M](https://arxiv.org/html/2506.13387v1)
- Metric Anything students are large (326M / 877M). iBims d1 86.6 without intrinsics and 92.0 with. Loss is a distance-balanced inverse-depth loss (C = 400). — [arXiv 2601.22054](https://arxiv.org/html/2601.22054)
- DisDepth-B0 (8.0M): NYU AbsRel 0.125 / d1 0.872 (in-domain). — [arXiv 2404.16386](https://arxiv.org/abs/2404.16386) [project notes]

### Inferences
- A paper can position b20dws as the smallest decoupled-scale metric student, with b20dws vs DepthART-S as the headline comparison. The novelty should sit in the scale-learning loss, specifically (b) FoV-equivariant scale consistency and (a+e) multi-teacher scale-posterior distillation, because that is where 16 of the missing iBims d1 points sit.
- Suggested ablation ladder, each step training-only:
  1. b20dws baseline.
  2. Add L_eq (crop+zoom equivariance on s_hat).
  3. Add a disagreement-weighted scale target (a).
  4. Add a bin soft-label CE from the multi-teacher mixture (e).
  5. Add an asymmetric near-band metric hinge (c).
  6. Optionally, a late sg(s_hat) alignment (d) as a negative or neutral control.
  Report IBims raw and scale-aligned d1 (aligned d1 should stay at about 91 if structure is untouched), NYU-654, SUN, and OCMR objAbsRel.

### Gaps
- I could not find a sub-10M metric model reporting iBims besides DepthART. UniDAC's size and losses were not available from the abstract.
