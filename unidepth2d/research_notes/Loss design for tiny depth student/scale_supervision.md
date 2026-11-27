# Losses and supervision for the global METRIC SCALE of monocular depth models with a decoupled scale head (2023-2026)

Scope note: this file adds to `docs/B20_RESEARCH.md` and does not repeat it. That file already covers the MoGe-2 Table 4 numbers, DepthART, Metric3D CSTM, DMD FoV augmentation, UniDepth camera/detach, ScaleDepth, RSA, "What the Averages Hide", FoundationGeo focal diversity, UniDAC and MetricAnything. This file adds:
- the exact loss code and equations,
- uncertainty, distributional and envelope scale losses,
- the 2026 zoom-equivariance evidence,
- Depth Pro loss ablations,
- a concrete loss and ablation plan for b20dws.

Notation used for b20dws: log D = U(r_c) + s_hat + c, with c = log(f_px/518.9). The target is s* = sg(median(log D_gt - U(r_c) - c)). The current loss is L_s = SmoothL1_{beta=0.05}(s_hat - s*). The delta1 threshold in log space is log 1.25 = 0.223, so an image whose structure is perfect still fails delta1 everywhere once |s_hat - s*| > 0.223. On IBims, b20dws scores raw d1 74.8 against scale-aligned d1 91.2.

## Q1. How do MoGe-2, Depth Pro, UniDepth/V2, Metric3D v2, DA-metric/DA3, ZoeDepth, DMD and recent work supervise metric scale? Exact loss forms

### Takeaway
Only MoGe-2 (and X-Lens, a multi-view model) use a truly decoupled scalar scale head. MoGe-2 trains it with a plain L2 loss in log space against a stop-gradient online optimal-alignment scale. Its code also ships an L1-in-log variant.
- Every other foundation model folds scale into a dense loss: a canonical-focal transform plus dense depth or inverse-depth losses (Metric3D, DA3, Depth Pro), or SILog / lambda-MSE in log space (UniDepth/V2).
- No surveyed paper uses an uncertainty-weighted, binned or ranking loss on the global scalar.
- b20dws (SmoothL1 in log space against a stop-gradient median target) is already in the MoGe-2 family. The design space left open is the shape of the target, the distance and the data, not the decoupling itself.

### Cited Findings
- MoGe-2 scale loss (Eq. 3): L_s = || log(s_hat) - stopgrad(log(s*)) ||_2^2. Here s* is the optimal scale between the predicted affine-invariant point map and the GT, computed online by the ROE solver ("an efficient parallel searching algorithm"). The paper does not say whether s* minimises truncated L1, median or least squares. The global geometry loss is the 1/z-weighted L1 after ROE alignment: L_G = sum_i (1/z_i) || s* p_hat_i + t* - p_i ||_1. No scale-loss weight and no FoV augmentation are reported ("follows MoGe for cropping and augmentation"). — [MoGe-2 arXiv 2507.02546](https://arxiv.org/html/2507.02546v1)
- MoGe-2 Table 4 (ViT-B, 10-dataset average) adds detail beyond B20_RESEARCH. Decoupling also improves metric POINT accuracy (Rel_p/d1):
  - entangled SI-log: 10.0/90.7
  - entangled shift-invariant: 8.99/92.1
  - decoupled conv head: 9.62/91.4
  - decoupled CLS-MLP: 9.20/91.9

  Boundary F1 rises from 10.7 (SI-log entangled) to 12.5 (decoupled). The entangled shift-invariant variant beats SI-log on everything except metric depth d1 (68.8 vs 68.6), so the large d1 gain (+4.2) comes specifically from the global-MLP head. — [MoGe-2 arXiv 2507.02546](https://arxiv.org/html/2507.02546v1)
- MoGe official code has two scale losses. `metric_scale_loss` is MSE on log(scale_pred) vs log(scale_gt). `metric_scale_l1_loss` is L1 on the logs. Both mask only scale_gt > 0, so there is no trimming or weighting inside the loss; s* is computed upstream. `decoupled_shift_invariant_global_loss(pred_metric_scale, pred_points, ..., detach=False)` also exists: it multiplies the points by the predicted scale, fits only a z-shift, and has a `detach` flag controlling whether metric-point gradients reach the scale. This is a second, dense path that supervises scale through the geometry. — [MoGe repo losses.py](https://raw.githubusercontent.com/microsoft/MoGe/main/moge/train/losses.py)
- UniDepthV2 depth loss is lambda-MSE in the pseudo-spherical output space: L = ||V[eps]||_1 + lambda^T (E[eps] ⊙ E[eps]), with lambda_theta = lambda_phi = 1 and lambda_z = 0.15 (SILog for log-depth). The total is L_lambdaMSE + 0.1 L_con + 1.0 L_EG-SSI + 0.1 L_L1(uncertainty).
  - The uncertainty is regressed with L1 onto sg(|Z_log - Z*_log|). It does not re-weight the depth loss.
  - The consistency loss is bidirectional L1 to a stop-gradient target under rescaling r ~ 2^U[-2,2] and translation U[-0.1,0.1].
  - No ablation of lambda_z or of SILog vs L1 is reported. Without camera conditioning, V2 drops from 54.5 to 49.3 d1 (ViT-S, 100k steps).
  - [UniDepthV2 arXiv 2502.20110](https://arxiv.org/html/2502.20110)
- Depth Pro predicts canonical inverse depth C. Metric depth is D_m = f_px / (w C). The metric loss is per-pixel MAE on C, discarding the top-20% error pixels per image on real data (no trimming on synthetic). Non-metric data is median/MAD-normalised. The FOV head is a separate small conv head predicting horizontal FOV with an L2 loss, trained after depth. — [Depth Pro arXiv 2410.02073](https://arxiv.org/html/2410.02073v2)
- Depth Pro ablations (Appendix B):
  - Stage-1 loss choice moves Apolloscape metric d1 from 45.6 (1A: MAE on metric data, SSI-MAE on non-metric) to 75.6 (1C: adds a scale-and-shift-invariant gradient loss on synthetic data). That is +30 d1 from the gradient-loss placement alone.
  - Inverse-depth vs log-depth vs depth targets on Hypersim d1 at 0-1 m: 0.730 / 0.700 / 0.657; at >16 m: 0.922 / 0.920 / 0.850. Inverse depth helps near range; log-depth is close.
  - FOV head d25%: depth-encoder-only 60.0; separate focal encoder 74.4; parallel encoders 78.2.
  - [Depth Pro arXiv 2410.02073](https://arxiv.org/html/2410.02073v2)
- Depth Anything 3 metric applies Metric3Dv2's canonical camera transform with canonical focal 300: D_metric = f_px * D_canon / 300. Teacher predictions are aligned to GT metric depth and used as supervision. Loss = depth + gradient + sky-mask (exact weights not verified). — [DA3 arXiv 2511.10647](https://arxiv.org/pdf/2511.10647); [DA3 metric model card](https://huggingface.co/APaul1/DA3METRIC-LARGE)
- Metric3D v2 decoder uses AdaBins-style bins (256 bins, 0.3-150 m for ConvNet models). Scale comes from the CSTM label transform plus dense losses, not a scalar head. ZoeDepth and Depth Anything (v1/v2 metric) use a metric-bins module in the decoder. — [Metric3D v2 arXiv 2404.15506](https://arxiv.org/html/2404.15506v4); [Region-aware scale adaptation arXiv 2507.14879](https://arxiv.org/pdf/2507.14879)
- X-Lens (2026, 0.04B params, multi-view) factorises the output as (D_hat normalised by its in-batch mean, confidence, a global scalar m_hat) with D_metric = m_hat · D_hat. The authors state that this "isolates the absolute physical scale into a one-dimensional pathway", letting all other geometric losses be scale-invariant. Its scale-loss form was not extracted. — [X-Lens arXiv 2607.12993](https://arxiv.org/pdf/2607.12993)

### Inferences
- b20dws vs MoGe-2: the student uses a median target and SmoothL1 with beta = 0.05. Since beta = 0.05 is much smaller than typical |Δs| ~ 0.1-0.4, this is effectively L1 in log space, matching MoGe's `metric_scale_l1_loss`. MoGe-2's paper uses L2. L2 penalises large scale misses (the ones that destroy d1) more heavily. A cheap first ablation is L1 vs L2 vs a margin-shaped loss (see Q2/Q4).
- The b20dws jump on IBims from b10 (66.8) to b20dws (74.8), +8 d1, is a 5M-scale replication of MoGe-2's +4.2. This is a publishable sanity result in its own right.
- MoGe has a second, dense path (`decoupled_shift_invariant_global_loss`, detach flag) in which the metric point loss can also reach the scale. b20dws has only the scalar path, so the scale head gets one scalar of supervision per image (~12k scalars per epoch).
- An ablation: add a dense metric loss on log D = U(r_c) + s_hat + c with the gradient detached from U. The scale then receives a robust, pixel-aggregated signal, such as a trimmed L1 whose effective target is a trimmed mean rather than the median. Structure stays protected.
- Depth Pro's results suggest a near-range-weighted or inverse-depth term matters for indoor scenes. The existing proximity-weighted SILog is the student's analogue.

### Gaps
- MoGe-2 does not state which estimator ROE uses for s* during training. The code fetched here takes scale_gt as an input; the computation is in the training script, which was not inspected.
- No paper found ablates median vs least-squares vs truncated-L1 targets for a scale head.
- The Depth Pro trimming ablation numbers were not in the visible appendix.
- The DA3 metric loss weights were not verified.
- PriorDA (Prior Depth Anything) and FoundationGeo exact scale-loss forms were not retrieved in this pass. PriorDA needs metric priors at input, so it is not applicable to b20dws.

## Q2. Uncertainty-aware, distributional, ranking/contrastive, binned and multi-hypothesis scale losses

### Takeaway
There is almost no published evidence for these losses on a global monocular scale scalar, which makes this a real novelty gap. The closest work:
- a 2026 caption-conditioned log-scale/log-shift "uncertainty envelope" trained with a softplus hinge, worth +0.3 to +0.6 d1 over RSA on NYU;
- Scalix, which adds a separate global-scale uncertainty head for SLAM.

Evidence from adjacent fields: confidence-weighting pixel losses hurts depth, while classification-style ("HL-Gauss") regression beats MSE in other domains. Distributional or binned scale heads are therefore the most promising, and untested, option. NLL re-weighting is the riskiest.

### Cited Findings
- "Language as Prior, Vision as Calibration" (arXiv 2601.01457) predicts global log-scale/log-shift θ for a FROZEN relative backbone (DPT-Hybrid, or DepthAnything-Small at 24.8M).
  - Loss: L_depth (per-pixel L1 on metric depth) + λ_env Σ_k softplus(|θ*_k - μ_k(T)| - r_k(T)) + λ_r ||r||_1 + λ_cal ||(α,β) - sg(θ*)||_1, with λ_env = 0.1 and λ_cal = 1.0. The radius r is clamped at 3. θ* is a closed-form least-squares fit in inverse depth.
  - Selector head: GAP over strides 4/8/16/32, concatenated, then a 3-layer MLP (256 hidden) with tanh.
  - Supervision ablation (DPT-H, NYU d1/AbsRel): without oracle target 0.908/0.103; without L_cal 0.914/0.098; without L_env 0.916/0.097; full 0.919/0.095.
  - DA-S on NYU: image-only scale head 0.749 vs full 0.781 d1.
  - Zero-shot SUN-RGBD (DA-S) AbsRel: image-based 0.279, RSA 0.238, theirs 0.213.
  - [arXiv 2601.01457](https://arxiv.org/html/2601.01457)
- In that work, an explicit stop-gradient oracle-scale regression term (L_cal) plus a dense metric L1 beats either alone. Removing the oracle target costs 1.1 d1 on NYU. This is direct evidence for combining a scalar target loss with a dense metric loss through the scale. — [arXiv 2601.01457](https://arxiv.org/html/2601.01457)
- Scalix (2026, SLAM) adds an uncertainty decoder to a base metric depth model and models GLOBAL scale uncertainty separately from per-pixel uncertainty. The head uses strided convs, adaptive average pooling and an MLP to a scalar. The authors argue that per-pixel depth errors are correlated through the shared scale, so they propagate the uncertainty into a full depth covariance. The exact NLL form and numbers were not extracted (PDF unreadable via fetch). — [Scalix arXiv 2608.17553](https://arxiv.org/pdf/2608.17553) (search-result summary only)
- UniDepthV2 trains its uncertainty only as a side output (L1 to the detached log error) and never re-weights the depth loss. — [UniDepthV2 arXiv 2502.20110](https://arxiv.org/html/2502.20110)
- Depth Pro 2 (2026; already logged in B20_RESEARCH) reports that confidence-weighted losses are harmful (boundary F1 -6.8). — see `docs/B20_RESEARCH.md` entry for arXiv 2611.26603 (not re-fetched here)
- The pitfalls of Gaussian NLL for heteroscedastic regression: the variance term down-weights hard samples and lets the mean underfit. The beta-NLL fix multiplies each sample's NLL by sg(σ^{2β}). — [Seitzer et al., ICLR 2022, arXiv 2203.09168](https://arxiv.org/abs/2203.09168) (from prior knowledge; not re-fetched in this session)
- Replacing MSE regression with cross-entropy over a histogram target (Gaussian-smoothed "HL-Gauss" label) improves performance and scaling of value regression in deep RL. — [Farebrother et al., "Stop Regressing", ICML 2024, arXiv 2403.03950](https://arxiv.org/abs/2403.03950) (from prior knowledge; not re-fetched)
- TIE-KD: KL to Gaussian soft labels over 257 depth bins plus a depth loss beats either alone for a 17.6M student (KITTI AbsRel 0.0654 vs 0.0718 / 0.0696). Already logged in B20_RESEARCH. — [TIE-KD arXiv 2402.14340](https://arxiv.org/abs/2402.14340)
- No source was found for a batch-contrastive or cross-image ranking loss on global scale. Ranking losses in depth are used intra-image as scale-free supervision. — [Plackett-Luce listwise depth ranking, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Lienen_Monocular_Depth_Estimation_via_Listwise_Ranking_Using_the_Plackett-Luce_Model_CVPR_2021_paper.pdf); [PatchRefiner arXiv 2406.06679](https://arxiv.org/abs/2406.06679)

### Inferences
- **Distributional scale head (recommended first novelty).** Replace the scalar regression by a K-bin softmax over log-scale, for example 48-64 bins over the s* range seen in training, with bin width ~0.03-0.05 (below 0.223/4).
  - Train with HL-Gauss cross-entropy against a Gaussian-smeared one-hot at s* (σ ≈ 0.75 bin width), plus SmoothL1 on the expectation E[s].
  - Inference uses E[s] or the mode: one softmax and a dot product over 64 values, effectively zero cost.
  - Why it should help a 5M student: classification losses give non-vanishing, bounded gradients and are less sensitive to label noise (teacher scale errors on 'own' rows). The histogram can also represent ambiguity (bimodal "small room vs big room") without the mean collapsing between modes. The entropy is a free per-image scale-confidence output, useful for the paper and for OCMR.
  - Evidence is indirect (RL value regression, TIE-KD dense bins). It is untested for a global depth scale, which is the novelty.
- **Heteroscedastic Laplace NLL on s** (L = |s_hat - s*|/b + log b) is risky. It lets the head give up on exactly the hard unseen-scene images that dominate the IBims error, consistent with the Depth Pro 2 confidence-weighting result and the project's own s6 NLL regression.
  - If uncertainty is wanted, use beta-NLL (β = 0.5), or train b as a detached side output (UniDepthV2 style: L1(b, sg|s_hat - s*|)) so it never re-weights the scale loss.
- **Cross-image (batch) relative-scale loss (novel).** Within a batch, supervise Δ_ij = s_hat_i - s_hat_j against s*_i - s*_j, using L1 or a logistic ranking loss on its sign with margin log 1.25.
  - For pairs from the SAME pseudo-label source this is invariant to any multiplicative teacher bias. "What the Averages Hide" reports dataset-dependent scale biases even for large models.
  - The absolute term can then be weighted more on GT rows (NYU/SUN) and less on teacher rows. This is an image-level analogue of PatchRefiner's "scale-free on pseudo, metric on GT" idea.
  - With batch 4 x accum 3, pairs exist only within a 4-image micro-batch, so a small memory bank of (feature, s*) from recent batches may be needed.
- **Envelope / interval loss.** The softplus-hinge envelope gives small but consistent gains on a frozen backbone (+0.3-0.5 d1). A text prior is not available at inference for b20dws. A train-only variant could use cached captions or scene labels to predict (μ, r) and use them as an auxiliary target or regulariser for the scale head. Expected gain is small.

### Gaps
- No paper reports binned/HL-Gauss, beta-NLL or ranking losses specifically for a monocular global scale scalar. Effects must be measured.
- The Scalix loss formula and numbers were not extracted.
- No multi-hypothesis (mixture-of-scales / winner-takes-all) global-scale head for monocular depth was found.

## Q3. What improves zero-shot metric scale for small models (camera/FoV augmentation, focal diversity, crop/zoom scale consistency, object-size and semantic priors)?

### Takeaway
The strongest new 2026 evidence is "Zoom In, Scale Out" (arXiv 2611.17903).
- Pixels-only metric models treat a digital zoom almost exactly as moving closer: DA-V2-S depth exponent α = 0.95, against an ideal of 0.
- Camera-predicting models fix only about a third of this (α 0.31-0.36).
- Models that take the TRUE camera in a canonical-focal form are nearly equivariant: Metric3D v2 α = -0.07.
- A label-free zoom self-consistency loss plus an anti-blur term brings α from 0.98 to 0.07 on NYU with NYU AbsRel roughly unchanged, but it costs accuracy on KITTI/DIODE-out.

For b20dws, which has a canonical-focal term c and known K, this yields an exact, label-free scale-equivariance constraint on the scale head. Combined with DMD-style FoV augmentation (iBims AbsRel 0.180 → 0.130), it is the best-evidenced lever for unseen-scene scale.

### Cited Findings
- Exact law: a centred crop by zoom factor s, resized back, equals the same scene shot with f' = s f (and c' adjusted). Metric depth of every surviving pixel is unchanged.
- Zoom audit, 800 images over KITTI/NYUv2/DIODE (α: depth exponent, ideal 0; β: focal exponent, ideal 1):

  | Model | α | β |
  |---|---|---|
  | DA-V2-S | 0.95 | n/a |
  | DA-V2-B | 0.85 | n/a |
  | ZoeDepth | 0.61 | n/a |
  | Depth Pro | 0.31 | 0.65 |
  | MoGe-2-S | 0.36 | 0.56 |
  | UniDepthV2-S | 0.34 | 0.45 |
  | Metric3D v2-S (true K) | -0.07 | n/a |
  | Metric3D v2-S (stale K) | 0.93 | n/a |

  — [Zoom In, Scale Out arXiv 2611.17903](https://arxiv.org/abs/2611.17903)
- The same paper explains why: a zoomed image is pixel-for-pixel the unzoomed scene "squashed along the viewing axis", so apparent size alone cannot tell the two apart (the zoom-squash ambiguity). Regression-trained focal heads "play safe" and shrink toward the training camera prior: β equals the cue reliability ρ = τ²/(τ²+σ²). Given the true camera, UniDepthV2-S α drops from 0.34 to 0.13 and its unzoomed scale error from 0.138 to 0.104. MoGe-2-S drops from 0.36 to -0.08, with scale error 0.214 → 0.146. — [arXiv 2611.17903](https://arxiv.org/pdf/2611.17903)
- Equivariance fine-tuning loss (Eq. 4) on unlabeled images: L = || U_b log D_θ(Z_b I) - log D_0(I)|_b ||_1 + λ || log D_θ(I) - log D_0(I) ||_1. Here s ~ LogUniform[1,3], D_0 is the frozen teacher on the unzoomed image, and λ = 1. The anti-shortcut term (Eq. 5, "EFT+B") adds γ || log D_θ(B_s' I) - log D_0(I) ||_1, where B_s' down-samples then up-samples (s' ~ LogUniform[1,3]) and γ = 1. — [arXiv 2611.17903](https://arxiv.org/pdf/2611.17903)
- Results (DA-V2-S, Table 4), α / AbsRel(1x) / blur sensitivity:

  | Dataset | Frozen | EFT (no anti-blur) | EFT+B |
  |---|---|---|---|
  | NYU | 0.98 / 0.344 / 0.7% | 0.12 / 0.313 / 4.2% | 0.07 / 0.301 / 0.4% |
  | KITTI | 1.03 / 0.138 / 1.1% | 0.18 / 0.156 / 29.6% | 0.05 / 0.175 / 4.9% |

  On the held-out DIODE-in set, α 0.70 → 0.37. AbsRel costs: KITTI 0.138 → 0.175 and DIODE-out 0.731 → 1.000, while NYU stays roughly unchanged across seeds (0.344 → 0.326 ± 0.026).
  - Without the blur term, the model learns "blurry means zoomed" (a 29.6% depth change under blur on KITTI).
  - A Zhong-style self-target (student's own stop-gradient prediction) drifts: KITTI AbsRel 0.222.
  - "EFT-S" learns one log-scale per image, D = D_0(I) e^{c(I)}, from a small MLP on frozen tokens. It reaches α 0.03 ± 0.05 (KITTI) and 0.09 ± 0.03 (NYU), i.e. a scale-only correction is enough.
  - [arXiv 2611.17903](https://arxiv.org/pdf/2611.17903)
- Training-free corrections from the same paper:
  - "FocalUnshrink" maps the focal estimate as μ + (log f_hat - μ)/β_hat. It takes Depth Pro from α 0.31 to 0.04 with unzoomed scale error unchanged (0.138 vs 0.136).
  - "ZoomFix" (scale depth by s^α when s is known) removes 38-70% of the 3x-zoom error.
  - The label-free |α| rank-correlates with the error a zoom adds (Spearman 0.67) but not with unzoomed accuracy (-0.09).
  - [arXiv 2611.17903](https://arxiv.org/pdf/2611.17903)
- Focal-length mismatch after crop/resize mainly causes a near-GLOBAL metric scale bias, while principal-point offsets cause spatially structured distortion. — [MDPI Electronics 15(10) 2180](https://www.mdpi.com/2079-9292/15/10/2180) (search-summary level; not fetched)
- Crop-and-resize equivariance via test-time averaging plus a self-consistency loss improves state-of-the-art supervised depth/normal predictors. — [Zhong et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhong_Improving_Equivariance_in_State-of-the-Art_Supervised_Depth_and_Normal_Predictors_ICCV_2023_paper.pdf)
- Depth Any Camera uses multi-resolution training; removing it drops Matterport3D d1 from 0.728 to 0.567. Already logged. — [DAC arXiv 2501.02464](https://arxiv.org/html/2501.02464v1)
- A frozen 24.8M DA-S with a GAP+MLP scale head over strides 4-32 reaches NYU d1 0.749 from image features alone and 0.781 with a caption prior. Semantic priors add ~3 d1 on top of an image-only scalar head at small scale. — [arXiv 2601.01457](https://arxiv.org/html/2601.01457)

### Inferences
- **b20dws is already in the "takes the camera" group (Metric3D-like),** because log D = U(r_c) + s_hat + c with c = log(f/518.9). In the canonical frame, a zoom by s raises c by log s. Depth stays correct exactly when the canonical scale s_hat + mean(U) drops by log s, i.e. when the network reads apparent size (which is natural).
  - Two shortcuts can break this. (a) The scale head's 16-d intrinsics embedding and the FiLM can learn to undo c, which is easy when training focal spread is tiny (NYU/SUN/own are all ~518-580 px). (b) The crop-context and FoV-uncrop augmentations must update K exactly; stale K gives α ≈ 0.93 per the audit.
  - **Action:** run the label-free zoom audit (α on NYU/IBims at s ∈ {1.25, 1.5, 2}) on b20dws. It costs nothing, and it is a strong analysis figure for the paper.
- **Proposed novel training-only loss, "canonical zoom-equivariance of the scale head" (L_zeq).** For a random centred or off-centre zoom Z_b with factor s ~ LogUniform[1, 2] (indoor, VGA) and K' updated:
  - Constraint: s_hat(Z_b I, K') = sg(s_hat(I, K)) - log s. Optionally also add the structure term U_b U(r_c)(Z_b I) ≈ U(r_c)(I)|_b up to a mean, which is shift-invariant because r_c is mean-centred.
  - Add the EFT+B anti-blur term on the scale head: s_hat(B I, K) = sg(s_hat(I, K)).
  - On labelled rows, zoom augmentation with an exactly transformed target already supervises this. The consistency term adds paired variance reduction and works on UNLABELLED images, so the scale head can see 10-100x more images. That matters because it gets only one scalar per image.
  - Unlike EFT on DA-V2-S, b20dws keeps K, so the constraint is satisfied by the correct solution. The KITTI/DIODE accuracy cost seen there (a pixels-only model forced to be equivariant without K) should not apply. This is an inference to verify.
- **Use FoV augmentation as a free scale-label multiplier,** with s* transformed exactly. DMD's iBims AbsRel went from 0.180 to 0.130 with only a small NYU cost (0.074 → 0.076).
  - Current settings (crop-context p 0.5, FoV uncrop p 0.25) may be too narrow in focal range. A sweep over the zoom range (e.g. 0.67-2.0) is a cheap ablation.
- **Order of expected impact on IBims raw d1** (inference): (1) more scale-labelled images (teacher s* on unlabeled indoor frames); (2) correct-K FoV/zoom augmentation plus L_zeq; (3) target and distance shaping (Q4); (4) semantic or caption priors (small).

### Gaps
- No small-model (≤10M) zoom-equivariance results exist; the EFT experiments are on DA-V2-S (24.8M).
- The paper's exact AbsRel cost for a K-conditioned model under EFT was not reported.
- Object-size priors as an explicit training loss: no 2023-2026 paper found with ablation numbers beyond the "adding objects reduces scale error" observation already in B20_RESEARCH.

## Q4. Quantitative ablations, and a concrete loss / ablation plan for b20dws (accuracy first, novelty second)

### Takeaway
The verified numbers point to:
- the decoupled global MLP scale head: +4.2 d1 in MoGe-2, +8.0 on IBims in this project;
- camera-correct FoV augmentation: iBims AbsRel -28% in DMD;
- correct camera conditioning: Metric3D α -0.07 vs 0.93 with stale K;
- oracle-scale regression plus a dense metric term: +1.1 d1 when adding the oracle term in arXiv 2601.01457.

The most defensible novel-yet-safe loss package for a 5M student is:
- (a) a distributional (HL-Gauss) log-scale head with a δ1-aware target;
- (b) canonical zoom-equivariance plus anti-blur consistency on the scale head, applied also to unlabeled teacher-free images;
- (c) a detached dense metric path into the scale head.

All are training-only.

### Cited Findings
- MoGe-2 decoupled CLS-MLP vs entangled SI-log: metric depth d1 68.6 → 72.8, Rel 17.9 → 16.5. A conv scale head gives no gain (68.4). — [MoGe-2](https://arxiv.org/html/2507.02546v1)
- UniDepthV2 (ViT-S, 100k-step zero-shot d1): camera conditioning +5.2 (49.3 → 54.5); output-space consistency +0.8 (54.5 → 55.3); EG-SSI +4.7 (55.3 → 60.0); spherical vs xyz output +10.9 (49.1 → 60.0). — [UniDepthV2](https://arxiv.org/html/2502.20110)
- Depth Pro: placing the gradient loss (SSI-gradient on synthetic data) moved Apolloscape metric d1 from 45.6 to 75.6. Inverse-depth targets beat log-depth at 0-1 m (0.730 vs 0.700 Hypersim d1). — [Depth Pro](https://arxiv.org/html/2410.02073v2)
- Language-prior calibration supervision ablation: NYU d1 0.908 without oracle, 0.914 without L_cal, 0.919 full. — [arXiv 2601.01457](https://arxiv.org/html/2601.01457)
- Zoom equivariance: α 0.98 → 0.07 (NYU) with NYU AbsRel roughly unchanged; anti-blur term needed (blur sensitivity 4.2% → 0.4% on NYU, 29.6% → 4.9% on KITTI). — [arXiv 2611.17903](https://arxiv.org/pdf/2611.17903)

### Inferences
**Proposed loss for b20dws.** Pixel terms are unchanged: they act on p_al and train only structure.

L_total = L_pix(p_al) + w_s · L_scale

L_scale = 1.0 · CE_HLGauss(π(s), N(s*_δ, σ)) + 0.5 · L1(E_π[s] - s*_δ) + 0.3 · L_dense_scale + 0.2 · L_zeq + 0.1 · L_blur [+ 0.2 · L_pair]

- **s*_δ, a δ1-aware alignment target (novel, ~10 lines).**
  - Replace the median by the window-mode of residuals ρ_i = log D_i - U_i - c: s*_δ = argmax_s Σ_i 1[|ρ_i - s| < log 1.25].
  - Compute it by a 1-D sliding-window max over sorted ρ, or as the mean-shift mode with a box kernel of half-width 0.223, in fp32 under no_grad.
  - The median minimises L1 (AbsLog). The window-mode is the exact per-image maximiser of oracle-aligned δ1. They differ when residuals are skewed (far walls, windows, teacher errors), which is where pseudo-labels fail.
  - Diagnostic before training: on IBims, compare the aligned d1 under median vs window-mode alignment. If the gap is under 0.3 d1, keep the median.
- **CE_HLGauss.** K = 64 bins over [s*_min - 0.3, s*_max + 0.3] from training statistics, with σ = 0.75 × bin width. Inference uses E_π[s] (zero cost). Keep the SmoothL1/L1 term on the expectation so the point estimate stays calibrated.
- **L_dense_scale.** A trimmed (top-10%) L1 on log D_pred = sg(U(r_c)) + s_hat + c vs log D_gt. Only s_hat receives gradient. This gives the scale head a pixel-aggregated robust signal, as in MoGe's `decoupled_shift_invariant_global_loss` with detach and arXiv 2601.01457's dense + L_cal combination. Add a proximity weight (<2 m) to match the indoor near-range emphasis.
- **L_zeq / L_blur.** Defined in Q3. Use s ~ LogUniform[1, 2], off-centre boxes, K updated exactly, and targets sg(s_hat(I)) - log s and sg(s_hat(I)) respectively. One extra forward pass of the student on a zoomed copy for ~25% of images (+~25% training compute, zero inference cost). Apply also to unlabeled indoor images, which need no teacher.
- **L_pair (optional).** Within-source pairwise L1 on (s_hat_i - s_hat_j) - (s*_i - s*_j), for teacher-bias robustness on 'own' rows.
- **Avoid:** Laplace/Gaussian NLL re-weighting of L_scale; per-pixel confidence weighting.

**Ablation table (each row vs A0).** All are training-only.
- A0: b20dws (SmoothL1 β = 0.05, median s*).
- A1: L2-log (the MoGe-2 paper form).
- A2: L1-log (the MoGe code form).
- A3: s*_δ window-mode target.
- A4: HL-Gauss bins + L1 on the expectation.
- A5: A4 + L_dense_scale.
- A6: + FoV/zoom augmentation with exact s* transform over a wider range (0.67-2.0).
- A7: + L_zeq.
- A8: + L_blur.
- A9: + L_zeq on unlabeled frames.
- A10: + L_pair.
- A11: beta-NLL, as a negative control.
- A12: the scale head's intrinsics embedding removed or detached. This tests the "undo c" shortcut.

**Report for each row:**
- IBims raw and scale-aligned d1/AbsRel;
- log-scale bias and the MAD of (s_hat - s*);
- failure rate (d1 < 0.5);
- the zoom exponent α (label-free);
- NYU-654 and SUN d1, as the in-domain guard;
- OCMR objAbsRel.

Use 3 seeds for the key rows. IBims is small (100 images), so per-image paired Wilcoxon tests on |s_hat - s*| are more sensitive than d1 averages.

**Paper framing (novelty without sacrificing accuracy):** "For tiny metric-depth students, scale is a one-scalar-per-image learning problem: data-starved, ambiguous and camera-entangled." The three contributions would be:
1. a δ1-optimal distributional scale target;
2. an exact canonical-focal zoom-equivariance self-supervision that multiplies scale labels for free;
3. a zoom-exponent analysis showing the 5M student becomes camera-equivariant (α → 0), where Depth Pro, MoGe-2 and UniDepthV2-S stay at 0.31-0.36.

### Gaps
- None of the proposed components (window-mode target, HL-Gauss scale head, scale-head zoom-equivariance with K) has published ablation numbers. They are hypotheses to test.
- Whether the gain survives at 5M params is unverified. DepthART (B20_RESEARCH) suggests a frozen-encoder second stage may be needed for the scale head to generalise.
- How exactly the current crop-context and FoV-uncrop augmentations update K and s* in `distill/dataset.py` was not inspected (read-only constraint; check before running A6-A9).
