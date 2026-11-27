# Relative-geometry (structure) losses for monocular depth, with a focus on compact students (b20dws)

Scope: structure-only losses (SSI variants, HDN, RPNL, EG-SSI, gradient matching, normal / virtual-normal, MoGe global and local point losses, edge/sharpness/frequency losses). This note is about what to apply to p_al = U(r_c) + c + s* in b20dws. Material already in docs/B20_RESEARCH.md is cited only by pointer: MoGe-2 decoupling, Depth Pro 2 trimming and boundary-loss harm, RPNL topline, EG-SSI topline, DAv2 trimming, PatchRefiner ranking, and Distill-Any-Depth normalization.

Project anchor numbers used below: IBims b20dws metric d1 74.8, scale-aligned d1 91.2, teacher V2-L 94.6 metric. So about 16 of the 25 missing d1 points are scale and about 9 are left after alignment. Some of those 9 are structure, some are the metric/aligned protocol difference.

## Q1. Exact formulations of the main structure losses, and which domain/alignment to use

### Takeaway
Use scale-only invariance in log-depth (a shift in log D), never depth-space affine (scale+shift) invariance as the main structure loss for a metric student. MoGe's ablation shows depth-affine SSI destroys scale-invariant geometry (point SI Rel 29.9 vs 9.8). Local and hierarchical normalizations (HDN, RPNL, EG-SSI, MoGe local spheres) are where the measured structure gains are. b20dws's median-in-log alignment is already the L1-optimal scale for an L1 log loss, so the remaining upgrades are about locality, robustness (truncation) and 3D-aware terms.

### Cited Findings
- **MiDaS/HDN base SSI normalization**: N_u(d_i) = (d_i − median_u(d)) / ((1/|u|) Σ_j |d_j − median_u(d)|), and L_SSI = mean_i |N_u(d_i) − N_u(d*_i)|, with one global context u. — [HDN, Zhang et al., NeurIPS 2022, ar5iv 2210.09670](https://ar5iv.labs.arxiv.org/html/2210.09670)
- **HDN loss**: L_HDN = (1/M) Σ_i (1/|U_i|) Σ_{u∈U_i} |N_u(d_i) − N_u(d*_i)|, where each pixel averages SSI over every context that contains it. The contexts are:
  - HDN-S: S×S spatial grids, S ∈ {1, 2, 4, 8}.
  - HDN-DR: equal-width depth-range bins, S ∈ {1, 2, 4}.
  - HDN-DP: equal-count depth-quantile bins.

  Implementation is just masked SSI per context, so it batches. — [HDN](https://ar5iv.labs.arxiv.org/html/2210.09670)
- **RPNL (Metric3D v2)**: M = 32 random crops of 0.125–0.5 of the image size. Each crop is normalized by median/MAD for both prediction and GT, and the L1 is averaged. The full depth loss is L_d = L_PWN + L_VNL + L_silog + L_RPNL. The total loss is 0.5·L_d + 1·L_n + 0.01·L_d-n (depth-normal consistency). — [Metric3D v2, arXiv 2404.15506](https://arxiv.org/html/2404.15506)
- **EG-SSI (UniDepthV2)**: patches are centred on pixels in the top 5% of RGB spatial gradient, with a random side of 4–8% of the short image side. Each patch is normalized as N_ω(D) = (D − median)/MAD, and L = Σ_ω ‖N_ω(D_ω) − N_ω(D*_ω)‖₁. Superpixels are explicitly rejected because they suppress thin structures. The total loss is L = L_λMSE (λ_z = 0.15, i.e. SIlog) + 0.1·L_con + **1.0**·L_EG-SSI + 0.1·L_unc-L1. The text calls the patched quantity inverse depth in one place and depth in another, so the domain is ambiguous. — [UniDepthV2, arXiv 2502.20110](https://arxiv.org/html/2502.20110)
- **MoGe global point loss**: L_G = Σ_{i∈M} (1/z_i)·‖s*·p̂_i + t* − p_i‖₁. (s*, t*) come from the ROE solver, which is exact for the weighted L1 objective; candidate optima lie where s·ẑ_k + t_z = z_k, and the search is O(N² log N). The residual is truncated as min(·, τ) with τ = 1. — [MoGe, arXiv 2410.19115](https://arxiv.org/html/2410.19115v3)
- **MoGe multi-scale local loss**: spheres S_j = {i : ‖p_i − p_j‖ ≤ r_j} around sampled GT anchors, with r_j = α·z_j·√(W² + H²)/(2f) and α ∈ {1/4, 1/16, 1/64}. Each sphere is aligned independently with ROE and the same (1/z)-weighted L1 is applied. The number of anchors is not stated. — [MoGe](https://arxiv.org/html/2410.19115v3)
- **MoGe normal loss**: L_N = Σ_i ∠(n̂_i, n_i), with normals from cross products of adjacent grid edges of the point map. It is shown only qualitatively, with no numeric ablation. — [MoGe](https://arxiv.org/html/2410.19115v3)
- **MoGe-2 scale loss** (for context): L_s = ‖log ŝ − stopgrad(log s*)‖²₂, with s* from ROE online. MoGe-2's local loss reuses MoGe's spheres, with no separate weight reported. — [MoGe-2, arXiv 2507.02546](https://arxiv.org/html/2507.02546)
- **Alignment method matters (MoGe Table 4, ViT-B, global loss only)**. Rel/δ1 for point-SI, point-affine, point-local and depth-affine:
  - Median affine-invariant: 10.9/89.0, 8.97/90.7, 9.44/90.7, 7.50/92.4.
  - ROE affine-invariant: 9.84/90.3, 7.88/92.1, 7.62/93.3, 6.29/93.7.
  - L2 (least-squares) affine-invariant is worst of the point variants: 13.5/84.2.
  - Depth-affine (MiDaS-style) SSI: point-SI 29.9/51.4 and disparity-affine 15.9/76.6. It wins only on its own metric (depth-affine 6.18/93.9).
  - ROE scale-invariant (no shift) sits between the two: 10.3/89.8 point-SI.

  — [MoGe](https://arxiv.org/html/2410.19115v3)
- **Depth Anything 3** (teacher loss ablation, Table 9): full loss δ1 0.919 / AbsRel 0.087, against 0.918 / 0.087 without distance normalization and 0.918 / 0.089 with MAE swapped in. Loss-variant differences are small at foundation scale. Taken from the search snippet; the full table was not read. — [Depth Anything 3, arXiv 2511.10647](https://arxiv.org/pdf/2511.10647)

### Inferences
- b20dws aligns in log space with s* = median(log D − U(r_c) − c). For an L1 loss on log residuals, the median is the exact L1-optimal shift: it is the 1-D equivalent of ROE, at O(N) cost instead of ROE's O(N² log N). So the "median vs ROE" gap in MoGe (which is about 3-D affine alignment of point maps with a 1/z weight) does **not** argue for changing the b20dws alignment.
  - One inconsistency remains. SILog's 15% mean term is L2-optimal at the *mean*, not the median, so a small residual mean-penalty survives alignment. A clean ablation is to replace SILog-on-p_al with a truncated L1 on log residuals. That would make the loss, alignment and trimming one robust family, as in MoGe's truncation (τ = 1 gave about −0.1 Rel on every metric).
- Do not add depth-space or inverse-depth affine SSI as a whole-image structure term. MoGe shows it decouples from scale-invariant 3D structure (point-SI Rel 29.9). The existing edge-guided SSI is patch-local, so its shift freedom is harmless locally. Check, however, whether distill/losses.py patches log-depth or inverse depth: UniDepthV2's own text is ambiguous.
- The UniDepthV2 teacher weights EG-SSI at 1.0 relative to SIlog. b20dws uses 0.3. The weight is a cheap one-knob ablation: try 0.3, 0.6 and 1.0.

### Gaps
- The HDN paper's final level configuration for HDN-DR is not stated in the text (it appears only in figures).
- The number of MoGe anchors per scale was not found. The numeric values of MoGe and MoGe-2 loss weights (λ) were not given in the read sections.
- The original Virtual Normal loss (Yin et al., ICCV 2019) formula and its numbers were not re-fetched. Only its definition was seen (normals of random 3-point triplets in reconstructed 3D) — [arXiv 2103.04216](https://arxiv.org/abs/2103.04216v4).

## Q2. Which structure losses give the biggest ablation gains, and which have evidence at small scale (<30M)

### Takeaway
The largest verified structure-loss gains come from locality:
- HDN-DR: −22% mean zero-shot AbsRel compared with image-level SSI. As an auxiliary to L1 in supervised NYU training: AbsRel 14.7 → 13.3, δ1 80.1 → 82.8, while plain SSI as an auxiliary gave nothing.
- EG-SSI: +4.7 zero-shot δ1 on UniDepthV2-S (34M).
- RPNL: NYU AbsRel 0.230 → 0.210.
- MoGe local spheres: point-local Rel 7.47 → 7.16.

Normal or depth-normal coupling gives small but positive gains even at ViT-S (NYU AbsRel 0.119 → 0.114). Higher-order and sharpness-specific losses repeatedly trade metric accuracy for crisper edges. Direct evidence below 10M parameters is close to absent.

### Cited Findings
- **HDN normalization ablation** (DPT-Hybrid, 16K-image subset, zero-shot AbsRel on DIODE/ETH3D/KITTI/NYU/ScanNet):

  | Method | DIODE | ETH3D | KITTI | NYU | ScanNet | Mean vs instance-level |
  |---|---|---|---|---|---|---|
  | Instance-level SSI | 47.1 | 19.4 | 13.2 | 9.6 | 10.22 | – |
  | HDN-S | 36.4 | 14.5 | 12.5 | 8.8 | 9.5 | −14% |
  | **HDN-DR** | 24.9 | 12.7 | 11.9 | 8.7 | 9.4 | **−22%** |
  | HDN-DP | 25.1 | 12.8 | 13.2 | 8.7 | 9.3 | −19% |

  Purely local contexts without the global level hurt: Local-S KITTI is +45% worse and Local-DP KITTI +76% worse. So the global level must stay in the hierarchy. — [HDN](https://ar5iv.labs.arxiv.org/html/2210.09670)
- **HDN as an auxiliary to a metric L1 loss** (fully supervised NYU):

  | Loss | AbsRel | δ1 |
  |---|---|---|
  | L1 | 14.7 | 80.1 |
  | L1 + SSI | 14.6 | 79.1 |
  | L1 + HDN | 13.3 | 82.8 |

  — [HDN](https://ar5iv.labs.arxiv.org/html/2210.09670)
- **HDN on iBims depth-boundary error** (DBE acc / comp / AbsRel): instance-level 2.55 / 5.84 / 9.81; HDN-S 2.35 / 5.15 / 8.58; HDN-DR 2.41 / 5.57 / 8.76. On the project's unseen benchmark, the spatial variant is best for boundaries. — [HDN](https://ar5iv.labs.arxiv.org/html/2210.09670)
- **RPNL** (AbsRel, KITTI / NYU): baseline 0.104 / 0.230; +SSIL 0.105 / 0.216; +RPNL 0.097 / 0.210. AbsRel is the only metric reported, measured on a small sampled training set. — [Metric3D v2](https://arxiv.org/html/2404.15506)
- **Joint depth-normal** (Metric3D v2, **ViT-Small**, Table XI): depth AbsRel DIODE-outdoor / NYU:

  | Setting | DIODE-outdoor | NYU |
  |---|---|---|
  | Without normal | 0.315 | 0.119 |
  | Without depth-normal consistency | 0.310 | 0.121 |
  | Full | 0.304 | 0.114 |

  This is the only small-backbone ablation found for normal-type structure losses. — [Metric3D v2](https://arxiv.org/html/2404.15506)
- **EG-SSI** zero-shot ablation (UniDepthV2-S, 100k steps): δ1 55.3 → 60.0, SIlog 16.2 → 15.3, F_A 56.1 → 57.9, ρ_A 78.2 → 79.8. Boundary F1 on IBims: V2-S 10.53, V2-B 12.93, V2-L 13.75. — [UniDepthV2](https://arxiv.org/html/2502.20110)
- **MoGe local and truncation** (ViT-B), Rel/δ1:

  | Variant | Point-local | Depth-affine |
  |---|---|---|
  | Full | 7.16 / 93.8 | 6.20 / 93.7 |
  | Without local loss L_S | 7.47 / 93.4 | 6.30 / 93.6 |
  | Without truncation | 7.12 / 93.8 | 6.31 / 93.5 |

  Without truncation, point-affine Rel is 7.91 against the full model's 7.83. The local loss mostly helps local geometry; global depth gains are about 0.1 Rel. — [MoGe](https://arxiv.org/html/2410.19115v3)
- **Sharpness vs metric accuracy trade-off (MoGe-2 Table 4, ViT-B)**:

  | Training data | Metric depth Rel / δ1 | Boundary F1 |
  |---|---|---|
  | Raw real-data labels | 15.8 / 75.7 | 10.3 |
  | Refined labels (mismatch filtering + log-space Poisson completion) | 16.5 / 72.8 | 12.5 |
  | Synthetic only | 21.7 / 65.0 | 13.3 |

  Sharpness gained by label processing cost 2.9 metric δ1. — [MoGe-2](https://arxiv.org/html/2507.02546)
- **Prior notes, not repeated in detail**: Depth Pro 2 found second-order or squared-gradient boundary losses cut metric δ1 (Waymo −3.3, ARKitScenes −11.2), and confidence weighting cut boundary F1 by 6.8. — see docs/B20_RESEARCH.md, [Depth Pro 2, arXiv 2611.26603](https://arxiv.org/abs/2611.26603)
- **Frequency-domain losses**: the only depth-specific FFT loss found (ApDepth) is added in a late phase of single-step diffusion fine-tuning, after spatial losses have fixed global metric accuracy. VistaDepth, FreSca and DepthMaster apply frequency modulation in diffusion latent or feature space, not as a loss. None of these is a compact feed-forward student. All of this comes from search snippets; I did not read the full text. — [search: VistaDepth arXiv 2504.15095](https://arxiv.org/html/2504.15095v1), [DepthMaster arXiv 2501.02576](https://arxiv.org/html/2501.02576v1), [FreSca arXiv 2504.02154](https://arxiv.org/pdf/2504.02154)
- **EfficientDepth (2025)**: loss = 0.4·SSI-L1 (median/MAD) + 0.2·Laplacian edge loss (RMSE of 3×3 Laplacian responses) + 0.4·LPIPS on min-max-normalized depth. Its only loss ablation reports validation "δ1" 0.426 → 0.366 with LPIPS, and the paper labels this δ1 as "lower is better". The metric definition is suspect and no parameter count was given, so treat the result as **unverified**. — [EfficientDepth, arXiv 2509.22527](https://arxiv.org/html/2509.22527)
- **Small-model evidence beyond the prior notes** is thin. ZipDepth (6.1M) uses SSI + 1:2 multi-scale gradient, and ACDepth's ordinal loss on a ResNet-18 student gives about 0.001 AbsRel. Both are already in docs/B20_RESEARCH.md. — [ZipDepth arXiv 2607.08771](https://arxiv.org/abs/2607.08771)

### Inferences
- **Expected ceiling for b20dws.** IBims scale-aligned d1 is already 91.2, and a 6M student trained on about 14M pseudo-labels saturates around aligned NYU d1 93 (ZipDepth, prior notes). So structure losses alone can plausibly add about 1–3 aligned d1 points.
  - Converted to metric d1 at fixed scale error, that is roughly +1–2 points.
  - **However**, better structure also makes s* a cleaner, less content-dependent target for the scale head (see Q3), so the indirect effect on metric scale may matter more than the direct one.
- **Best-evidenced structure additions, in order:**
  1. HDN-DR + HDN-S levels on log-depth, as a replacement or superset of the current whole-image SILog-on-p_al term. It is the only structure loss with evidence that it helps *as an auxiliary to a metric loss in supervised NYU training* (+2.7 δ1), and it improves iBims boundaries.
  2. EG-SSI weight raised toward the teacher's 1.0.
  3. A scale-invariant 3D normal term from depth plus the existing ray map (Metric3D v2 ViT-S: −4% NYU AbsRel).
- **What to avoid.** Higher-order gradient, Laplacian and LPIPS-type sharpness terms, and FFT losses, have no evidence of metric-accuracy gains for students. MoGe-2 and Depth Pro 2 both show sharpness-for-accuracy trade-offs. If any are tried, keep them small-weight and restrict them to teacher ('own') rows, where edges are exact. Kinect GT on NYU/SUN has edge noise.

### Gaps
- No paper found ablates HDN, RPNL, EG-SSI or MoGe local losses on a <10M feed-forward model. All gains above come from 22M–300M+ backbones, so transfer to 5M is unverified.
- No quantitative ablation of MoGe's normal loss, or of a pure virtual-normal loss, was found for 2023–2026 small models.
- I found no reliable evidence that frequency-domain (FFT) losses help feed-forward metric students.

## Q3. Interaction with metric losses when scale is decoupled (structure supervised after optimal alignment)

### Takeaway
Decoupling (stop-grad optimal scale target for a global head, structure losses only on the aligned output) improves *both* metric and relative geometry. MoGe-2's entangled SI-log variant is worse even on affine-invariant structure, because scale errors inject large gradients into the geometry branch. Within a decoupled design, the structure losses that pair naturally with it are exactly the scale-invariant ones: log-space aligned L1/SILog, local and hierarchical normalizations, and normals, which are invariant to global scale by construction. The open design opportunity is to make the structure loss shape the *quality of the scale target s** itself.

### Cited Findings
- **MoGe-2 Table 4** (ViT-B, 10 datasets), metric depth Rel/δ1 / affine-invariant point Rel/δ1 / boundary F1:

  | Configuration | Metric depth | Affine-inv point | F1 |
  |---|---|---|---|
  | Entangled SI-log | 17.9 / 68.6 | 10.3 / 88.8 | 10.7 |
  | Entangled shift-invariant | 16.9 / 68.8 | 9.05 / 90.2 | 11.8 |
  | Decoupled conv head | 17.7 / 68.4 | 9.15 / 90.0 | 12.7 |
  | Decoupled CLS-MLP | 16.5 / 72.8 | 8.87 / 90.6 | 12.5 |

  The scale loss is L2 in log space against stopgrad(log s*), with s* from ROE. — [MoGe-2](https://arxiv.org/html/2507.02546)
- **HDN as an auxiliary**: a whole-image SSI auxiliary on top of a metric L1 gives no gain (14.7 → 14.6 AbsRel, δ1 down 1.0), while a hierarchical auxiliary gives a large one. When a metric (or aligned-global) loss is already present, another *global* invariant loss is redundant; only *local or hierarchical* invariants add information. — [HDN](https://ar5iv.labs.arxiv.org/html/2210.09670)
- **UniDepthV2** keeps an SIlog main loss (λ_z = 0.15, so the scale term is heavily down-weighted) and adds EG-SSI at weight 1.0. Its uncertainty is a side output trained with L1 on detached log error and does not re-weight the depth loss. — [UniDepthV2](https://arxiv.org/html/2502.20110)
- **Metric3D v2** mixes metric SILog, RPNL, VNL and pair-wise normal losses with a small-weight (0.01) depth-normal consistency term. Its ViT-S ablation removing normal or consistency hurts NYU depth by 0.005–0.007 AbsRel. — [Metric3D v2](https://arxiv.org/html/2404.15506)
- **Prior notes**: Distill-Any-Depth showed that local-crop distillation must be normalized or it fights global metric scale, and PatchRefiner found SILog on pseudo-labels hurt compared with SSI/ranking. — docs/B20_RESEARCH.md

### Inferences

**Properties of the current b20dws setup**

- Because s* is a stop-grad median of the *student's own* residual, every pixel loss on p_al is exactly invariant to a log-shift in L1 and approximately so for SILog. The structure branch therefore never sees a scale gradient, which is the MoGe-2 recipe. The scale head's target, though, is only as good as the structure: if r_c has low-frequency layout errors, such as a far wall pulled closer or a tilted floor, then median(residual) shifts with image *content and crop*. The scale head is then asked to predict a target that depends on its own sister branch's errors.

**Proposed structure losses for b20dws (novelty candidates)**

All four are train-only, have zero inference cost, and are built from cited components.

1. **Region-Scale Agreement loss (RSA-loss, low-frequency structure).** Sample K regions R_k: HDN-S grid cells at 2×2 and 4×4 plus HDN-DR depth bins. Compute the local optimal log-scale m_k = median_{R_k}(log D − U(r_c) − c), and penalize L_RSA = mean_k |m_k − s*|, with s* stop-grad and smooth-L1.
   - If structure were perfect up to scale, every region would give the same scale. This term targets exactly the *between-region* error that HDN's within-region normalization ignores. It is also what makes s* crop-stable, which matters because training uses crop-context and FoV-uncrop augmentations.
   - Relation to prior work: it is the log-depth, O(N), scale-only analogue of HDN or MoGe local alignment, but it supervises the *alignment parameters' consistency* rather than the aligned residual. I found no paper that does this, but treat novelty as unverified until a targeted search is done.
   - Pair it with HDN-DR/S within-region terms: m_k-aligned L1 inside each region.
2. **Camera-aware local scale-invariant loss.** Define local regions in metric 3D using the ray map and the target depth, with radius r = α·z·diag/(2f) and α ∈ {1/4, 1/16, 1/64} as in MoGe. Align each region with its own log-median, the closed-form L1 optimum, instead of ROE.
   - Unlike pixel-grid patches (RPNL, EG-SSI), the physical neighbourhood stays consistent under FoV augmentation and across NYU/SUN/own cameras.
   - It costs roughly as much as EG-SSI. It is cheaper than MoGe because the alignment is 1-D (log-shift) rather than 3-D affine.
3. **Scale-free normal loss.** Unproject X = D·ray with both p_al and the target. Compute grid cross-product normals as in MoGe's L_N, and take the angular loss.
   - Normals are invariant to the global scale s, so this is a pure structure term that cannot leak into or fight the scale head.
   - Restrict it to teacher ('own') rows, or smooth NYU/SUN GT normals first, because Kinect edge noise is large.
   - Evidence of value at small scale: Metric3D v2 ViT-S −0.005 NYU AbsRel.
4. **Loss-family consistency.** Replace SILog(λ = .85)-on-p_al with truncated L1 on log residuals (τ chosen in log units, e.g. 0.3), so that the alignment (median), the loss (L1) and the trimming are one robust estimator. The supporting evidence is MoGe's truncation gain and the "median vs mean" mismatch noted in Q1.

**Interaction risks**

- Strong local losses at high weight can dominate a 96-channel decoder, as in HDN's purely-local rows: KITTI +45–76%. Keep the global level, i.e. the aligned whole-image L1, in every hierarchy.
- The proximity-weighted SILog should be computed on the same aligned residual, so its near-band gain does not reintroduce scale. This holds already if the weights come from target depth only.

**Suggested ablation ladder** (each step against b20dws; report IBims metric d1 / aligned d1 / bias_log, NYU-654 d1, SUN d1):

| Step | Change |
|---|---|
| A0 | b20dws baseline |
| A1 | SILog-on-p_al → truncated log-L1 (loss/alignment consistency) |
| A2 | + HDN-DR (3 levels) + HDN-S (1, 2, 4), log-depth, replacing whole-image SSI if present |
| A3 | EG-SSI weight 0.3 → 1.0, patches on RGB-gradient top 5%, 4–8% side |
| A4 | + scale-free normal loss on own rows (w 0.1–0.5) |
| A5 | + RSA-loss (w 0.1–0.5) |
| A6 | Camera-aware 3D local loss replacing A2's spatial grid |
| A7 | Gradient matching restricted to own rows only (Depth Pro 2 guidance, prior notes) |
| A8 | Negative control: + Laplacian/LPIPS or FFT sharpness term, to document the trade-off |

The expected paper story is A5/A6, a camera-aware, scale-decoupled structure objective, with A2–A4 as the evidence-backed base.

### Gaps
- No published study measures how structure-loss choices change the *variance of the per-image optimal-scale target s** or the downstream scale-head error. The RSA-loss benefit on metric scale is a hypothesis to test, not a cited result.
- MoGe-2 does not ablate its local or normal losses inside the decoupled setting, so their marginal value *with* decoupling is unknown.
- Novelty of RSA-loss and the camera-aware log-median local loss was not checked against a dedicated literature search beyond HDN, MoGe and Metric3D v2. A targeted search is recommended before claiming novelty in a paper.
