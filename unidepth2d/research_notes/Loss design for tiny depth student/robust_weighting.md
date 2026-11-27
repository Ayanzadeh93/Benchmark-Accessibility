# Robust losses, pixel/sample weighting, and multi-term loss balancing for depth regression with noisy pseudo labels (b20dws student)

Scope note: docs/B20_RESEARCH.md already covers top-10% trimming (DAv2, Depth Pro 2), confidence-weighting harm (Depth Pro 2), the decoupled scale head (MoGe-2, ScaleDepth), ranking losses (PatchRefiner, ACDepth), EG-SSI (UniDepthV2), CSTM (Metric3D v2) and multi-teacher sampling (Distill Any Depth). These notes avoid repeating that material. They add robust M-estimators, set-valued (superset) losses, teacher-consistency weighting, imbalanced (long-tail) regression, multi-term balancing evidence, and range weighting, each mapped onto b20dws.

Verification legend: [V] = number read from the primary paper or its table this session. [S] = from a search snippet of the primary source only. [K] = classic from prior knowledge whose URL is known but was not re-fetched this session; treat any number tagged [K] as unverified.

Key project fact driving everything below: IBims scale-aligned d1 is 91.2 against raw d1 74.8. So about 16 d1 points are lost to per-image global scale, and only about 9 to structure. Loss work on the scale branch (L_s and its targets) has roughly twice the headroom of loss work on the pixel/structure terms.

## Q1. Robust losses and sample/pixel rejection (Barron adaptive, GNC, trimming, truncation, curriculum/self-paced, set-valued losses)

### Takeaway
Robust pixel losses give small, inconsistent gains in depth. Lienen et al.'s controlled study shows Barron's loss did not beat plain L1 on clean cross-dataset tests, while set-valued (epsilon-insensitive / "superset") losses did. Trimming, teacher-consistency soft weighting and epsilon-insensitive losses are the robust tools with depth-specific evidence. For b20dws the highest-value novel use of robustness is on the per-image SCALE target, not the pixels: use an interval loss whose width comes from teacher disagreement (V2-L/B/S caches).

### Cited Findings
- **Barron general loss.** rho(x, alpha, c) = |alpha-2|/alpha * [ ((x/c)^2/|alpha-2| + 1)^(alpha/2) - 1 ]. alpha=2 gives L2, alpha=1 Charbonnier (smooth L1), alpha=0 Cauchy, alpha=-2 Geman-McClure, alpha=-inf Welsch. [S] — [Barron, CVPR 2019, arXiv:1701.03077](https://arxiv.org/abs/1701.03077); the formula is reproduced as Eq. 3 in [Lienen et al., ACML 2021](https://proceedings.mlr.press/v157/lienen21a.html) [V].
- **Barron on depth (KITTI, unsupervised monodepth).** The adaptive alpha beat annealed and all fixed alpha. Replacing the baseline loss with the adaptive loss on wavelet coefficients cut average error by about 17%. Adaptive alpha must be trained through the NLL of the general distribution, otherwise alpha collapses to the smallest value. [S] — [Barron 2019](https://arxiv.org/pdf/1701.03077)
- **Controlled robust-loss study for supervised depth.** EfficientNet-B0 U-Net (~15M), trained on NYUv2 at 224x224, iBims-1 used as the validation set, tested zero-shot on DIODE-indoor. Hyperparameters came from a 20-trial random search, with 3 seeds. [V] — [Lienen, Nommensen, Ewerth, Hullermeier, ACML 2021](https://proceedings.mlr.press/v157/lienen21a.html)
  - DIODE AbsRel (10k NYU training images), lower is better [V]:

    | Loss | DIODE AbsRel |
    |---|---|
    | L2 | .446 |
    | Barron | .458 (worst of the robust losses) |
    | Huber | .441 |
    | Ruber | .427 |
    | BerHu | .431 |
    | L1 | .432 |
    | trimmed L1 (MiDaS 20%) | .430 |
    | weighted L2 (Kinect noise model sigma = .0012 + .0012(x-.4)^2) | .433 |
    | OSL_L2 (epsilon-insensitive L2) | .423, with the best RMS 1.757 and delta1 .339 |
    | FOSL_L1 | .413 |
    | OSL_L1 | .417 |
    | scaled SILog | .411 |

  - On the noisy in-domain NYU test, scaled SILog is best (AbsRel .196, delta1 .702). The authors note that SILog "shows less robust behavior" on clean DIODE and argue that NYU test scores reward fitting Kinect noise. [V]
- **Superset / OSL loss definition.** L*(Y, y_hat) = min_{y in Y} L(y, y_hat). With Y = [y - eps, y + eps], OSL_L1 equals the SVR epsilon-insensitive loss: zero inside the interval, linear outside. FOSL uses a trapezoidal fuzzy set (eps, delta) and gives 0 inside eps, a quadratic ramp, then linear (Eq. 8). Intervals "could in principle also be customized for each observation individually". [V] — [Lienen et al. 2021](https://proceedings.mlr.press/v157/lienen21a.html)
- **Teacher-consistency soft weighting for noisy pseudo-labels (EndoOmni, 2024).**
  - Labeled pixels use w = exp(-rho(g(x), d)^2 / tau1) with tau1 = 0.5. Here rho is the SSI discrepancy between a pretrained teacher g and the label d.
  - Pseudo-labeled pixels use v = exp(-rho(g(x), g(x_aug))^2 / tau2) with tau2 = 0.1, where the consistency is between teacher outputs on 2 augmentations (flip/rotation).
  - The loss is normalised by the sum of weights: L = (1/sum v) * sum v_i rho(f(x_i) - g_bar(x_i)). [V]
  - Their analysis: the loss on clean pixels converges fast while noisy pixels stay high, and teacher inconsistency correlates with pseudo-label noise, rising sharply above about the 80th percentile. [V]
  - Ablation (Fig. 6, relative to L_wssi = 100%): L_wssi > L_wssi-labeled-only > trimmed SSI (L_tssi) > L_ssi on SERV-CT and Hamlyn. Exact deltas are only plotted (roughly 1-10% relative on AbsRel). [V]
  - Source: [EndoOmni, arXiv:2409.05442](https://arxiv.org/abs/2409.05442)
- **Conflicting evidence on confidence weighting.** Depth Pro 2 reports that confidence-weighted losses cost boundary F1 -6.8 (already in B20_RESEARCH.md). EndoOmni's weights come from the TEACHER (external, fixed), not from the student's own uncertainty. That distinction probably matters (inference). — [EndoOmni](https://arxiv.org/abs/2409.05442) vs Depth Pro 2 (B20_RESEARCH.md)
- **Loss truncation and self-paced learning.** In a 2025 noisy-label classification comparison, fixed-percentile loss truncation collapsed to near chance on CIFAR-100N, and self-paced (keep ratio rising to 1) gave only marginal gains. Small-loss selection is "mostly studied for classification", with few regression studies. [S] — [Step-E, arXiv:2511.17040](https://arxiv.org/pdf/2511.17040); [arXiv:2609.04009](https://arxiv.org/pdf/2609.04009)
- **Classics, not re-fetched this session [K].**
  - GNC: anneal a surrogate from convex to the target robust (Geman-McClure/TLS) loss with control parameter mu; no initial guess needed. — [Yang et al., RA-L 2020, arXiv:1909.08605](https://arxiv.org/abs/1909.08605)
  - Loss truncation: drop the highest-loss fraction of examples after a hot-start period. — [Kang and Hashimoto, ACL 2020, arXiv:2004.14589](https://arxiv.org/abs/2004.14589)
  - Self-paced learning: binary per-sample weights v_i = [l_i < 1/K], with K annealed. — [Kumar et al., NeurIPS 2010](https://papers.nips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html)

### Inferences
- **Do not swap SILog for Barron/Cauchy on the pixel terms.** Lienen's controlled study puts Barron worst on clean zero-shot data, and SILog best or near-best. Pixel robustness is already handled by the top-10% trim. Expected gain from a Barron pixel swap is about 0 and risky.
- **Novel, cheap, and backed by Lienen + EndoOmni: an epsilon-insensitive SCALE loss with teacher-derived per-image intervals ("interval scale supervision").**
  - For 'own' rows, compute the per-image optimal log-scales from the three caches: s*_L, s*_B, s*_S (same median formula against the student's U(r_c) + c).
  - Set the interval I = [min(s*_k) - eps0, max(s*_k) + eps0].
  - Loss: L_s = SmoothL1(dist(s_hat, I)), i.e. zero inside the interval and Huber outside. Use eps0 of about 0.02 (about 2% depth). On GT rows (NYU/SUN), use eps0 only, or a sensor-specific eps (SUN has 4 sensors).
  - Why it should help: the student stops chasing teacher-specific scale noise that it cannot predict from 5M-param features, which reduces overfitting of the global head (the term most responsible for the IBims gap).
  - Novelty: Lienen used fixed eps tuned on validation. Per-sample intervals derived from multi-teacher disagreement on a GLOBAL scale variable appear unpublished (no source found).
  - Risk: if the interval is too wide, the head under-fits. Ablate eps0 in {0, .02, .05} and the interval width source (L/S only vs L/B/S).
- **Pixel side: EndoOmni-style weight v = exp(-(r_L - r_S)^2 / tau) on 'own' rows**, using the scale-aligned log residual between cached V2-L and V2-S, as a soft replacement for or complement to the hard top-10% trim.
  - Normalise by sum(v) so the effective loss scale is unchanged.
  - Keep the trim as a fallback, because Depth Pro 2 found confidence weighting harms boundaries. Mitigate this by applying v only to SILog and NOT to the gradient/edge terms.
- **Curriculum on the trim ratio: 0% for epochs 1-3, then ramp to 10%.** Loss-truncation and EndoOmni both show that noisy samples are identifiable only after a warm-up. The current recipe trims from epoch 1, when residuals still reflect the untrained model, not label noise. Expected effect is small (<0.5 d1) but free.

### Gaps
- No published depth-specific result for GNC or classic self-paced learning was found. Trimming and teacher-consistency weighting are the only noisy-pseudo-label depth techniques with numbers.
- The exact EndoOmni ablation numbers are only plotted. Its numeric table was not found.
- Barron's KITTI per-metric table was not opened; the ~17% figure is from the abstract snippet.

## Q2. Multi-term loss balancing (Kendall UW, GradNorm, DWA, PCGrad, Nash-MTL, FAMO) and structure-vs-scale gradient conflict

### Takeaway
On NYUv2 depth, 12 balancing/gradient-surgery methods land within about +/-5% of plain linear scalarization. Several (PCGrad, MGDA, RLW) are worse, and Kurin et al. show tuned equal weighting matches specialized MTL optimizers. b20dws already removes the main structure-vs-scale conflict at the head by stop-gradient. Learned balancing should therefore be low priority, apart from a single learned (Kendall-style) weight on the scale term vs the pixel sum, which costs nothing.

### Cited Findings
- **FAMO Table 1, NYU-v2 (3 tasks, SegNet/MTAN setup, 3 seeds).** Depth Abs Err / Rel Err [V]:

  | Method | Abs Err | Rel Err |
  |---|---|---|
  | STL | .6754 | .2780 |
  | LS (equal sum) | .5493 | .2263 |
  | SI (sum of log losses) | .5354 | .2201 |
  | RLW | .5759 | .2410 |
  | DWA | .5510 | .2285 |
  | UW (Kendall) | .5446 | .2260 |
  | MGDA | .6070 | .2555 |
  | PCGrad | .5550 | .2325 |
  | GradDrop | .5455 | .2279 |
  | CAGrad | .5486 | .2250 |
  | IMTL-G | .5426 | .2256 |
  | Nash-MTL | .5261 | .2171 |
  | FAMO | .5474 | .2194 |

  The best improvement over LS on depth is about 4% (Nash-MTL). PCGrad and MGDA are worse than LS. — [FAMO, NeurIPS 2023, arXiv:2306.03792](https://arxiv.org/abs/2306.03792)
- **SI baseline.** "Scale-invariant (SI) baseline that minimizes sum_k log L^k", which is "invariant to any scalar multiplication of task losses". It is second-best on depth after Nash-MTL and costs zero. [V] — [FAMO](https://arxiv.org/abs/2306.03792)
- **FAMO cost.** FAMO uses O(1) space/time per step (no per-task gradients) and works in log-loss space. [S] — [FAMO](https://arxiv.org/abs/2306.03792)
- **Kurin et al.** Unitary scalarization plus standard regularization "matches or improves upon" specialized MTL optimizers. Their gains "can be matched by tuning the scalarization coefficients", and many optimizers act as regularizers. [S] — [Kurin et al., NeurIPS 2022, arXiv:2201.04122](https://arxiv.org/abs/2201.04122)
- **Dense-prediction survey.** GradNorm improved over fixed weights (+1.45%) while UW reduced performance (-0.23%). Fixed uniform weights beat UW, MGDA and DWA on segmentation and depth in their NYUD setting. [S] — [Vandenhende et al., TPAMI, arXiv:2004.13379](https://arxiv.org/abs/2004.13379)
- **Gradient magnitude vs angle.** A 2023 study found "no evidence" that angular gradient conflict is unique to MTL, and emphasizes magnitude imbalance instead. [S] — [arXiv:2311.04698](https://arxiv.org/abs/2311.04698)
- **Uncertainty-weighting critique.** An IJCV 2025/26 paper identifies limitations of UW and proposes UW-SO, analytic weights with softmax normalization. [S] — [IJCV s11263-025-02625-x](https://madoc.bib.uni-mannheim.de/71719/1/s11263-025-02625-x.pdf)
- **Kendall UW formula [K].** L = sum_k (1/(2 sigma_k^2)) L_k + log sigma_k, implemented as exp(-s_k) L_k + s_k with s_k = log sigma_k^2. — [Kendall et al., CVPR 2018, arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
- **Classics, not re-fetched [K].**
  - GradNorm equalizes per-task gradient norms on the last shared layer toward rate-based targets (alpha hyperparameter). — [Chen et al., ICML 2018, arXiv:1711.02257](https://arxiv.org/abs/1711.02257)
  - DWA sets w_k(t) proportional to exp(r_k(t-1)/T), with r_k = L_k(t-1)/L_k(t-2). — [Liu et al., MTAN, CVPR 2019, arXiv:1803.10704](https://arxiv.org/abs/1803.10704)
  - PCGrad projects a conflicting task gradient onto the normal plane of the other. — [Yu et al., NeurIPS 2020, arXiv:2001.06782](https://arxiv.org/abs/2001.06782)
- **beta-NLL [K].** Heteroscedastic Gaussian NLL under-fits high-variance samples. beta-NLL multiplies by stopgrad(sigma^(2 beta)), with beta = 0.5 recommended. — [Seitzer et al., ICLR 2022, arXiv:2203.09168](https://arxiv.org/abs/2203.09168)

### Inferences
- **Where conflict actually exists in b20dws.**
  - The scale head reads GAP of the last two encoder stages, so L_s gradients enter the shared encoder alongside the four pixel terms. The pixel losses are scale-free (s* is stop-gradient), so they push the encoder toward scale-INVARIANT features, while L_s needs scale-DISCRIMINATIVE features. That is a genuine representational tension, and it is plausibly why NYU d1 dropped (86.7 to 85.4) when the scale head was added.
  - A cheap diagnostic: log cos(g_pixel, g_scale) on the encoder's last-stage params every N steps. Mostly negative values would justify PCGrad-on-encoder-only (two backward passes, train-time only). Values near 0 mean magnitude, not angle, is the issue, so tune the L_s weight.
- **Recommended balancing ablation (ordered by evidence and cost).**
  - (a) Sweep the L_s weight {0.5, 1, 2, 4}. This is the most likely win (Kurin).
  - (b) Use SI log-sum for the two groups: log(L_pix) + log(L_s). This is zero-cost and invariant to SmoothL1-beta magnitude (FAMO table).
  - (c) Use a Kendall learned weight on 2 groups only. Never apply it per pixel term.
  - (d) Use FAMO if (a-c) disagree.
  - Skip GradNorm, Nash-MTL and PCGrad on all terms: they need per-term gradients, and the gains are at most 4% on NYU depth.
- **A heteroscedastic scale head (predict s_hat and log b, Laplace NLL on the scale) could make the scale loss robust to label-scale noise per image.** Use beta-NLL-style stopgrad weighting to avoid the under-fitting pathology. The predicted b also gives a free per-image scale-confidence output for OCMR. This competes with the interval loss in Q1; ablate one vs the other.

### Gaps
- No paper was found that measures structure-vs-scale gradient conflict in a decoupled-scale metric depth model. The PCGrad-on-encoder idea is untested.
- The GradNorm and DWA numbers for depth in a single-task, multi-loss setting (as opposed to multi-task) were not found. All balancing evidence is from seg/depth/normal MTL.

## Q3. Range-aware / near-field / distance-binned / long-tail depth weighting

### Takeaway
Two families have numbers: (i) imbalanced-regression reweighting by depth-bin frequency (LDS, Balanced MSE, ConR on NYUD2-DIR), which helps rare depth ranges but costs the frequent range; and (ii) distance-dependent per-pixel weights (inverse-depth losses in Depth Pro, VistaDepth's sigmoid distance weight). For b20dws the most promising novel step is to apply imbalance-aware weighting to the per-image SCALE distribution (the long tail of scene scales), not to pixels. The project already has a near-field pixel weight.

### Cited Findings
- **DIR / LDS on NYUD2-DIR (ResNet-50 encoder-decoder).** Depth 0.7-10 m in 0.1 m bins; many/medium/few-shot by training pixels per bin. RMSE and delta1, All / Many / Med / Few [V]:

  | Method | RMSE All | RMSE Many | RMSE Med | RMSE Few | delta1 All | delta1 Many | delta1 Med | delta1 Few |
  |---|---|---|---|---|---|---|---|---|
  | Vanilla | 1.477 | .591 | .952 | 2.123 | .677 | .777 | .693 | .570 |
  | +LDS | 1.387 | .671 | .913 | 1.954 | .672 | .701 | .706 | .630 |
  | +FDS | 1.442 | .615 | .940 | 2.059 | .681 | .760 | .695 | .596 |
  | +LDS+FDS | 1.338 | .670 | .851 | 1.880 | .705 | .730 | .764 | .655 |

  - LDS: p_tilde(y') = integral k(y, y') p(y) dy, Gaussian kernel size 5, sigma 2, with weights proportional to 1/p_tilde. Many-shot delta1 DROPS by 4.7-7.6 points. [V] — [Yang et al., ICML 2021, arXiv:2102.09554](https://arxiv.org/abs/2102.09554); code: [github.com/YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression)
- **Balanced MSE** (CVPR 2022) reports SOTA on NYUD2-DIR using a 16-component GMM for the label prior. Exact numbers were not extracted. A later paper warns of "large variability across different runs ... especially on the NYUD2-DIR dataset" and averages 3 seeds. [S] — [Ren et al., arXiv:2203.16427](https://arxiv.org/abs/2203.16427); [arXiv:2308.10603](https://arxiv.org/pdf/2308.10603)
- **ConR** (contrastive regularizer for imbalanced regression) reports NYUD2-DIR RMSE 1.265 overall and 1.667 few-shot. [S] — [ConR, ICLR 2024, arXiv:2309.06651](https://arxiv.org/abs/2309.06651)
- **Depth Pro trains on canonical INVERSE depth** because it "prioritizes areas close to the camera over farther areas". Its metric loss is MAE with top-20% per-image trimming on real data, MAD-normalized for non-metric data. [V] — [Depth Pro, arXiv:2410.02073](https://arxiv.org/abs/2410.02073)
- **VistaDepth BiasMap.**
  - Distance weight: w_dist = sigmoid((z_mean - mu)/(sigma + eps)), with per-image mu and sigma. It is multiplied by a normalized Sobel edge weight restricted above the 70th depth percentile. [V]
  - Ablation (diffusion model, NYUv2 AbsRel/delta1): none 6.7/94.8; distance only 5.7/95.9; edge only 5.5/96.3; both 5.4/96.8. KITTI: 11.1/89.1 to 9.9/91.4. [V]
  - Their analysis says uniform loss over-emphasizes near range. This is the opposite emphasis to Depth Pro.
  - Source: [VistaDepth, arXiv:2504.15095](https://arxiv.org/abs/2504.15095)
- **Weighted L2 with a Kinect noise model** (sigma(x) = .0012 + .0012(x - .4)^2, i.e. down-weighting far pixels) gave DIODE AbsRel .433, no better than L1 .432. [V] — [Lienen et al. 2021](https://proceedings.mlr.press/v157/lienen21a.html)

### Inferences
- **Near vs far weighting cuts both ways.** Depth Pro (near-priority) and VistaDepth (far-priority) both report gains, and fixed sensor-noise weighting gave none (Lienen). For an indoor robot/OCMR target, keep the existing near band (<2 m, gain 2) but ABLATE it. With scale fixed by s*, the near-weight only reshapes structure, and on NYU-dominated data it may simply re-weight the many-shot range. Expected effect is small.
- **Novel, cheap and arguably the best paper angle: "scale-imbalanced scale learning".**
  - The scale head's target s* (per-image log-scale) has a long-tailed distribution. The NYU/SUN/own mix is concentrated in small rooms, while IBims has corridors and large halls.
  - Apply LDS (Gaussian-smoothed inverse or inverse-sqrt frequency over s* bins, about 0.05 log units wide) or Balanced-MSE to L_s, at the IMAGE level.
  - DIR shows the expected pattern: few-shot gains (+6-8 delta1) at a many-shot cost. To protect NYU, use inverse-sqrt weights clipped to [0.5, 3], or sampler-level rebalancing instead of loss weights.
  - Image-level DIR applied to a decoupled metric-scale head appears unpublished (no source found).
  - Measure IBims raw d1 and the |bias_log| distribution per scene type.
- **Pixel-level DIR on log-depth bins is less attractive.** Pixel losses are already scale-free (structure only), and the many-shot cost would hit NYU in-domain, which the user does not want to lose.
- **A distance-binned ERROR monitor (not a loss)** is needed to decide all of the above. Log per-bin (0-1, 1-2, 2-4, 4-8, >8 m) AbsRel on IBims/NYU for every ablation, mirroring VistaDepth Table II.

### Gaps
- No depth paper was found that reweights the per-image global scale target by its frequency. The idea is supported only by analogy to DIR.
- The NYUD2-DIR numbers for Balanced MSE and FDS-only were not verified beyond the DIR table.
- VistaDepth's gains are on a diffusion model with latent-space loss; transfer to a 5M CNN regressor is unknown.
- No safety-weighted (obstacle/near-field risk) depth loss with numbers was found. The search surfaced only detection-side depth weighting (arXiv:2602.05730, not depth estimation).

## Q4. Synthesis: proposed loss and ablation plan for b20dws (inferences grounded in the above)

### Takeaway
Keep the structure losses roughly as they are, and put the novelty and the effort into the scale term. The proposed name is a "teacher-interval, imbalance-aware scale loss":

L = L_pix(structure; trimmed with warm-up, optional teacher-consistency weight) + lambda_s * w_img(s*) * Huber(dist(s_hat, I_teacher))

Here I_teacher = [min_k s*_k - eps0, max_k s*_k + eps0] over the V2-L/B/S caches (eps0 only on GT rows), and w_img is an LDS inverse-sqrt weight over s* bins. All components are train-time only, so inference is unchanged.

### Cited Findings
- epsilon-insensitive / superset losses gave the best clean cross-dataset (iBims-val to DIODE) results among 12 losses — [Lienen et al. 2021](https://proceedings.mlr.press/v157/lienen21a.html) [V]
- Teacher-consistency soft weights beat trimming and plain SSI for noisy pseudo-labels — [EndoOmni](https://arxiv.org/abs/2409.05442) [V]
- LDS improves few-shot depth ranges at a many-shot cost — [DIR](https://arxiv.org/abs/2102.09554) [V]
- Balancing methods move NYU depth by at most about 4% vs equal weights; SI log-sum is zero-cost and second-best — [FAMO](https://arxiv.org/abs/2306.03792) [V]

### Inferences
Ablation table, run on 12k samples x 40 epochs with the same seed. Report IBims raw d1/AbsRel, scale-aligned d1, |bias_log| median, NYU d1, SUN d1 and OCMR objAbsRel, using 2 seeds for the final rows, since DIR-style results are seed-sensitive:

| ID | Change | Expected effect / risk |
|---|---|---|
| A0 | b20dws as-is | baseline |
| A1 | L_s weight 0.5/2/4 | cheapest; scale is the dominant error source |
| A2 | trim warm-up (0% for 3 epochs, then 10%) | small, safe |
| A3 | interval scale loss, eps0 .02, from L/B/S | main novelty #1; risk of under-fitting if intervals are wide |
| A4 | LDS inverse-sqrt img-weights on L_s | main novelty #2; watch NYU d1 |
| A5 | A3 + A4 | proposed |
| A6 | teacher-consistency pixel weight v = exp(-(r_L - r_S)^2/tau), tau in {.01, .05}, on SILog only | may help the own rows; may hurt edges (Depth Pro 2) |
| A7 | SI log-sum balance log L_pix + log L_s | zero-cost alternative to A1 |
| A8 | Laplace beta-NLL scale head (instead of A3) | alternative robust scale; also yields scale confidence |
| A9 | remove the proximity SILog / swap it to an inverse-depth L1 term | range-weighting check |
| A10 | encoder-only PCGrad between pixel and scale groups (only if cos<0 is measured) | diagnostic-driven |
| A11 | Barron adaptive alpha on SILog residuals | expected null (Lienen); include as a negative control for the paper |

### Gaps
- None of the proposed scale-side losses (A3, A4, A8) has a published depth result. Their expected gains are inferred. The reviewer-facing novelty claim rests on this gap, which was checked only by the searches above, not by an exhaustive literature review.
- The teacher-disagreement interval requires that V2-B/S caches exist for all 'own' images and that the GT rows have no teacher maps. If V2 caches exist for NYU/SUN as well, the intervals could be teacher-vs-GT, which is untested.
