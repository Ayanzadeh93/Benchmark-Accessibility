# Codebase + experiment-history audit: b20dws loss path

Read-only audit, 2026-11-25. Files read: `distill/losses.py`, `distill/train.py`, `distill/student.py`,
`distill/dataset.py`, `config.yaml`, `scripts/run_b20dws.ps1`, `outputs/checkpoints/b20dws/train_summary.json`,
the b20dws section of `outputs/b20_queue.log`, `results_bench/*b10*|*b20*|*S16*` JSON, and docs
EXPERIMENTS / RESULTS_MASTER / RESULTS_DIGEST / PAPER_TABLES / SCALE_GAP / B20_SPEC / B20_RESULTS / B20_RESEARCH,
plus `outputs/checkpoints/ablation_table.md`. No repo code was edited.

---

## 0. Corrections to the task brief (what the code actually runs)

1. **The b20dws recipe also includes `range`** (`train_summary.json`: `s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt`).
   `range` sets `range_weight_alpha=1.0`, so `weight = 1 + exp(-d/3)` per pixel (dataset.py:418-419). Near pixels get
   about 2x weight, 3 m about 1.37x, 10 m about 1.04x. That weight multiplies `valid` into `eff_mask`
   (losses.py:580-582), which feeds SILog, prox-SILog, grad and eg_ssi. Prox-SILog then adds up to another 3x
   (`1+gain*sigmoid((2-d)/0.5)`) on top, so near pixels carry about 6x the weight of far ones in prox-SILog.
2. **Teacher caches.** I found only **V2-L** depth and conf for own rows (`outputs/v2/cache/own/{depth_mm,conf}`,
   7,931 rows) and the older stretched copy in `outputs/data_gt/own`. `data_gt/*/assist_mm` holds the **b2_mix student**
   (S8 "assistant"), not V2-B/S (scripts/cache_assistant.py:1-6). B20_SPEC section 6 item 12 also says
   "only V2-L depth PNGs are cached". V2-B/S exist only as benchmark runs. Any multi-teacher loss therefore needs a new
   teacher pass first.
3. `outputs/data_gt/own_consensus_scale.json` (N=5-view V2-L consensus, delta mean -0.021, std 0.208, sigma median
   0.030) is keyed to the **stretched data_gt/own** pseudo-labels. b20dws uses the **v2cache** (aspect-true, a
   different teacher pass), so `csd`/`rel` cannot be applied to b20dws rows without rebuilding the consensus on
   the v2cache.
4. `use_teacher_confidence: false` and the v2cache conf maps are unused (`conf_mode` is "none", so `weight` = range only).
5. `select_metric` resolves to `gt` (selgt), meaning mean(nyu, sun) val AbsRel. Those val rows are a random 3% of
   NYU/SUN **train** frames (dataset.py:154-166), so they share scenes with training. Best epoch = 39 = last.

---

## 1. The b20dws training step as executed (train.py:376-462)

```
parts = model(img, K, return_parts=True)          # r_up = U(r - mean_hw(r)), s_hat, c = log(f_px/518.9)
s*, ok = dsh_scale_target(r_up, c, logD, valid)    # fp32 nanmedian over valid (pre-trim, pad excluded), no grad
p_al  = r_up + (c + s*)                            # cast to fp16 under AMP
valid_loss = trim_mask_own(p_al, logD, valid, is_gt, 0.10)   # own rows, epoch>=1
L_pix = 1.0*SILog(p_al; eff_mask=valid_loss*range_w, lam .85)
      + 0.5*proxSILog(...) + 0.5*GradMatch(4 scales) + 0.3*EG-SSI(pad-aware)
L_s   = SmoothL1_{beta .05}(s_hat - s*) over rows with >=500 valid px, weight 1.0
loss  = (L_pix + L_s)/3 ; GradScaler ; clip 1.0 ; AdamW ; cosine ; EMA .999
```

Epoch-39 running averages (b20_queue.log):

| term | raw value | x weight | share of total 0.307 |
|---|---|---|---|
| silog | 0.115 | 0.115 | 37% |
| prox_silog | 0.096 | 0.048 | 16% |
| grad | 0.011 | **0.0055** | **2%** |
| eg_ssi | 0.282 | 0.085 | 28% |
| scale_head (L_s) | 0.054 | 0.054 | 18% |
| s_hat_err = mean abs(s_hat - s*) | **0.074** | | (in-sample, train mode) |
| trim_frac | 0.087 | | (target was 0.10) |
| pad_frac | 0.097 | | about 10% of all pixels are synthetic pad |

Validation (EMA) at epoch 39: nyu 0.073, sun 0.152, **own 0.182** AbsRel (own = against V2-L pseudo-labels).

---

## 2. Key measured fact the brief misses: dsh made STRUCTURE slightly worse everywhere

From `results_bench/*.json` (`depth` block; d1_si = d1 after a per-image median SCALAR; medianlog = 100·mean|median log-ratio|):

| set | model | d1 | d1_si | d1_ssi | silog | arel_si | per-image scale err (medianlog) |
|---|---|---|---|---|---|---|---|
| IBims | b10 | 66.8 | **87.3** | 91.4 | **13.97** | .111 | 17.4 |
| IBims | b20core | 74.3 | 86.6 | 91.4 | 14.44 | .116 | 12.8 |
| IBims | b20dws | 74.8 | 86.5 | 91.2 | 14.37 | .116 | **12.7** |
| IBims | S16 (B2) | 70.7 | 88.8 | 92.8 | 12.64 | .101 | 15.1 |
| NYU | b10 | 86.7 | **90.6** | 93.0 | **12.12** | .093 | 7.14 |
| NYU | b20dws | 85.4 | 89.6 | 92.8 | 12.70 | .098 | 7.06 |
| SUN | b10 | 83.5 | **91.4** | 93.6 | **12.01** | .092 | 8.99 |
| SUN | b20dws | 83.0 | 90.5 | 93.1 | 12.61 | .097 | 8.80 |

Reading:
- The **IBims gain is entirely scale**: per-image scale error drops 17.4 -> 12.7 (about -27%).
- **In-domain scale is unchanged** (NYU 7.14 -> 7.06, SUN 8.99 -> 8.80). So the NYU -1.3 d1 / SUN -0.4 regression is
  **not** a scale problem. It is a **structure** regression: d1_si drops 0.8-1.0 points and silog rises 0.4-0.6 on
  **all three** sets, IBims included.
- b20core shows the same pattern, so **dws4 is not the cause**. The cause lies in the bundle
  {dsh, cstm, fov, ownar, trim}. The leave-one-out ladder in B20_SPEC section 4 was **never run**.
- Remaining headroom on IBims: perfect scale -> d1 about 86.5 (+11.7). Perfect scale and structure at V2-S level ->
  d1_ssi 97.5 (vs 91.2). Scale is still the bigger lever, but a structure-repair loss could also buy about 1 d1 in-domain.
- The scale head **underfits in-sample**: train s_hat_err 0.074 (about 7.7% per-image scale error on training
  images) vs 0.127 on IBims. Its target is noisy on own rows (see 3.5) and its input is narrow (see 3.7).

---

## 3. Correctness issues and weaknesses (ranked)

### 3.1 [BUG, confirmed by measurement] EG-SSI edge weight is dominated by GT-hole borders
`eg_ssi_loss` (losses.py:111-143) runs Sobel on `target_log`. Invalid pixels carry `log_depth = 0`
(dataset.py:530-531), so every hole border becomes a step of |log d| in the edge map. Only the synthetic pad is
suppressed (losses.py:135-137). The weight `w = mask*edge_w` then keeps the hole-adjacent **valid** pixels at the
maximum weight, which are the least reliable GT pixels. `emax` is set by hole borders, so real depth edges are
compressed toward `edge_floor`.
Measured on 40 random training images per source: invalid fraction NYU 4.1% mean, SUN 20.6% mean (median 14%).
emax including hole borders is 8.9 (NYU) and 9.8 (SUN), against an interior-only emax of 3.9 and 4.5. Hole borders
dominate emax by more than 1.5x in **85% of NYU and 82% of SUN images**, and 60% of training samples are GT rows.
Fix (zero cost, train-only): `edge *= 1 - maxpool3x3(~valid)` before normalising (the same trick as `pad`), or Sobel
on a hole-inpainted target. Better still, follow UniDepthV2 and use RGB-gradient patches. EG-SSI is 28% of the
total loss, so this is a real lever on structure.

### 3.2 [WEAKNESS] Under dsh the SILog lambda is only a skew penalty, and it is mis-weighted
With p_al median-aligned and s* detached, every pixel loss is shift-invariant except the `-lam*mu^2` part of
SILog and prox-SILog. Gradient matching is shift-invariant, and EG-SSI is shift- and spread-invariant. So
`SILog = sqrt(Var_w(e) + (1-lam)*mu_w^2)` with `mu_w` = the **range-weighted** mean residual, while s* is the
**unweighted median**. Because r_c is mean-centred, a global shift cannot reduce `mu_w`. The only way to reduce it
is to **tilt** near against far log-depth so the near-weighted mean residual matches the median. That is a
spurious structure force, and plausibly one source of the d1_si/silog regression in section 2. Untested.
Ablation: `silog_lambda = 1.0` under dsh, or compute s* with the same weights as the estimator (weighted median),
or drop the range weight from the variance term.

### 3.3 [WEAKNESS] Nothing in L_pix supervises log-depth SPREAD except SILog
EG-SSI normalises both maps by median and mean-abs-deviation **in log space** (losses.py:79-91), so it cannot see
a compressed log range (`log D' = a*log D`, with a < 1). The measured student failure is exactly that compression
(EXPERIMENTS 7.1: median(pred/GT) is 1.32 at under 1 m and 0.91 at 5 m and beyond for B1). Gradient matching is
weak (2% of the total). Only SILog and prox-SILog penalise spread. A spread term is a natural loss to add:
per-image log-MAD/std ratio, or an affine-in-log slope fit, gated to rows with exact labels. Without dsh, scale
and spread were entangled in one SILog. With dsh, scale has its own head and spread does not.

### 3.4 [WEAKNESS] Mean-centring gauge includes unsupervised pad pixels
`r_c = r - mean_hw(r)` (student.py:295) averages over the whole stride-4 grid, pad included. Pad pixels are never
supervised: `valid &= pad==0` and `s*` is computed on valid pixels. With pad fraction a (0 to 64% on fov rows,
mean 10%), `s* = median(logD - r) + mean_all(r)` contains `a*(mean_pad(r) - mean_valid(r))`. That is an arbitrary
quantity the scale head must predict from GAP features that also see the noise canvas. This adds target noise
and a train/test mismatch for exactly the scalar that drives IBims. Fix: centre over non-pad pixels during
training (pass the pad mask to `_dsh_output`), or add a weak loss tying r on the pad to its valid mean. At
inference (no pad) the two are identical.

### 3.5 [WEAKNESS] Scale target is noisy and source-mixed
- Own rows (40% of samples): s* is the V2-L pseudo-label scale. The family-wide over-range is +0.021 to +0.026 log
  (RESULTS_DIGEST C2). The consensus file shows a cached-pass vs multi-view std of 0.208 log on the stretched copy.
  None of this is weighted or denoised for b20dws (no `dshrel`, no csd on the v2cache).
- `L_s` is uniform over GT and own rows (spec C1). SmoothL1 with beta 0.05 is close to L1, so the loss is robust,
  but a 7.7% in-sample error says the head cannot fit these targets.
- **BatchNorm with batch 4** (hypothesis, not tested): s_hat is a regression from GAP of BN features. In train
  mode the batch mean of each BN output channel is pinned to beta, so GAP features partly carry only *relative*
  information between the 4 images. At eval, running stats replace that. This couples s_hat to the other images in
  the batch, including mixed sources, and creates a train/eval gap for a global scalar. Diagnostic: s_hat_err on
  training images in eval mode vs train mode. Mitigations: freeze encoder BN stats after warm-up for the scale
  path, or read the scale head from features after a non-BN op.
- s_hat_err is never logged per source, so the GT-vs-own split of the scale error is unknown.

### 3.6 [OPEN QUESTION] L_s gradient reaches the trunk (`scale_head_detach` False)
g = [GAP F4, GAP F3, GAP x_td, e(k)] is not detached (student.py:300-303), so L_s (18% of the loss, L1-like,
per-image gradient of constant magnitude) shapes encoder stages 3-4 and the fused decoder feature in a 5M net.
That is MoGe-2's design (ViT-B, 100M). At 5M it may compete with structure capacity, which is consistent with the
d1_si drop in section 2. **The `dsh_detach` tag exists and was never run**: it is a cheap, highly informative
ablation. A middle ground is a gradient-scale layer (for example 0.1x into the trunk).

### 3.7 [WEAKNESS] Scale head inputs are narrow
GAP throws away layout. Indoor scale cues are room-layout and vanishing geometry, floor/ceiling extent, and known
object sizes. The head gets no statistics of the predicted structure (r_c percentiles or spread). k_embed is
16-d ReLU(4 numbers); with cstm, c already absorbs focal. Hidden width 128, untested. No auxiliary target
regularises the scale latent (ScaleDepth uses scene-class CE; SUN has scene labels).

### 3.8 [MINOR] Normalisation inconsistencies across terms
SILog and prox-SILog are per-image (sqrt, then batch mean). Gradient matching and EG-SSI pool over the batch
(sum num / sum den), so SUN rows with 20% holes get less weight, and large-valid own rows get more. Gradient
matching downsamples by `::2` striding, which aliases. Gradient matching is effectively **off** (0.0055 of 0.307).
The Depth Pro 2 / DAv2 advice (gradient term on precise pseudo-labels only, at weight 1-2x SSI) was never tried.

### 3.9 [MINOR] Trim is computed after s*, the realignment is a no-op, and the rate is 8.7% not 10%
Trim sits after the s* computation (by design), so s* still uses the trimmed-out outliers. The median is robust,
so this matters little. `median_align` inside trim is a no-op under dsh. The rate is 8.7% because the quantile is
taken on the `::2` grid and the count includes GT rows in `n_v` (only own pixels are trimmed). Harmless. Trim's
effect on structure (it removes the hardest 10% of own pixels, often true edges, glass and far range) is unablated.

### 3.10 [MINOR] fp16 notes
`p_al` is cast to fp16 (train.py:384-385). Log-depth up to 3.4 has a step of about 0.002 (0.2%) in fp16, well
below the errors. The autocast reductions (sum) run in fp32, and `dsh_scale_target` and `scale_head_loss` are
fp32. No fp16 bug found. The b20 rep4 divergence of the scale-head loss after about epoch 10 is documented but
undiagnosed (rep4 BN branches). dws4 is stable.

### 3.11 [MINOR] Augmentation is weak for a pseudo-label student
Photometric aug is only gain ±15% and bias ±10 (dataset.py:521-524), with no colour jitter, blur or noise. DAv1
shows naive pseudo-labelling is flat without strong student-side perturbation. `aug2` was deferred. CutMix was
rejected, correctly, because it corrupts s*.

---

## 4. Inventory of everything already tried (to avoid repeats)

Protocols differ: rows marked (old) use the legacy gt_eval protocol (pred-clip, `ablation_table.md`), and the
rest use the v2 harness. Do not compare across them.

### 4.1 B1 single-tag ablations on S2-mix (old protocol; NYU AbsRel/d1, SUN AbsRel/d1)
| tag | what | NYU | SUN | verdict |
|---|---|---|---|---|
| S2-mix base | own+NYU+SUN, FiLM, EMA | .118/86.6 | .119/86.3 | reference |
| s4 (S4-edge) | EG-SSI 0.3 | .118/86.5 | .116/87.0 | ~0 |
| s5 (S5-crop) | crop p .5, K-updated | .117/86.6 | .116/87.3 | ~0 (prox1m F1 .235 -> .146) |
| s6 (S6-unc) | beta-NLL + rank conf | .121/85.9 | .120/86.3 | negative (predates weighting fix) |
| s7 (S7-task) | task-weight maps + obj-dist 0.3 | .125/84.9 | .126/84.3 | negative |
| s8 (S8-assist) | b2_mix agreement weighting | .131/83.7 | .135/83.4 | clearly negative |
| ray (S11) | 6-ch ray input | .117/86.8 | .120/86.4 | ~0 |
| bins (S12) | 64 log-bins + CE (arch change) | .118/86.9 | .120/86.4 | ~0 |
| norm (S13) | normal consistency 0.3 | .122/85.6 | .119/86.6 | slightly negative |
| range (S14) | 1+exp(-d/3) weight | .117/86.4 | .118/86.6 | ~0 |
| S9-ultra | s4,s5,ray,range | .122/84.8 | .119/85.9 | negative on B1 |

### 4.2 B2 (15.5M), v2 harness (NYU d1 / SUN d1 / IBims d1)
| run | recipe | NYU | SUN | IBims | note |
|---|---|---|---|---|---|
| S10-ultra-L | s4,s5,ray,range | 87.8 | 85.5 | 65.6 | |
| S15-anch-icd | + anch (N4 affine head) + icd (flip consistency 0.3) | 87.5 | 85.3 | 65.6 | null |
| **S16-prox** | + prox-SILog 0.5 | **88.4** | **86.0** | **70.7** | +5.1 IBims, best B2 |
| S18-zsclean | S10 without SUN | 87.5 | 81.7 | 67.7 | SUN is partly in-domain |
| S19-sced | S16 warm +15 ep, scale + sced | 88.5 | 85.9 | 68.9 | IBims -1.8 |
| S21-scale2 | S16 warm +15 ep, scale-bias on all rows | 88.8 | 86.1 | 68.9 | IBims -1.8 |
| S21-scale2-full | scratch | - | - | - | died at epoch 5, unusable |
| SSDI | S21 + V2-L r0 scale at test | 90.1 | 89.8 | 82.75 | test-time donor, not a loss |

### 4.3 B1 b-series, v2 harness (IBims d1 / NYU d1 / SUN d1 / OCMR objAbsRel)
| run | recipe | IBims | NYU | SUN | OCMR |
|---|---|---|---|---|---|
| b1_mix | plain | 60.4 | 86.1 | 81.9 | .173 |
| b10 | s4,s5,ray,range,prox | 66.8 | **86.7** | **83.5** | **.155** |
| b20core | + dsh,cstm,fov,ownar,trim,selgt | 74.3 | 85.6 | 82.7 | .162 |
| **b20dws** | b20core + dws4 | **74.8** | 85.4 | 83.0 | .160 |
| b20 | b20core + rep4 | 62.8 | 69.3 | 68.3 | .219 (L_s diverged at about epoch 10) |

### 4.4 Implemented or planned but never trained or measured
`dsh_detach`, `dshft`, `dshrel` (named only, not implemented), the b20 leave-one-out ladder (minus dsh, minus
cstm+fov, minus ownar, minus trim), P-series P0-P5 (`ssionly`, `csd`, `rel`), `anchloss` (deferred: own objects
are 99.5% class-less), `aug2`, the RGB-edge EG-SSI variant, gradient matching on own rows only, and
silog_lambda / L_s weight / beta sweeps.

### 4.5 Training-free (inference-side) results, for context
Post-hoc affine log recalibration helps students' prox<1m F1 (S10 .506 -> .658). Ground-lock helps V2 but hurts
students. Size-prior SMA is negative. Resolution fusion is null.

### 4.6 Lessons that constrain new loss proposals
1. **In-domain scale losses do not transfer OOD** (S19/S21: NYU up, IBims -1.8). Any scale loss must be judged on
   IBims, not NYU-val.
2. **A conv-map affine head (anch) is null, and a GAP-MLP decoupled head (dsh) is +8 IBims.** Scale needs a global
   pathway.
3. **Confidence or uncertainty weighting of the depth loss hurts** (S6; Depth Pro 2 agrees). Reweighting the
   *scale* loss by teacher reliability is untested in the dsh setting.
4. **Prox-SILog was the only aux loss with a large OOD gain** (S16, +5.1 IBims). That was in an *entangled* model,
   where it shaped scale. Under dsh it only shapes structure.
5. **Multi-teacher or assistant agreement weighting is harmful** (S8). Averaging teachers is harmful in the
   literature (Distill Any Depth).
6. Flip consistency (icd) is null because the flip leaves K unchanged. sced (crop virtual camera) is null to
   negative as a pixel loss. Neither has been tried as a **scale-only** consistency on s_hat.

---

## 5. Concrete hooks for new loss terms

| Where | Location | What you can do there |
|---|---|---|
| Tag parsing | `train.py:176-250` (add `elif t == "...":` before `else` at :249) | new recipe tag -> `dcfg["loss"][key]` |
| Compatibility | `train.py:72` `DSH_INCOMPATIBLE`, `:75-90` `check_recipe_compat` | sced/icd are blocked under dsh because they call `model()`. A scale-only consistency must use `return_parts=True` |
| Loss config | `losses.py:533-569` `DistillLoss.__init__` (keys under `distill.loss.*`) | new weights and hyperparameters |
| Pixel losses | `losses.py:571-647` `DistillLoss.forward`; add before `terms["total"]` at :646 | receives `pred_log` (= p_al under dsh), `target_log`, `mask` (trimmed bool), `weight`, `is_gt`, `intrinsics`, `pad`, `scale_sigma` |
| Scale target | `losses.py:446-470` `dsh_scale_target` | weighted median, robust or consensus target, pad-excluded gauge, per-source targets |
| Scale loss | `losses.py:473-479` `scale_head_loss`; call site `train.py:406-413` | NLL with predicted sigma_s, reliability weights, ranking or pairwise scale loss across the batch, per-source logging |
| Trim | `losses.py:493-530`; call site `train.py:391-394` | progressive or adaptive trim, GT-row trimming, trim before s* |
| EG-SSI | `losses.py:111-143` | hole-mask fix, RGB-edge patches, spread-aware normaliser |
| SILog | `losses.py:32-48`, prox `:271-300` | lambda under dsh, weighted-median consistency |
| Gradient matching | `losses.py:51-76` | own-row gating, avg-pool pyramid, weight |
| Model parts | `student.py:291-318` `_dsh_output` (g at :300-301, s_hat at :304, parts dict at :316) | expose g, r_c stats, predicted sigma_s, or a second scale head (train-only); pad-aware centring at :295 |
| Second view | `train.py:426-448` (sced block, the template for a crop virtual-camera view with K update) | two-view s_hat equivariance; must call `model(..., return_parts=True)` |
| Per-pixel weight | `dataset.py:407-429` | compose new weights (range, own-row conf) |
| Batch fields | `dataset.py:537-558` out dict | add `source` id (already a string), teacher conf, a second crop's params, scene label |
| Aug | `dataset.py:462-497` (s5/fov), `:521-524` (photometric) | strong student-only photometric aug |
| Logging | `train.py:474-480`; W&B `train/*` at :499 | log s_hat_err per source (GT vs own), per-term values |
| Tests | `tests/test_dsh.py`, `test_trim.py`, `test_fov_ownar.py`, `test_b10_parity.py` | gradient-isolation tests (dL_pix/d scale_mlp = 0) must stay green |

Config keys that exist today (distill.loss): `silog_weight, silog_lambda, grad_weight, grad_scales, ssi_weight,
eg_ssi_weight, nll_weight, nll_beta, obj_dist_weight, bins_weight, normal_weight, prox_silog_weight, prox_band_m,
prox_gain, anchor_weight, scale_bias_weight, scale_bias_all, scale_rel, scale_rel_tau, scale_head_weight,
scale_head_beta, trim_frac_own, trim_start_epoch`. Other keys live under distill: `range_weight_alpha, crop_context,
fov_uncrop, own_source, icd_weight, sced_weight, sced_p, select_metric, conf_mode`.

---

## 6. Untested knobs, cheapest first (screen with 15-epoch runs, gate on IBims paired bootstrap plus NYU d1_si)

1. EG-SSI hole-border fix (bug, 3.1).
2. `dsh_detach` (exists).
3. `silog_lambda = 1.0` under dsh (3.2).
4. Pad-excluded mean-centring (3.4).
5. Gradient-matching weight and gating (0.5 -> 1-2, own rows only).
6. L_s weight {0.3, 1, 3} and beta {0.02, 0.05, L2}.
7. Leave-one-out of {trim, fov, ownar, cstm} to find the structure regression in section 2.
8. Per-source s_hat_err logging, plus the train-vs-eval-mode s_hat check (BN hypothesis, 3.5).
9. Spread loss (3.3).
10. Scale-only two-view equivariance on s_hat (4.6, item 6).
11. Rebuild the consensus on the v2cache, then reliability-weight L_s on own rows (`dshrel`).
12. Strong student-only photometric aug (`aug2`).

Dev-set hygiene: own-val (0.182, never trained on, OOD-like) is a usable dev signal. NYU/SUN val share scenes with
training. IBims (n=100) is the test set: if it is used for screening, say so, as B20_SPEC section 4 already notes.
