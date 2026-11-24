# UniDepth2D V2 Plan: CoDeQ

Status: plan only. Nothing has been trained. Date: 2026-11-18.
Scope: a small metric-depth model for indoor navigation by blind and low-vision users, built to survive int8 and export.

## 1. Summary

1. **What is built.** CoDeQ is an RGB-only metric-depth family on MobileNetV4-Conv encoders with a decoder closed over five operators. The primary model, CoDeQ-N, has 4.32M parameters and 4.80 GMAC at 480x640. Its int8 ONNX file is 4.85 MB.
2. **Where it comes from.** The plan starts from CoDe (data and teacher first, ranked first by two of three judges). It takes the op-closed graph, release gates and QAT design from QDepth, and the block-deletion nano, batch-12 training and CPU scale gate from SplitDepth.
3. **Budget.** Tier 1 is 22.40 GPU-hours and runs on data already on disk. Tier 2 adds 45.60 GPU-hours.
4. **What it should achieve.** My estimate for CoDeQ-N on IBims-1 is raw delta1 of about 67 (range 61 to 72), with int8 within 1 point of float. That would beat b1 (60.4 at 5.1M) at a smaller size.
5. **What it will not claim.**
   - It is not expected to reach UniDepthV2-S (82.5 measured, 86.6 published).
   - No zero-shot number can be reported until Z: is reconnected. Tier 1 produces development results only.
   - Monocular depth at this accuracy must not be the sole ranging source for a safety product (section 9.3).

## 2. Evidence checked while writing this plan

These were measured on this machine on 2026-11-18, using CPU prototypes with random weights unless stated. They support structural conclusions only, not accuracy. Scripts and JSON are in `C:\Users\aydina1\AppData\Local\Temp\claude\5e71390d066b023a\f410b8b0-855a-42ba-bf6c-2f64a738e291\scratchpad\final`. That folder is temporary, so task 12 ports the prototype into the repo.

| # | Check | Result | Consequence |
|---|---|---|---|
| E1 | Plain rebuild of timm MobileNetV4 (Conv, BatchNorm2d, ReLU, explicit Add) against timm | Max absolute feature difference 0.0 for all three variants | QAT can use the plain encoder with pretrained weights |
| E2 | Family size at 480x640 | Table in section 3.1 | Sizes are measured, not surrogates |
| E3 | Static ONNX export, opset 18 | CoDeQ-N: Conv 80, Relu 47, Add 18, Resize 5, DepthToSpace 1. No Shape, Slice, Concat or Gather | Allowlist test is achievable |
| E4 | U8S8 QDQ after ONNX Runtime 1.32 optimization | All 80 Conv become QLinearConv and all 18 Add become QLinearAdd. 5 Resize, 1 DepthToSpace and 6 Quantize/Dequantize pairs remain | Report the Resize path as not pure integer |
| E5 | ONNX Runtime 1.32 `max_intermediate_outputs=4`, MinMax, true range [-5, 5] | Returned [-1, 1]: full chunks are discarded silently | **Never set CalibMaxIntermediateOutputs** |
| E6 | Repeated `collect_data` in chunks of 3 | MinMax [-5, 5]; Percentile [-5, 4.99] | Use chunked calibration |
| E7 | `TensorQuantOverrides` with rmin -5, rmax 5 | Written range [-5.02, 4.98] | Chunked ranges can be passed to the quantizer |
| E8 | Composite label mask, 60 own images, squashed cache | Per-pixel test keeps 46.7% of strong-edge gradient pairs. Low-pass test at sigma 16 keeps 89.0% and masks 2.7% of pixels | Use the low-pass mask |
| E9 | Local weight cache | efficientvit_b1, efficientvit_b2 and ghostnet_100 present. MobileNetV4 weights absent | Small download or local pretraining (section 11) |
| E10 | Weight-only int8 by DequantizeLinear on Conv weights | CoDeQ-N 17.16 MB to 4.47 MB. It ran slower than fp32 on x86 in a noisy test | Weight-only is a size option, not a speed option |

Single-thread timings from these checks were noisy (CoDeQ-S int8 measured faster than CoDeQ-N), so none are quoted as results.

### Where reviewers disagreed

1. **Camera input for old students.** Two judges asked for K=None. The third measured that K=None skips FiLM and collapses S16 on NYU validation from delta1 .978 to .467. Decision: use a fixed default K of (.8, 1.08, .5, .5) with FiLM on, and record `camera_mode`.
2. **Calibration memory.** One judge proposed `CalibMaxIntermediateOutputs`. E5 shows that loses data silently. Decision: chunked `collect_data`.
3. **Development set.** One judge wanted SUN held out, another objected because SUN contains NYU images. Decision: SUN hold-out is the only many-scene set on disk, so it is the default. NYU-derived frames are removed by hash and disclosed. DIML/CVL test (847 MB) is the upgrade if approved.
4. **Quantization format.** U8S8 replaces S8S8, because S8S8 left 51 of 83 Conv in float on trained b1.
5. **Nano route.** Block deletion replaces Torch-Pruning, which becomes one optional Tier 2 ablation.

## 3. Model family

### 3.1 Variants (measured at 480x640)

| Model | timm encoder | Encoder M | Decoder k | Total M | GMAC | fp32 MB | Weight-int8 MB | Full int8 MB |
|---|---|---|---|---|---|---|---|---|
| CoDeQ-P | `mobilenetv4_conv_small.e2400_r224_in1k` | 1.137 | 32.1 | 1.169 | 1.48 | 4.64 | 1.26 | 1.47 |
| **CoDeQ-N** | `mobilenetv4_conv_medium.e500_r256_in1k`, 5 blocks deleted | 4.257 | 62.3 | 4.320 | 4.80 | 17.16 | 4.47 | 4.85 |
| CoDeQ-S | `mobilenetv4_conv_medium.e500_r256_in1k` | 6.955 | 62.3 | 7.018 | 5.61 | 27.89 | 7.21 | 7.72 |

### 3.2 Encoder

1. Build with `timm.create_model(name, pretrained=True, features_only=True, out_indices=(0,1,2,3,4))`.
2. CoDeQ-N only: delete indices 2, 3, 5, 8 and 9 of `blocks[3]`. All five are 256-to-256, stride 1, with residual skips.
3. All variants: drop `blocks[4]`, the 1x1 conv to 960 channels. This saves about 0.25M parameters. The stride-32 feature becomes the output of `blocks[3]`.
4. Convert to the plain encoder:
   - `BatchNormAct2d` becomes `nn.BatchNorm2d` plus the same activation (ReLU or Identity).
   - Each skip becomes an `Add` module so an observer can attach to it.
5. Feature shapes for CoDeQ-N and CoDeQ-S: f0 32@240x320, f1 48@120x160, f2 80@60x80, f3 160@30x40, f4 256@15x20. CoDeQ-P channels are 32, 32, 64, 96, 128.

### 3.3 Decoder and head

Definitions:
- `CBR(a,b,k)` = Conv(k, bias off) + BatchNorm + ReLU.
- `DS(a,b)` = `CBR(a,a,3,groups=a)` then `CBR(a,b,1)`.
- `up2` = Resize bilinear x2 with constant scales, half-pixel.
- Channels: dc=64, d2=32 for N and S; dc=48, d2=24 for P.

Graph:
1. `L1..L4 = CBR(f_i, dc, 1)`.
2. `y4 = DS(L4)`; `y3 = DS(L3 + up2(y4))`; `y2 = DS(L2 + up2(y3))`; `y1 = DS(L1 + up2(y2))`.
3. `y0 = DS(CBR(f0, d2, 1) + up2(CBR(y1, d2, 1)))`, shape d2@240x320.
4. Coarse: `c = Conv1x1(DS(y0, d2) to 1)`, bias initialized to ln(sqrt(0.2*30)) = 0.896.
5. Residual: `r = PixelShuffle(2)(Conv3x3(y0 to 4))`, weights and bias initialized to zero.
6. Output: `log_depth = up2(c) + r`, 1@480x640.

Decoder modes used by the ladder:
- `s4`: predict from y1 and upsample x4 bilinearly.
- `s2`: output `up2(c)` only.
- `ps`: the full graph above.

Contract:
- Input is a float NCHW tensor with ImageNet mean and std, matching the pretrained weights.
- No intrinsics, ray map or FiLM.
- Exp and clamp to 0.2-30 m stay outside the graph.
- Allowed ONNX operators: Conv, Relu, Add, Resize, DepthToSpace. Run T2-13 onward may add AveragePool, only inside its own config.

## 4. Losses

Notation: p = predicted log depth, y = target log depth, d = exp(y), e = p - y, M = valid mask (0.2 m <= d <= row dmax).

**Base loss, all rows, all runs from R1 onward:**

`L = 1.0*L_SI + 0.5*L_prox + 0.5*L_grad`

1. `L_SI = mean_i sqrt( max(0, sum(w e^2)/sum(w) - 0.45*(sum(w e)/sum(w))^2) + 1e-6 )`, with `w = M*(1 + exp(-d/3))`, applied once.
   - The legacy `silog_loss` multiplies e by w before squaring. S16 therefore trained with a global-scale coefficient of .52 to .64.
   - Lambda 0.45 gives 0.55 with correct weighting, so the fix does not silently weaken the scale penalty.
2. `L_prox` = existing `proximity_weighted_silog` (lambda .85, band 2 m, gain 2, soft .5).
3. `L_grad` = existing 4-scale `gradient_matching_loss` on binary M.

Sensor-GT rows and pseudo-labelled rows use the same loss.

**Legacy loss, R0 only:** `silog_loss` (w squared, lambda .85) + 0.5 prox + 0.5 grad + 0.3 eg_ssi, with the range tag.

**Added only by their own Tier 2 run:**

| Run | Term | Rows |
|---|---|---|
| T2-03 | L_grad weight 2.0 | Pseudo-labelled |
| T2-04 | Drop the 10% of pixels with largest abs(e - median(e)), detached, from all terms | Pseudo-labelled |
| T2-05 | Colour jitter .4/.4/.4/.1 at p=.8, Gaussian blur sigma .1-2 at p=.5, grayscale p=.1 | Student input, pseudo-labelled |
| T2-06 | CutMix p=.5, with L_SI computed per region at lambda 1 | Pseudo-labelled |
| T2-07 | L_SI lambda .85 | All |
| T2-08 | w = M | All |
| T2-14 | L_SI at lambda 1 on y_rel, plus Huber(s - stopgrad(median(y - y_rel)), delta .1) with weight .5 | All |

**QAT loss (only if QAT runs):** `L_QAT = L + 1.0*mean|p_q - p_f| + 10*(mean(p_q - p_f))^2`, where p_f is the frozen float twin in eval mode.

**Training settings:** AdamW lr 3e-4, wd .01, linear warmup then cosine, AMP, grad clip 1.0, EMA .999, 480x640, batch 12 without accumulation, 12,000 draws per epoch (1,000 steps). Seeds 2027, 2028, 2029. Probes run 10 epochs and finalists 40. The final-epoch EMA is always the checkpoint, so no dataset selects a checkpoint.

## 5. Data and teachers

### 5.1 Dataset roles (frozen in `configs/v2/roles.yaml` before the first run)

| Role | Data | Rule |
|---|---|---|
| Train | own (7,308 rows less group-leak exclusions), NYU (15,386 rows, keep-NYU branch only), pool X | Never a benchmark |
| Development | D_SUN: 5,285 local SUN rows less the 80 rows with 8x depth, less pHash near-duplicates of local NYU frames | All accept/reject decisions. SUN is permanently not zero-shot for V2 |
| Calibration | 300 own training rows, excluding the 397 frozen ids, with at least 30 low-light images (mean luma below 60) | Int8 only |
| Frozen own test | 397 ids | Never trained, calibrated or used for decisions |
| Report-only zero-shot | IBims-1; ETH3D indoor, DIODE Indoor val and TUM subset if approved | Scored once, after the recipe is frozen |

Statistics on D_SUN use a paired bootstrap (10,000 resamples) over near-duplicate clusters. Clusters are connected components at 64-bit pHash Hamming distance 8 or less, because the local SUN copy has no scene ids.

**NYU leak rule.** SUN RGB-D contains NYU's labelled images, including the official Eigen test frames, and those cannot be matched while Z: is offline. The NYU-free branch may be called zero-shot on NYU only after three steps: hash D_SUN against the official 654 test frames, remove matches, and confirm every ladder decision is unchanged.

### 5.2 Corpus

| Pool | Images | Source | Labels |
|---|---|---|---|
| own | 7,308 | Existing | V2-L cache in `outputs/depth_mm` |
| X-indoor | Up to 3,397 before dedupe | 1,028 extra stills, plus 1 fps from 39:29 of indoor video (supermarket, Dollar Tree, UMBC, BWI) | New V2-L |
| X-IN | 15,000 kept of up to 18,000 candidates | ImageNet-100 train, seeded random, min side at least 300 px, teacher median 0.5-15 m, p95/p5 at least 2 | New V2-L |
| NYU | 15,386 | Existing sensor GT | Sensor |

This is about 1.4x the current corpus, and Tier 2 reaches about 1.9x. Published data-scale gains start at 100k images and above, so they do not predict the effect at this scale.

Draw weights:

| Run | own + X-indoor | NYU | X-IN |
|---|---|---|---|
| R0 to R5 | .50 (own only) | .50 | 0 |
| K1 | .50 | .50 | 0 |
| K2 | .35 | .35 | .30 |
| C0 | 1.00 (own only) | 0 | 0 |
| C1 | 1.00 | 0 | 0 |
| C2 | .70 | 0 | .30 |

### 5.3 Image pipeline

- R0 to R4d use the existing squashed 640x480 cache and the s5 zoom crop.
- R5 onward use the cover cache: scale by max(640/W, 480/H), keep aspect, random-crop 480x640 for training, centre-crop for evaluation, with no zoom. RGB-only models have no camera input to compensate for zoom.
- Leak blocker: SHA-1 and 64-bit pHash (Hamming distance 4 or less) against D_SUN, the frozen test, and benchmarks when reachable.
- Group splits are by video and pre-hash stem. Any video with a frame in the frozen test is excluded.

### 5.4 Composite labels (pseudo-labelled rows only)

Both teachers run on the same original-aspect image. V2-L uses the existing longest-side 1280 setting. MoGe-2-L (`Ruicheng/moge-2-vitl-normal`) runs at resolution level 9 in env `depth_cmp`.

1. `L_U = log(V2-L)`, `L_M = log(MoGe-2-L)`. Fill invalid MoGe pixels from L_U.
2. Fit `L_M' = a*L_M + b` to L_U by robust IRLS, with a clamped to [.8, 1.25].
3. Solve in the DCT-II domain: `L_C_hat = (lam*U_hat + mu*M_hat)/(lam + mu)`, where `mu = (2 - 2cos(pi*ky/H)) + (2 - 2cos(pi*kx/W))` and `lam = (pi/48)^2`. Mean log depth therefore comes only from V2-L.
4. Mask: keep a pixel when V2-L is valid and `|G16(L_M') - G16(L_U)| <= ln(1.25)`, where G16 is a Gaussian with sigma 16 px at short side 480.
5. Save 16-bit mm PNG labels.

Teachers cannot correct a shared error on glass or mirrors.

## 6. Integrity work before any GPU run

1. Fix `boundary_f1_si` so skipped thresholds score 0, then re-score all rows.
2. Add `camera_mode` (none, default, gt) to every runner and result.
3. Add image ids to per-image rows and `--limit` to output filenames.
4. Fix the SUN decode heuristic and blacklist the 80 bad rows.
5. Add run manifests and the append-only registry.
6. Add seeding, atomic saves, resumable optimizer/scaler/EMA state, unique run folders, a GPU lock and a stop-on-failure runner.

## 7. Ablation ladder

Notation:
- **D1** = raw metric delta1 (%) on D_SUN.
- **BF** = corrected boundary F1 on D_SUN.
- **m** = max(2s, 0.5 points), where s is the sample standard deviation of D1 across R5a, R5b and R5c.
- Every run changes one factor against a named parent. `run_ladder.py` rejects a config that differs from its parent in more than one declared factor.

### 7.1 Tier 1 (22.40 GPU-hours, data on disk)

| Run | Parent | One change | GPU-h | Gate |
|---|---|---|---|---|
| T1-00 | none | Timing: 200 training steps, 1,000 images per teacher | 0.20 | Epoch at most 300 s; V2-L at least 3.5 img/s; MoGe-2-L at least 3 img/s |
| T1-01 | none | V2-L labels for X-indoor and X-IN candidates | 1.40 | At least 98% valid; at least 15,000 X-IN kept; 0 leak hits |
| R0 | none | Legacy reference: efficientvit_b1, S16 recipe, own .5 + NYU .5, batch 12 | 0.75 | D1 at least 65; no NaN |
| R1 | R0 | Corrected L_SI, lambda .45 | 0.75 | D1 at least R0 - 1.0 |
| R2 | R1 | Remove eg_ssi | 0.75 | D1 at least R1 - 0.5; BF at least R1 - .005 |
| R3 | R2 | Remove ray map and FiLM (RGB only) | 0.75 | D1 at least R2 - 1.5 |
| R3r | R3 | Hardswish to ReLU6 | 0.75 | Int8 control: report pixels within 1.25x of fp32 against R3 |
| R4 | R3 | Encoder to the CoDeQ-N plain encoder | 0.75 | **Hard gate:** PTQ keeps at least 99% of pixels within 1.25x of fp32; D1 at least R3 - 2.0 |
| R4d | R4 | Legacy FPN to the closed decoder, mode s4 | 0.75 | D1 at least R4 - 1.0; allowlist test passes |
| R5a | R4d | Cover-crop pipeline replaces squash and zoom | 0.75 | D1 at least R4d - 1.0 |
| R5b | R5a | Seed 2028 | 0.75 | Used for m |
| R5c | R5a | Seed 2029 | 0.75 | s at most 1.0, otherwise m uses the measured s |
| K1 | R5a | Add X-indoor | 0.75 | Accept if D1 at least R5 mean + m |
| K2 | best of K1, R5a | Add X-IN | 0.75 | Accept if D1 at least parent + m |
| C0 | R5a | Remove NYU | 0.75 | Report the cost; flag if above 6 points |
| C1 | C0 | Add X-indoor | 0.75 | Accept if D1 at least C0 + m |
| C2 | best of C1, C0 | Add X-IN | 0.75 | Accept if D1 at least parent + m |
| L0 | none | Score V2-L and composite labels against sensor GT on 500 D_SUN frames | 0.10 | Composite delta1 at least V2-L - 0.2; delta1_ssi at least V2-L - 0.1; BF at least V2-L + .005; edge-pair survival at least 85% |
| LBL-M | L0 | MoGe-2-L labels for the chosen training rows only | 1.80 | Runs only if L0 passes |
| L1 | P* | Composite labels on pseudo-labelled rows | 0.75 | Accept if (BF at least P* + .01 and D1 at least P* - 0.3) or D1 at least P* + m |
| E1 | W1 | Decoder mode s4 to s2 | 0.75 | Accept if BF at least W1 + .01 and D1 at least W1 - 0.3 |
| E2 | best of E1, W1 | Add PixelShuffle residual (mode ps) | 0.75 | Same gate, plus PTQ agreement at least 99% |
| F1 | W2 | CoDeQ-N, 40 epochs | 3.40 | D1 at least last probe + 1.0; int8 gates in section 8.3 |
| O1 | F1 | Output precision: uint8, int16, float tail | 0.10 | Keep uint8 unless D1 loss above 0.3 or BF loss above 5% |
| Q1 | F1 | QAT, 6 epochs, only if PTQ fails | 1.20 | Passes I1 to I6, otherwise ship weight-int8 + Float16 labelled not fully int8 |
| X1 | F1 | Evaluate artefacts and baselines; W8 and full int8 of S16 and b1 | 0.70 | All policy checks pass; every result has a registry record |

**Branch rule, fixed in advance.**
- K* is the best accepted keep-NYU run and C* is the best accepted NYU-free run.
- Choose P* = C* if D1(C*) is at least D1(K*) - 1.0 and b_std(C*) is at most b_std(K*) + .01. Otherwise P* = K*.
- If K* wins, NYU stays in-domain and C* continues as a separate clean track in Tier 2 (F5).
- W1 is the accepted result of L1 (or P*). W2 is the accepted result of E1 and E2 (or W1).

**Bridge rule.** R1 to R3 and R4d to R5 are descriptive. A missed gate is recorded and examined before F1 but does not stop the chain. Only R4 is a hard gate.

**If R4 fails:** run per-layer sensitivity, keep at most 5 layers in float, and retest. If it still fails, run Q1 on R4 before continuing.

**Cut order if time runs short:**
1. Drop R3r.
2. Drop R5c and use m = 0.5.
3. Merge K1 into K2 and C1 into C2, and state that the pool was tested as a whole.
4. Run probes at 8 epochs.

If an epoch takes above 420 s, stop and diagnose before continuing.

**Run order:** T1-00, T1-01, R0, R1, R2, R3, R3r, R4, R4d, R5a, R5b, R5c, K1, K2, C0, C1, C2, L0, LBL-M, L1, E1, E2, F1, O1, Q1, X1.

### 7.2 Tier 2 (45.60 GPU-hours)

W* is the final Tier 1 recipe. Each probe gate is D1 at least W* + m unless stated.

| Run | Parent | One change | GPU-h | Gate |
|---|---|---|---|---|
| T2-01 | none | Re-baseline all rows on IBims and official NYU/SUN; audit V2-S and MoGe-2-L | 2.00 | V2-S gap to 86.6 and MoGe-2-L gap to 83.0 explained or recorded as unresolved |
| T2-02 | none | Competitors in the same harness | 1.50 | YOLO26 adapter reproduces vendor NYU median-aligned delta1 within 1 point |
| T2-03 | W* | Gradient weight 2.0 | 0.75 | BF at least W* + .01 and D1 at least W* - 0.3 |
| T2-04 | W* | 10% trimming | 0.75 | Standard |
| T2-05 | W* | Strong photometric perturbation | 0.75 | Standard |
| T2-06 | W* | CutMix | 0.75 | Standard |
| T2-07 | W* | Lambda .85 | 0.75 | Standard |
| T2-08 | W* | Remove range weight | 0.75 | Drop the weight if D1 at least W* - 0.3 |
| T2-09 | W* | RGB-guided residual replaces PixelShuffle | 0.75 | BF at least W* + .01 and D1 at least W* - 0.3 |
| T2-10 | W* | Composite crossover 96 px | 0.75 | Standard |
| T2-11 | W* | X-IN 15,000 to 30,000, including labelling | 2.75 | Standard |
| T2-12 | W* | Coverage sampler replaces random X-IN | 0.85 | Standard |
| T2-13 | W* | Global scale head, loss unchanged | 0.75 | D1 at least W* + m and b_std down by at least .01 |
| T2-14 | T2-13 | Decoupled scale loss | 0.75 | Same gate against T2-13 |
| T2-15 | W* | Freeze encoder and decoder, refit head only | 0.30 | Same gate against W* |
| T2-16 | T2-14 | Frozen-encoder scale stage | 0.30 | Same gate against T2-14 |
| T2-17 | T2-16 | Teacher CLS distillation, including the cache | 1.30 | Runs only if leave-one-source-out ridge R^2 is at least .2 |
| T2-18 | W* | Input 384x512 | 0.60 | Report accuracy per GMAC |
| T2-19 | W* | CoDeQ-P encoder, 10 epochs | 0.50 | Recipe ranking holds |
| T2-20 | W* | CoDeQ-S encoder, 10 epochs | 0.90 | Recipe ranking holds |
| T2-21 | F3 | Torch-Pruning CoDeQ-S to 4.3M | 1.50 | Keep only if D1 at least F2 mean + m |
| F2a, F2b, F2c | W** | CoDeQ-N, 40 epochs, 3 seeds | 10.20 | Int8 IBims delta1 at least b1 + 5 points |
| F3 | W** | CoDeQ-S, 40 epochs | 4.40 | IBims delta1 at least S16 - 1 point |
| F4 | W** | CoDeQ-P, 40 epochs | 2.60 | Report the size point |
| F5 | C* | Clean-track CoDeQ-N if K* won | 3.40 | Report NYU zero-shot only under the NYU leak rule |
| Q2 | F2, F3, F4 | QAT where PTQ fails | 3.00 | I1 to I6 |
| T2-E | all | Frozen final benchmark run | 2.00 | Policy check passes on every table row |

W** is W* plus accepted Tier 2 changes. Final leave-one-out claims use Holm-Bonferroni correction.

**Not repeated (proven dead ends):** view or crop consistency for scale, intrinsics conditioning as the scale fix, in-domain per-image scale supervision, task maps, assistant-agreement weighting, bin head, normal loss, semantic size anchors, ground-lock, test-time multi-resolution fusion.

## 8. Compression and export

### 8.1 Artefact matrix

Each of S16, b1 and CoDeQ (N, then P and S in Tier 2) gets:

| Artefact | Method |
|---|---|
| fp32 ONNX | Opset 18, static 480x640 |
| Weight-only int8 (W8) | `edge/weight_int8.py`: per-channel int8 Conv weights with DequantizeLinear |
| Full int8 (PTQ) | ONNX Runtime static QDQ, U8S8, per-channel |
| QAT int8 | Only if PTQ fails |
| Core ML | Float16 plus 8-bit or 6-bit palettized weights, converted in WSL |

S16 and b1 keep their FiLM inputs with the default K as constants.

### 8.2 PTQ procedure

1. Pre-process with `quant_pre_process`.
2. Calibrate on the 300 calibration rows by calling `collect_data` repeatedly in chunks of at most 8 images. **Never set CalibMaxIntermediateOutputs.**
3. Compare MinMax and Percentile 99.99 on the own validation slice, never on a benchmark or D_SUN.
4. Pass ranges through `TensorQuantOverrides` to `quantize_static`.
5. Audit the optimized runtime graph, not the QDQ file.

### 8.3 Int8 gates (proposed; owner confirms before release)

| Gate | Requirement |
|---|---|
| I1 | At least 99% of pixels within 1.25x of fp32 |
| I2 | p99 abs(delta log) at most ln(1.10) on pixels nearer than 2 m |
| I3 | Near-field mean signed shift within +/-0.01 log units |
| I4 | D1 drop at most 1.0 point with paired 95% upper bound at most 1.5; AbsRel rise at most .005; BF drop at most 10% |
| I5 | All Conv and Add run as QLinear kernels in ONNX Runtime x86; remaining float nodes are listed |
| I6 | Safety regression gates in section 9.2 |

### 8.4 Recovery ladder

**QAT trigger:** PTQ loses more than 1.0 point of D1, or misses I1, I2 or I3.

1. Per-layer sensitivity with `qdq_loss_debug`.
2. Keep at most 5 sensitive layers in float.
3. QAT.
4. Weight-int8 with Float16 activations, labelled not fully int8.

### 8.5 QAT implementation

- Do not call `prepare_qat_fx` on stock timm MobileNetV4. It inlines BatchNorm with training=True, so 44 of 45 BN layers keep updating.
- Use the plain encoder with custom eager fake-quant in `distill/v2/qat.py`:
  1. Fold every BN into its Conv.
  2. Per-channel symmetric int8 weights.
  3. Per-tensor uint8 activations after ReLU, Add and Resize.
- Schedule: 6 epochs, lr 1e-4 to 1e-5, AMP off, observers frozen after epoch 4, observer buffers excluded from EMA.
- Export float weights and pass QAT ranges through `TensorQuantOverrides`.

### 8.6 Other formats

- **ONNX parity:** max abs(delta log) below 1e-3 against PyTorch.
- **Core ML:** WSL venv with torch==2.7.0, coremltools==9.0, numpy<2.4 and timm==1.0.27. Use `torch.jit.trace` on a wrapper that prepends ImageNet normalization, and load with `skip_model_load=True`. Label as **converted, not executed** until a Mac checks parity, Neural Engine placement and latency.
- **LiteRT (optional):** WSL, Python 3.11, litert-torch 0.10.0. Run parity on Windows with ai-edge-litert 2.3.0.
- **Pruning:** only T2-21, in a separate `depth_edge` clone, against the directly trained CoDeQ-N.

## 9. Evaluation, safety and policy

### 9.1 Evaluation protocol

**Metrics per model and set:** raw delta1, AbsRel, RMSE, delta1_ssi, delta1 after per-image scale alignment, delta1 after one per-dataset scale, corrected boundary F1, mean bias, b_std. No TTA.

**Size and speed:**
- Parameters, GMAC at equal resolution, and exact artefact size on disk.
- CPU latency on trained weights at 480x640, with pinned P-cores, 1 and 4 threads, 20 warm-up and 300 timed runs, median and p95.
- Laptop numbers are labelled as an x86 proxy.
- YOLO comparisons use MACs against MACs (Ultralytics FLOPs are 2x MACs) and int8 against int8.

**Zero-shot matrix (computed by the policy check):**

| Model | IBims-1 | ETH3D indoor | DIODE Indoor | NYU | SUN |
|---|---|---|---|---|---|
| CoDeQ keep-NYU | clean | clean | clean | in-domain | dev, never |
| CoDeQ NYU-free | clean | clean | clean | clean only under the NYU leak rule | dev, never |
| S16, b1 | clean | clean | clean | in-domain | in-domain |
| UniDepthV2, UniK3D | clean | clean | clean | clean | clean |
| MoGe-2 | lineage unverified | unverified | unverified | unverified | unverified |
| YOLO26-Depth | clean | clean | in-domain | not clean | in-domain |
| DepthART | paper-reported | unverified | unverified | fine-tuned | unverified |
| LightlyTrain dav3-metric-tiny | unverified | unverified | unverified | clean per vendor | unverified |

**Baselines:** UniDepthV2-S/B/L, UniK3D-S/B, MoGe-2-S/B/L, S16, b1, SSDI, YOLO26n/s/m/l/x-depth, Depth Anything V2 Metric Indoor Small, DepthART-S/B, LightlyTrain dav3-metric-tiny and tiny-plus, and ZipDepth (aligned columns only).

**While Z: is offline:** report D_SUN as development results only. Local NYU frames may be used only as a labelled proxy for NYU-free runs.

**Registry:** `benchmark/registry_v2.jsonl`, append-only and hash-chained. Each record holds run id, parent, factor changed, checkpoint SHA-256, code tree hash (many harness files are untracked by git), config hash, manifest and split hashes, seed, package versions, teacher ids, camera mode, gates, status and result-file hashes.

### 9.2 Safety harness (`benchmark/safety.py`)

Thresholds are provisional until the owner confirms them.

| Gate | Definition | Requirement |
|---|---|---|
| SF1 | Dangerous over-estimation: P(pred > 1.25*gt given gt < 2 m) | At most 10%; int8 rise at most 0.5 point |
| SF2 | Band miss rate at 0-1, 1-2, 2-4 m | Each no worse than R0 by more than 1 point |
| SF3 | Corridor alert: ROI x in [.25, .75]W, y in [.35, 1]H; event = GT p5 below T for T = 1 and 2 m; conformal threshold on pred p5 | Test miss rate at most 6% at a 5% target; false-alarm rate at most 30% |
| SF4 | Quantization regression after per-artefact recalibration | Miss rate change at most 1 point; false-alarm change at most 3 points |
| SF5 | Perturbations: low light, motion blur, JPEG q40, roll +/-15 degrees | D1 drop at most 8 points each |
| SF6 | Temporal: p95 frame-to-frame change in median log depth at 5 fps on local indoor video | At most .05; compare with R0 |

A false-alarm limit is required because b1 reaches a 0.9% miss rate only at 86.2% false alarms. Calibrate on one dataset and test on another once a second set is reachable. Until then the split is within D_SUN and labelled development.

**Not scored today:** glass and mirrors (excluded from IBims validity), drop-offs, curbs and thin obstacles. No safety claim may cover these until the gold capture exists.

### 9.3 Deployment statement

Monocular zero-shot accuracy near 65-73 raw delta1 is not sufficient as the sole ranging source. A separate sparse-anchor track simulates about 150 GT points per frame on CPU, with noise, and aligns by robust log-median. It is never merged into zero-shot tables.

### 9.4 Policy checks (`benchmark/policy_check.py`)

Each check fails CI and blocks table generation.

| Check | Rule |
|---|---|
| P1 | A zero-shot label requires the benchmark to be absent from training, development, selection and calibration |
| P2 | No SHA-1 or pHash overlap between training and benchmark images |
| P3 | Result `camera_mode` is none or default for zero-shot rows |
| P4 | Every table number traces to a result JSON whose hash is in the registry |
| P5 | Registry parent exists and exactly one factor changed |
| P6 | Calibration ids are all own training rows, excluding frozen ids |
| P7 | Checkpoint is the final-epoch EMA with no selection |
| P8 | Teacher lineage is recorded; unverified lineage prints a qualifier |
| P9 | Core ML rows carry the claim level; converted-only rows cannot be called validated |
| P10 | Competitor rows carry their own zero-shot status from the matrix |

### 9.5 Unit tests

| File | Covers |
|---|---|
| `tests/test_v2_model.py` | Shapes, timm parity, parameter range, strict load, K ignored |
| `tests/test_v2_losses.py` | Weight applied once, global-scale coefficient .55, legacy regression |
| `tests/test_v2_data.py` | Aspect, crop bounds, role disjointness, blacklist, leak blocker |
| `tests/test_composite.py` | Solver residual, mean-log preservation, edge survival at least 85% |
| `tests/test_protocol_metrics.py` | Boundary F1 zero-handling (updated) |
| `tests/test_camera_mode.py` | Default K keeps FiLM on, mode recorded |
| `tests/test_calib_chunked.py` | Regression for E5, range equality, source rejection |
| `tests/test_v2_export.py` | Parity, allowlist, static shapes, no NaN, runtime kernel counts |
| `tests/test_qat.py` | BN frozen, every Conv-BN pair folded, observers outside EMA |
| `tests/test_safety.py` | Synthetic over-estimation, conformal coverage, ties, ROI |
| `tests/test_policy.py`, `tests/test_registry.py` | Each rule fails on a crafted violation; tamper detection |
| `tests/test_ladder.py` | One-factor check, stop on failure, resume |

## 10. Implementation tasks

Tasks in the same work package can be written in parallel unless a dependency is listed.

| # | File | Work | Depends on |
|---|---|---|---|
| 1 | `benchmark/protocol.py` | Boundary F1 zero-handling, count of contributing images | none |
| 2 | `benchmark/gt_eval.py`, `scripts/build_gt_dataset.py` | SUN decode fix | none |
| 3 | `benchmark/provenance.py` | Hashes, environment stamp, run manifest | none |
| 4 | `benchmark/registry_v2.py` | Append-only hash-chained registry | 3 |
| 5 | `benchmark/run_bench.py` | camera_mode, ids, limit in filename, manifest, `v2:` spec | 3 |
| 6 | `edge/onnx_runner.py` | Camera-mode parity with PyTorch | 5 |
| 7 | `benchmark/policy_check.py` | P1 to P10 and the zero-shot matrix | 3, 4 |
| 8 | `benchmark/safety.py` | SF1 to SF6 and sparse-anchor simulation | 5 |
| 9 | `benchmark/latency.py` | Pinned-core latency for all runners | 5 |
| 10 | `benchmark/adapters/` | YOLO26, DAv2 Metric, DepthART, LightlyTrain, ZipDepth runners | 5 |
| 11 | `benchmark/ablation_v2.py`, `benchmark/paper_tables.py` | Tables generated from the registry | 4, 7 |
| 12 | `distill/v2/encoder.py` | Plain MobileNetV4, block deletion, timm key map | none |
| 13 | `distill/v2/model.py` | Decoder modes, family config, scale head option | 12 |
| 14 | `distill/v2/losses.py` | Corrected loss and optional terms | none |
| 15 | `distill/v2/data.py` | Roles, cover crop, draw weights, perturbation | 17 |
| 16 | `distill/v2/train.py` | Trainer, seeding, resume, atomic saves, final EMA | 13, 14, 15, 4 |
| 17 | `scripts/v2/build_roles.py`, `configs/v2/roles.yaml` | D_SUN, clusters, blacklist, groups, leak blocker | none |
| 18 | `scripts/v2/build_cover_cache.py` | Aspect-preserving cache | 17 |
| 19 | `scripts/v2/build_pool.py` | Frames, stills, ImageNet-100 candidates | 17 |
| 20 | `scripts/v2/label_unidepth.py` | V2-L labels at longest side 1280 | 19 |
| 21 | `scripts/v2/label_moge.py` | MoGe-2-L labels in depth_cmp | 19 |
| 22 | `scripts/v2/fuse_composite.py` | Solver, mask, label validation | 20, 21 |
| 23 | `scripts/v2/run_ladder.py`, `configs/v2/*.yaml` | Ladder runner, one-factor check, GPU lock, cut order | 16 |
| 24 | `edge/calib_chunked.py` | Chunked calibration with overrides | none |
| 25 | `edge/ptq.py` | U8S8 PTQ and int8 gates | 24 |
| 26 | `edge/weight_int8.py` | Weight-only int8 | none |
| 27 | `edge/sensitivity.py` | Per-layer and per-group sensitivity | 25 |
| 28 | `edge/export_v2.py` | Static export, allowlist, parity | 13 |
| 29 | `edge/verify_runtime.py` | Optimized-graph kernel audit | 25 |
| 30 | `distill/v2/qat.py` | Eager QAT | 12, 13, 14 |
| 31 | `edge/coreml_convert.py` | WSL Core ML route | 28 |
| 32 | `tests/` and `pytest.ini` | All tests in section 9.5 | matching modules |

## 11. Owner decisions

Work continues on the stated default for each.

1. **Reconnect Z: or set DATASETS_ROOT.** Default: Tier 1 runs on D_SUN and reports development results only, with no zero-shot claim.
2. **Approve two small MobileNetV4 weight downloads from the timm Hugging Face hub.** Default: pretrain the encoders on local ImageNet-100 for about 1.5 GPU-hours, drawn from the Tier 1 contingency, and flag the runs.
3. **Approve extra zero-shot sets:** ETH3D indoor (about 5.3 GB), DIODE val (2.6 GB), TUM subset (1.94 GB), DIML/CVL test (847 MB). Default: IBims-1 is the only clean set and the several-datasets claim is dropped.
4. **Approve a separate env for YOLO26-Depth** (ultralytics 8.4.104 or newer, weights about 245 MiB, AGPL-3.0). Default: no head-to-head row and no claim of beating it.
5. **Name the primary device and provide Mac access if Core ML must be validated.** Default: ONNX int8 on CPU is the validated format and Core ML stays converted, not executed.
6. **Confirm the int8 and safety thresholds before any release claim.** Default: the provisional values in sections 8.3 and 9.2 apply to research reporting only.
7. **Decide whether V2 may ship commercially.** Default: research-only, because UniDepthV2-L is CC BY-NC 4.0.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Per-scene scale still does not transfer | Track mean bias and b_std; test scale head separately; no parity claim |
| MobileNetV4 fails int8 on trained weights | Hard gate at R4, sensitivity, QAT, labelled fallback |
| Removing NYU costs too much accuracy | Pre-registered branch rule and separate clean track |
| ImageNet-100 teaches poor indoor scale | K2 and C2 test it alone; dropped unless it clears m |
| D_SUN contains NYU-derived frames | Hash removal, disclosure, NYU leak rule |
| SUN is Kinect-class GT, so BF is noisy | BF gates use a margin; final claims rest on laser-scanned sets |
| 10-epoch rankings differ from 40 epochs | Finalists and leave-one-out runs confirm at full length |
| Training runs slower than planned (S21-full ran 3x slower) | T1-00 timing and the cut order |
| Another process writes to the repo or uses the GPU | GPU lock, unique run folders, code tree hash |
| Teacher errors on glass and mirrors | Hazard stays unscored and is excluded from claims |

## 13. Prior art and claim limits

- Two-teacher distillation for tiny metric depth exists: YOLO26-Depth uses DA3MONO-LARGE for structure and DA3METRIC-LARGE for scale.
- The composite solver adapts MoGe-2's log-space Poisson refinement.
- The stride-2 skip and learned upsampling follow ZipDepth.
- The frozen-encoder scale stage follows DepthART. DepthART reports only some cells of that factorial, so T2-13 to T2-16 fill cells relevant to an RGB-only model.
- A claim of int8 accuracy across several zero-shot benchmarks at this size needs a check against the MetaDepth-CPU paper, which could not be read in full.

## 14. Expected results

These are estimates, not measurements.

| Model | IBims raw delta1 | AbsRel | delta1_ssi | Boundary F1 |
|---|---|---|---|---|
| b1 today (5.1M) | 60.4 | .210 | 91.4 | .047 |
| S16 today (15.5M) | 70.7 | .176 | 92.8 | .047 |
| CoDeQ-P (1.17M) | 60 (54-66) | .20-.23 | 90-92 | .06-.10 |
| **CoDeQ-N (4.32M)** | **67 (61-72)** | **.17-.20** | **92-94** | **.07-.12** |
| CoDeQ-S (7.02M) | 70 (65-75) | .16-.19 | 93-95 | .08-.13 |
| UniDepthV2-S (34M) | 82.5 | .142 | 97.5 | .162 |

Today's student rows used ground-truth intrinsics and the old boundary metric, so they will move when re-baselined.

---

# Binding amendments from the completeness review

These override the plan text above wherever they conflict. Verdict of the review: NOT READY TO LAUNCH. 22 gaps: 6 blockers, 10 major, 6 minor. The architecture, op-closed export path and budget arithmetic hold up (Tier 1 sums to 22.40 h, Tier 2 to 45.60 h), but the data plan fails on measured facts.

1. The ImageNet-100 pool cannot be built: every sampled local image has a 160 px short side, so the 300 px filter keeps none.
2. The video-exclusion rule removes at least 2,781 own rows. Corrected, the corpus shrinks to about 0.80x instead of growing 1.4x.
3. The D_SUN hash filter misses most NYU-derived frames: about 595 of 5,285 rows match NYU, and raw pHash at distance 4 finds 0.2% of them.
4. S16 and b1 trained on the D_SUN rows, so Tier 1 has no valid comparison against the current models.
5. With Z: still unmounted, neither the zero-shot goal nor the beat-YOLO goal has a route to a result.

Independent review is the one owner requirement missing entirely. Leave-one-out runs are cited but never scheduled.

Checked and fine: 85 non-frozen own images are low-light (mean gray below 60), so the 30-image calibration quota is feasible. MoGe-2-L weights are cached, and all 7,931 own originals are on disk.

All checks were read-only and CPU-only. Row counts for the video groups are a lower bound, because they only catch names containing a frame number. GPU speed was not measured. Evidence scripts and results are in C:\Users\aydina1\AppData\Local\Temp\claude\5e71390d066b023a\f410b8b0-855a-42ba-bf6c-2f64a738e291\scratchpad\critic (leak.py, leak.json, groups.py, in100b.py).

## A1 [blocker]

**Gap.** The X-IN pool cannot be built as specified. The local ImageNet-100 copy (clane9/imagenet-100) is already downscaled: all 6,800 images sampled across its 17 train files have a short side of exactly 160 px, so the 'min side at least 300 px' filter keeps 0 images. T1-01's gate of 15,000 kept fails, and K2, C2, T2-11 and T2-12 cannot run. Dropping the filter is not a fix, because 160 px images upscaled 3x would feed blurred inputs to both teachers.

**Required fix.** Remove X-IN from the default path and delete K2, C2, T2-11 and T2-12 from the no-download plan. This frees about 1.2 h of V2-L labelling, about 1.2 h of MoGe labelling and 1.5 h of probes in Tier 1, plus 3.6 h in Tier 2. Add an owner decision for a real replacement, with MIT Indoor67 first (15,620 images, about 2.4 GB), then the Places365 indoor subset. Re-add K2 and C2 only with that pool, and change the T1-01 gate to X-indoor only.

## A2 [blocker]

**Gap.** The rule 'any video with a frame in the frozen test is excluded' removes far more data than the plan assumes. All 15 named video groups in own have frozen frames: 2,937 rows, 156 of them frozen. That removes at least 2,781 training rows, taking own from 7,308 to about 4,610 (-37%), and they are the walking-POV footage closest to the navigation task (Walmart, MIT, Grand Central, Whole Foods, UMBC, ITE corridor). With X-IN also gone, the keep-NYU corpus is about 23,400 rows, or 0.80x today's 29,078 rather than the claimed 1.4x. The NYU-free branch has only about 8,000 rows.

**Required fix.** Invert the rule. Keep the large videos in training and remove their 156 frames from the frozen test, leaving 241 ids. Then hold out a few whole small groups as a new group-disjoint video test (for example ITE_corridor 64, MP_f1_1 41, vegas_Sphere 36, Restroom_m2 19, lab_ITE_Building 12). Freeze this in roles.yaml before R0, and restate the corpus size and growth claim in sections 1 and 5.2 from the measured row counts.

## A3 [blocker]

**Gap.** D_SUN keeps most NYU-derived frames because the planned pHash match does not survive SUN's border crop. Thumbnail correlation finds 595 of 5,285 local SUN rows (11.3%) matching a local NYU training frame at NCC 0.90 or higher. Raw pHash finds only 0.2% of those at Hamming 4 and 9.1% at Hamming 8. Keep-NYU runs are therefore scored on scenes they trained on, which inflates K* and biases the pre-registered K*-versus-C* branch rule. Own also has 18 rows at NCC 0.90 or higher against NYU and 2 against SUN, so the 'NYU-free' branch is not yet NYU-free.

**Required fix.** In build_roles.py, hash NYU after cropping to rows 45-472 and columns 40-601, which raised recall to 97.5% at Hamming 8 and 99.8% at Hamming 12. Remove the union of crop-aligned Hamming 12 or less and thumbnail NCC 0.85 or higher, then inspect a sample of the matches. Apply the same blocker to own and X-indoor. Add a regression test using a known matched pair, and report the final D_SUN size. Re-run the blocker when benchmarks become reachable, and define the action on a hit: drop those benchmark images and disclose it.

## A4 [blocker]

**Gap.** Tier 1 has no valid comparison against the current models. D_SUN is built from the same 5,285 local SUN rows that S16 and b1 used for training and validation, so X1's baseline rows for them are in-domain on nearly every image. The fallback comparison is also confounded: F1 trains 40 epochs while the retrained reference R0 trains 10. The headline 'beats b1 at smaller size' therefore cannot be supported by any Tier 1 number.

**Required fix.** Bar S16 and b1 from D_SUN accuracy tables through a policy rule. In Tier 1, report only the equal-length comparison W2 versus R0 at 10 epochs, plus clean foundation baselines. Add F0 to Tier 2: the legacy b1 recipe without SUN, 40 epochs, about 3.4 h, as the parent comparator for F1 and F2.

## A5 [blocker]

**Gap.** The zero-shot goal has no route to a result by default. Z: is still unmounted today, so no clean benchmark is on disk. The plan's fallback produces development numbers only, and if extra sets are not approved it drops the several-datasets claim altogether. The plan also overlooks that the repo can already re-fetch data locally: benchmark/fetch_tier1.py accepts --root and DATASETS_ROOT, and C: has about 201 GB free.

**Required fix.** Add a pre-flight step, before R0, with one consolidated owner approval: fetch IBims-1 with 'python -m benchmark.fetch_tier1 --only ibims --root C:/Tim/datasets', set DATASETS_ROOT, and fetch DIML/CVL test (847 MB) and the TUM subset (1.94 GB). Check teacher lineage for each set first, since MoGe-2's training list is unverified. Name IBims-1 plus TUM as the minimum set for the several-datasets claim. If approval is refused, state in the Tier 1 report that goal (c) is unmet.

## A6 [blocker]

**Gap.** The owner's first goal, beating the YOLO depth model, has no success criterion and no executable path. T2-02 only validates an adapter, and its gate needs the official NYU 654 frames, which are offline. YOLO26-Depth is clean only on IBims-1 and ETH3D, neither of which is on disk, and D_SUN is in-domain for it. The expected-results table has no YOLO26 row, and the default outcome is simply 'no claim'.

**Required fix.** Pre-register the win condition before training: CoDeQ-N int8 against YOLO26n int8, and CoDeQ-S against YOLO26s, on IBims-1 plus one more clean set. Compare raw delta1, AbsRel, delta1 after one per-dataset scale, corrected boundary F1, artefact MB, and x86 latency at equal 480x640 input. Require a paired-bootstrap win on raw delta1 and no loss on size. Run the YOLO baseline as soon as the clean set arrives, and add an alternative adapter check that needs no NYU download, such as reproducing the vendor IBims-1 figures with its TTA and log-least-squares alignment.

## A7 [major]

**Gap.** R1 and R5a each change several factors, which breaks the plan's one-change rule. Legacy training passes the range-weighted eff_mask to all three loss terms (distill/losses.py:482, 492, 507). The R1 loss applies range weighting only to L_SI and uses binary M for the gradient term, so R1 changes the squared-weight fix, lambda (.85 to .45), the prox weighting and the gradient weighting together. R5a changes both the resize method (squash to cover) and removes the s5 zoom crop.

**Required fix.** Split R1 into R1a (weight applied once with lambda .45, eff_mask kept for prox and grad) and R1b (range weight removed from prox and grad). Split R5a into R5a-cover (aspect-preserving, zoom crop kept) and R5a-nozoom. Have run_ladder.py compare resolved loss weights and pipeline parameters, not just declared factor names. Each split costs about one probe.

## A8 [major]

**Gap.** L1 bundles two changes, one of which resembles a listed dead end, and it is tested where it cannot show its benefit. The composite target also adds a teacher-agreement mask, while S8 assistant-agreement weighting was harmful (-2.8 NYU delta1 points) and the plan lists it as not repeated. L1 also runs at the s4 decoder, and history shows boundary F1 stays at .047-.0525 at stride 4 regardless of loss or capacity. Its BF gate can therefore reject composite labels that would help once the decoder can express edges.

**Required fix.** Run E1 and E2 before the label test, or run a 2x2 of labels (V2-L, composite) by decoder (s4, ps). Split L1 into L1a (composite target, no agreement mask) and L1b (add the low-pass mask), and keep the mask only if L1b clears the margin on its own. State explicitly how the mask differs from S8.

## A9 [major]

**Gap.** Report-only zero-shot sets are used as decision gates, which contradicts the plan's own 'scored once, after the recipe is frozen' rule. F2 is gated on int8 IBims delta1 against b1 and F3 on IBims delta1 against S16. If Z: returns during Tier 1, X1 could also score F1 on IBims before Tier 2 keeps changing the recipe. Policy check P1 only inspects dataset roles, so it would not catch this. These gates also cannot execute while IBims is offline.

**Required fix.** Move the F2 and F3 gates to D_SUN against F0. Restrict T2-01 to third-party baselines and legacy models, and block any CoDeQ checkpoint from a report-only set until a recipe-freeze record exists in the registry. Add a policy rule that fails when a zero-shot result predates that record. Report T2-E outcomes as final, with no retuning afterwards.

## A10 [major]

**Gap.** The owner's independent-review requirement is not covered. The plan's 'reviewers disagreed' section concerns judges of the plan itself. No step has someone other than the training pipeline verify results, and the existing gate_check.py reproduction gate is recorded as not passed.

**Required fix.** Add a required stage IR before any table is released, plus a benchmark/independent_audit.py task. A separate session with no access to training code should recompute metrics from saved predictions using separately written metric code, verify the registry hash chain, rebuild one int8 artefact from its checkpoint hash, re-run a random 10% of result rows, and trace each paper claim to a registry record. Require gate_check.py to pass or record each unresolved row as excluded.

## A11 [major]

**Gap.** Leave-one-out runs are cited but never scheduled. Section 7.2 applies Holm-Bonferroni to 'final leave-one-out claims' and the risk table relies on them to confirm rankings at full length, but neither the Tier 2 table nor ablation_runs contains one. The honest-ablation deliverable therefore has only greedy forward steps at 10 epochs.

**Required fix.** Add LOO-1 to LOO-k: remove each accepted component from W** one at a time, at 10 epochs (about 0.85 h each, at most 5 components, about 4.25 h). Confirm the largest effect at 40 epochs if budget allows. Funding comes from the removed T2-11 and T2-12 (3.6 h), and Tier 2 stays near 49.7 h after also adding F0.

## A12 [major]

**Gap.** The Tier 1 estimate is optimistic and has almost no reserve. Probes assume 270 s/epoch, but S16 and S18 measured 305 and 296 s/epoch, the project's own 5M figure implies about 360 s, and F1 itself is budgeted at 306 s. The T1-00 gate even allows 300 s. D_SUN evaluation after each run is not budgeted, and the 1.5 h pretraining fallback comes from a contingency that is only 1.6 h. At 305 s/epoch Tier 1 is about 25 h with evaluation (about 26.6 h with fallback); at 360 s it is about 28.5-30 h.

**Required fix.** Re-budget at the T1-00 measured speed plus about 3 minutes of evaluation per run, and set the T1-00 gate from the rate the budget actually assumes. After removing X-IN work and adding the R1 and L1 splits, 18 probes at 305 s total about 22.7 h, and dropping R3r brings it to about 21.8 h. Keep F0 in Tier 2. Make the cut order automatic once projected total exceeds 24 h.

## A13 [major]

**Gap.** Evaluation code for most planned sets is missing from the task list. benchmark/datasets_tier1.py:330 registers only nyu, sun, ibims and kitti_obj, and no implementation task adds D_SUN, ETH3D indoor, DIODE Indoor, TUM or DIML/CVL. Every ladder decision depends on a D_SUN loader. Centre-crop evaluation also conflicts with the harness contract, which resizes a full-frame prediction to GT, so 3:2 ETH3D images are undefined for a fixed 480x640 model.

**Required fix.** Add a benchmark/datasets_v2.py task with loaders, split hashes and tests for each set, and extend fetch_tier1.py for approved downloads. Define non-4:3 handling in the v2 runner: cover-resize, predict, and evaluate all models on the same crop mask, recording that mask in the result JSON.

## A14 [major]

**Gap.** The owner's request for 8-bit weights on the current model is left to the last Tier 1 step with no pass criterion. X1's gate only checks policy and registry records. Current full-int8 agreement is 10.2% (S16) and 21.5% (b1) of pixels within 1.25x of fp32, no recovery ladder is applied to those models, and D_SUN cannot measure their accuracy because they trained on it.

**Required fix.** Add CPU-only run X0 before R0: export S16 and b1 with default K, build weight-only int8 and U8S8 PTQ, and report paired float-versus-int8 change on the frozen own test, plus IBims when reachable. Apply gates I1 to I4 and run the sensitivity step on failures. Deliver weight-only int8 as the guaranteed artefact, and label full int8 as failed if it stays below the gates.

## A15 [major]

**Gap.** The no-download fallback for MobileNetV4 weights confounds the key R4 encoder decision. R4 would compare EfficientViT with full ImageNet-1k pretraining against an encoder pretrained for about 1.5 h on 160 px ImageNet-100. A poor R4 result could then reflect weak pretraining rather than the architecture, and R4 is the plan's only hard gate.

**Required fix.** Make the two small weight downloads a required pre-flight approval. If refused, do not run R4 as specified: pretrain both encoders under the same ImageNet-100 protocol and compare those, flagging the whole chain as non-standard pretraining.

## A16 [major]

**Gap.** The tool environment is not specified, and the main env cannot run the export work. The depth env has no onnx, onnxruntime or onnxscript; only depth_cmp has them (1.23.2, 1.32.0, 0.7.2). Export, PTQ and calibration tasks and their tests name no env, so tests/test_v2_export.py and test_calib_chunked.py will fail at import in depth. Neither env has imagehash, torch_pruning or torchao. The depth_edge clone and WSL venv installs are also missing from the owner decisions.

**Required fix.** Add an environment table mapping every task and test file to depth, depth_cmp, depth_edge or a WSL venv, with pinned versions. Use pytest.importorskip plus a CI job per env so missing dependencies cannot silently skip coverage. Implement pHash with cv2.dct to avoid a new dependency, and add the env creations to the owner decisions.

## A17 [minor]

**Gap.** Pruning, an explicit owner requirement, exists only as one optional Tier 2 run (T2-21). It depends on an unapproved env and has no Tier 1 or no-install result, so the deliverable can disappear without any decision being made.

**Required fix.** Mark T2-21 as required. Add a no-install fallback: L1-norm channel pruning written directly for the plain Conv encoder, or report the block-deletion series as a structured-pruning ablation at the same recipe.

## A18 [minor]

**Gap.** Edge deployability has no latency gate. CoDeQ-N is 4.80 GMAC, the plan flags that phone latency may disappoint, and latency is only reported. A model could pass every gate and still be too slow for navigation use.

**Required fix.** Add a provisional gate for owner confirmation: int8 median latency at 480x640 on 4 pinned P-cores no worse than b1 int8, with a stated minimum FPS. Make T2-18 (384x512) the pre-declared fallback if the gate fails.

## A19 [minor]

**Gap.** Policy rule P5 (parent exists, exactly one factor changed) will reject valid records. Root runs, third-party baselines, legacy S16 and b1, and finalists built on W** with several accepted changes do not fit it. P4 also needs registry records for all those baseline rows.

**Required fix.** Add record types to the registry (root, ablation, finalist, baseline, legacy-import). Apply the one-factor rule only to ablation records, require finalists to list their accepted component runs, and backfill S16 and b1 with hashes and a legacy provenance flag.

## A20 [minor]

**Gap.** Boundary F1 gates use a fixed +.01 margin that is not tied to measured noise. D1 gets a seed-derived margin m, but BF does not, even though SUN uses Kinect-class GT and the corrected metric will shift all BF values.

**Required fix.** Compute s_BF from R5a, R5b and R5c on the corrected metric, and use m_BF = max(2*s_BF, .01) in the L1, E1, E2, T2-03 and T2-09 gates.

## A21 [minor]

**Gap.** Safety gate SF6 (temporal stability) is measured on local indoor video, but the same supermarket, Dollar Tree, UMBC and BWI footage is added to training as X-indoor. The stability score would come from footage the model has trained on.

**Required fix.** Hold out Dollar_Tree and Airport_BWI (2:20 in total) from X-indoor and reserve them for SF6, and record that split in roles.yaml.

## A22 [minor]

**Gap.** R3r is not a valid int8 control. It swaps Hardswish for ReLU6 while warm-starting from Hardswish-pretrained weights, and the cited EfficientViT-B0 deployment reports that ReLU6 fine-tuning failed and had to be trained from scratch.

**Required fix.** Drop R3r from Tier 1, which saves about 0.85 h. If the control is still wanted, run it in Tier 2 as a from-scratch ReLU6 training and label it as such.

