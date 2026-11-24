# Final specification: b10 and b20 compact metric-depth students

**Scope.** Repo `C:\Tim\nature\unidepth2d`, stack `distill/train.py`. The distill/v2 CoDeQ stack is not used (see section 6).

**What I checked in code** (read-only, nothing edited, no GPU used):
- `student.py`: FiLM sits at :101-108 and :146-149. `head` at :70-73 is `_ConvBNAct(96,96,3)` followed by a 1x1. `forward` does not clamp; `infer_metres` clamps.
- `train.py`: recipe tags are parsed at :127-175. Selection is `val_per[select_metric]` at :360-361, and `last.pt`/`best.pt` are saved at :373-378.
- `dataset.py`: invalid pixels get `log_depth = 0` with `valid = False` (:351-365). `is_gt` is already in the batch (:368).
- `run_bench`: names its output after the checkpoint's parent directory and reads `cfg["student"]["train_size"]` (run_bench.py:94-99).

## 0. Vote tally and decisions

| Component | Judges keeping it | Decision |
|---|---|---|
| Decoupled global scale head, GAP+MLP, stop-grad median target (`dsh`) | 3/3 | **KEEP** |
| Camera embedding e(K) as scale-head input | 3/3 | KEEP (part of dsh) |
| Attention pooling in the scale head | 0/3 | reject |
| Gauge term, and mean-based a* with gradient through r | 0/3 | reject (mean-centring r_c replaces it) |
| Reliability weighting of the own-row scale loss | 1 for, 1 against | not in b20; optional tag `dshrel` |
| CSTM canonical-focal term (`cstm`) | 3/3 | **KEEP** (only together with fov) |
| Aspect-true own rows, crop form (`ownar`) | 3/3 | **KEEP** |
| Letterbox form of ar | 0/3 | reject |
| Zoom-out uncrop added on top of the unchanged s5 (`fov`) | 3/3 | **KEEP** |
| Symmetric fov that replaces s5 | 0/3 | reject |
| Top-10% residual trim on own rows only (`trim`) | 2/3 | **KEEP** (first thing to cut if short on time) |
| Trim on GT rows | 0/3 | reject |
| MobileOne re-param of the two stride-4 convs (`rep4`) | 3/3 | **KEEP** (final rung, ablated with b20core) |
| Lite decoder / mres / tak / nofilm | 0/3 each | reject (section 6) |
| Fold FiLM, ray map and cstm into constants at export for a fixed camera | 2/3 (3rd says optional) | KEEP (deployment only, applied to b10 and b20) |
| `dshft` fallback (retrain the scale head only) | 2/3 | KEEP as a fallback only |
| Keep the scale head in fp16 under W8 | 2/3 | KEEP |
| Select checkpoints on own-val | dropped | b20 selects on GT-val (`selgt`); compare last.pt against last.pt |

**What is new in b20.** b20 is "CanonScale-Rep". It combines three things on a 5M CNN student:
1. A camera-canonical (CSTM) depth space.
2. A MoGe-2-style decoupled global scale head with a stop-grad target, trained on aspect-true, focal-diverse views whose labels are exact.
3. A re-parameterised stride-4 decoder that cuts about 41% of MACs.

None of the cited prior work combines these. The target is the measured failure: student IBims error is about 85% global scale (d1_ssi .93 vs raw .66-.71).

---

## 1. b10: exact recipe (S16 recipe moved to B1)

**Status.** b10 is already running (`outputs/checkpoints/b10`, started 18:24, about 4 min per epoch, about 2.7 h in total). Do **not** start any other GPU job until it finishes; the `*_TAINTED_concurrent_run` folders show why.

```powershell
cd C:\Tim\nature\unidepth2d; $env:PYTHONPATH="C:\Tim\nature\unidepth2d"
conda run -n depth --no-capture-output python -m distill.train --config config.yaml --data gt --use-intrinsics `
  --backbone efficientvit_b1.r224_in1k --recipe s4,s5,ray,range,prox --ckpt-subdir b10 --name b10
```

**Model**
- timm `efficientvit_b1.r224_in1k`, features_only. Channels 32/64/128/256 at strides 4/8/16/32.
- 6-channel stem (RGB plus unit rays).
- FiLM on (fx/W, fy/H, cx/W, cy/H), applied to x_td.
- 96-channel FPN decoder. Output is log-depth, clamped to [0.2, 30] m at inference.
- 5.119 M parameters and 6.84 GMAC at 480x640. Of that, the encoder is 3.02 GMAC and the decoder 3.82. smooths[0] and head[0] are 1.593 GMAC each.

**Losses**
- SILog (λ 0.85, weight 1.0)
- Gradient matching (0.5, 4 scales)
- s4 EG-SSI (0.3)
- prox SILog (0.5, 2 m band, gain 2)
- range α 1.0
- s5 crop (p 0.5, min_scale 0.5)

**Data and optimiser**
- Mix own .4 / nyu .4 / sun .2. 12k samples per epoch, 40 epochs.
- AdamW 3e-4, wd 0.01, cosine schedule, EMA 0.999, AMP, batch 4 × accum 3, input 480x640.

**Checkpoint selection.** Keep whatever the running config uses (reportedly `select_metric: own`). Report both `best.pt` and `last.pt` (final EMA). The primary b10-vs-b20 comparison is last.pt against last.pt, because it involves no selection.

**Why this recipe**
1. S16 is the best measured student: IBims 65.6 → 70.7 on B2, OCMR .146, bias_log about 0.
2. The B1 single-tag ablations measured in-domain own-val only. That metric cannot see camera or scale generalisation: UniDepth reports OOD gains of 10-20 δ1 against in-domain gains under 1.5.
3. Nothing untested is added: no s6, s7, s8, norm, bins, anch, icd, sced or scale2.

**Expected b10** (estimate): IBims d1 63-67 (b1_mix 60.4), d1_ssi about .92, NYU about 86-87, SUN about 84-86, OCMR .155-.165, about 13.9-14 ms.

**Contingency.** If b10's IBims d1 comes out below b1_mix (60.4), do not rerun b10. b20's `cstm` removes the s5 label inconsistency that is the likely cause. Note it in the report.

---

## 2. b20 components

**b20 command** (after b10 finishes and the smoke test passes):
```powershell
conda run -n depth --no-capture-output python -m distill.train --config config.yaml --data gt --use-intrinsics `
  --backbone efficientvit_b1.r224_in1k `
  --recipe s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,rep4,selgt --ckpt-subdir b20 --name b20
```

**Control run b20core:** the same recipe without `rep4`, with `--ckpt-subdir b20core --name b20core`. It isolates rep4's accuracy cost.

**Notation.** Network input is H=480, W=640. Normalised intrinsics at the network input (after every crop, uncrop and resize) are k = (f̂x, f̂y, ĉx, ĉy). Encoder features are F1..F4 (strides 4-32). x_td is the post-FiLM fused decoder feature (96 × 120 × 160). log D is the target log-depth, and Ω is the set of valid pixels (pads and holes are invalid). U is the bilinear ×4 upsample.

### C1 `dsh`: decoupled global scale head

**Algorithm**
- r = head(x_td) ∈ R^{1×120×160}, taken before the existing upsample at student.py:165.
- Mean-centre it: r_c = r − mean_{hw}(r). This removes the gauge freedom exactly, so no gauge loss is needed.
- Camera embedding: e(k) = ReLU(W_k·[log f̂x, log f̂y, ĉx−½, ĉy−½] + b_k), with W_k ∈ R^{16×4}.
- Global token: g = [GAP(F4) (256); GAP(F3) (128); GAP(x_td) (96); e(k) (16)], 496 dimensions.
- Scale: ŝ = w₂ᵀ·ReLU(W₁g + b₁) + b₂, with W₁ ∈ R^{128×496}.
- Initialisation: w₂ = 0 and b₂ = log√(0.2·30) = 0.896, the same as the current head bias. At init the output is a constant map.
- Inference: log D̂ = U(r_c) + ŝ + c(k), where c comes from C2 (c = 0 if cstm is off). The clamp to [log .2, log 30] is applied in `infer_metres` and in the export, as today.

**Training** (fp32, per image)
- s* = stopgrad( median_{p∈Ω} [log D(p) − U(r_c)(p) − c] ). Use `torch.nanmedian` on (B, HW) with invalid pixels set to NaN. If an image has fewer than 500 valid pixels, give it scale-loss weight 0.
- Aligned prediction: p_al = U(r_c) + c + s*. All existing pixel losses get p_al instead of the raw output. Because s* is detached, a scale error never sends gradient into the structure output.
- Scale loss: L_s = 1.0 · mean_b SmoothL1_{β=0.05}(ŝ_b − s*_b), applied uniformly to all rows (GT and own).
- Total: L = L_pix(p_al) + L_s.
- L_s gradient may flow into the encoder and decoder through the GAP inputs, as in MoGe-2. Optional ablation tag `dsh_detach` detaches g.

**Hyperparameters:** `scale_head_weight` 1.0, `scale_head_beta` 0.05, hidden width 128, embedding width 16. Same optimiser; the head is included in the EMA.

**Recipe tag `dsh`:**
- sets `cfg.student.scale_head=True` and `dcfg.loss.scale_head_weight=1.0`;
- asserts that none of `bins`, `anch`, `scale`, `scale2` or `ssionly` is present (scale_bias must be off).

**Files and functions**
- `distill/student.py` (about 90 LOC)
  - `StudentConfig` (:21-34): add `scale_head: bool=False`, `cstm_fc: float=0.0`, `rep4: bool=False`.
  - `__init__`: add `self.k_embed` and `self.scale_mlp`.
  - `forward(x, intrinsics, return_parts=False)`: keep `feats`. After `out = self.head(x_td)` (:164), when `scale_head` is on, centre r at stride 4, compute ŝ and c, and return `U(r_c)+ŝ+c`.
  - With `return_parts=True`, return `{'r_up': U(r_c), 's_hat': ŝ (B,1), 'c': c (B,1)}`.
  - The default return stays a tensor, so run_bench, fps_bench, validate and export_onnx are unchanged.
  - `build_student` reads the new keys with defaults off, so old checkpoints still load.
- `distill/losses.py` (about 40 LOC):
  - `dsh_scale_target(r_up, c, target_log, valid) -> (s_star, ok)`, under no_grad, in fp32.
  - `scale_head_loss(s_hat, s_star, ok, beta)`.
- `distill/train.py` (about 35 LOC):
  - Tag parsing goes at :127-175.
  - At :280, when dsh is on, call `model(img, intr, return_parts=True)`, build `p_al`, and pass `p_al` as `pred_log` to `criterion` (:288).
  - Add `w·L_s` to `loss` before the `/accum`, and log `s_hat_err = mean|ŝ−s*|`.

**Unit test** (`tests/test_dsh.py`):
1. At init the output is constant (0.896 + c).
2. On a synthetic map where target = r_c + 0.7 + noise, s* recovers 0.7 within 1e-3.
3. ∂L_s/∂head[-1].weight = 0 exactly, and ∂L_pix/∂scale_mlp.* = 0 exactly.
4. Loading an old checkpoint (no dsh keys) gives an output identical to the current code.

**Expected gain:** IBims raw d1 +1 to +4 over the same model without dsh. The judges cut the design's +4 to +10 because MoGe-2's evidence is a ViT-B CLS token. bias_log should move toward 0, OCMR objAbsRel by −0.005 to −0.015, and NYU/SUN by ±1.

**Gate:** paired-bootstrap IBims raw d1 ≥ b10 + 2 with |bias_log| no worse; otherwise run `dshft`.

**Citations:** MoGe-2 Table 4 (decoupled MLP scale 72.8 vs entangled 68.6; a conv-head scale gives 68.4, which matches this repo's null `anch` result); ScaleDepth; "What the Averages Hide". DepthART must **not** be cited as evidence for a scale head alone: its .713 includes the adapters, and its scale-head-only result (.534) is worse than full fine-tuning (.572).

### C2 `cstm`: canonical-focal depth (CSTM-label)

**Algorithm**
- f_px = √((f̂x·W_in)(f̂y·H_in)) is the focal length in pixels at the network input.
- c = log(f_px / f_c), with f_c = 518.9 px (NYU Kinect at 640x480, so c ≈ 3e-4 on NYU).
- The network predicts canonical log-depth, and c restores metric depth exactly. Under a zoom by factor k, the content magnifies by k, f_px is multiplied by k and c shifts by log k. The canonical target is therefore invariant to the crop.
- With dsh, c enters only the s* target and the output. Without dsh, the output is `out + c`.
- Bonus: inference at a lower resolution (section 5) automatically stays label-consistent, because f_px scales with the input.
- No parameters.

**Tag `cstm`:** sets `cfg.student.cstm_fc=518.9`.

**Files and functions**
- student.py: helper `_cstm_term(intrinsics, H, W)` (about 10 LOC).
- train.py: the tag (about 3 LOC).
- export_onnx: c exports as Log/Add on the (1,4) intrinsics input (no change beyond a parity check).

**Unit test** (`tests/test_cstm.py`):
- Zoom by k = 1.5 with the K update, and check c increases by exactly log 1.5.
- NYU K gives |c| < 1e-3.

**Expected gain:** about 0 on its own. Together with fov and ownar, +1 to +4 IBims d1 jointly (judge range).

**Citation:** Metric3D/v2 Table 9 (NYU AbsRel: CamConvs .333 vs CSTM-label .212).

### C3a `ownar`: aspect-true own rows

**Algorithm.** For own rows, read the image, depth and conf from `outputs/v2/cache/own` (7,861 rows, aspect kept, 480 high for landscape). Use its `manifest.csv` (orig/cover sizes and scale) in place of the anisotropically stretched `data_gt/own`.

- **Intrinsics.**
  1. Convert the `intrinsics.json` K of the original image to pixel units (multiply by orig w/h if it is stored normalised).
  2. Multiply by the manifest `scale` to get K_px for the cache image (H_c, W_c).
  3. Assert that cx_px / W_c is in [0.35, 0.65] for 99% of rows.
- **Crop or pad each axis independently to 480x640.**
  - Larger than the target: random crop when training (offset x0 or y0; cx −= x0, cy −= y0). Centre crop for val rows.
  - Smaller than the target (portrait or narrow): paste at a random offset (centred for val), with cx += ox. Fill the pad with noise, image ~ N(127, 40²) clipped to uint8. Depth = 0 there (so invalid), weight = 0, `pad = 1`.
- **No letterbox and no resize.** Resolution is preserved.
- Then the normal pipeline runs: s5/fov, flip, photometric.
- Frozen own test stems and the 3% val split stay keyed by id. The 70 rows missing from the cache fall back to the stretched copy.
- Own conf and boxes are not used in b20, since s6 and s7 are off.

**Tag `ownar`:** sets `dcfg.own_source="v2cache"` and `dcfg.v2cache_manifest="outputs/v2/cache/own/manifest.csv"`.

**Files and functions:** `distill/dataset.py`, about 90 LOC.
- `MultiSourceDataset.__init__` (:123-170): load the cache manifest map id → (img, depth, conf, scale).
- `__getitem__` (:236-275): an `own_source=='v2cache'` branch plus `_crop_or_pad_to(H, W, img, depth, weight, K, rng, train)`.
- Add a `"pad"` (1,H,W) uint8 tensor to the out dict (:359-378); it is zero everywhere for other rows.

**Unit test** (`tests/test_fov_ownar.py::test_ownar_backproject`): the 3D point back-projected from a cache pixel with K_px lands at the same 3D location after crop or pad with the updated K, to within 1e-4. The pad has `valid == False`.

**Expected gain:** part of the joint +1 to +4 IBims. It fixes the measured defect: about 94% of own rows (5,793 wider than 4:3 and 1,595 narrower) currently train on stretched geometry.

**Citations:** Depth Pro 2 (aspect/FoV diversity is needed for metric scale; this citation could not be verified by the judges); DMD.

### C3b `fov`: zoom-out uncrop added on top of the unchanged s5

**Algorithm.** Draw one u ~ U(0,1) per sample.
- u < 0.5: the existing s5 crop, exactly as now. Zoom factor k_in = 1/U(0.5, 0.9) ∈ [1.11, 2].
- 0.5 ≤ u < 0.75: uncrop.
  1. Draw z ~ U(0.6, 0.95). Set (h', w') = (round(zH), round(zW)).
  2. Resize the image with INTER_AREA, and depth and weight with NEAREST.
  3. Draw offsets ox ~ U{0..W−w'} and oy ~ U{0..H−h'}, and paste into a canvas of N(127, 40²) noise. Depth = 0, weight = 0 and `pad = 1` outside the pasted image.
  4. Update K in pixels: fx' = fx·w'/W, fy' = fy·h'/H, cx' = cx·w'/W + ox, cy' = cy·h'/H + oy. Boxes are remapped.
- Otherwise: no geometric change.
- Applies to every source. Depth values are never changed, so the metric labels stay exact. The effective focal range becomes ×0.6 to ×2.
- The flip happens afterwards (cx → 1−cx, as now).
- Fix in `eg_ssi_loss` (:111): when `pad` is given, multiply `edge` by (1 − maxpool3x3(pad)) **before** max-normalisation. Otherwise the 0-filled target at the pad border creates spurious edges that dominate `emax`. Without pads the function behaves exactly as now, so b10 parity is kept.

**Tag `fov`:** sets `dcfg.fov_uncrop={p:0.25, zmin:0.6, zmax:0.95}`, and asserts `cstm` and `ray` are present.

**Files and functions**
- dataset.py: a `_fov_uncrop()` next to the S5 block (:296-325), sharing that block's K and box update. Restructure the s5 draw to use the shared u (about 55 LOC).
- losses.py: `eg_ssi_loss(..., pad=None)` (about 5 LOC). `DistillLoss.forward` passes `pad` through.

**Unit tests:**
- `test_uncrop_backproject`: on a synthetic constant-depth plane, the back-projected 3D extent of the valid region is unchanged within 1%.
- `test_s5_rate_unchanged`: over 10k draws the s5 rate is 0.50 ± 0.01.
- `test_egssi_pad`: with pad=None the output is identical to the old function.

**Expected gain:** jointly with C2 and C3a, +1 to +4 IBims d1 and +0 to +1 SUN-raw; NYU −0.5 to +0.5.

**Citation:** DMD Table 7 (iBims AbsRel .180 → .130, NYU .074 → .076).

### C4 `trim`: scale-aligned top-10% trimming, own rows only

**Algorithm**
- Starting from the 2nd epoch, for rows with is_gt = 0:
  - ρ = |p_al − log D| on Ω, detached.
  - τ = the 0.90 quantile of ρ, via `torch.kthvalue` on ρ[..., ::2, ::2].
  - valid ← valid · 1[ρ ≤ τ].
- Trimming is applied to `valid` **before** the criterion call, so silog, grad, eg_ssi and prox, which all consume `eff_mask = valid·weight`, see the trimmed mask.
- s* is computed **before** trimming.
- GT rows are never trimmed.
- Without dsh, p_al = pred − median(pred − target).

**Tag `trim`:** sets `dcfg.loss.trim_frac_own=0.10` and `trim_start_epoch=1`. The epoch is passed to the step.

**Files and functions:** losses.py `trim_mask_own(p_al, target_log, valid, is_gt, frac)` (about 25 LOC); train.py call before :288 (about 6 LOC).

**Unit test** (`tests/test_trim.py`):
- 10% ± 0.5% of the valid own pixels are removed and GT rows are untouched.
- Adding a global shift of +1.0 to the prediction does not change which pixels are trimmed.

**Expected gain:** +0 to +1 d1, within noise. Its role is to protect the dsh structure branch from V2-L failures on glass, mirrors and far range. Check IBims boundary metrics, and drop it first if GPU time is short.

**Citations:** Depth Anything V2 (ignores the top-10% loss regions on pseudo-labels); distill/v2 T2-04 precedent.

### C5 `rep4`: re-parameterised depthwise-separable stride-4 blocks (the latency lever)

**Algorithm.** Replace `smooths[0]` and `head[0]` (dense 3x3 96→96 at 120x160, 1.593 GMAC each) with `RepDWSep(96)`.

Training form:
- DW stage: y = ReLU( Σ_{j=1..2} BN_j(DW3x3_j(x)) + BN_3(DW1x1(x)) + BN_id(x) )
- PW stage: z = ReLU( Σ_{j=1..2} BN_j(PW1x1_j(y)) + BN_id(y) )

Fusion:
- For each branch: K'_b = K_b·γ/√(σ²+ε) and b'_b = β − μγ/√(σ²+ε).
- The 1x1 depthwise kernel is padded to 3x3 at the centre. The identity is a centre-δ depthwise kernel, or I₉₆ for the pointwise stage.
- K = Σ K'_b and b = Σ b'_b.
- Deploy form: DW3x3(+b) → ReLU → PW1x1(+b) → ReLU.

Cost:
- Each replaced block costs 19,200 × (96·9 + 96²) = 0.194 GMAC. The model goes from 6.84 to about 4.04 GMAC (−41%).
- Parameters change by −145.7k.
- The encoder and the stride-8/16/32 smooths are unchanged. `head[1]` (1x1 96→1) is unchanged.

**Tag `rep4`:** sets `cfg.student.rep4=True`.

**Files and functions**
- New `distill/rep.py` (about 90 LOC): `RepDWSep(c, n_branches=2)` with `forward` and `reparameterize()`, plus `reparameterize_model(m)`.
- student.py (:65, :70): build `RepDWSep` when `rep4` is set, and add `DepthStudent.reparameterize()` (about 15 LOC).
- `benchmark/fps_bench.py` and `edge/export_onnx.py`: call `model.reparameterize()` after loading if it exists, and assert fused-vs-unfused max |Δ log D| < 1e-4 (about 10 LOC).
- `run_bench` can use the unfused form; it is numerically identical.

**Unit test** (`tests/test_rep.py`):
- In fp64 eval mode with random BN statistics, fused and unfused outputs differ by less than 1e-10.
- Parameter count after fusion matches the analytic value.
- The ONNX op set is within {Conv, Relu, Add, Mul, Resize, ReduceMean, Gemm, Log, Clip, Concat}.

**Expected gain:** accuracy −0.5 to +0.2 d1; latency −41% MACs.

**Gate:** b20 vs b20core IBims d1 drop ≤ 0.5 and boundary F1 drop ≤ 1 pt. Otherwise retrain as `rep4h`, which converts only `smooths[0]` (saves 1.4 GMAC).

**Citations:** MobileOne; RepVGG; ZipDepth (6.1M re-param depth student on the ANE; unverified citation).

### Supporting changes (no new accuracy claims)

1. **Tag `selgt`** sets `dcfg.select_metric="gt"`. At train.py:361, `"gt"` maps to mean(val_per['nyu'], val_per['sun']) (about 4 LOC). Reason: ownar changes the own-val split, so own-val selection is not comparable with b10.
2. **`--max-iters N` flag** for a smoke test (about 6 LOC).
3. **Tag `dshft`** (fallback only), used with `--init-from outputs/checkpoints/b20/best.pt --epochs 5 --lr 1e-3`. It sets `requires_grad=False` on everything except `scale_mlp` and `k_embed`.
4. **`edge/quant_sim.py` and `edge/weight_int8.py`:** exclude parameter names starting with `scale_mlp` or `k_embed` from 8-bit quantisation (about 64k params kept in fp16/fp32).
5. **`edge/fold_camera.py`** (new, about 70 LOC), for a fixed deployment K and applied to **both** b10 and b20:
   - (a) fold FiLM γ into the head[0] kernel per input channel, and β into its bias. This is exact in the interior; at the borders it is inexact because of zero padding, so report the maximum border error. If it exceeds 1e-3, keep β as a constant (1,96,1,1) Add.
   - (b) replace the ray map with a constant buffer.
   - (c) turn c into a constant add.
   - This removes the intrinsics input from the Core ML graph.
   - Unit test `tests/test_fold_camera.py`: interior parity < 1e-4.

Total new code is about 560 LOC including tests.

---

## 3. Inference-time guarantee for b20

| | b10 | b20 (deploy, fused) |
|---|---|---|
| Params | 5.119 M | 5.119 + 0.064 (scale head) − 0.146 (rep4) ≈ **5.04 M** |
| MACs at 480x640 | 6.84 G | 6.84 − 2·1.593 + 2·0.194 + ~0.0001 ≈ **4.04 G (−41%)** |
| Largest stride-4 tensors | 96×120×160 | unchanged widths (96) |
| Added ops | — | 3 × ReduceMean (GAP), 2 small Gemm, 1 ReduceMean + Sub (centring), scalar Add, Log on a (1,4) input |
| Removed | — | two dense 3x3 convs replaced by dw3x3 + pw1x1 |

**Why b20 cannot be slower than b10**
- Every added op is O(C) or a scalar, about 0.06 MFLOP and under 0.05 ms.
- The removed compute is 2.8 GMAC.
- All ops are ANE-native: Conv, ReLU, Add, Resize, ReduceMean, Gemm, Mul (broadcast (1,C,1,1)), Log, Clip.
- No new op types appear beyond GAP and Gemm. FiLM is kept, and it can be folded away at export (C5 support item 5).

**How much faster it should be**
- On the RTX 5070 Ti at batch 1 in fp32, PyTorch is partly launch-bound and depthwise convs are memory-bound. Expect 11.5-14 ms against b10's about 13.9 ms. The hard rule is "never above b10 + 0.2 ms".
- On compute-bound backends (ORT-CPU, Core ML GPU/ANE in fp16), expect roughly 25-40% lower latency.
- Optional: running at 384x512 (only if section 5's post-hoc test passes) gives about 2.6 GMAC, −62% vs b10.

**Deployment format:** Core ML fp16 with 8-bit weight-only, which is measured lossless for this family; the scale head stays in fp16. Not full INT8: measured d1 collapses to 16-24.

**Must be measured on the Mac.** EfficientViT LiteMLA fp16 parity: max |Δ log D| < 0.01 on 50 IBims images, plus an Xcode Core ML performance report for b10 and b20.

---

## 4. Training plan and time budget (one RTX 5070 Ti laptop, sequential, never concurrent)

| Step | What | GPU time | Notes |
|---|---|---|---|
| 0 | b10 training (running) | ~2.7 h (ends ~21:10) | Meanwhile write the b20 code and run unit tests on CPU only (`$env:CUDA_VISIBLE_DEVICES=""; python -m pytest tests/test_dsh.py tests/test_cstm.py tests/test_fov_ownar.py tests/test_trim.py tests/test_rep.py tests/test_fold_camera.py`) |
| 1 | b10 quick check: IBims only | ~2 min | Confirms b10 ≥ b1_mix |
| 2 | b20 smoke: `--max-iters 30`, full b20 recipe | ~3 min | Check: no NaN, `s_hat_err` falling, trim fraction about 0.10, pad pixels invalid, epoch-time projection |
| 3 | **b20** (full recipe), 40 ep | ~2.8-3.1 h | Estimated 4.2-4.6 min per epoch: b10's ~4 min + ~5% nanmedian/kthvalue + ~0-5% CPU data aug (workers ≥ 6). rep4's training branches are small at 96 channels |
| 4 | **b20core** (no rep4), 40 ep | ~2.8-3.0 h | Control for rep4 attribution |
| 5 | Evaluation of b10, b20, b20core (section 5) | ~1.8 h | SUN full is about 70 of those minutes |
| F1 | Fallback `dshft` | ~20 min | Only if the dsh gate fails |
| F2 | Fallback `rep4h` | ~3 h | Only if the rep4 gate fails |

- **Total GPU after b10:** about 7.5-8 h, i.e. one overnight queue for steps 2-4 and the morning for step 5.
- **Coding:** about 6-8 h (about 560 LOC), overlapping step 0.
- **If time is short:** drop b20core first (b20 is still evaluable on its own; only rep4 attribution is lost), then drop `trim` from b20.
- **Optional, a separate night:** a 15-epoch leave-one-out ladder (b20 minus each of dsh, cstm+fov, ownar, trim), about 1.1 h each. Gate each on IBims with paired bootstrap, and treat IBims as dev for these screens, so the final IBims claim is reported with that caveat.

---

## 5. Evaluation plan

**Setup** (PowerShell, env `depth`; ONNX steps use env `depth_cmp`):
```powershell
cd C:\Tim\nature\unidepth2d; $env:PYTHONPATH="C:\Tim\nature\unidepth2d"
```

**Step 1: isolate checkpoints into their own directories.** run_bench names its output `student-<parent dir>`, so best.pt and last.pt from the same directory would overwrite each other.
```powershell
foreach ($m in "b10","b20","b20core") { foreach ($k in "best","last") {
  New-Item -ItemType Directory -Force "outputs/checkpoints/${m}_$k" | Out-Null
  Copy-Item "outputs/checkpoints/$m/$k.pt" "outputs/checkpoints/${m}_$k/best.pt" } }
$C="student:C:/Tim/nature/unidepth2d/outputs/checkpoints"
$M="--model $C/b10_last/best.pt --model $C/b20_last/best.pt --model $C/b20core_last/best.pt --model $C/b10_best/best.pt --model $C/b20_best/best.pt"
```

**Step 2: dense evaluation**, official v2 protocol, GT-K for students. Pass all models in one call to amortise IO.
```powershell
python -m benchmark.run_bench --dataset ibims --boundary --save-per-image $M.Split(' ')
python -m benchmark.run_bench --dataset nyu --variant filled --save-per-image $M.Split(' ')
python -m benchmark.run_bench --dataset sun --variant raw --save-per-image `
   --model $C/b10_last/best.pt --model $C/b20_last/best.pt --model $C/b20core_last/best.pt   # full, ~70 min
python -m benchmark.compare_paired results_bench/ibims__student-b10_last.json results_bench/ibims__student-b20_last.json --metric d1 --metric arel
python -m benchmark.compare_paired results_bench/nyu-filled__student-b10_last.json results_bench/nyu-filled__student-b20_last.json --metric d1 --metric arel
python -m benchmark.compare_paired results_bench/ibims__student-b20core_last.json results_bench/ibims__student-b20_last.json --metric d1 --metric arel
```
Check the exact JSON names in `results_bench/` before running compare_paired.

**Step 3: OCMR-SUN object distance**, one model per call.
```powershell
foreach ($m in "b10_last","b20_last","b20core_last") { python -m benchmark.ocmr_eval_ann --model "$C/$m/best.pt" }
```

**Step 4: 8-bit weight-only robustness.**
```powershell
python -m benchmark.run_bench --dataset ibims --boundary --save-per-image --model "q:8:C:/Tim/nature/unidepth2d/outputs/checkpoints/b10_last/best.pt" --model "q:8:C:/Tim/nature/unidepth2d/outputs/checkpoints/b20_last/best.pt"
```

**Step 5: zero-training low-resolution test**, replacing mres.
- New script `scripts/set_train_size.py --ckpt X --hw 384 512 --out Y` (about 20 LOC) copies the checkpoint with `cfg["student"]["train_size"]=[384,512]`.
- Run IBims and NYU on `b10_r384` and `b20_r384`.
- If the d1 drop is ≤ 1, a 384x512 deployment is viable at about −36% further MACs. If the drop is > 1, try BN recalibration (`scripts/bn_recal.py`, 200 batches, BN buffers only). Only adopt mres in a future rung if the drop still exceeds 1.5.

**Step 6: latency**, all in one session.
- (a) Static cost: new `scripts/count_macs.py` (Conv/Linear hooks, after `reparameterize()`), which prints params and GMAC for b10, b20 and b1_mix.
- (b) RTX fps_bench. First copy `outputs/checkpoints/fps_bench.json` to `fps_bench_prev.json`, because the run overwrites it. Re-measure the references in the same call:
  ```powershell
  python -m benchmark.fps_bench --config config.yaml --model $C/b10_last/best.pt --model $C/b20_last/best.pt `
    --model $C/b1_mix/best.pt --model $C/S16_prox/best.pt --model v2-s --res-level 9
  ```
- (c) ORT-CPU fp32. In env `depth_cmp`: `python -m edge.export_onnx --ckpt outputs/checkpoints/b20_last/best.pt --out outputs/edge/b20/fp32.onnx`, which reparameterises and checks parity (do the same for b10). Then new `edge/ort_latency.py --onnx ... --threads 4 --threads 1 --warmup 20 --iters 200`, reporting median and p90 (about 40 LOC).
- (d) Mac, run by the user: new `edge/export_coreml.py`, which applies fold_camera, then traces and converts with `coremltools.convert(compute_precision=FLOAT16, compute_units=ALL)`. Collect the fp16 parity on 50 IBims frames and the Xcode performance report for b10 and b20.

**What counts as b20 beating b10.** All comparisons are last.pt against last.pt; best.pt is reported as well.

- **Primary (OOD accuracy):** IBims-100 raw d1(b20) ≥ d1(b10) + 2.0. The paired-bootstrap 95% CI lower bound must be > 0, IBims ARel must not be worse, and |bias_log| must be ≤ b10's.
- **Non-regression:**
  - NYU-filled d1 ≥ b10 − 0.5.
  - SUN-raw d1 ≥ b10 − 0.5.
  - OCMR objAbsRel ≤ b10 + 0.003.
  - IBims boundary metrics no worse than b10 by more than 1 pt.
  - q:8 IBims d1 within 1.0 of fp32.
- **Latency:**
  - Fused params ≤ 5.2 M.
  - GMAC ≤ 0.65 × b10.
  - ORT-CPU median ≤ 0.80 × b10.
  - RTX fps_bench ≤ b10 + 0.2 ms.
  - On the Mac (when measured), Core ML latency < b10, with fp16 parity < 0.01.
- **Verdicts:**
  - All three groups pass: **b20 wins** (accuracy kept or improved, latency reduced).
  - Primary passes but rep4 fails its gate (b20 vs b20core): ship b20core accuracy and retrain with `rep4h`.
  - Primary fails: run `dshft`. If it still fails, b20 is a latency-only result: a comparable-accuracy check of b20 vs b10 within ±0.5 d1 plus the latency criteria.
- **Context:** report S16 (70.7 / 88.4 / 86.0, 17.8 ms) and official V2-S (82.5 IBims, 40 ms) in the same tables. Label SUN rows as partly in-domain, since SUN is in the training mix.

---

## 6. Rejected ideas and reasons

1. **tak: online frozen-S16 teacher with CWD feature KD** (0/3). +40-50% train time and +1.2 GB. The scale-free response cannot transfer S16's main advantage, which is global scale. CWD on x_td can import S16's SUN in-domain bias. The repo's earlier assistant teacher (s8) broke (.199). It also confounds attribution. Defer to a later `b20+tak` rung.
2. **lite decoder (48/64/96, dw5x5)** (0/3). It replaces the whole decoder and halves the stride-4 width, for only about 0.7 GMAC more saving than rep4. The ANE "2 MB tile" argument is unverified.
3. **mres multi-resolution self-distillation** (0/3). Low-res train-mode forwards pollute the BN statistics and EMA of the full-resolution model, and it costs +45% training. The evidence is from classification (RS-Nets). Replaced by the zero-training low-resolution test in section 5.
4. **nofilm** (0/3). FiLM is part of the proven recipe and was never ablated on B1, and the saving is about 0.1-0.3 ms. Export folding gets the same benefit at zero risk.
5. **Attention-pooled scale head** (0/3). No evidence it beats GAP, and it adds a softmax op.
6. **Gauge loss, and mean-based a* with gradient through r** (dropped). Mean-centring r_c is exact and free, and the median stop-grad target is robust to outliers.
7. **Reliability weighting of the own-row scale loss** (split vote). It was only ever tried inside scale/scale2, which cost −1.8 IBims. Available as optional tag `dshrel`.
8. **Letterbox form of ar** (rejected). It loses 25% resolution on about 74% of own rows; crop-or-pad avoids that.
9. **Symmetric fov replacing s5** (rejected). It changes the proven s5 setting (zoom-in capped at 1.4×) and confounds the b10 comparison.
10. **Trimming GT rows** (2/3 against). It removes hard but correct sensor pixels, and the evidence covers pseudo-labels only.
11. **Own-val checkpoint selection for b20.** Not comparable once ownar is on, and blind to OOD gains. Replaced by `selgt` plus last.pt-vs-last.pt.
12. **More V2-L pseudo-labels (100k-1M frames), shared-context crops with the teacher re-run, multi-teacher L/B/S, feature KD from UniDepth.** All need new teacher passes: only V2-L depth PNGs are cached, with no features and no V2-B/S maps. This is the biggest long-term lever; propose it as b30.
13. **bins/TIE-KD, s6 NLL/confidence weighting, s7, s8, norm, anch, icd, sced, scale/scale2, ground-lock, SMA priors.** Measured neutral or negative here; Depth Pro 2 also reports that confidence weighting hurts.
14. **CutMix/CutDepth.** Mixing scenes from different cameras corrupts the per-image s* target.
15. **Strong student-only photometric augmentation.** Plausible but not aimed at scale; deferred as tag `aug2`.
16. **Backbone swap (RepViT-M0.9, FastViT-T8, MobileNetV4) and the distill/v2 CoDeQ stack.** The swap confounds b10 vs b20, and measured MACs at 480x640 are not lower (T8 7.08 G, RepViT 8.87 G). CoDeQ has never run end to end, its loss and train tests are missing, it has no intrinsics path, and its frozen roles exclude SUN. Both belong in a b30 latency track with Mac profiling.
17. **Stride-2 skip, PixelShuffle, DySample, convex upsampling.** They add latency and ANE-unfriendly ops for at most about +0.5 d1.
18. **DepthART frozen encoder with ray cross-attention adapters, and the UniDAC spatial scale map.** DepthART's ablation is inconsistent, and attention at stride 4 is risky on the ANE. A single scalar already closes 82-86% of the scale gap.
19. **DepGraph pruning of S16, RS-Nets private BN, ordinal/ranking losses, CLIP/language scale cues, test-time adaptation.** Each is over budget, has tiny gains, or needs data or outputs that do not exist.
20. **Full INT8 / W8A8 PTQ or QAT, QARepVGG, Hilbert-curve output.** Full INT8 is measured to collapse (d1 16-24). The ANE is fp16-native, and W8A8 helps only on M4/A17 Pro. Ship fp16 with 8-bit weights.
21. **Larger or progressive training resolution.** NYU and SUN are natively 640x480, so it only upsamples them.

---

## 7. References

Items marked † are 2026 preprints cited by the research pass that I could not verify independently; treat their numbers as provisional.

- MoGe-2: Wang et al., 2025. https://arxiv.org/abs/2507.02546
- Metric3D: Yin et al., ICCV 2023. https://arxiv.org/abs/2307.10984
- Metric3D v2: Hu et al., 2024. https://arxiv.org/abs/2404.15506
- DMD, zero-shot metric depth with FoV-conditioned diffusion: Saxena et al., 2023. https://arxiv.org/abs/2312.13252
- ScaleDepth: Zhu et al., 2024. https://arxiv.org/abs/2407.08187
- UniDepth: Piccinelli et al., CVPR 2024. https://arxiv.org/abs/2403.18913
- UniDepthV2: https://arxiv.org/abs/2502.20110
- Depth Anything V2: Yang et al., 2024. https://arxiv.org/abs/2406.09414
- Depth Anything (V1): https://arxiv.org/abs/2401.10891
- Depth Pro (v1): https://arxiv.org/abs/2410.02073
- † Depth Pro 2: https://arxiv.org/abs/2611.26603
- Distill Any Depth: https://arxiv.org/abs/2502.19204
- Beyer et al., "Knowledge distillation: A good teacher is patient and consistent": https://arxiv.org/abs/2106.05237
- MobileOne: Vasu et al., CVPR 2023. https://arxiv.org/abs/2206.04040
- RepVGG: https://arxiv.org/abs/2101.03697
- QARepVGG: https://arxiv.org/abs/2212.01593
- EfficientViT: Cai et al. https://arxiv.org/abs/2205.14756
- RepViT: https://arxiv.org/abs/2307.09283
- FastViT: https://arxiv.org/abs/2303.14189
- MobileNetV4: https://arxiv.org/abs/2404.10518
- iFormer: https://arxiv.org/abs/2501.15369
- DySample: https://arxiv.org/abs/2308.15085
- RS-Nets: https://arxiv.org/abs/2007.09558
- Depth Any Camera: https://arxiv.org/abs/2501.02464
- TAKD: https://arxiv.org/abs/1902.03393
- Channel-wise distillation (CWD): https://arxiv.org/abs/2011.13256
- Distillation Scaling Laws: https://arxiv.org/abs/2502.08606
- DisDepth (as cited by the designs): https://arxiv.org/abs/2404.16386 (ID not independently checked)
- † DepthART: https://arxiv.org/abs/2607.17099 (its .713 includes the adapters; the scale head alone is .534)
- † "What the Averages Hide": https://arxiv.org/abs/2611.20518
- † FoundationGeo: https://arxiv.org/abs/2607.11588
- † ZipDepth: https://arxiv.org/abs/2607.08771
- † Apple Neural Engine latency study (Bryngelson): https://arxiv.org/abs/2606.22283

**Key repo files:**
- C:\Tim\nature\unidepth2d\distill\student.py
- C:\Tim\nature\unidepth2d\distill\losses.py
- C:\Tim\nature\unidepth2d\distill\dataset.py
- C:\Tim\nature\unidepth2d\distill\train.py
- C:\Tim\nature\unidepth2d\benchmark\run_bench.py
- C:\Tim\nature\unidepth2d\benchmark\fps_bench.py
- C:\Tim\nature\unidepth2d\edge\export_onnx.py
- C:\Tim\nature\unidepth2d\edge\quant_sim.py
- C:\Tim\nature\unidepth2d\edge\weight_int8.py
- C:\Tim\nature\unidepth2d\outputs\v2\cache\own\manifest.csv
- C:\Tim\nature\unidepth2d\outputs\checkpoints\b10\

**New files:**
- C:\Tim\nature\unidepth2d\distill\rep.py
- C:\Tim\nature\unidepth2d\edge\fold_camera.py
- C:\Tim\nature\unidepth2d\edge\ort_latency.py
- C:\Tim\nature\unidepth2d\edge\export_coreml.py
- C:\Tim\nature\unidepth2d\scripts\count_macs.py
- C:\Tim\nature\unidepth2d\scripts\set_train_size.py
- C:\Tim\nature\unidepth2d\scripts\bn_recal.py (only if needed)
- tests: test_dsh.py, test_cstm.py, test_fov_ownar.py, test_trim.py, test_rep.py, test_fold_camera.py

**Open risks to watch:**
1. The format of `intrinsics.json` (pixel or normalised) must be confirmed before ownar; the K assertion in C3a catches a wrong guess.
2. The ~0.5-1 h of SUN evaluation is the long pole.
3. All Mac and ANE claims stay unverified until the user runs `export_coreml.py` on the target Mac.