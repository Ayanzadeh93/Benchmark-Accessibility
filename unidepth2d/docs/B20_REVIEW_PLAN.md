# b20dws improvement plan: one prioritized list

Scope: every finding that both verifiers marked `correctness.real = true` (all 24 qualified), plus the loss-report fix pack A0' (from `reports/Loss design for tiny depth student.md` and `research_notes/Loss design for tiny depth student/codebase_audit.md`) folded in as Tier A. Where the two verifiers disagreed, the plan uses the corrected proposal that is technically sound. Any proposal part one verifier rejected is moved to the "Not worth doing" list.

Tier definitions:
- **Tier A**: implement now behind a config or CLI flag whose default reproduces today's outputs bit for bit, and add a unit test. Several Tier A items (A01, A02, A07-A14, A16) are data or loss fixes, so measuring their effect still takes a retrain (see the schedule).
- **Tier B**: training-recipe ablation.
- **Tier C**: architecture variant.

Accuracy flag: any item where either verifier rated accuracy_risk medium or high carries **flag: may reduce accuracy - only adopt if the ablation shows no loss**.

Priority: **P0** blocks a credible CVPR/WACV paper (data leakage, wrong labels, wrong metric, broken deployment numerics). **P1** is a cheap, measured accuracy or efficiency gain. **P2** is a speculative ablation. **P3** is deferred.

---

## 0. Summary table

| ID | Title | Tier | Pri | Retrain to measure | Acc. flag |
|---|---|---|---|---|---|
| A01 | Apply roles.yaml split hygiene to the legacy loader | A | P0 | yes (paper table) | - |
| A02 | SUN RGB-D: drop 8x rows, rebuild with GT K and official decode | A | P0 | yes | - |
| A03 | LiteMLA fp16 overflow: exact k-fold, denominator guard, new Core ML parity gate | A | P0 | no | - |
| A04 | Boundary F1: switch tables to v2 + DepthPro metric | A | P0 | no | - |
| A05 | One K resolution path (K=None no longer silently disables FiLM) | A | P0 | no | - |
| A06 | Seeding and reproducibility in distill/train.py | A | P0 | no | - |
| A07 | A0'-a SILog consistent weighted moment, fp32 two-pass | A | P1 | yes | - |
| A08 | A0'-b lambda = 1 under dsh | A | P1 | yes | - |
| A09 | A0'-c EG-SSI hole-border mask, q99 normaliser | A | P1 | yes | - |
| A10 | A0'-d pad-aware r_c centring and pad-masked GAP | A | P1 | yes | - |
| A11 | A0'-f dsh_detach fix (k_embed keeps gradient) and GradScale gamma | A | P1 | yes | - |
| A12 | A0'-g native-stride gradient matching (strides 4-32) | A | P1 | yes | - |
| A13 | A0'-h EG-SSI zero-MAD init guard (step-0 1/eps explosion) | A | P1 | no ablation | - |
| A14 | A0'-e trim warm-up flag and trim unit tests | A | P2 | leave-one-out only | - |
| A15 | Post-hoc PreciseBN on EMA weights | A | P1 | no | - |
| A16 | INTER_NEAREST_EXACT for every label resize | A | P1 | yes (bundled) | - |
| A17 | Closed-form ray map (about -40 ms on torch CPU) | A | P1 | no | - |
| A18 | One shared inference preprocess with an aspect policy | A | P1 | no | - |
| A19 | Optional guided-upsampling post-process (+GF row) | A | P1 | no | - |
| A20 | 384x512 operating point, eval only | A | P1 | no | **flag** |
| A21 | Core ML uint8 ImageType input, normalisation in graph | A | P2 | no | - |
| A22 | ORT int8: opt-in mixed precision plus collapse gate | A | P2 | no | - |
| A23 | Training throughput flags plus per-group grad-norm logging | A | P1 | no | - |
| A24 | Export plumbing: derive SCALE_HEAD_SHAPES, out_indices field | A | P1 | no | - |
| B01 | Batch 12x1 instead of 4x3, optional late BN freeze | B | P1 | yes | - |
| B02 | Student-side photometric aug, blur/zoom decoupling | B | P1 | yes | - |
| B03 | NYU white border as pad, plus a quantisation check | B | P2 | yes | - |
| B04 | Optimizer param groups (no-wd on 1-D params, per-group clip, encoder LR) | B | P2 | yes | **flag** |
| B05 | Virtual-focal augmentation, gated on diagnostics | B | P2 | yes | **flag** |
| B06 | EfficientViT-B1 r288 / r256 pretrained weights | B | P2 | yes | **flag** |
| B07 | M7-S0 GT-anchored own-row scale rectifier (loss report) | B | P2 | yes | - |
| B08 | b20 leave-one-out ladder (loss report) | B | P2 | yes | - |
| B09 | noray / zero-init ray stem (only inside a retrain that runs anyway) | B | P3 | yes | - |
| C01 | DW-separable smooths at stride 8/16 (`dws_levels`) | C | P1 | yes | - |
| C02 | Stride-2 residual refine head | C | P2 | yes | - |
| C03 | Scale-head masked attention pooling, gated on a frozen-feature probe | C | P2 | yes | - |
| C04 | Encoder Hardswish to ReLU (gradual swap) | C | P3 | yes | **flag** |
| C05 | Conv-only backbone (MNv4) in a separate b30 latency track | C | P3 | yes | **flag** |

---

## 1. Tier A: implement now

### A01: Apply roles.yaml split hygiene to the legacy loader (P0)
- **Location**: distill/dataset.py:139-166 (MultiSourceDataset drops only frozen_test.json ids before the 3% val carve); configs/v2/roles.yaml; reuse distill/v2/data.py (load_roles, held_out_ids).
- **Change**: add `data_gt.roles` (default None, so the b20dws split stays bit-identical). When it is set, filter before the val carve using the v2 loader, not hand-parsed YAML. Ignore the `counts` key.
  - own: drop excluded.own, nyu_family_own.ids, test.video_own, test.frozen_own, sf6_temporal.own_ids.
  - nyu: drop excluded.nyu. This rule is stricter than Eigen and must be stated in the paper. The minimum is the a3_match_of_ibims1 rows plus the confirmed Eigen matches.
  - sun: drop blacklist.sun_8x, nyu_derived_sun.match_an_eigen_test_frame, and any excluded.sun. Dropping nyu_derived_sun.ids (1,045 NYU-train duplicates) is optional.
  - Log drop counts per source and reason. Write roles_sha256 to run_info.
- **Measured state**: the train split has 27,820 rows. Of those, 48 are confirmed Eigen-test frames (covering 50 test ids, 7.6% of NYU-654), 168 of 189 are video_own test frames, all 247 excluded.nyu rows are present (10 of them a3-match IBims), 77 of 80 are sun_8x, and 16 are SUN Eigen matches.
- **Free step, no retrain**: re-evaluate the existing b20dws and b10 on NYU-654 minus the 50 leaked ids (about 604 frames) and on video_own minus the 168 trained frames. Report old and clean numbers side by side. Until the retrain, mark NYU-654 and own/video numbers as contaminated. The b10-vs-b20 deltas stay fair because both share the split.
- **Expected effect**: removes about 2.5-4% of rows. NYU d1 probably drops a few tenths, which is inflation being removed, not accuracy lost. IBims can only get cleaner. Required for any zero-shot or NYU claim.
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: `tests/test_split_hygiene.py`. With the flag on, train and val ids must not intersect test.*, the confirmed Eigen-test list, the IBims a3 matches, or blacklist.sun_8x. Assert per-source counts. With the flag off, assert the split equals the legacy 27,820 rows (hash of the sorted ids).
- **Run**: R1 (b20dws-clean) and R2 (b10-clean), 40 epochs each. These become the new reference rows. Re-check IBims afterwards.

### A02: SUN RGB-D: drop 8x rows, rebuild with GT K and official decode (P0)
- **Location**: scripts/build_gt_dataset.py:89-96 and :113 (median-heuristic decode, then depth > 8 m zeroed), :120-121 (K blank); distill/dataset.py:129, :169-174, :267-275 (fallback to teacher-predicted outputs/data_gt/sun_intrinsics.json); evaluation uses GT intrinsics.txt (benchmark/datasets_tier1.py:160, run_bench.py:117-121).
- **Change**:
  1. Stop-gap with zero risk: the 80 sun_8x rows are already dropped by A01. Add a load-time assertion that the fraction of valid stored-mm values divisible by 8 is below 0.5 per GT row, and the same check at build time.
  2. Real fix: write a new `manifest_sungt.csv` selected by flag `data_gt.manifest`. The old manifest stays as the default.
     - Map each sun_XXXXXX id to its official Z:/Datasets/SUNRGBD alltrain dir by thumbnail NCC > 0.97, not by HF index order. Keep ids and row order identical so val carves and roles stay fixed.
     - Decode raw PNGs with `benchmark.datasets_tier1._sun_decode_depth`. Use the same depth variant (raw or bfx) as the HF repack, chosen by NCC against the existing depth_mm. Never apply the unconditional decode to HF arrays.
     - Write fx_n = K00/W0, fy_n = K11/H0, cx_n = K02/W0, cy_n = K12/H0 from intrinsics.txt, with W0,H0 the ORIGINAL size, so the values are invariant to the 640x480 stretch.
     - Unmatched rows: per-sensor K by original size (730x530 kv2, 561x427 kv1, 591x441 xtion, 681x531 realsense), or drop the row.
     - Keep the existing nyu_derived_sun handling.
  3. Diagnostic: measure log(f_teacher/f_gt) on own rows that have calibrated K, to check whether the teacher has the same +5% bias there.
- **Expected effect**: on the only multi-camera GT source (20% of the mix), removes 5-12% scale-label noise and a +2.08 log error on 77 rows. Also removes the contradiction between NYU and SUN kv1 rows, where the same sensor gets about a 10% different f. Expect lower SUN medianlog and s_hat_err and better near-range scale. Do not claim an IBims gain until it is measured.
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: with the new manifest, every SUN row has non-empty fx..cy; fx_n falls within 1% of one of the four per-sensor values; sun_intrinsics.json is never opened (mock it); ids and order match the old manifest; the frac8 assertion passes on all rows and fails on a synthetic `mm<<3` array.
- **Run**: R3 = R1 + A02 (+ A16), 40 epochs. Log per-source s_hat_err and L_s. Prerequisite for B07 and C03.

### A03: LiteMLA fp16 overflow: exact k-fold, denominator guard, new Core ML parity gate (P0)
- **Location**: timm efficientvit_mit.py:359-365 (LiteMLA._attn upcasts to fp32 via `.float()`, a no-op once traced from an fp32 model); edge/export_coreml.py:140-173 (FLOAT16, no patch), :182-205 (gate `worst < 0.01`); distill/student.py:300-304 (GAP scale head spreads one inf to every pixel); docs/B20_RESULTS.md:123.
- **Measured state**:
  - max|q@kv| is 73k-122k at stride 16 (1,200 tokens), above the fp16 maximum of 65,504. The minimum denominator is exactly 0 on real images.
  - In fp16 emulation, 1-7% of IBims/own frames come out fully non-finite (for example corridor_02). On 40 images, d1_ssi falls 91.38 to 89.45.
  - Even a healthy fp16 model has max|dlogD| about 0.018, so the current gate fails it.
  - The current gate also cannot see NaN: `max(worst, nan)` keeps the old worst.
- **Change** (export-only flag `--mla-kscale {0,4,6}`, default 0 = current graph; the Core ML recipe in the docs uses 4):
  1. Fold s = 2^-4 (2^-6 if measured headroom is below 4x) into the **k output rows of each qkv 1x1 conv only**. With heads of [q,k,v] blocks of dim 8, those are channels h*3*dim+dim to h*3*dim+2*dim-1. Scale any bias or BN on those rows too. Do **not** also scale the aggreg convs: they are linear, bias-free and group-confined, so the scale already propagates, and scaling them again applies s^2. Do **not** use the k/N variant: eps/N underflows to 0/0 in fp16.
  2. Denominator guard: eps = 1e-5 is subnormal in fp16, and the ANE may flush it, which would give 0/0 where q is all zero. Monkeypatch `_attn` before tracing to drop the `.float()` casts and use `den.clamp_min(2**-14)`, or eps' = 2^-14. This changes the effective eps slightly. Accept it only if fold-vs-unfold fp32 max|dlogD| stays below 1e-3. Note: one verifier suggested eps' = eps/s as "exact". That is inverted. Exactness would need eps' = s*eps, which is even more subnormal, so use the clamp.
  3. Keep the three scale-head GAPs and scale_mlp in fp32 via an op_selector (negligible cost). Optionally add an on-device isfinite(s_hat) fallback.
  4. Replace the gate. Fail on any non-finite output first. Then require, per image, p99 |dlogD| < 0.01 and max < 0.03, plus delta1 agreement > 99.5% and |dAbsRel| < 0.002, over at least 50 real images sampled (not the first 50 alphabetically) from IBims and own. Log max|q@kv|*s per block.
  5. Fallback if the fold misbehaves on device: `ct.transform.FP16ComputePrecision(op_selector=...)` to keep only the attention MatMul/Add/Div chain in fp32. Measure the ANE partition cost in the Xcode performance report.
  6. Update B20_RESULTS.md: fp16 parity requires the fold.
- **Expected effect**: prevents all-NaN maps on 1-7% of unseen frames on Core ML, the ANE and any fp16 ORT/TensorRT build. fp32 numbers are unchanged (fold max|dlogD| about 5e-4, identical d1/d1_ssi/AbsRel). Prerequisite for any on-device latency or accuracy claim.
- **Accuracy risk**: none. **Inference cost**: 0 with the weight fold.
- **Test**: `tests/test_fp16_attention.py`:
  - (a) fold vs no-fold, fp32: max|dlogD| < 1e-3 on 20 IBims images, plus a synthetic check that k-row indexing matches the head layout;
  - (b) torch fp16 (CUDA half, or CPU emulation) with pure-fp16 attention and the patch, over the full IBims set at 480x640: 0 non-finite maps and within the new gate;
  - (c) the gate function returns FAIL on a NaN array.
- **Run**: none (no retrain). Needs one Mac session for export_coreml.py, parity and ANE latency.

### A04: Boundary F1: switch tables to v2 + DepthPro metric (P0)
- **Location**: benchmark/run_bench.py:321-343 (legacy protocol.boundary_f1_si, which drops images whose prediction has no edges); benchmark/protocol_v2.py:12-24, :208; benchmark/paper_tables.py:102; docs/ALL_RESULTS.html:152.
- **Change** (additive keys; the legacy key stays unchanged and is labelled legacy):
  - Under `--boundary`, compute per image `bf1_si_v2` (protocol_v2.boundary_f1_si_v2, t=5..25, both_empty='zero') and `bf1_depthpro` (si_boundary_f1_depthpro, once with valid=mask and once with valid=None).
  - Store per-image rows with ids. Aggregate with aggregate_boundary_f1 and store n_contributing.
  - Re-score b10, b20core, b20dws **and the foundation baselines** (UniDepthV2-S/B/L, UniK3D, MoGe-2) on the same images. Regenerate ALL_RESULTS and PAPER_TABLES with the DepthPro-protocol bF1 as the headline and the metric named in the caption.
  - Restate the B20_SPEC dws4 gate as bF1v2 with a paired bootstrap (pair_rows_by_id + paired_bootstrap_clustered); accept when the 95% CI includes 0.
- **Verified values** (IBims, last.pt, 99/100 images contributing):

  | Model | v2 | DepthPro |
  |---|---|---|
  | b10 | .0123 | .0074 |
  | b20core | .0083 | .0044 |
  | b20dws | .0087 | .0051 |

  dws4 is edge-neutral. The loss is in the b20 bundle, but that is not a causal claim until B08 runs.
- **Expected effect**: no model change. Gives the correct attribution and keeps dws4 as a free decoder-MAC saving. In the paper, state the real limitation: every 5M student has near-zero boundary F1 against the large models.
- **Accuracy risk**: none. **Inference cost**: 0.
- **Test**: an image whose prediction has no edges scores 0 under v2 (not None) and is counted; n_contributing equals the image count; legacy values match the stored results_bench JSON for b20dws (.03185).
- **Run**: eval only, about 1 GPU-hour for all rows.

### A05: One K resolution path (K=None no longer silently disables FiLM) (P0)
- **Location**: distill/student.py:238-241 (FiLM only when K is not None) vs :194-212, :227, :286, :296 (_DEFAULT_K fallbacks); distill/dataset.py:50,120 and distill/v2/data.py:74 (DEFAULT_INTR=(0.7,0.9,.5,.5)); callers benchmark/nyu_eval.py:59, distill/visualize.py:77, run_bench.py:103 and :116-122, benchmark/safety.py:1071-1072.
- **Change**:
  - Add StudentConfig `k_none_policy: {legacy, default_k}`, default `legacy`. With the default, every GT-K output is bit-identical and the old K=None numbers reproduce. Under `default_k`, resolve `k = self._k_or_default(intrinsics, B, x.device, x.dtype)` once at the top of forward and use it for rays, FiLM, k_embed and cstm.
  - Export one module-level `DEFAULT_K` from student.py and import it in dataset.py, v2/data.py, ort_int8.py, export_v2.py, v2/model.py and run_bench.py:103. Check whether any training row actually hits the (0.7,0.9) fallback before unifying it.
  - Fix the callers so the legacy branch is unreachable from repo code: nyu_eval passes the NYU GT K, visualize passes K or DEFAULT_K, safety 'default-k' passes DEFAULT_K, so 'none' and 'default-k' are no longer identical. Flip the default to `default_k` once the legacy numbers are archived.
  - Label any no-K column "assumed default K", not "uncalibrated".
- **Expected effect** (corrected): GT-K rows unchanged; torch K-less output now matches ONNX/Core ML; NYU no-K AbsRel 0.108 to 0.073; own no-K 0.33 to 0.27; **IBims no-K d1 about 72.3 to 70.6** (FiLM-off was masking the default-K focal mismatch). This is a consistency fix, not an accuracy gain.
- **Accuracy risk**: none on GT-K rows. **Inference cost**: 0.
- **Test** (tests/test_dsh.py): for every b20 config with `default_k`, forward(x, None) == forward(x, DEFAULT_K) exactly; with `legacy`, outputs equal the current code; GT-K outputs are identical under both policies; there is one DEFAULT_K object (identity check across modules).
- **Run**: eval only. K-dropout is excluded (see Not worth doing).

### A06: Seeding and reproducibility in distill/train.py (P0)
- **Location**: distill/train.py:118-350 (no seeding), :286 (WeightedRandomSampler without generator), :287 (no worker_init_fn), :502-508; port from distill/v2/train.py:422-427 and :777-780.
- **Change**: add `--seed` (default None = current behaviour).
  - Call seed_everything(seed) before build_student, so the init weights are seeded.
  - Rebuild the sampler each epoch with `generator=torch.Generator().manual_seed(seed*1000+epoch)`.
  - Pass `worker_init_fn=_seed_worker` and `generator=g` to the loader.
  - Write the seed into the ckpt, train_summary.json, the registry and wandb.
  - Optional `--snapshot-every N` (saves ema_state e{N}.pt).
  - Optional own-val Roboflow base-image dedup (strip `.rf.<hash>` before splitting), behind its own flag because it changes the val split.
  - Keep reporting on last.pt (final EMA). Do not select on DIODE.
- **Expected effect**: no direct accuracy change. Required to back every sub-point attribution (dws4, PreciseBN, trim, and so on) with seed statistics. Run 54e73nhr (peak at e11, then divergence) shows run-to-run variance is large.
- **Accuracy risk**: none. **Inference cost**: 0.
- **Test**: on CPU, two 20-step runs with the same seed (tiny model, deterministic algorithms) give identical loss traces; different seeds differ; seed=None leaves global RNG state untouched.
- **Run**: none of its own. Every run in the schedule uses it.

### A07-A14: Loss fix pack A0' ("b20dws-fixed")

The definitions come from the loss report. None of these changes the inference graph: forward and ONNX export are bit-identical at pad = None. Each part gets its own flag. The bundle tag `a0p` turns on A07(a2), A08, A09, A10, A11(gamma=1), A12 and A13. A14 stays off.

Two reminders from the loss report:
- The SILog moment bug and the EG-SSI normaliser bug were already present in b10. They cannot explain the b10 to b20 structure regression; only B08 can attribute that.
- Expected gains from A0' sit near seed noise (IBims-100 paired CI about +/-4.9 d1). Judge them on per-image paired statistics.

#### A07: A0'-a SILog consistent weighted moment, fp32 two-pass (P1)
- **Location**: distill/losses.py:32-48 (`diff=(p-t)*w` squared, giving mu2 = sum(w^2 e^2)/sum(w)); prox-SILog :271-300.
- **Change**: compute `e = p_al.float() - t`, `mu = sum(v*e)/sum(v)`, `L = sqrt(sum(v*(e-mu)^2)/sum(v) + (1-lambda)*mu^2 + 1e-6)`, a two-pass fp32 variance. Arm a2 (default under `a0p`) uses v = eff_mask^2 and keeps the legacy 4:1 near emphasis, now consistent. Arm a1 uses v = eff_mask. Flag `loss.silog_moment: {legacy, a1, a2}`, default legacy. This also fixes the fp16 cancellation at lambda = 1.
- **Expected effect**: part of A0' (NYU d1 +0.2 to +0.6, d1_si +0.2 to +0.6, IBims raw +0 to +0.5, all estimates).
- **Accuracy risk**: low. Arm a1 halves the near emphasis, which a2 hedges. **Inference cost**: 0.
- **Test**: with all weights equal to 1, legacy, a1 and a2 agree to 1e-7; legacy mode is bit-identical to the current function; an fp16 input with a large constant offset gives the same value as fp64.
- **Run**: inside R4. a1 vs a2 as a 15-epoch screen (S-a1).

#### A08: A0'-b lambda = 1 under dsh (P1)
- **Location**: losses.py SILog and prox-SILog lambda (config `silog_lambda`, prox lambda).
- **Change**: when `a0p` or `loss.dsh_lambda1` is set and scale_head is on, use lambda = 1 for both. Under a median-aligned decoupled scale, the (1-lambda)*mu^2 term can only be reduced by tilting near against far, which is a spurious structure force.
- **Expected effect**: removes the spurious tilt (about 1-2% of the gradient).
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: with lambda = 1, adding a constant to p_al leaves both losses unchanged (shift invariance); with the flag off, lambda stays 0.85.
- **Run**: inside R4.

#### A09: A0'-c EG-SSI hole-border mask, q99 normaliser (P1)
- **Location**: losses.py:111-143 (Sobel on target_log, where holes carry log d = 0; hole borders set emax in 85% of NYU and 82% of SUN images); train.py around :376/:400 must pass the **pre-trim** valid mask.
- **Change**: `edge <- edge * (1 - maxpool3x3(~valid_pretrim))` before normalising, with `edge_w = 0.2 + 0.8*clamp(edge/q99(edge), 0, 1)` in fp32. This replaces the separate pad path. Do not detect holes with `target_log == 0`, because NYU pixels at exactly 1 m have log 0. Do not use the trimmed mask, because trimmed pixels are not holes.
- **Expected effect**: real edges stop being squashed to the 0.2 floor on GT rows. EG-SSI carries 28% of the loss.
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: on a synthetic target with a hole plus one real step edge, the real edge gets weight 1.0 and the hole border gets 0; with no holes and no pad, the old and new edge weights match up to the max vs q99 normaliser (compare with the legacy flag).
- **Run**: inside R4.

#### A10: A0'-d pad-aware r_c centring and pad-masked GAP (P1)
- **Location**: student.py:295 (`r_c = r - mean_hw(r)` averages over pad), :300-301 (GAP taps see the noise canvas); train.py passes `pad` into `_dsh_output`.
- **Change** (training only):
  - `m4 = 1 - avgpool4(pad)`; `r_c = r - sum(m4*r)/max(sum(m4), 1)` in fp32.
  - Masked GAP of each tap: `sum(f*w)/sum(w).clamp_min(1e-3)` with w = 1 - avgpool(pad) at that tap's stride. This is the zero-parameter part of the scale-head pooling finding.
  - When pad is None, take the existing `.mean` branch, so inference and export stay bit-identical.
- **Expected effect**: removes the `a*(mean_pad r - mean_valid r)` target noise from s* and the pad contamination of the GAP inputs, for the scalar that drives IBims.
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: pad=None gives bit-identical r_c and g; with pad, the mean of r_c over non-pad stride-4 cells is 0 to 1e-6; the masked GAP equals the plain GAP of the valid region on a synthetic map.
- **Run**: inside R4.

#### A11: A0'-f dsh_detach fix (k_embed keeps gradient) and GradScale gamma (P1)
- **Location**: student.py:300-304 (dsh_detach detaches the whole concatenated input, so k_embed stays at random init).
- **Change**: `t = cat(GAP F4, GAP F3, GAP x_td)`; `t = GradScale(t, gamma)`, where gamma = 0 is a true detach; `g = cat(t, k_embed(k))`. k_embed always gets the full gradient. Flag `student.dsh_grad_scale` (default 1.0 = current undetached behaviour, bit-identical). The legacy `dsh_detach` maps to gamma = 0 with k_embed attached.
- **Expected effect**: makes the gamma = 0 arm valid. Default behaviour is unchanged.
- **Accuracy risk**: none at the default. **Inference cost**: 0.
- **Test**: with gamma = 0, dL_s/d(encoder) = 0 while dL_s/d(k_embed) is nonzero; with gamma = 1, gradients match the current code exactly; the existing gradient-isolation test (dL_pix/d scale_mlp = 0) stays green.
- **Run**: the gamma sweep belongs to loss-report Phase 2, after this plan.

#### A12: A0'-g native-stride gradient matching (strides 4-32) (P1)
- **Location**: losses.py:51-76 (`::2` aliasing, batch-pooled, starts at stride 1 although the output is a bilinear upsample from stride 4; currently 2% of the loss).
- **Change**: flag `loss.grad_mode: {legacy, native4}`.
  - Masked pooling: `T4 = pool4(t*w)/pool4(w)`, `w4 = 1[pool4(w) >= .75]`.
  - Pyramid at strides 4/8/16/32 on `R = r_c - T4`, per-image normalised, in fp32, using the **untrimmed** own-row mask.
  - Set lambda_g so that ||d(lambda_g*L_grad4)/d r_c|| / ||d SILog/d r_c|| = 0.25 on a fixed probe batch, logged once.
  - This also removes the stride-1 staircase pull on 8-bit NYU targets (see B03).
- **Expected effect**: IBims d1_si +0.2 to +0.5 and DBE, raw about 0 to +0.3 (estimates).
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: a step edge reproduced at stride 4 costs less than a linear ramp; per-image normalisation keeps the loss invariant to how many images share the batch; legacy mode is bit-identical.
- **Run**: inside R4. Leave-one-out screen S-g (A0' without A12).

#### A13: A0'-h EG-SSI zero-MAD init guard (P1, no dedicated ablation)
- **Location**: student.py:192 (zero-init `head[-1].weight[:1]`); losses.py:87-89 (`scale = mad.clamp(min=1e-6)`), used by eg_ssi_loss:141.
- **Measured state**: at init r_c = 0 and p_al is constant, so the EG-SSI gradient is about 1e6. The first finite gradient norm is about 1e5. GradScaler skips about 15 steps (scale 65536 to 2) during warm-up.
- **Change** (flag `loss.ssi_pred_guard`, default off): drop EG-SSI rows whose **prediction** MAD is below 1e-4 (multiply the row weight by `mad_pred > 1e-4`). The target-side normaliser stays unchanged. Keep the spec-C1 zero init. Optionally `GradScaler(init_scale=2**12)`. No log-meaningful floor such as 1e-2: it would reweight near-planar crops late in training.
- **Expected effect**: hygiene only. No skipped warm-up steps, and the final metrics change negligibly. Ships inside R4. Report as an implementation detail.
- **Accuracy risk**: none. **Inference cost**: 0.
- **Test**: build DepthStudent(scale_head=True, pretrained_backbone=False) and run one AMP forward/backward of eg_ssi_loss at init. With the guard on: finite unclipped grad norm < 1e3 and at most 1 scaler skip. With it off: grad norm > 1e4, which documents the old behaviour.
- **Run**: none of its own. It rides in R4.

#### A14: A0'-e trim warm-up flag and trim unit tests (P2)
- **Location**: losses.py:493-530 (trim_mask_own), train.py:391-394.
- **Change**: `loss.trim_start_epoch` already exists; under `a0p` it stays at 1. Add the missing trim tests and a `trim_frac` log broken down by source.
- **Expected effect**: attribution only. Trim's effect on structure has never been ablated.
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test** (closes the "trim untested" item): trim only touches own rows; the realised fraction on own valid pixels is within 1 point of 10% once the quantile is computed on the full grid, or the 8.7% is documented; GT rows are never trimmed; the trimmed mask is never passed to EG-SSI's hole detector.
- **Run**: trim_start 3 and no-trim live inside B08.

### A15: Post-hoc PreciseBN on EMA weights (P1)
- **Location**: distill/train.py:30-47 (EMA blends BN running_mean/var), :483-507; new scripts/bn_recal.py (already named in B20_SPEC.md:400).
- **Change**: standalone script `--ckpt --out --n-batches 100-200 --bs 12 --seed`.
  - Load the EMA state. For every _BatchNorm: reset_running_stats(), momentum=None, affine unchanged.
  - Put only the BN layers in train mode. Run no_grad fp32 forwards on **augmented** rows from the training mix (own .4 / NYU .4 / SUN .2, same crop-context and FoV-uncrop augmentation).
  - Restore momentum. Save `best_pbn.pt`; never overwrite best.pt.
  - Run it before any rep4 reparameterisation or QAT fusion, and export ONNX/Core ML from the recalibrated checkpoint.
  - Clean rows are only an ablation: they regress IBims AbsRel .1792 to .1846.
  - Optional `--precise-bn` in train.py to recalibrate before each validate().
- **Verified on one checkpoint**:

  | Variant | IBims d1 / d1_si / AbsRel | NYU-300 d1 / d1_si |
  |---|---|---|
  | Original EMA | 74.83 / 86.50 / .1792 | 84.63 / 89.79 |
  | PreciseBN, augmented rows | 75.06 / 87.32 / .1764 | 85.07 / 90.35 |

  d1_ssi does not move, so do **not** claim this recovers the b20 structure regression.
- **Expected effect**: about +0.8 d1_si and +0.2 to +0.4 d1, for free. BN folds at export.
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: on a toy conv+BN model, cumulative recalibration equals the exact dataset mean/var; best.pt bytes are unchanged; the ONNX export of best_pbn.pt matches torch eval to 1e-5.
- **Run**: eval only. Use 3 recalibration seeds on full NYU-654, SUN and IBims with a per-image paired bootstrap. Adopt if metric d1 does not drop on any dataset and AbsRel is not significantly worse. Apply the same step to both arms of any final comparison. Optional diagnostic: also run it on b10 to separate batch-4 BN noise from EMA as the cause.

### A16: INTER_NEAREST_EXACT for every label resize (P1)
- **Location**: distill/dataset.py:81-82, :376-378 (fov), :501-504 (s5); distill/v2/data.py:451; scripts/v2/build_cover_cache.py:71; scripts/build_gt_dataset.py:56, :160, :163; scripts/v2/fuse_composite.py:455 (validity mask NEAREST next to LINEAR values: half-pixel mask leak); label_unidepth.py:277, teacher/run_teacher.py:100, the label_moge save chain.
- **Measured state**: OpenCV 4.13 INTER_NEAREST shifts labels 0.31-0.70 source px toward the top-left at each stage, while RGB uses AREA/LINEAR. The flip mirrors the x bias into jitter; the y bias stays systematic.
- **Change**: flag `label_interp: {nearest, nearest_exact}`, default nearest. Under nearest_exact, use cv2.INTER_NEAREST_EXACT for every depth/weight/pad/conf/validity resize. Rebuild the cover cache and SUN (inside A02). The 1024-px teacher PNGs keep one stage of offset unless they are re-saved; document this. Requires OpenCV >= 4.5.1.
- **Expected effect**: d1 within noise. Small gain on boundary metrics, prox F1 and the edge-weighted losses.
- **Accuracy risk**: none to low. **Inference cost**: 0.
- **Test**: a ramp test over the ratios actually used (853->640, 1024->853, 640->384, 448->640, 730->640, 1024->640) asserts |mean offset| <= 0.15 source px for NEAREST_EXACT (tie rounding gives about +0.1 at 1024->640, so 0.05 would fail) and that INTER_NEAREST **fails** the same test. Also run the full online chain (s5 up, fov down, flip) and assert a label-vs-RGB offset below 0.15 dst px on both axes.
- **Run**: bundled into R3 as a data-correctness fix. No standalone ablation.

### A17: Closed-form ray map (about -40 ms on torch CPU) (P1)
- **Location**: distill/student.py:196-207.
- **Change**: build fx,fy,cx,cy as (B,1,1,1) views; `x = ((u-cx)/fx.clamp(min=1e-4)).expand(B,1,H,W)`, same for y; `inv = torch.rsqrt(x*x + y*y + 1)`; return `cat([x*inv, y*inv, inv], 1)`. Drop the redundant 1e-6 clamp, since the norm is always >= 1. The cause is the slow dim-1 vector-norm reduction on this torch CPU build (it is slow on contiguous tensors too), not expanded views.
- **Expected effect**: _build_rays 40.1 ms to 0.21 ms, max diff 6e-8. Only PyTorch-CPU paths benefit (eval, demo, ExecuTorch). The reported ORT 33.8 ms and Core ML numbers are unchanged; do not claim a deployment gain.
- **Accuracy risk**: none. **Inference cost**: -33 to -44 ms torch CPU.
- **Test**: parity with the old formula (atol 1e-6) for default K and random K, B > 1, odd H/W; ONNX export still matches torch.
- **Run**: none.

### A18: One shared inference preprocess with an aspect policy (P1)
- **Location**: benchmark/run_bench.py:113-121, benchmark/gt_eval.py:106 and :110 (always default_intr), distill/evaluate.py:32, distill/predict_objects.py:47, edge/ort_int8.py:40 (+ onnx_runner), edge/export_coreml.py:99-104 vs dataset.py:323-359 and :451-461 (ownar: square pixels).
- **Measured state**: inference stretches to 640x480. A 16:9 frame then has fy_px/fx_px = 1.33, fx_n/fy_n = 0.56 (training is always about 0.75) and c about 0.144 log off. KITTI is worse (fx_n/fy_n about 0.30).
- **Change**: new module (for example `distill/preprocess.py`) used by all of the paths above, with `--aspect {stretch (default, bit-identical), budget, crop}`.
  - **budget**: isotropic resize to the largest shape with H*W <= 307,200 and both sides multiples of 32. 4:3 gives 480x640 (identical to today), 16:9 gives 416x736 (0.306 MP), 3:1 KITTI gives about 288x960. Scale K with the image and normalise by the new size. Upsample the prediction back for metrics.
  - **crop**: reuse `_crop_or_pad_to(train=False)`, as a diagnostic only.
  - Core ML: export EnumeratedShapes or one fixed model per bucket (4:3, 16:9, 3:4, 9:16), not RangeDim; check ANE residency per shape.
  - Also fix gt_eval to pass the true K when it is known.
- **Expected effect**: no change to the 4:3 tables. Own-val evidence for 16:9 is mixed (d1 +5, AbsRel +0.005). Adopt budget as the default only if it is no worse on AbsRel and better on d1 or OCMR on wide sets.
- **Accuracy risk**: low. **Inference cost**: 0 (same or fewer pixels).
- **Test**: the 4:3 input under budget is bit-identical to stretch; the 16:9 shape is 416x736 with fx_px/fy_px within 1% of 1; normalised K is unchanged under the isotropic resize; one preprocess object is imported by every listed caller (grep test).
- **Run**: eval only. Compare stretch, budget and crop on own frozen-test (16:9 subset), OCMR-KITTI and the 4:3 sets (sanity), with a paired bootstrap.

### A19: Optional guided-upsampling post-process (+GF row) (P1)
- **Location**: distill/student.py:311-312 (bilinear x4 of r_c); an eval/export flag.
- **Change**: flag `--post {none (default), gf, gup}`.
  - `gf` is the measured variant: guided filter r=8, eps=1e-3, gray guide from the 480x640 input, on clamped log-depth. Use separable 1x17 and 17x1 AvgPools with reflect-equivalent borders.
  - `gup` is the ANE-preferred form: fast guided upsampling of r_c. Compute a and b at 120x160 with r=2 on the downsampled gray guide, bilinear x4, then a*I + b at full resolution, then add s_hat + c. This replaces the bilinear x4, leaves the scale path untouched and avoids fp16 cancellation on log-depth.
  - Ship as a separate graph variant or output. Report as a separate "+GF" row; the unfiltered row stays primary.
- **Verified for `gf`**: IBims bF1v2 .0087 to .0373 (4.3x), d1 74.84 to 74.74, d1_ssi 91.18 to 91.14. `gup` is not yet measured.
- **Expected effect**: about +0.03 bF1v2, training-free. Keep only if it beats both control rows (nearest x4, and an unsharp mask tuned to the same d1) on bF1v2 and DepthPro bF1, with d1/d1_ssi CIs overlapping zero, and Core ML fp16 agrees with fp32.
- **Accuracy risk**: low (about -0.1 d1). **Inference cost**: about +1 ms.
- **Test**: with post=none, outputs are bit-identical; gf on a constant guide returns the input; ONNX/Core ML vs torch parity within the A03 gate.
- **Run**: eval only. Re-score the foundation models with v2 (A04) before stating any ratio.

### A20: 384x512 operating point, eval only (P1) - flag: may reduce accuracy - only adopt if the ablation shows no loss
- **Location**: cfg.student.train_size; StudentModelRunner (run_bench.py:94); student.py:48-54; B20_SPEC section 5 Step 5 (planned, never run: scripts/set_train_size.py and scripts/bn_recal.py do not exist).
- **Change**:
  - Add an eval-only `--hw` override, or set_train_size.py to copy the ckpt with train_size=[384,512]. K stays normalised to the original image.
  - Run run_bench on IBims, NYU and SUN at 384x512 and 480x640. Report d1, d1_si, d1_ssi, AbsRel and the per-dataset median log(pred_384/pred_480).
  - Diagnostic: compute c as if W were 640. This shows whether s_hat follows visual scale or normalised K. Normalised K is collinear with c in training, so expect a scale bias of about 0.08 log, not exact absorption.
  - If the d1 drop is above 1, try BN-only recalibration at 384 (A15 script). It will not fix a K-driven bias.
  - Measure latency on ORT (4 threads) and on the Core ML ANE with a fixed 384x512 graph, not torch eager.
  - Ship 384x512 as an optional "fast" graph only if IBims d1 stays within 1 point. Never replace the default.
- **Expected effect**: about -36% MACs and -34% torch CPU latency, at an accuracy cost still to be measured. Gives the paper an accuracy-latency curve.
- **Accuracy risk**: medium (one verifier). **Inference cost**: -36% MACs.
- **Test**: the override leaves the checkpoint untouched; c shifts by exactly log 0.8 at 384x512; at 480x640 the output is bit-identical.
- **Run**: eval only. Do not run the multi-resolution fine-tune (see Not worth doing).

### A21: Core ML uint8 ImageType input, normalisation in graph (P2)
- **Location**: edge/export_coreml.py:166, :79-82, :99-104, :184; distill/dataset.py:25-26.
- **Change**: flag `--image-input`; default is the current TensorType path, which stays identical to ONNX.
  - The wrapper registers mean/std as (1,3,1,1) buffers and computes `x = image*(1/(255*std)) - mean/std` before the student.
  - Declare `ct.ImageType(name='image', shape=(1,3,H,W), color_layout=RGB, scale=1.0, bias=[0,0,0])`. ImageType cannot express a per-channel std, so the multiply-add stays in the graph. Do not fold into the stem: zero padding would then mean black, not the mean colour.
  - Parity feeds PIL uint8 images (a seeded random one plus the parity set). Record `input_kind`.
  - Document that the app must resize to exactly HxW (Vision `.scaleFill`) or rescale K. With `--fixed-k`, a centre crop silently breaks metric scale.
  - Optionally add the same flag to export_onnx.py.
- **Expected effect**: removes the host float conversion and the 3.7 MB float buffer. The resize still remains. Measure end-to-end time on device rather than quoting 1-3 ms. No model, GMAC or ANE-kernel claim.
- **Accuracy risk**: none (fp16 normalisation error is inside the gate). **Inference cost**: one in-graph elementwise op.
- **Test**: in torch, the wrapper on a uint8 tensor equals the student on the normalised tensor (atol 1e-6); the default export is byte-identical to the current spec.
- **Run**: none. Needs the Mac.

### A22: ORT int8: opt-in mixed precision plus collapse gate (P2)
- **Location**: edge/ort_int8.py:90-110 (head_nodes), :141-147 (quantize_static over all op types).
- **Measured state**: default int8 pins the output at the clamp (log 30 m everywhere). Conv+Gemm only, with the early convs kept float, reaches 99.5% of pixels within 1.25x of fp32 at 13.4 MB (-36%). Div is not quantized by ORT by default.
- **Change**:
  - Default unchanged. Add `--mixed`: `op_types_to_quantize=['Conv','Gemm']`, and `nodes_to_exclude` every Conv whose initializer name starts with student.encoder.stem_res0, stages_0, stages_1 or stages_2.blocks.0 (select by name, not topological index).
  - Add an agreement gate that runs in both modes. It reuses int8_agreement on held-out own rows and marks the JSON failed when fewer than 95% of pixels are within 1.25x or the output is constant.
  - In the paper, give the root cause: the LiteMLA normaliser range plus the high-activation early convs. Keep weight-only W8 (edge/weight_int8.py, Core ML --w8; about 4x smaller) as the size and deployment result.
- **Expected effect**: no silent garbage int8 builds, and an explained analysis result. No latency gain (x86 44.1 vs 44.8 ms) and no size gain over W8.
- **Accuracy risk**: none. **Inference cost**: n/a (opt-in artefact).
- **Test**: the gate rejects a constant output map; the mixed mode excludes exactly the named initializers (graph inspection); default mode produces the same node list as today.
- **Run**: none (CPU only).

### A23: Training throughput flags plus per-group grad-norm logging (P1)
- **Location**: distill/train.py:287-297 (loaders), :38-44 (Python EMA loop), :474-475 (float() sync on every term every micro-iteration), :467-469 (clip, no logging).
- **Change**:
  - Flag `--fast-io` (default off, because cudnn.benchmark and persistent workers change RNG and kernel choice): torch.backends.cudnn.benchmark=True, persistent_workers=True, prefetch_factor=4, loss terms kept as detached GPU tensors with .item() only at the log interval, and EMA via `torch._foreach_mul_/_foreach_add_`.
  - Always-on logging every 50 steps (logging only): pre-clip grad norm in total and per group (encoder; decoder+film+head; scale_mlp+k_embed), clip-fire fraction, GradScaler scale, skipped-step count, and s_hat_err per source. This closes loss-report diagnostic D5.
- **Expected effect**: probably 10-20% shorter runs (bs4x3 is mostly compute-bound, about 219 of about 250 ms per step), plus the diagnostics that B04 needs.
- **Accuracy risk**: none. **Inference cost**: 0.
- **Test**: foreach EMA equals loop EMA bitwise on CPU; GPU-accumulated terms equal float accumulation to 1e-6; with the flag off, loader kwargs equal today's.
- **Run**: turn it on from R1 onwards and measure steps/s.

### A24: Export plumbing: derive SCALE_HEAD_SHAPES, out_indices field (P1)
- **Location**: edge/export_coreml.py:52 (hard-coded (128,496)) and the "exactly one linear op" assertion around :118; edge quant_sim.py / weight_int8.py FLOAT_KEEP_PREFIXES; student.py:98-104 (no out_indices).
- **Change**: derive the kept-fp16 shape from `tuple(student.scale_mlp[0].weight.shape)`. Add `StudentConfig.out_indices` (default None = today's call to timm), passed through to timm.create_model. This prevents silent quantisation of the scale head under any C03/C05/B06 variant.
- **Expected effect**: no change for b20dws. Required before C03 and C05.
- **Accuracy risk**: none. **Inference cost**: 0.
- **Test**: the derived shape is (128,496) for b20dws; a config with a different g_dim yields the new shape and the scale head stays fp16 in the W8 path.
- **Run**: none.

---

## 2. Tier B: training-recipe ablations

All B screens run on top of the new baseline **B\*** = R4 config (clean split + SUN fix + A16 + A0' + A23 + seeded). Rules:
- 15-epoch screens use a **compressed** 15-epoch cosine and are compared only with a 15-epoch B\* control.
- Promote an arm only if it clears the acceptance rule in section 5.

### B01: Batch 12x1 instead of 4x3, optional late BN freeze (P1)
- **Location**: run_info batch_size 4 / grad_accum 3; student.py:300-304 (GAP of BN features); train.py:352-355 (the dshft BN-freeze pattern to reuse).
- **Change**: arm (i) batch_size 12, grad_accum 1, same lr 3e-4, warmup and 1000 steps/epoch. Arm (ii) = (i) + BN layers in eval mode (affine still trainable) for the last 25% of epochs. No LayerNorm on g, no bf16 (see Not worth doing).
- **Measured state**: the train-vs-eval s_hat difference is 0.036 log, but eval-mode |s_hat - s*| is 0.0678 vs 0.0689 in train mode. Inference is therefore not limited by BN noise, and loss-report gate D3 (gap > 0.02) is not met. Step time 219 to 162 ms at 3.56 GiB. This setting also aligns with V2_PLAN's batch 12.
- **Expected effect**: about 26% less wall-clock per run (more seeds per week). Accuracy change speculative, 0 to +1.5 IBims d1.
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: config parse gives the effective batch of 12 for both settings; the BN-freeze hook leaves affine params trainable and running stats unchanged after the freeze epoch.
- **Run**: screens S-B01i and S-B01ii (15 epochs). Report IBims d1/d1_si/bias_log, NYU, SUN and per-source eval-mode s_hat_err. If neutral or better, adopt for speed in the final stack.

### B02: Student-side photometric aug, blur/zoom decoupling (P1)
- **Location**: dataset.py:521-524 (gain +/-15%, bias +/-10 only), :467-506 (s5 crop U(0.5,0.9), then upsample with INTER_AREA), :500 and :506; build_cover_cache.py:56-61.
- **Change** (flags, default off):
  - A0 (in every arm): INTER_LINEAR when upscaling at dataset.py:500/:506; INTER_AREA only for shrinking.
  - A1 "decouple": on all samples, Gaussian blur sigma U(.3,1.2) at p=.25 plus a random down-up resample with scale U(.6,1) at p=.2. Softness then no longer marks a crop.
  - A2 "aug2-mild": A1 + ColorJitter(.3,.3,.3, hue .03) at p=.5, gamma U(.8,1.3) at p=.2, Gaussian noise sigma U(0,5)/255 at p=.2, JPEG q U(50,95) at p=.2. Student input only, after geometric augs and flip; refill pad pixels afterwards. No grayscale.
  - A3 (only if A1 or A2 helps own rows): rebuild the own cover cache at short side about 576 (the label ceiling); `_crop_or_pad_to` must then resize rather than crop, so the FoV distribution does not change.
  - Leave the s5 range alone here.
  - The justification is the texture/domain gap. It is not a K shortcut: K enters exactly.
- **Expected effect**: +0.2 to +1.5 IBims d1 is plausible from the literature but speculative; better robustness on phone JPEG, blur and low light (the V2_PLAN SF5 gate).
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: augs never touch depth, valid or pad; pad pixels equal the canonical fill afterwards; flags off give bit-identical samples for a fixed seed (needs A06).
- **Run**: screens S-B02a1 and S-B02a2, plus a perturbed-IBims eval (blur, JPEG q40, low gamma, noise). Accept an arm only if IBims d1 improves by at least 0.5 beyond the seed spread and NYU/SUN drop by no more than 0.3.

### B03: NYU white border as pad, plus a quantisation check (P2)
- **Location**: scripts/build_gt_dataset.py:65-80 (HF vikhyatk/nyu_depth_v2: per-image 8-bit depth, a 5-8 px saturated-white border with valid labels); dataset.py:449-531.
- **Change**:
  - Flag `nyu_border: {keep, pad, crop}`, default keep. `crop`: rows 6:474, cols 7:632; cx -= 7, cy -= 6, fx/fy unchanged in px; then the normal resize+K path. `pad`: mark the frame as pad AND overwrite the RGB with the noise canvas. Loss masking alone leaves the input cue in place.
  - Add a build-time warning or test when the median unique-value step equals max/255.
  - Disclose in the paper that NYU train **and eval** GT come from the 8-bit repack.
  - The stride-1 staircase bias in gradient matching is already removed by A12 (native4 starts at stride 4), so no separate NYU-only gradient gating is needed.
- **Expected effect**: removes about 2.4% of NYU pixels supervised on saturated input and a possible dataset-ID cue. Plausibly +0.2-0.5 NYU d1_si (unmeasured).
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: crop mode updates cx/cy exactly and leaves no pixel with RGB > 250 in the outer 5 px; keep mode is bit-identical.
- **Run**: screen S-B03 (`crop`). Optionally a linear dataset-ID probe on the GAP features before and after.

### B04: Optimizer param groups (P2) - flag: may reduce accuracy - only adopt if the ablation shows no loss
- **Location**: distill/train.py:300-301 (one AdamW group, lr 3e-4, wd .01), :467-469 (global clip 1.0); student.py:189.
- **Change** (flag `optim.groups`, default off):
  - Arm (b): no weight decay for p.ndim <= 1, BN/LN affine, k_embed and scale_mlp[-1].bias; same lr everywhere; clip **per group** at 1.0 (scale head = scale_mlp + k_embed; everything else = trunk).
  - Arm (c): (b) + encoder lr 0.5x (1.5e-4). The stem conv (adapted ray channels), FiLM, decoder and scale head stay at 3e-4.
  - Do not raise the global clip. Do not go to 0.1x encoder lr.
  - Decide whether clipping matters at all from A23's clip-fire log first. Adam is invariant to a constant gradient scale, so expect small effects.
- **Expected effect**: 0 to +1 IBims d1 (low confidence).
- **Accuracy risk**: low to medium. **Inference cost**: 0.
- **Test**: group membership covers every parameter exactly once; no-wd params match the ndim/name rule; flag off gives the identical single group.
- **Run**: screens S-B04b and S-B04c. Adopt only if the IBims gain has the same sign in 2 seeds and NYU d1_si is within 0.3.

### B05: Virtual-focal augmentation, gated on diagnostics (P2) - flag: may reduce accuracy - only adopt if the ablation shows no loss
- **Location**: dataset.py:462-497 (K changes only together with zoom); student.py:226-241, :296-306.
- **Measured state**: on NYU the K-only slope is 0.61 (FiLM about -0.06 log, k_embed about -0.03, rays about 0). log fx_n and log fy_n in k_embed duplicate c at a fixed input size.
- **Change**:
  - Step 1, free diagnostics on the existing checkpoint:
    - (i) per-path K-only decomposition;
    - (ii) the on-manifold crop+K zoom slope, with default upsampling and with a sharpened or downsample-matched version;
    - (iii) a regression of per-image log scale error on log f_px across SUN sensors (after A02) and DIODE-indoor (report only).
  - Step 2, only if (iii) shows |slope| > 0.15 or (ii) is below 0.85 on sharp inputs: flag `aug.vfocal`. After all geometric augs, with p about 0.15 and a = exp(U(-0.1, 0.1)): fx,fy *= a; depth *= a; recompute valid against dmin/dmax; scale box_dists by a if box terms are on; image unchanged. Describe it as a CSTM-label-consistent depth stretch, not an exact pinhole identity.
  - Keep FiLM and k_embed. An optional sub-arm feeds k_embed only (cx-.5, cy-.5).
- **Expected effect**: slope toward 0.9 or above, so scale transfers better to new cameras. IBims anywhere from -1 to +2. Possible in-domain cost if K currently acts as a dataset ID.
- **Accuracy risk**: medium (both verifiers). **Inference cost**: 0.
- **Test**: under vfocal the CSTM canonical depth target (logD - c) is unchanged by the augmentation; valid respects dmax after scaling; flag off is bit-identical.
- **Run**: screen S-B05 only if the gate passes. Gate on IBims, SUN per sensor and own-val. NYU d1 may not drop by more than 0.3.

### B06: EfficientViT-B1 r288 / r256 pretrained weights (P2) - flag: may reduce accuracy - only adopt if the ablation shows no loss
- **Location**: student.py:23; run_info cfg.student.backbone.
- **Change**: backbone='efficientvit_b1.r288_in1k', with r256 as an optional second arm. Same architecture, ImageNet top-1 80.4 vs 79.4, pretrained at 324 tokens vs 196 for r224. Needs a weight download from the HF hub, which the user must approve.
- **Expected effect**: 0 to +0.5 d1. Fine-tuning at 480x640 largely erases the pretraining-resolution prior, so expect it within seed noise.
- **Accuracy risk**: low to medium. **Inference cost**: 0.
- **Test**: the state_dict loads with the same keys and shapes as r224; feature_info channels are unchanged.
- **Run**: screen S-B06 over 2 seeds. Adopt only if the IBims d1 gain exceeds the seed spread.

### B07: M7-S0 GT-anchored own-row scale rectifier (from the loss report core) (P2)
- **Location**: new scripts/cache_teacher_scale.py; dataset.py:401-405 under tag `tsp`.
- **Change**: s*_own <- s*_own - a_bar, where a_bar is the per-sensor Huber mean of median(logD_L - logD_GT) on a SUN-train calibration split with NYUv2-origin frames removed. Apply it to s* directly, not through the csd depth rescale. **Depends on A02**: with teacher K and 8x rows, the calibration split is itself biased.
- **Expected effect**: 0 to +0.5 IBims (corrects the V2-family +0.021 to +0.026 log over-range).
- **Accuracy risk**: low. **Inference cost**: 0.
- **Test**: a_bar is computed on rows disjoint from every val/test role; with the tag off, s* is unchanged.
- **Run**: screen S-B07 after R3 exists. Needs one V2-L teacher pass on the calibration split (about 1-1.5 GPU-h offline).

### B08: b20 leave-one-out ladder (from the loss report) (P2)
- **Change**: remove one of {trim, fov, ownar, cstm} at a time from B\*. This is the only experiment that can attribute the b10 to b20 structure regression (d1_si -0.8 to -1.0) and the boundary-F1 loss (A04).
- **Expected effect**: attribution, not accuracy.
- **Accuracy risk**: n/a. **Inference cost**: 0.
- **Test**: none beyond existing tag parsing.
- **Run**: 4 screens at 15 epochs, 1 seed. Report bF1v2 and d1_si with paired CIs. Use 2 seeds only for an arm that moves bF1v2 or d1_si by more than 0.5.

### B09: noray / zero-init ray stem (P3)
- **Change**: only inside a retrain that is happening for other reasons. Either a 3-channel input with K only through FiLM, scale head and cstm, or ray input filters zero-initialised with full pretrained RGB stem weights instead of timm's repeat-and-halve.
- **Expected effect**: +/-0.3 d1. A simplification, not an improvement.
- **Accuracy risk**: low. **Inference cost**: -33 MMAC (noray).
- **Test**: the zero-init variant at step 0 gives encoder features identical to the 3-channel pretrained network.
- **Run**: optional extra arm in Phase 3 only if GPU time is left.

---

## 3. Tier C: architecture variants

### C01: DW-separable smooths at stride 8/16 (`dws_levels`) (P1)
- **Location**: distill/student.py:115-123; config builder around :374.
- **Change**: `StudentConfig.dws_levels` (tuple, default () = today; dws4 implies {0}). Arm `(0,1,2)`: smooths[1] and [2] become _DWSep(dc) with a 3x3 depthwise kernel. Keep smooths[3] (stride 32, 25 MMAC) dense, because it carries the global context that feeds the x_td GAP. Add a `dw_k=5` option only if the 3x3 version regresses boundaries. Persist the field to run_info so eval and export rebuild the same graph.
- **Measured state**: smooths.1 is 398 MMAC, the largest decoder op.
- **Expected effect**: -0.44 GMAC (about -11%), -0.17M params, a few % ORT CPU latency. The ANE gain is unknown: depthwise convs can run at low ANE utilisation, so measure before claiming.
- **Accuracy risk**: low. This touches the scale-head input (x_td), so check the IBims d1 minus d1_si gap and |s_hat - s*|. **Inference cost**: -0.35 to -0.46 GMAC.
- **Test**: the default config builds the same module tree and state_dict keys as today; dws_levels=(0,1,2) reports the expected MAC delta (+/-1%) at 480x640; old checkpoints still load.
- **Run**: screen S-C01 (15 epochs), then 2 seeds × 40 epochs if it passes. Gate: IBims d1 drop <= 0.3, no bF1v2 drop, NYU d1_si within paired CI. Report GMAC including the LiteMLA matmuls (4.09 convention).

### C02: Stride-2 residual refine head (P2)
- **Location**: student.py:128-129, :229-237, :311-312.
- **Change**:
  - `r2 = U2(r_c) + PW(ReLU(PW(ReLU(DW3x3(cat(U2(r_c), stem_proj))))))`.
  - stem_proj is 16 to 32 from encoder.stem_res0. features_only omits the stem, so take it via a hook or out_indices (A24).
  - Keep the full stride-4 _DWSep(96) head. Zero-init the last PW. Mean-centre the residual so the dsh gauge is preserved. Keep the scale head on x_td.
  - Supervise with native-stride gradient (A12) extended to stride 2. Check the ANE working set (32x240x320 fp16 = 4.9 MB).
- **Expected effect**: GT oracle caps at stride 4 are bF1v2 .081-.087 vs .257-.285 at stride 2, while the d1 ceiling moves at most +0.2. So the case is boundary-only: plausibly .04-.08 bF1v2.
- **Accuracy risk**: low. **Inference cost**: about +34 MMAC, +1-2 ms CPU; memory-bound, so measure.
- **Test**: at init (zero last PW) the output is bit-identical to b20dws; the residual has zero mean.
- **Run**: 40 epochs only after A19 is measured. Gate: bF1v2 at least +0.02 over A19 `gup`, d1 and d1_ssi not worse within paired CI, measured CPU and ANE latency.

### C03: Scale-head masked attention pooling, gated on a frozen-feature probe (P2)
- **Location**: student.py:183-186, :300-304; edge/export_coreml.py:52 (via A24).
- **Change**:
  - Step 1, a frozen-feature probe that takes minutes. Dump f4, f3 and x_td from the b20dws EMA checkpoint (eval mode) on the training mix and IBims, and fit three heads to s*: (a) the current GAP-MLP, (b) GAP + 2x2-pool, used only as a probe, (c) 1-2-query attention pooling initialised as GAP. Also verify the claimed f4 dominance via per-group ||W1[:,group] g_group|| on real data.
  - Step 2, only if (c) cuts IBims s_hat_err by at least 0.01 log beyond (a) and after A02/A10 have landed: 1-2 learned 64-d queries, W_k 256 to 64, logits `q^T W_k f4 / 8 + log(w + 1e-4)` with the pad weight w from A10. Init q = 0, so it starts as masked GAP. Concatenate with GAP f4. Keep the new params fp32 in the W8 paths.
- **Expected effect**: 0 to +2 IBims d1, speculative and low confidence.
- **Accuracy risk**: low. **Inference cost**: +17k params, about 0.02 MMAC.
- **Test**: with q = 0 and no pad, the pooled vector equals GAP to 1e-6; export keeps scale_mlp fp16 (A24).
- **Run**: probe in Phase 0. A 15-epoch screen only if the probe passes.

### C04: Encoder Hardswish to ReLU, gradual swap (P3) - flag: may reduce accuracy - only adopt if the ablation shows no loss
- **Location**: timm efficientvit_mit.py:766 (act_layer=nn.Hardswish, 30 modules); student.py:98-103.
- **Change**:
  - Step 1, gate: re-profile the real b20dws ONNX on an **idle** machine (4 threads, interleaved) to get the true HardSigmoid+Mul+Reorder share at the 33.8 ms baseline, and time a ReLU Core ML graph on the ANE (random weights are fine).
  - Step 2, only if the idle CPU gain is above 15% and CPU is a stated target: `enc_act: {hardswish (default), relu}`. Anneal `act = (1-a)*hswish + a*relu` with a from 0 to 1 over 5-10 epochs, warm-started from b20dws, with L2 feature distillation on feats[-1] and feats[-2] against frozen b20dws. Also try a partial arm: ReLU only in the stem and early blocks.
  - A cold swap is known to fail (V2_PLAN.md:670, A22).
- **Expected effect**: about -29% ORT CPU latency (verified interleaved, under load) at the same MACs. ANE is likely unchanged.
- **Accuracy risk**: medium to high. **Inference cost**: 0 MAC; CPU-only gain.
- **Test**: with a = 0 the model is bit-identical to b20dws; module count of swapped activations equals 30 (or the partial set).
- **Run**: 5-10 epoch screen only after the gate passes. IBims d1 and the d1 minus d1_si gap must stay within 0.3. Deferred by default.

### C05: Conv-only backbone (MNv4) in a separate b30 latency track (P3) - flag: may reduce accuracy - only adopt if the ablation shows no loss
- **Location**: student.py:23, :98-104; distill/v2 (CoDeQ MNv4 encoders already exist); edge/export_coreml.py:52.
- **Change**: no swap in the paper's main model. Open the b30 track only if the A03 Mac measurement shows LiteMLA falling off the ANE, poor ANE latency or fp16 drift. Then run a param-matched MNv4-Conv-M (about 5M, block-deleted CoDeQ-N encoder with the 960-ch head conv dropped) with the b20dws decoder, scale head, cstm and loss unchanged, paired seeds, and IBims d1 as the deciding metric. MNv4-Conv-S is a separate "tiny" row only. Cite B20_RESEARCH.md and V2_PLAN.md for the ANE argument.
- **Expected effect**: encoder-only ORT -20 to -30% (MNv4-M 44.6 vs 63.8 ms under load), a likely larger ANE gain (unmeasured), and no LiteMLA fp16 hazard. Depth accuracy unknown.
- **Accuracy risk**: medium. **Inference cost**: depends on the variant; the unslimmed MNv4-M is +2.7M params and +2.1 GMAC.
- **Test**: out_indices (A24) produce 4 levels with the expected channels; SCALE_HEAD_SHAPES is derived.
- **Run**: deferred, outside this schedule.

---

## 4. Not worth doing (with reasons)

| Proposal | Reason |
|---|---|
| Make mixed-precision ORT int8 the default, or present QDQ int8 as a latency/size result | x86 latency 44.1 vs 44.8 ms (no gain); weight-only W8 is about 4x smaller vs 1.7x; changing the default changes the meaning of existing int8 results. Kept only as opt-in (A22). |
| LSQ+ QAT for full W8A8 on the ANE | Retraining for a benefit that exists only on M4/A17 Pro. B20_SPEC already chose fp16 + 8-bit weights. |
| LiteMLA k/N monkeypatch; scaling the aggreg convs in addition to the qkv k rows | eps/N underflows to 0/0 in fp16 (28/40 to 63/120 images). Scaling aggreg is redundant (applies s^2). |
| Dedicated ablation or retrain for the step-0 EG-SSI gradient explosion; random-init the head | About 15 skipped steps out of 40k, all in warm-up at lr 1.5e-5, with no measurable path to final accuracy. Random init breaks spec C1 and still skips 4 steps. Fixed by flag inside R4 (A13). |
| Raise the global clip to about 10; the 3x3 encoder-LR x clip grid; 0.1x encoder LR | The rep4 run's scale loss diverged; the clip is the guard. Adam is scale-invariant, so the claimed throttling is second-order. 9 runs for a low-confidence effect. 0.1x comes from ViT/DINOv2 practice. |
| bf16 autocast without GradScaler | 7 mantissa bits worsen the known SILog cancellation unless every reduction is forced to fp32. |
| LayerNorm on g before scale_mlp | Eval-mode s_hat error is not BN-noise-limited (0.0678 vs 0.0689 train). |
| PreciseBN-then-freeze training arm (loss report M3) | Gate D3 fails: the train-vs-eval s_hat_err gap is about 0.001, below 0.02. The post-hoc A15 is the useful part. |
| PreciseBN on clean rows | Regresses IBims AbsRel .1792 to .1846. |
| DIODE-indoor as a selection/dev set; 25-epoch schedule screen; e25-e40 snapshot averaging | DIODE is reserved as a report-only zero-shot test (V2_PLAN.md:135, :363). The +0.0006 gt-val gap is on scene-overlapping val. Averaging needs BN recompute and adds a confound. |
| 3 seeds of every legacy b10/b20 variant | About 28 GPU-h that measures variance only. Seeds go to the final recipe and its clean baselines. |
| Remove FiLM or k_embed for efficiency; K-dropout now | 12.6k params (0.25%), one elementwise op; export already folds FiLM for a fixed camera; nofilm was rejected (0/3 judges). K-dropout could hurt the GT-K scale head; at most a later optional arm. |
| Virtual-focal augmentation without its diagnostic gate | Slope 1 is not obviously optimal for a small model; the probe is off-manifold; medium risk. Gated in B05. |
| 2x2 grid pooling for the scale head | Breaks crop and translation invariance under the crop-context and FoV-uncrop augs. |
| Anisotropic-stretch augmentation, a 480x848 graph, tiled crops, centre crop as deploy default | Undoes ownar; +33% or 2x pixels; centre crop drops 25-60% of the FoV (breaks OCMR). Budget mode (A18) keeps full FoV at the same pixel count. |
| Constant-folded conv_ray(default K); standalone noray ablation | +0.0024 AbsRel for under 1% ORT gain; noray is a simplification that only belongs in a retrain that happens anyway (B09). |
| Cold Hardswish to ReLU swap with plain fine-tuning | Documented failure (V2_PLAN A22); no ReLU ImageNet B1 weights exist. Only the gated gradual C04. |
| MNv4-Conv-M or FastViT-T8 as the paper's main backbone; MNv4-Conv-S as the main model | 7.77M / +50% GMAC changes the size class and confounds every b10/b20 row; FastViT-T8 expected -1 to -2 d1; MNv4-S likely loses several d1. Separate b30 track (C05). |
| Multi-resolution {384, 480} fine-tune; 384x512 as the new default | mres was rejected in B20_SPEC (BN/EMA pollution, +45% cost); 384 is an extra operating point only (A20). |
| Gaussian dequantisation of NYU targets or a float-source NYU rebuild, for now | Below d1 tolerance; the stride-1 bias is removed by A12; a rebuild changes the corpus and breaks comparability with every logged run. |
| DWS on smooths[3] (stride 32); removing the stride-32 lateral | Saves 22-32 MMAC and touches the global-context path of the scale head. |
| The guided filter as the headline metric configuration; folding ImageNet normalisation into the stem conv | GF costs about 0.1 d1 and is a known post-process, so it is a separate "+GF" row. Stem folding breaks border padding semantics. |
| Claiming any ANE latency, fp16 parity or DWS ANE gain before a Mac run | Never measured (B20_RESULTS.md:123, RESULTS_MASTER:124/209). |
| Loss-report negative controls as recipes: NLL/confidence weighting of the scale loss, the raw log-sum balance, a Barron adaptive pixel loss, the M12 near hinge | Measured or literature negatives (S6, Depth Pro 2, Lienen et al.); M12 is bounded to about 1% log. Table rows only. |
| Merge-stage drops: FPN stride-4 bilinear Resize micro-optimisation; PPM/ASPP | Speculative 1-2 ms with an unknown ANE cost model; LiteMLA already provides global context. |

---

## 5. Run schedule for one RTX 5070 Ti laptop GPU

Assumptions:
- A 40-epoch b20dws run takes about 3.3 h; a compressed 15-epoch screen about 1.25 h.
- A23 is expected to save 10-20% and B01 (bs12) about 26% more if adopted. The hours below are not discounted.
- Only one GPU job runs at a time. Never measure latency while a training job runs; use an idle machine (thermal and background load inflated earlier numbers by up to 78%).
- A night is about 10 h of queue time.

**Phase 0: code and eval only (days 1-3, about 4-6 GPU-h of evaluation, no training)**
1. Implement and unit-test A01-A24, every flag defaulting to today's behaviour, and run the full test suite.
2. Eval jobs on existing checkpoints:
   - A04: re-score all rows, including the baselines.
   - A01: clean NYU-604 and video_own-21 re-eval.
   - A05: K-less re-eval.
   - A15: PreciseBN, 3 recalibration seeds, full NYU/SUN/IBims.
   - A20: 384x512 eval.
   - A18: stretch/budget/crop eval.
   - A19: GF, gup and both controls.
   - A03: fp16 emulation over full IBims.
   - A22: mixed int8 agreement.
3. Free diagnostics:
   - loss-report D1, D2, D4, D7, D8, D9 (D3 already answered, D5 via A23);
   - B05 steps (i)-(iii);
   - C03 frozen-feature probe;
   - C04 idle re-profile.
4. Build the A02 SUN mapping and `manifest_sungt.csv` (CPU), and rebuild the cover cache under A16.
5. Whenever a Mac is available: A03 Core ML export, parity gate and ANE latency; A21.

**Phase 1: new clean baselines (nights 1-2, about 17 GPU-h)**

| Run | Config | Epochs | GPU-h |
|---|---|---|---|
| R1 | b20dws-clean: A01 + A06 seed 0 + A23 | 40 | 3.3 |
| R2 | b10-clean: A01 + A06 seed 0 + A23 | 40 | 3.3 |
| R3 | R1 + A02 (SUN GT-K + raw decode) + A16 | 40 | 3.3 |
| R4 = B\* | R3 + A0' bundle (A07 a2, A08, A09, A10, A11 gamma=1, A12, A13) | 40 | 3.3 |
| S-ctl0, S-ctl1 | B\* control, seeds 0 and 1 | 15 | 2.5 |
| S-a1 | B\* with A07 arm a1 | 15 | 1.25 |

- R1 vs R3 isolates the data-correctness fixes; R3 vs R4 isolates A0'. Both are single-seed, so confirm in Phase 3.
- Apply A15 to every 40-epoch checkpoint at eval time (both arms equally) and report pre- and post-PreciseBN.

**Phase 2: Tier B/C screens (nights 3-4, about 17 GPU-h)**, each 15 epochs, seed 0, against S-ctl0/S-ctl1:
- Night 3: S-B01i (bs12, about 0.95 h), S-B01ii (bs12 + late BN freeze), S-B02a1, S-B02a2, S-B03, S-B04b, S-C01 (dws_levels 0,1,2). About 8.5 h.
- Night 4: S-g (A0' minus A12), S-B04c, S-B06 (r288, if the download is approved), S-B07 (after the calibration pass), S-B05 (only if its gate passed), S-C03 (only if the probe passed), then seed-1 repeats of the two best arms. About 7.5-8.5 h.

Screen stop rule: drop an arm that loses more than 0.5 NYU d1_si, raises the clip-fire rate above 30%, or regresses bF1v2.

Screen promote rule: IBims raw d1 at or above control with the per-image paired bootstrap favouring it, NYU/SUN d1 and d1_si within 0.3, the same sign in 2 seeds, and the mechanism metric moving as predicted (for example s_hat_err for scale-side arms).

Selection signals: own-val (Roboflow-dedup per A06) and NYU/SUN val. IBims is the test set; if it is used for screening, say so in the paper.

**Phase 3: confirmation (weekend, about 28-32 GPU-h)**
- Final stack F = B\* + promoted arms (bs12 if adopted cuts each run to about 2.5 h): 3 seeds × 40 epochs, 7.5-10 h.
- B\* seeds 1 and 2 × 40 epochs, 6.6 h. This gives the 3-seed A0'/data-fix row.
- b10-clean seeds 1 and 2 × 40 epochs, 6.6 h, for the headline b10 vs b20 comparison with seed statistics.
- B08 ladder (-trim, -fov, -ownar, -cstm) × 15 epochs on B\*, 5 h. Optional, but it is the only attribution of the structure and boundary regression.
- If C01 passed its screen: C01 × 2 seeds × 40 epochs, 6.6 h.
- Report:
  - mean ± std over seeds next to the per-image bootstrap;
  - bF1v2 and DepthPro bF1 (A04);
  - PreciseBN rows (A15);
  - the 384x512 point (A20);
  - ORT latency (idle, 4 threads), plus Core ML/ANE latency only if measured on the Mac.

**Later (outside the core schedule)**: C02 (40 epochs, after A19), loss-report Phases 2-4 (gamma/beta sweep, CPSD vs DCST, PVS with L_rel) on top of F, C04 (only if its gate passes), and the C05 b30 track (only if the Mac measurement calls for it).

Total: about 60-66 GPU-h, roughly 1.5-2 weeks of nights plus one weekend.

**Hard ordering constraints**
- A01 before any number goes into a paper table.
- A02 before B07, B05 step (iii) and C03.
- A03 before any on-device latency or accuracy claim.
- A04 before the dws4/C01 decision.
- A24 before C03, C05 and B06 exports.
- A06 seeds in every run from R1 onwards.

No files were written to the repo. The plan is based on the verified findings and the two loss-audit files, which were read only.