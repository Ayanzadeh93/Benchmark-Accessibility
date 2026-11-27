# B20 evaluation fixes

Evaluation-side changes only. Nothing here touches training, losses, training data loading,
the SUN/NYU training data, SUN intrinsics or train/test splits (`distill/train.py`,
`distill/losses.py`, `distill/dataset.py`, `distill/v2/*`, `scripts/build_gt_dataset.py` and
the training configs are unchanged). No model was retrained. Existing `results_bench/*.json`,
checkpoints and caches are not modified. New results go to `results_rescore/`.

## A04 / A05 (2026-11-25, branch b20-lossfix, spec: docs/B20_REVIEW_PLAN.md A04, A05)

### A04: corrected boundary F1 in `benchmark/run_bench.py --boundary`

| What | Detail |
|---|---|
| Legacy key | `boundary_f1_si` is computed the same way as before (`protocol.boundary_f1_si`, mean over the images that return a value). A new `boundary_f1_si_legacy_note` key marks it as LEGACY. |
| New per-image rows | `boundary_f1.per_image`: `{id, bf1_si_legacy, bf1_si_v2, bf1_depthpro_masked, bf1_depthpro_nomask}`. `bf1_si_v2` is `protocol_v2.boundary_f1_si_v2` (t = 5..25 %, `both_empty='zero'`). `bf1_depthpro_*` is `protocol_v2.si_boundary_f1_depthpro` with `valid=mask` (a masked extension) and with `valid=None` (the DepthPro reference). `None` means the image does not contribute. |
| Aggregates | `boundary_f1.aggregate[<key>]` = `protocol_v2.aggregate_boundary_f1` -> `{bf1, n_images, n_contributing, n_no_gt_boundary}`. `boundary_f1.definitions` names each metric. |
| Helpers | `run_bench.boundary_row(id, gt, pred, valid)` and `run_bench.boundary_summary(rows)`. |
| New flags (all opt-in) | `--out-dir DIR`: the JSON and `summary.csv` go to DIR instead of `results_bench/`. `--device cpu\|cuda`: the default is still "cuda if available". |
| Default behaviour | Without `--boundary` the output is unchanged. With `--boundary` the old keys and values are unchanged and the new keys are added. |
| Comparison script | `python -m scripts.rescore_compare --dir results_rescore` writes `a04_a05_comparison.{json,md}`. It covers legacy (with the stored results_bench value), v2 and DepthPro bF1 with n_contributing. It also restates the dws4 gate as bF1v2 with a paired image bootstrap (`pair_rows_by_id` + `paired_bootstrap_clustered`, accept when the 95 % CI contains 0). |

### A05: one K resolution path (`k_none_policy`)

| What | Detail |
|---|---|
| `distill/student.py` | Adds a module-level `DEFAULT_K = (0.8, 1.08, 0.5, 0.5)`. `DepthStudent._DEFAULT_K` is the same object. Adds `K_NONE_POLICIES = ("legacy", "default_k")`, `StudentConfig.k_none_policy` (default `"legacy"`), `DepthStudent.set_k_none_policy()`, and `build_student` reads `student.k_none_policy` (default legacy). The policy is a plain attribute, so the state_dict and the init RNG stream are unchanged. |
| `legacy` (default) | Same code path as before. With K=None, FiLM is skipped while rays, k_embed and cstm fall back to DEFAULT_K. |
| `default_k` | With K=None, K is resolved once at the top of `forward` (`_k_or_default`) and used for rays, FiLM, k_embed and cstm, so `forward(x, None) == forward(x, DEFAULT_K)` exactly. When a K is given, outputs are identical under both policies. |
| `benchmark/run_bench.py` | `--k-none-policy {legacy,default_k}` (default legacy) applies to torch students (`student:`, `hybrid:`, `q:`). Non-legacy runners get a `+kdef` name suffix so they never share a filename with a legacy row. `--no-k` does not pass the dataset K to the models (an "assumed default K" eval) and adds `__noK` to the filename; GT K is still used for the 3D metrics. `--no-k` together with `--gt-cam` is rejected. The fusion parity K now uses `DEFAULT_K` (same values). |
| `benchmark/nyu_eval.py` | `--student-k {none,gt,default}` (default `none` = historical K-less call). `gt` passes the NYU GT K (`datasets_tier1.NYU_K`, normalised by the original image size). `default` passes DEFAULT_K explicitly. Adds `--out`. For a non-default mode the default output name gets a `__k<mode>` suffix, so `eval_nyu_gt.json` is never overwritten. |
| `distill/visualize.py` | `--student-k {none,default}` (default `none`). `default` passes DEFAULT_K explicitly. |
| `benchmark/safety.py` | `--camera-mode default-k` now builds students with `k_none_policy='default_k'` (name suffix `+kdef`), so it is no longer identical to `none`. `none` (the default) and `gt-k` are unchanged. |
| Not changed (out of scope: training or export code) | The plan also asks to import DEFAULT_K in `distill/dataset.py`, `distill/v2/data.py`, `distill/v2/model.py`, `edge/ort_int8.py` and `edge/export_v2.py`. Those files are training or export code and were not touched. `v2/model.py`, `ort_int8.py` and `export_v2.py` keep literal copies with equal values. `dataset.py` and `v2/data.py` define a different `DEFAULT_INTR = (0.7, 0.9, .5, .5)`. Whether any training row reaches that fallback was not checked, because that is training scope. The default stays `legacy`. Flip it only after the legacy numbers are archived. |
| Labelling | A K-less column is "assumed default K", never "uncalibrated". |

### Tests (run each file on its own: `python -m pytest -q -p no:cacheprovider <file>`)

| File | Covers |
|---|---|
| `tests/test_boundary_v2.py` | An edgeless prediction scores 0 under v2 (not None) and is counted. `n_contributing` equals the image count. An image with no GT boundary is excluded and counted. Rows match the protocol functions exactly. On IBims-1, b20dws: the legacy key equals the ORIGINAL code (branch b20-canonscale, `git archive` into tmp) bit-for-bit on the same device, and equals the stored results_bench value (0.0318484) within 1.5e-4 (see note). |
| `tests/test_k_policy.py` | `default_k`: `forward(x, None) == forward(x, DEFAULT_K)` exactly (B = 1, 2) for b10 and every b20 config (core, dws4, rep4, cstm-only, FiLM-only) plus the real b10/b20/b20core/b20dws/b1_mix checkpoints. `legacy` equals the original `distill/student.py` from b20-canonscale for K=None and GT K. GT K is identical under both policies. K=None differs between policies exactly when FiLM is live. State_dict and init are unchanged. A bad policy is rejected. There is one DEFAULT_K object. nyu_eval K modes are covered. safety `default-k` builds `default_k` students. run_bench runner name and GT-K equality are checked. |
| Bit-parity guard | `python scripts/parity_golden.py compare --ref <golden>`: PASS 296/296 (b10, b20core, b20dws, b1_mix; random + IBims inputs; GT K and K=None; tensors and metrics). |

Note on the stored legacy bF1: `results_bench/ibims__student-b20dws_last.json` stores
0.03184836 (2026-11-23). On 2026-11-25 the original, unchanged code gives 0.03188149 on CUDA and
0.03176185 on CPU, and the new code gives exactly the same values on each device. The
difference comes from the numerical stack (depth metrics move by about 1e-5), not from the
code. The legacy metric thresholds per-pixel ratios, which amplifies that drift.

### Rescoring results (IBims-1, `results_rescore/`)

See `results_rescore/a04_a05_comparison.md` (generated by `scripts/rescore_compare.py`) and
the run logs in `results_rescore/logs/`.

### Review follow-ups (minors and the nomask major)

| Item | Resolution |
|---|---|
| DepthPro `nomask` on datasets with GT holes | DepthPro clips inverse depth at 1e-6, so GT holes (depth 0) become 1e6 and every hole border becomes a GT edge no prediction can match. Headline = `bf1_depthpro_masked` for IBims/NYU. `bf1_depthpro_nomask` is labelled "DepthPro reference, unmasked; GT holes counted as edges" in the JSON definitions and in the comparison table, and is only a headline number on dense-GT sets. |
| Legacy aggregate field name | For `bf1_si_legacy`, a None also means "prediction has no edge". The aggregate now also carries `n_dropped` (= `n_no_gt_boundary`, kept for key-set stability) plus a note: for legacy it counts every dropped image. |
| Stored legacy value drift | The stored `results_bench` legacy bF1 (0.03184836) is not reproduced bit-for-bit by today's stack, even with the original code (0.0318815 cuda / 0.0317619 cpu). Do NOT compare old JSON values to new ones at 1e-4 precision. The test uses a 1.5e-4 tolerance against the stored value; bit-exactness is checked against the original branch run on the same device. |
| `--boundary` JSON keys | With `--boundary`, the JSON gains additive keys (`boundary_f1`, `boundary_f1_si_legacy_note`) and the v2/DepthPro variants add runtime. `boundary_f1_si` is identical. Runs without `--boundary` are unchanged. Not gated behind a separate flag. |
| `distill/student.py` edit | No parameter, buffer, RNG or state_dict change. The legacy forward is untouched. **Training configs must not set `student.k_none_policy`.** It is an eval-only switch; set it with `set_k_none_policy()` or `--k-none-policy`. |
| `default_k` dtype | Under `default_k`, K is built with `x.dtype`. `forward(x,None)==forward(x,DEFAULT_K)` is tested in fp32 only. Under fp16/autocast, pass K in the same dtype as x for equality. Not changed: building K in fp32 would promote rays to fp32 and break the cat with half inputs. |
| One DEFAULT_K | Holds for eval only (student.py, run_bench, nyu_eval, visualize, safety). The training-side fallback `DEFAULT_INTR = (0.7, 0.9, .5, .5)` in `distill/dataset.py` and `distill/v2/data.py` was deliberately left alone, per the user's no-training-change scope. |
| `tests/_orig.py` | Pre-existing shared helper imported by tracked tests (test_b10_parity, test_cstm, test_dsh, test_fov_ownar, test_rep, test_trim). The leading underscore stops pytest collecting it. Kept; the docstring now says so. |
