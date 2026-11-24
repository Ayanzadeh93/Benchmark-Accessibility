# Surpass-UniDepthV2 — implementation status, corrections, and run-plan (2026-07-13)

Answers three questions for every plan item: **implemented**, **reviewed+corrected this
session**, or **remaining (run-plan)**. No pipeline code was executed in this session (per
request); everything under "run-plan" is listed as commands to run, not run.

---

## 1. Implementation status by phase

| Phase | Item | Status | Where |
|---|---|---|---|
| 0 | Related-work dossier | ✅ implemented | [docs/RELATED_WORK_DOSSIER.md](RELATED_WORK_DOSSIER.md) |
| 0 | Tab. V/VI zero-shot-vs-FT settled | ✅ implemented | PROTOCOL_NOTES §1 (FT-confirmed) |
| 1 | Official metric port (δ1/ARel/SqRel/SILog/τ/d_AUC/SSI/SI/F_A/ρ_A/AUSE/boundary) | ✅ implemented + self-test | [benchmark/protocol.py](../benchmark/protocol.py) |
| 1 | Pred never clipped, GT-mask only, native `infer()` | ✅ implemented | [benchmark/run_bench.py](../benchmark/run_bench.py) |
| 1 | Tier-1 loaders: NYU-654 / SUN alltest (NYU-filtered) / IBims-1 | ✅ implemented | [benchmark/datasets_tier1.py](../benchmark/datasets_tier1.py) |
| 1 | Tier-1 fetcher (resumable, mirrors) | ✅ implemented | [benchmark/fetch_tier1.py](../benchmark/fetch_tier1.py) |
| 1 | Reproduction gate | ✅ implemented, ⚠️ **corrected** (raw-δ1 bug) | [benchmark/gate_check.py](../benchmark/gate_check.py) |
| 1 | TUM-RGBD + KITTI-Eigen loaders | ⛔ remaining (deferred, documented) | fetch_tier1 header |
| 2 | OCMR metric family (+ per-band, prox-F1, AUSE@obj) | ✅ implemented | [benchmark/ocmr_metrics.py](../benchmark/ocmr_metrics.py) |
| 2 | OCMR-SUN GT builder (57.7k obj / 4,379 img) | ✅ implemented (cache built) | [benchmark/ocmr_sun.py](../benchmark/ocmr_sun.py) |
| 2 | OCMR eval — annotation track (box5/maskmed/center) | ✅ implemented | [benchmark/ocmr_eval_ann.py](../benchmark/ocmr_eval_ann.py) |
| 2 | OCMR eval — detector-in-loop track (YOLO 5-pt) | ✅ implemented | [benchmark/ocmr_eval.py](../benchmark/ocmr_eval.py) |
| 2 | Gold-set capture tool + protocol | ✅ implemented | [benchmark/gold_capture.py](../benchmark/gold_capture.py) |
| 2 | Object-level split-conformal intervals | ✅ **added this session** | ocmr_metrics.`conformal_log_intervals` |
| 3 | Post-hoc affine recal + size anchor (CSV track) | ✅ implemented | [prototypes/sma_anchor.py](../prototypes/sma_anchor.py) |
| 3 | Ground-lock + SMA live track (A1+A4) | ✅ implemented | [prototypes/anchor_lock.py](../prototypes/anchor_lock.py) |
| 3 | Resolution-marginalization (A3-lite) | ✅ **added this session** | [prototypes/res_marginalize.py](../prototypes/res_marginalize.py) |
| 4 | Loss library (SILog/grad/SSI/EG-SSI/β-NLL/bins/normal/obj-dist/ICD) | ✅ implemented | [distill/losses.py](../distill/losses.py) |
| 4 | Learned affine head (N4 "anch") | ✅ implemented | [distill/student.py](../distill/student.py) |
| 4 | **anchor-consistency loss** (A1-as-loss) | ✅ **added this session** | losses.`anchor_consistency_loss` + `anchloss` tag |
| 4 | **proximity-weighted SILog** (safety band) | ✅ **added this session** | losses.`proximity_weighted_silog` + `prox` tag |
| 4 | 15-variant registry (S0–S15) + trained ckpts | ✅ implemented | [benchmark/registry.json](../benchmark/registry.json) |
| 4 | Zero-shot-clean (no-SUN) retrain | ⛔ remaining (run-plan) | train.py `--data gt` w/ no-SUN mix |
| 5 | Paper plan / experiments / benchmark docs | ✅ implemented | docs/{PAPER_PLAN,EXPERIMENTS,BENCHMARK_SURPASS}.md |
| 5 | Legacy ablation table (gt_eval ecosystem) | ✅ implemented | [benchmark/ablation_table.py](../benchmark/ablation_table.py) |
| 5 | **Paper-grade table assembler** (results_bench/results_ocmr) | ✅ **added this session** | [benchmark/paper_tables.py](../benchmark/paper_tables.py) |

---

## 2. Corrections made this session (review findings)

1. **Gate metric bug (critical)** — `gate_check.py` compared measured **`d1_ssi`** against the
   published Table I δ1, but the repo's own PROTOCOL_NOTES §2b #1 established (today) that those
   published numbers are **raw metric δ1** for metric methods (measured V2-L IBims raw d1 ≈94.2–94.6
   == published 94.5; SSI variant ≈98.5). The gate would have spuriously FAILED V2-L. Fixed: HARD
   check now uses raw `d1`; `d1_ssi` demoted to an informational SOFT line; V2-S checkpoint-drift
   caveat auto-annotated; docstring + PROTOCOL_NOTES §3 rule 1 corrected to "raw δ1".
2. **Latent cv2 bug** — `run_bench.py` resized model points with `cv2.resize(pts[c], (w,h),
   cv2.INTER_LINEAR)` (flag passed into the `dst` positional). Would raise if points ever came at
   non-GT resolution. Fixed to `interpolation=cv2.INTER_LINEAR`. (Swept the repo — no other
   positional-interpolation cases.)

Metric-port correctness was re-verified against `UniDepth-main/.../evaluation_depth.py`: ddof=1
SILog, lower-median semantics, no-pred-clip, squared-distance F_A compat, SSI/SI closed forms — all
faithful. `protocol.py`'s self-test remains the bit-exact check (run it, item 0 below).

---

## 3. Remaining coding completed this session

- `prototypes/res_marginalize.py` — uncertainty-weighted multi-resolution log-depth fusion over
  UniDepthV2 `resolution_level` (confidence=precision), boundary-F1-vs-compute curve; uniform-avg
  and precision-weighted modes; student input-scale variant via the σ head.
- `distill/losses.py` — `proximity_weighted_silog` (proper weighted-variance SILog, smooth <2 m
  up-weight) and `anchor_consistency_loss` (label-free `d_geo = fy·H/h_box` supervision of scale);
  both wired into `DistillLoss`, self-gating, weight-0 by default (inert until enabled).
- `distill/train.py` — recipe tags `prox` and `anchloss`; `box_sizes` plumbed to the criterion.
- `distill/dataset.py` — emits `box_sizes` (optional npz `sizes`, backward-compatible; kept in sync
  through crop/truncation/flip); box emission now also triggers on `anchor_weight>0`.
- `scripts/build_task_weights.py` — writes per-object `sizes` (physical-height prior by class) into
  the box npz so `anchloss` has data; older caches without `sizes` stay valid (→ self-gated).
- `benchmark/ocmr_metrics.py` — `conformal_log_intervals` (split-conformal object range intervals,
  plain + uncertainty-normalized) for claim C4 / Tab. 6.
- `benchmark/paper_tables.py` — assembles Dense / OCMR / anchoring / res-marg tables from
  `results_bench/` + `results_ocmr/` into `docs/PAPER_TABLES.md` (+ CSVs), all rows `[ours]`.

**Enablement note:** `anchloss` needs intrinsics in the batch → train with `--use-intrinsics`, and a
box cache rebuilt by the updated `build_task_weights.py`. `prox` needs no data changes.

---

## 4. Run-plan (NOT executed — list of what to run, in order)

Envs: `depth` (official/teacher), `IJCAI` (students). GPU: RTX 5070 laptop 12 GB.

**0. Unit-verify the metric port (fast, do first):**
```
conda run -n depth python -m benchmark.protocol          # self-test vs official torch, must pass
```
**1. Fetch Tier-1 data (~15–20 GB, resumable):**
```
python -m benchmark.fetch_tier1 --root data_bench --only nyu,sun,ibims
python -m benchmark.datasets_tier1 inspect nyu ; ... inspect sun ; ... inspect ibims
```
**2. Baselines through native infer (res-level 9) — reproduction gate inputs:**
```
conda run -n depth python -m benchmark.run_bench --dataset sun   --model v2-l --model v2-s --res-level 9
conda run -n depth python -m benchmark.run_bench --dataset ibims --model v2-l --model v2-s --res-level 9 --boundary
python -m benchmark.gate_check                          # writes docs/GATE_REPORT.md (raw-δ1 gate)
```
**3. Full dense suite (V2-S/B/L, V1-L, students):**
```
conda run -n depth python -m benchmark.run_bench --dataset nyu --variant filled --model v2-l --model v2-b --model v2-s --model v1-l --res-level 9
# students: --model student:outputs/checkpoints/S10_ultra_L/best.pt  (repeat per registry entry)
```
**4. OCMR object-range (GT cache already built):**
```
conda run -n depth python -m benchmark.ocmr_eval_ann --model v2-l --res-level 9
# repeat for v2-b, v2-s, v1-l, and each student ckpt
```
**5. Training-free anchoring + resolution-marginalization (new):**
```
conda run -n depth python -m prototypes.anchor_lock       --model v2-l --res-level 9
conda run -n depth python -m prototypes.res_marginalize   --dataset ibims --model v2-l --levels 3,6,9 --boundary
conda run -n IJCAI python -m prototypes.sma_anchor --cal <nyu_obj.csv> --test <sun_obj.csv>   # CSV track
```
**6. Phase-4 trainings (new losses; single-GPU):**
```
python -m scripts.build_task_weights --config config.yaml     # regenerate box npz WITH sizes (for anchloss)
# proximity-weighted SILog student:
python -m distill.train --config config.yaml --data gt --name S16-prox   --recipe s4,s5,ray,range,prox   --use-intrinsics --backbone efficientvit_b2.r224_in1k
# anchor-consistency student (needs intrinsics + sized boxes):
python -m distill.train --config config.yaml --data gt --name S17-anchloss --recipe s4,s5,ray,range,anchloss --use-intrinsics --backbone efficientvit_b2.r224_in1k
# zero-shot-clean (edit data_gt.mix to own+nyu only, SUN=0):
python -m distill.train --config config.yaml --data gt --name S18-zsclean  --recipe s4,s5,ray,range --use-intrinsics
```
**7. Assemble tables:**
```
python -m benchmark.paper_tables      # docs/PAPER_TABLES.md from results_bench/ + results_ocmr/
python -m benchmark.ablation_table --config config.yaml    # legacy gt_eval ecosystem (still valid)
```
**8. FPS/efficiency + gold set (as available):**
```
python -m benchmark.fps_bench --config config.yaml
python -m benchmark.gold_capture annotate <folder>         # user collects 200–500 measured ranges
```

---

## 5. Known deferrals & risks (unchanged from plan, re-affirmed)

- **TUM-RGBD / KITTI-Eigen loaders**: deferred — official repo ships no TUM split (any TUM number
  is our-split-only) and KITTI is account-gated. Add only if a defensible split is fixed; note the
  deviation from the locked "Tier-1 + KITTI sanity" scope.
- **V2-S published-row mismatch** (checkpoint drift, §2b #5): gate won't pass V2-S on IBims → use
  measured baselines everywhere with the footnote (already the documented policy).
- **anchloss data**: current box npz predates the `sizes` field → rebuild via build_task_weights
  before training `anchloss`; without a rebuild the term self-gates to zero (safe, but inactive).
- **SUN class canonicalization** (plural/synonym) still a pre-release todo (PAPER_PLAN §3.2).
- **Conformal intervals**: implemented but need a cal/test object split wired in the OCMR eval to
  report Tab. 6 numbers (small follow-up in ocmr_eval_ann).

---

# Update 2026-07-14 — PDF verified, first full results in, KITTI unlocked

## What ran since the audit (user-executed; all four run-plan steps 0–4 for the core slice)
- Tier-1 data fetched + extracted (NYU labeled.mat, SUN zip+toolbox, IBims core); OCMR-SUN GT
  cache built (4,379 imgs / 57,677 objects).
- Dense baselines: IBims v2-l/v2-s (default + r9), SUN-raw v2-l+r9 & v2-s+r9 **full n=4396**,
  SUN-bfx probe (ruled bfx out), NYU smoke.
- OCMR-SUN full table: v2-l+r9, v2-s+r9, S10-ultra-L, b1-mix × {box5, maskmed, center}.

## Verified from the actual PDF (2502.20110v2, text-extracted)
All PROTOCOL_NOTES §1 facts confirmed verbatim (Tables I/II/IV/V/VI/VII, no-TTA, same-weights,
native-resolution inference, Tab. V/VI = fine-tuned). **No eval-resolution override exists in the
paper** → remaining ARel gap (+0.7–1.0pt, model-independent) is data-packaging/ckpt-drift, not
protocol (PROTOCOL_NOTES §2c). Policy: measured baselines everywhere.

## Gate verdict (post raw-δ1 fix, full-n)
IBims V2-L d1 **PASSES** (+0.09); SUN V2-L d1 −0.69 / ARel +1.00; ARel misses ±0.5 on all
slices by a consistent offset. Formal ±0.5 double-gate: NOT PASSED → published rows enter only
[paper]-tagged with the drift/packaging footnote, or are omitted (rules §3).

## First-results reading (paper-claim evidence already in hand)
- **C1 evidenced**: V2-L dense d1 95.7 (saturated) vs proxF1@1m 0.669 on 57.7k objects.
- **C2 target quantified**: V2-L bias_log +0.024, V2-S +0.026 (systematic over-ranging);
  students ≈0 — the anchor/ground-lock has a concrete, measured drift to remove.
- Protocol ablation is monotone for all models (maskmed < box5 < center error) → box5 is a
  sound deployed proxy (+~1–1.5pt objAbsRel over the mask oracle).
- Object-level ranking ≠ dense ranking (S10 beats V2-S dense-NYU but trails on objAbsRel) —
  more C1 material.

## Corrections/code this update
1. **gate_check matching bug**: `+r9` runs were invisible (SUN misreported "NOT RUN"); now
   matches the JSON `model` field (base or `+rN`), excludes `+gtcam`, prefers largest-n.
2. **benchmark/ocmr_kitti.py (new)**: OCMR-KITTI from `Z:\Datasets\KITTI` object benchmark
   (7,481 frames; pedestrian_subset 1,779/4,487 available as a class slice). GT conventions
   z_front/z_center/euclid; outdoor bands; proxF1@{5,10}m. Dense KITTI-Eigen stays impossible
   from this drop (no velodyne) and has nothing zero-shot to reproduce (Tab. VI is FT).
3. **ocmr_metrics**: `core`/`summarize` take `bands`/`prox_thresholds` (indoor defaults
   unchanged); added `RANGE_BANDS_OUTDOOR`.
4. **paper_tables**: OCMR section now slice-aware (sun/kitti), protocol `center` included,
   proximity keys picked dynamically.

## Comparability hardening (2026-07-14, user directive "everything must be comparable")
Contract codified as PROTOCOL_NOTES §3 rules 6–11 and enforced mechanically:
1. **One box5 implementation** — `ocmr_metrics.box5_sample` (≥3-of-5 validity) is now the
   only sampler; `ocmr_eval_ann`, `ocmr_kitti`, and `prototypes/anchor_lock` delegate to it
   (previously three copies with a divergent min-valid rule between the OCMR and anchoring
   tables).
2. **paper_tables guards** — GT-cam probe runs auto-excluded; warns on: n mismatch across
   models within a (dataset, variant) group; mixed resolution policy (`v2-l` + `v2-l+r9`)
   in one group; OCMR object-pool drift >0.5% across models per slice/protocol. Warnings
   print AND land in a "Comparability warnings" section at the top of PAPER_TABLES.md.
3. **fps_bench rewritten** — now benches v2-s/b/l + v1-l + students on the SAME rig with
   peak-VRAM, GPU/torch/precision/warmup metadata stamped into fps_bench.json (paper
   Tab. VIII A6000 rows only ever appear in a separately-labeled block).
   ablation_table.py reads both old and new fps formats.
4. **det_box5 on the official split** — `ocmr_eval_ann --det-yolo <w>` adds the
   detector-in-the-loop protocol on the SAME NYU-filtered alltest split (YOLO boxes
   identical across models, sensor-GT std≤1m gate, shared sampler). Legacy
   `ocmr_eval.py` (HF streams) is header-marked SUPERSEDED for paper numbers.
5. **Slices are not cross-comparable** — sun/kitti/gold have different GT conventions;
   paper_tables carries a slice column and the rule is documented.

## Next runs (updated queue)
```
python -m benchmark.gate_check                                  # re-emit with r9-aware matching
conda run -n depth python -m benchmark.run_bench --dataset nyu --variant filled --model v2-l --model v2-s --model v2-b --model v1-l --res-level 9
conda run -n depth python -m benchmark.run_bench --dataset sun --variant raw --model v2-b --model v1-l --res-level 9
conda run -n depth python -m benchmark.run_bench --dataset ibims --model v2-b --model v1-l --res-level 9 --boundary
python -m benchmark.ocmr_kitti build                            # one-time GT cache from Z:
conda run -n depth python -m benchmark.ocmr_kitti eval --model v2-l --res-level 9
conda run -n depth python -m benchmark.ocmr_kitti eval --model v2-s --res-level 9
conda run -n depth python -m benchmark.ocmr_kitti eval --model student:outputs/checkpoints/S10_ultra_L/best.pt
conda run -n depth python -m prototypes.anchor_lock --model v2-l --res-level 9    # C2: remove the +2.4% drift
conda run -n depth python -m prototypes.anchor_lock --model student:outputs/checkpoints/S10_ultra_L/best.pt
conda run -n depth python -m prototypes.res_marginalize --dataset ibims --model v2-l --levels 3,6,9 --boundary
# remaining registry students on dense SUN/IBims + OCMR-SUN, then:
python -m benchmark.paper_tables
```

## Verification sweep + KITTI closure (2026-07-14, later)
User asked to "apply" the session's corrections/coding list — ALL items re-verified present
by grep/existence checks: gate raw-d1 + `+rN` matching + gtcam exclusion; run_bench cv2
keyword fix (line 173); prox/anchloss losses + wiring + box_sizes path (losses/train/
dataset/build_task_weights); conformal_log_intervals + shared box5_sample + outdoor bands;
res_marginalize.py / paper_tables.py / ocmr_kitti.py on disk; PROTOCOL_NOTES §3 raw-δ1.
Nothing was missing; no re-application needed.

KITTI closure: `raw/` = exactly the 3 object-benchmark zips (11.7 GB; full object dataset,
already extracted). No velodyne/sequences/depth found at any depth ⇒ dense Eigen impossible
today. Added `datasets_tier1.iter_kitti_obj` (dataset key `kitti_obj`): velodyne-projected
sparse-depth dense eval over the 7,481 object frames, official constants (min 0.05/max 80,
garg crop), self-gating with an actionable error until `data_object_velodyne.zip` (~29 GB,
new `fetch_tier1 --only kitti` entry) is downloaded + extracted. Sparse GT ⇒ no boundary-F1
on this slice; rows [ours]-only.
```
