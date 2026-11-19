# Protocol notes — UniDepthV2 evaluation, verified facts, and our deviations

Source of truth for every benchmark decision in this repo. Everything below was extracted
from the UniDepthV2 paper PDF (arXiv 2502.20110v2, 14-page TPAMI-style version, extracted
2026-07-11), the official `UniDepth-main` code in this repo, or measured locally. Any change
to the harness must be reflected here.

## 1. Verified paper facts (arXiv 2502.20110v2)

### Training data (Sec. IV-A)
23 datasets, 16M images: A2D2, Argoverse2, ARKit-Scenes, BEDLAM, BlendedMVS, DL3DV,
DrivingStereo, DynamicReplica, EDEN, HOI4D, HM3D, Matterport3D, Mapillary-PSD, MatrixCity,
MegaDepth, NianticMapFree, PointOdyssey, ScanNet, ScanNet++, TartanAir, Taskonomy, Waymo,
WildRGBD.

**NOT in training** (⇒ zero-shot for UniDepthV2): NYU-Depth-v2, KITTI, SUN RGB-D, IBims-1,
TUM-RGBD, ETH3D, Sintel, DDAD, nuScenes, HAMMER, Booster, FLSea.

### Zero-shot eval suite (Tables I–III)
- Indoor: SUN RGB-D, IBims-1, TUM-RGBD (validation splits).
- Outdoor: ETH3D, Sintel, DDAD, nuScenes.
- Challenging: HAMMER, Booster, FLSea.

### Zero-shot protocol (Sec. IV-A "Evaluation Details")
- No test-time augmentation; same weights for all zero-shot evals; checkpoints NOT
  fine-tuned on KITTI/NYU.
- **Headline metrics: δ1^SSI, F_A, ρ_A** — NOTE the δ1 in Tables I–III is
  **scale-shift-invariant aligned** (least-squares scale+shift in depth space, their
  `evaluation_depth.ssi`), not raw metric δ1. The ARel column is raw metric ARel.
- F_A: AUC of point-cloud F1-score up to 1/20 of the dataset's max depth.
- ρ_A: AUC of average angular error of predicted camera rays, up to 15°.
- Uncertainty (Table VII): AUSE **w.r.t. δ1** (rank by predicted uncertainty, progressively
  mask, integrate gap to oracle), nAUSE (0 = oracle, 1 = random), Spearman ρ.

### Tables V/VI are FINE-TUNED (settled 2026-07-11)
"Table V and Table VI show results for models fine-tuned on the NYU and KITTI training sets
and evaluated on their respective validation splits." ⇒ NYU δ1 98.8 / ARel 4.68 and KITTI
δ1 98.9 / ARel 3.73 are **in-domain fine-tuned** numbers. There is **no published zero-shot
NYU row for UniDepthV2** (NYU is not in the ten-set suite). Do not cite Tab. V/VI as
zero-shot. Context for NYU zero-shot: UniDepthV1 README reports 98.4 δ1 (V1, zero-shot).

### Published rows we may reuse AFTER the reproduction gate passes
(δ1 below = δ1^SSI; ARel = raw metric, in %)

Table I (indoor, zero-shot):
| Model | SUN δ1 / ARel / F_A / ρ_A | IBims-1 δ1 / ARel / F_A / ρ_A | TUM δ1 / ARel / F_A / ρ_A |
|---|---|---|---|
| UniDepthV2-S | 90.8 / 10.5 / 74.2 / 87.7 | 86.6 / 13.5 / 62.4 / 67.5 | 69.0 / 23.6 / 50.6 / 86.1 |
| UniDepthV2-B | 94.4 / 8.4 / 79.9 / 91.1 | 89.7 / 11.1 / 68.5 / 76.5 | 77.5 / 20.7 / 57.3 / 89.4 |
| UniDepthV2-L | 96.4 / 6.8 / 84.6 / 93.4 | 94.5 / 7.8 / 70.9 / 74.1 | 90.5 / 22.1 / 62.9 / 89.6 |
| UniDepthV1-L | 94.3 / 10.4 / 78.6 / 85.8 | 15.7 / 41.0 / 30.3 / 76.6 | 72.3 / 17.2 / 54.8 / 86.8 |
| DepthPro | 83.1 / 13.3 / 71.1 / 89.3 | 82.3 / 17.0 / 62.8 / 75.9 | 56.9 / 19.9 / 48.1 / 96.5 |
| Metric3Dv2†‡ | 81.2 / 13.3 / – / – | 68.4 / 20.7 / – / – | 63.0 / 9.3 / – / – |
| MASt3R | 80.1 / 14.4 / 71.5 / 92.0 | 61.0 / 19.7 / 55.7 / 76.0 | 52.4 / 27.9 / 44.1 / 93.7 |
| ZoeDepth† | 80.9 / 13.6 / – / – | 49.8 / 21.5 / – / – | 55.6 / 33.6 / – / – |

(† needs GT camera for 3D; ‡ needs GT camera for 2D depth.)

Table IV (scale-invariant boundary F1, DepthPro protocol; DepthPro at fixed 1536²):
ETH3D / IBims-1 / Sintel — V2-S 2.13/10.53/20.58, V2-B 2.99/12.93/28.25,
V2-L 2.95/13.75/33.08, DepthPro† 4.04/19.38/29.24, Metric3Dv2 2.86/13.48/23.03.

Table VII (uncertainty): zero-shot AUSE/nAUSE/ρ — V2-S .040/.539/.289,
V2-B .034/.637/.291, V2-L .032/.645/.299; in-domain — V2-S .020/.221/.682,
V2-B .018/.212/.721, V2-L .017/.199/.744.

Table VIII (efficiency; A6000, fp16, 0.5MP): V2-S 23.0ms/34.18M/0.29T/0.66GiB,
V2-B 35.1/114.4/0.82/1.32, V2-L 65.4/353.8/2.17/3.47, DepthPro† 270.6/952.0/19.3/8.42,
Metric3Dv2 133.6/411.9/3.47/3.50, ZoeDepth 64.8/346.1/2.08/2.02.

### Official metric implementations (UniDepth-main/unidepth/utils/evaluation_depth.py)
- Per-image metrics over `mask & (gt <= max_depth)`; predictions bilinear-resized to GT
  size; **predictions are never clipped**.
- `ssi(gt, pred)`: closed-form least-squares scale+shift of pred onto gt in DEPTH space.
- `si(gt, pred)`: median-ratio scaling.
- `arel` helper in their file applies median scaling internally when called via the
  "_ssi/_si" variants loop; plain `arel` in DICT_METRICS is raw.
- `silog` = 100 · std(log pred − log gt) (std-only flavor, not Eigen λ-form).
- Extra metrics available: sqrel, log10, medianlog, d_auc (AUC of δ over exponents
  0.01–5), tau@3%.
- 3D: chamfer + F1-AUC on point clouds, computed at ≤240×320 (their OOM guard downsample,
  nearest-exact).
- NYU eval mask: border crop `[..., 45:-9, 41:-39]` on 480×640 (== rows 45:471, cols
  41:601 == our existing `eigen_crop_mask`). NYU min/max depth 0.005/10.0.
- SUN: min/max 0.005/8.0, split `alltest.txt` from their HDF5 pack, per-image intrinsics.
  README: "The common split between SUN-RGBD and NYUv2 is removed from SUN-RGBD validation
  set for evaluation."

## 2. Local audit findings (2026-07-11) and fixes

| # | Issue | Fix |
|---|---|---|
| 1 | NYU via `vikhyatk/nyu_depth_v2` repack; teacher δ1 0.947 vs V1-paper ≈0.984 → repack fidelity suspect; repro gate fails | Rebuild NYU test from official `nyu_depth_v2_labeled.mat` + Eigen 654 split; keep both filled `depths` (standard) and `rawDepths` (strict) variants |
| 2 | SUN via `wyrx/SUNRGBD_seg` first-N-streamed; no NYU-overlap filter (leakage path: students train on NYU-train; SUN test contains NYU frames) | Official SUNRGBD.zip + toolbox `allsplit.mat`; drop `kv1/NYUdata` entries; full filtered alltest; per-image intrinsics from metadata |
| 3 | `gt_eval.py` clips predictions to [1e-3, max_d] pre-metric | Mask-only (GT-based); never clip predictions |
| 4 | Missing metrics vs paper (δ1^SSI, F_A, ρ_A, SILog, SqRel, τ, d_AUC, AUSE/nAUSE/ρ, boundary F1) | `benchmark/protocol.py` implements the full set; unit-checked against official code on synthetic tensors |
| 5 | Teacher eval forced to 640 longest-side | Baselines run through native `model.infer()` (internal 0.2–0.6MP budget); resolution recorded per run |
| 6 | MIN_D 1e-3 vs official 0.005 | Use per-dataset official min/max (NYU 0.005/10, SUN 0.005/8) |
| 7 | Students trained on SUN-train ⇒ not zero-shot on SUN | Phase 4 retrains a zero-shot-clean variant (no SUN); all tables carry a "train data" column |
| 8 | own-corpus numbers are teacher-relative | Always labeled "distillation agreement", never absolute accuracy |

## 2b. Gate-run discoveries (2026-07-13, measured on IBims-1, n=100)

1. **Tables I–III δ1 is RAW metric δ1 for metric methods, not δ1^SSI.** Measured V2-L
   IBims raw d1 94.21 (r-default) / 94.59 (res-level 9) vs published 94.5; the SSI
   variant measures 98.5 — far above. The paper's "δ1^SSI" notation evidently applies
   the SSI alignment only where a method is not metric (e.g., MASt3R). Our tables
   report BOTH (d1 and d1_ssi) to be unambiguous.
2. **F_A quirk (bug-compat)**: official knn_points returns SQUARED distances
   (pytorch3d convention) and f1_score/auc compare them against metre thresholds →
   effective threshold sqrt(t). With squared-compat enabled our F_A lands within
   ±1.4pt of published (V2-L 72.2 vs 70.9; V2-S 61.0 vs 62.4). protocol.py keeps
   `squared_compat=True` by default and documents it.
3. **UniDepthV2 "confidence" output is ERROR-like** (training target: L1 on
   |log d − log d*|): higher = less certain. Treat it directly as uncertainty for
   AUSE/Spearman. (This also explains the confidence-weighting collapse noted in
   this repo's config comments.)
4. **Resolution policy**: `model.infer()` without `resolution_level` keeps native
   pixels within [0.2, 0.6]MP. res-level 9 (upper budget slice) matches published
   V2-L IBims best (d1 Δ+0.09, ARel Δ+0.68). Our baseline runs use res-level 9,
   recorded per run.
5. **V2-S residual gap**: measured IBims d1 82.5 (r9) vs published 86.6 — persists
   across res levels while V2-L matches; suspected HF-checkpoint drift vs the
   paper-time Small checkpoint (HF release 28.02.2025). All our tables therefore use
   MEASURED baselines; published rows quoted only with this note attached.
6. eval_3d masks do NOT apply the max-depth cut (official prepare_points_eval passes
   raw depth_mask) — mirrored in protocol.eval_3d_image.
7. ρ_A interpretation (dataset-level AUC of per-image mean ray angular error ≤15°)
   lands within 1.5pt of published on IBims V2-L — kept, marked as interpretation.

## 2c. Direct-PDF verification + full-run gate analysis (2026-07-14)

**PDF verified** (local file `2502.20110v2 (2).pdf`, 14 pages, text-extracted): Tables I, II,
IV, V, VI, VII values match §1 exactly. Sec. IV-A confirms verbatim: 23-dataset training list;
ten-set zero-shot suite; "we do not exploit any test-time augmentations"; "same weights for all
zero-shot evaluations"; "checkpoint ... not fine-tuned on KITTI or NYU"; Tab. V/VI captions =
fine-tuned (settled again, from the PDF itself). Tab. IV note: DepthPro at fixed 1536², "other
models ... are run at the native input resolution" → our native-`infer()` policy is the paper's.
**The paper states NO eval-time resolution override anywhere** → no further protocol knob exists;
residual baseline gaps are data-packaging/checkpoint-level, not protocol-level.

**Full-run gate state** (results_bench, n=full):
| slice | model | d1 meas/pub (diff) | ARel meas/pub (diff) | F_A meas/pub (diff) |
|---|---|---|---|---|
| SUN-raw n=4396 | v2-l+r9 | 95.71 / 96.4 (−0.69) | 7.80 / 6.8 (+1.00) | 84.24 / 84.6 (−0.36) |
| SUN-raw n=4396 | v2-s+r9 | 89.52 / 90.8 (−1.28) | 11.18 / 10.5 (+0.68) | 75.31 / 74.2 (+1.11) |
| IBims n=100 | v2-l+r9 | 94.59 / 94.5 (+0.09) ✅ | 8.48 / 7.8 (+0.68) | 73.17 / 70.9 (+2.27) |
| IBims n=100 | v2-s+r9 | 82.53 / 86.6 (−4.07, drift) | 14.16 / 13.5 (+0.66) | 63.08 / 62.4 (+0.68) |

Reading: d1/F_A/ρ_A land within ~0.1–2pt (IBims V2-L d1 within 0.09!). ARel runs +0.7–1.0pt
high across ALL models/sets — a consistent, model-independent offset that (per the PDF check
above) cannot be a missing protocol switch; candidates are (a) their internal HDF5 repack of
SUN/IBims depth (source variant/quantization), (b) HF-checkpoint drift (proven for V2-S).
SUN `bfx` is ruled out (probe: SILog 18.6, ARel 16.2 — far off; `raw` is the eval variant).
**Consequence**: the ±0.5pt double gate (d1+ARel) is not reachable for ARel; per §3 policy all
tables use MEASURED baselines with the drift/packaging footnote; published rows may be quoted
[paper]-tagged only with that caveat attached, or omitted entirely.

Also fixed 2026-07-14: `gate_check.latest_result` previously missed `+r9`-suffixed runs (glob
required exact `__<model>.json`), reporting full SUN runs as "NOT RUN"; now matches the JSON's
`model` field (base or `+rN`), excludes `+gtcam`, prefers largest-n.

**KITTI (Z:\Datasets\KITTI, reachable 2026-07-14)**: verified = the complete KITTI
**object-detection benchmark** (training/{image_2,label_2,calib} 7,481 frames, testing 7,518,
raw/ holds exactly the three object zips 11.7 GB total, curated pedestrian_subset 1,779
frames / 4,487 pedestrians 2–74 m). Recursive search confirms **no velodyne, no raw
sequences, no depth annotations** ⇒ dense Eigen eval impossible from this drop (and Tab. VI
is fine-tuned anyway — nothing zero-shot to reproduce). Implemented:
1. **OCMR-KITTI** (`benchmark/ocmr_kitti.py`, runnable today) — 3D-box GT → per-object
   range, gates h≥25px / trunc≤0.5 / occ≤2 / 2–80 m, GT conventions z_front (default,
   visible-surface) vs z_center vs euclid as a protocol ablation, outdoor bands
   0-5-10-20-40-80 m, proxF1@{5,10}m.
2. **Dense outdoor sanity, velodyne-gated** (`datasets_tier1.iter_kitti_obj`, dataset key
   `kitti_obj`) — single-sweep LiDAR projection into cam2 over the same 7,481 frames,
   official constants ported from their kitti.py (min 0.05 / max 80, garg crop
   [0.40810811H:0.99189189H, 0.03594771W:0.96405229W]); sparse GT ⇒ never run boundary-F1
   on it; our-protocol slice, rows always [ours]-only. Unlocks by downloading ONE zip:
   `data_object_velodyne.zip` (~29 GB; `fetch_tier1 --root Z:/Datasets/KITTI --only kitti`,
   then extract). Until then the loader raises with that instruction.

## 2c-bis. Campaign facts (2026-07-19; details in RESULTS_DIGEST.md)
- **V1-L is infeasible in this env** (needs removed `xformers.components`; legacy xformers
  incompatible with Blackwell/torch-2.11). Measured V1 rows dropped everywhere; published
  V1 rows only [paper]-tagged. **Never install xformers into `depth`** — xformers 0.0.35
  silently broke ALL V2 inference (verified, uninstalled, V2 re-verified).
- First full campaign done: dense NYU/SUN/IBims (officials+students), OCMR-SUN 6 models,
  OCMR-KITTI 3 models, anchoring both models, gate re-run. Ground-lock on V2-L: ARel
  −9.6% rel + bias eliminated (C2 met). SMA negative indoors (occlusion). center>box5
  outdoors, maskmed<box5 indoors (protocol finding).
- S17-anchloss training deferred: own corpus 99.5% class-less (class_id = per-image
  open-vocab index, not a global vocabulary). Revive via gold-set classes or YOLO relabel.

## 2d. Canonical dataset locations (moved 2026-07-14)

Source datasets live on the datasets drive, one titled folder each (override via the
`DATASETS_ROOT` env var; legacy `data_bench/{nyu,sun,ibims}` is an automatic fallback):

| Dataset | Location | Contents |
|---|---|---|
| NYU Depth v2 | `Z:\Datasets\NYUv2` | nyu_depth_v2_labeled.mat, splits.mat |
| SUN RGB-D | `Z:\Datasets\SUNRGBD` | SUNRGBD.zip, SUNRGBDtoolbox.zip, SUNRGBD/ (extracted) |
| IBims-1 | `Z:\Datasets\IBims-1` | ibims1_core_mat.zip (+ extracted) |
| KITTI | `Z:\Datasets\KITTI` | object benchmark (training/testing/raw zips, pedestrian_subset) |
| Gold (ours) | `Z:\Datasets\OCMR-Gold` | gold_capture sessions (created on first capture) |

Derived artifacts (ocmr_sun_gt.jsonl, ocmr_kitti_gt.jsonl, legacy results) stay in
`unidepth2d/data_bench/`. `fetch_tier1` now defaults `--root Z:/Datasets` with the titled
relative paths. All loaders (`datasets_tier1`, `ocmr_sun`, `ocmr_eval_ann`, `ocmr_kitti`,
`gold_capture`, `anchor_lock` via `sun_extract`) resolve through
`datasets_tier1.{NYU_DIR,SUN_DIR,IBIMS_DIR,KITTI_ROOT}` — no other file hardcodes a
dataset path.

## 3. Comparability rules for the paper
1. Published UniDepthV2 rows may appear in our tables only after our harness reproduces
   SUN + IBims-1 rows for V2-L and V2-S within RAW δ1 ±0.5pt / ARel ±0.5pt (raw δ1 because
   the published Table I δ1 is raw-metric for metric methods — §2b #1; `benchmark/gate_check.py`
   compares measured raw `d1`, and carries `d1_ssi` only as an informational line). V2-S is
   expected to miss the gate on IBims by checkpoint drift (§2b #5) — measured rows are used
   everywhere with that footnote.
2. Any table mixing published and measured numbers marks each row [paper] / [ours].
3. All object-level (OCMR) numbers are measured by us for every method — published dense
   numbers never enter OCMR tables.
4. Every measured row records: checkpoint id, inference resolution policy, dataset split
   hash, harness git hash, date.
5. δ1 without qualifier in our text = RAW metric δ1 (no alignment). SSI-aligned values are
   always written δ1^SSI. (The paper's Tables I–III headline δ1 is δ1^SSI notation but behaves
   as raw for metric methods — §2b #1.)

### Comparability contract (mechanically enforced 2026-07-14)
6. **One resolution policy per table**: officials at `+r9` (recorded in the model name),
   students at their deploy train-size. `paper_tables.py` WARNS when a dataset group mixes
   `v2-x` and `v2-x+r9` rows; only one may be quoted.
7. **Zero-shot only**: `--gt-cam` probe runs never enter main tables — auto-excluded by
   `paper_tables.py` (and by `gate_check.py`); they may appear only in a separate, clearly
   marked analysis block.
8. **Same split, same n**: `paper_tables.py` WARNS when models within a (dataset, variant)
   group have different image counts, and when an OCMR slice's object pool drifts >0.5%
   across models (fix by recomputing on the common subset from the per-object CSVs).
9. **One sampling implementation**: the 5-point recipe lives ONLY in
   `ocmr_metrics.box5_sample` (≥3-of-5 validity); the SUN annotation track, OCMR-KITTI, and
   the anchoring prototype all import it. No track may redefine sampling.
10. **Slices are not cross-comparable**: SUN (polygon×raw-depth median), KITTI (3D-box
    z_front), and gold (laser point) are different GT conventions — compare models WITHIN a
    slice, never absolute values across slices. `paper_tables.py` carries a `slice` column.
11. **Efficiency rows are same-rig**: `fps_bench.py` stamps GPU/torch/precision/iters into
    its output; paper Tab. VIII (A6000) rows may only appear in a separate block labeled
    with their rig.

## 4. Local baseline numbers (pre-fix harness, for regression reference only)
Old harness (vikhyatk NYU, first-1500 wyrx SUN, pred-clip, min 1e-3, teacher@640):
teacher V2-L NYU 654: AbsRel .0794 δ1 .9469 | SUN-1500: .0875/.9459
official V2-S NYU: .1146/.8925 | SUN-1500: .1169/.8956
S10-ultra-L NYU: .1101/.8771 | SUN-1500: .1064/.8908 (trained on NYU+SUN+own)
These numbers are superseded by the v2 harness and must not appear in the paper.
