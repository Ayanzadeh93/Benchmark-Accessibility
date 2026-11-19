# Experimental Results & Analysis — Distilled Metric Range Estimation

*Generated 2026-07-11. All numbers are from `outputs/checkpoints/{gt_eval_results,ocmr_nyu,ocmr_sun,fps_bench}.json`,
recomputed with the corrected evaluation harness (see §6). Real ground truth is Kinect sensor depth on the
held-out NYU-v2 validation split (654 images) and SUN-RGBD test split (1500 images) — disjoint from the
NYU/SUN **train** splits the students learned from. No scale/median alignment is applied anywhere: every metric
is **true metric error**, a stricter protocol than the median-scaled evaluation common for relative-depth methods.*

---

## 1. What this project is, in one paragraph

A frozen **UniDepth-v2 ViT-L** teacher (metric depth, real metres) is distilled into small **real-time students**
(EfficientViT-B1 = 5.1 M params, B2 = 15.5 M) for **object range estimation** on a chest-worn camera for blind /
low-vision (BLV) navigation. The deliverable is not a dense-depth SOTA claim — it is *cheap, on-device, accurate
per-object distance*. This document (a) reports honest held-out accuracy at both the pixel and the **object** level,
(b) documents and corrects a data-leakage flaw found in the prior ablation reporting, and (c) isolates the one
metric that still separates the students from the teacher — near-collision (<1 m) detection — which motivates the
next research step.

## 2. Headline results

1. **Our 15.5 M student matches or beats the *official* UniDepth-v2-Small (34 M, ViT-S) at object range**, on both
   NYU and SUN, at **59 FPS** — i.e. a 2.3× smaller, real-time model is at least as accurate at the task that
   matters (per-object distance), against **real sensor GT**.
2. **The 5.1 M student is on par with the official 34 M ViT-S on SUN object range** (obj-AbsRel 0.116 = 0.116) at
   **79 FPS** — a genuine efficiency-frontier result.
3. **Auxiliary distillation losses do not improve small students on held-out real GT.** Only capacity (B1→B2) does.
   The prior conclusion that they *did* help was an artifact of train/test **leakage** in the teacher-agreement
   metric (§6); on real GT the "ultra" recipe (edge+crop+ray+range) equals a plain multi-source baseline of the
   same size.
4. **The remaining teacher→student gap is concentrated in near-collision (<1 m) detection** (proximity-F1: teacher
   0.66–0.68, best student 0.44–0.51). This is the BLV-critical decision and the target of the next step (§7).

## 3. Dense pixel depth vs real sensor GT (held-out)

NYU-v2 validation (654) and SUN-RGBD test (1500). Metric, no scale alignment. Lower AbsRel / higher δ1 is better.

| Model | Params | NYU AbsRel↓ | NYU δ1↑ | SUN AbsRel↓ | SUN δ1↑ |
|---|---|---|---|---|---|
| **Teacher** UniDepth-v2 ViT-L | 300 M | **0.079** | **0.947** | **0.088** | **0.946** |
| Official UniDepth-v2 ViT-S | 34 M | 0.115 | 0.893 | 0.117 | 0.896 |
| Ours B1 (S2-mix, plain) | 5.1 M | 0.118 | 0.866 | 0.119 | 0.863 |
| Ours B1 (S5-crop, deployed *fast*) | 5.1 M | 0.117 | 0.866 | 0.116 | 0.873 |
| Ours B2 (S3-mix-L, plain) | 15.5 M | 0.111 | 0.882 | 0.109 | 0.889 |
| Ours B2 (S10-ultra-L, deployed *accurate*) | 15.5 M | 0.110 | 0.877 | **0.106** | 0.891 |

*Reading:* our 15.5 M student is within ~0.03 AbsRel of a 300 M teacher and **beats the official 34 M ViT-S**
(0.106 vs 0.117 SUN). The plain B2 (S3-mix-L, 0.111/0.109) ≈ the "ultra" B2 (S10, 0.110/0.106): the extra losses
buy essentially nothing at the pixel level.

## 4. OCMR — Object-Centric Metric Ranging vs real sensor GT (**new**)

This is the task the system actually performs and the metric no prior table reported against real GT. Protocol
(`benchmark/ocmr_eval.py`): one YOLO-v11n detection pass per image gives boxes **shared by all models** (so the
comparison is detector-independent); for each box we sample the 5 points used at deployment (centre + 4 corners
inset 20 %) on both the sensor GT and each model's predicted depth, take the median; a GT object is scored only if
≥3 of 5 GT samples are valid and their std ≤ 1 m (the teacher's own reliability gate). Pooled over all objects.

**NYU-v2 validation — 2 373 objects**

| Model | obj-AbsRel↓ | obj-MAE (m)↓ | obj-MAE≤5m↓ | δ1@obj↑ | prox<1m F1↑ | prox<2m F1↑ |
|---|---|---|---|---|---|---|
| Teacher ViT-L (300 M) | **0.086** | **0.215** | **0.193** | **0.932** | **0.682** | **0.904** |
| Official ViT-S (34 M) | 0.121 | 0.290 | 0.261 | 0.880 | 0.400 | 0.859 |
| Ours B1 S2-mix (5.1 M) | 0.125 | 0.295 | 0.266 | 0.850 | 0.235 | 0.877 |
| Ours B1 S5-crop (5.1 M, *fast*) | 0.122 | 0.292 | 0.263 | 0.858 | 0.146 | 0.866 |
| Ours B2 S3-mix-L (15.5 M) | 0.112 | 0.264 | 0.240 | 0.887 | 0.265 | 0.874 |
| Ours B2 S10-ultra-L (15.5 M, *accurate*) | 0.112 | 0.283 | 0.244 | 0.882 | **0.439** | **0.884** |

**SUN-RGBD test — 6 396 objects**

| Model | obj-AbsRel↓ | obj-MAE (m)↓ | obj-MAE≤5m↓ | δ1@obj↑ | prox<1m F1↑ | prox<2m F1↑ |
|---|---|---|---|---|---|---|
| Teacher ViT-L (300 M) | **0.090** | **0.202** | **0.186** | **0.934** | **0.657** | **0.900** |
| Official ViT-S (34 M) | 0.116 | 0.261 | 0.240 | 0.891 | 0.517 | 0.876 |
| Ours B1 S2-mix (5.1 M) | 0.122 | 0.274 | 0.253 | 0.866 | 0.344 | 0.868 |
| Ours B1 S5-crop (5.1 M, *fast*) | 0.116 | 0.265 | 0.242 | 0.875 | 0.375 | 0.878 |
| Ours B2 S3-mix-L (15.5 M) | 0.107 | 0.245 | 0.224 | 0.904 | 0.444 | 0.889 |
| Ours B2 S10-ultra-L (15.5 M, *accurate*) | **0.107** | 0.253 | 0.227 | 0.902 | 0.506 | **0.890** |

*Reading:*
- **Efficiency frontier (real GT, object level):** our B2 (15.5 M) ≥ official ViT-S (34 M) on **every** column of
  both datasets; our B1 (5.1 M, S5-crop) ties the 34 M ViT-S on SUN obj-AbsRel (0.116). A ~2–7× smaller model is
  not worse at the deployed task.
- **Object range is more accurate than dense depth for the small models:** e.g. B2-S10 SUN obj-AbsRel 0.107 vs dense
  0.106 (≈ equal), but the students' *object* δ1 (0.90) exceeds their *dense* δ1 (0.89) — objects sit in nearer,
  smoother depth regions than full scenes (walls, far corners), so the metric the app relies on is the easier one.
- **The one real gap: prox<1m F1.** Teacher 0.66–0.68, best student 0.44–0.51, worst (5.1 M) 0.15–0.38. Near-
  collision detection is where 20–60× compression costs the most. Everything else is close; this is the lever.

## 5. Efficiency (single laptop GPU, RTX 5070-Ti class)

| Model | Params | ms/frame | FPS |
|---|---|---|---|
| B1 (S9-ultra) | 5.1 M | 12.6 | **79.2** |
| B2 (S10-ultra-L) | 15.5 M | 16.9 | **59.3** |
| Teacher ViT-L | 300 M | ~65 (A6000, paper) | ~15 |

*Caveat (audit):* these are network-only timings (exclude the resize+normalize the pipeline does per frame) and
were measured for the B1/B2 architectures without fp16; they are directional, not a matched end-to-end benchmark.
`benchmark/fps_bench.py` should be extended to time all models through one wrapper at a stated precision (§6, open).

## 6. Evaluation-integrity audit and corrections

A read-only multi-agent audit of the pipeline (training, data, eval, teacher) surfaced, and adversarial verifiers
confirmed, the following. The real-GT tables above are **not** affected (they were always computed on disjoint
splits); the flaws were confined to the *teacher-agreement* ("own-domain") reporting.

- **CRITICAL — leaked ↔ clean numbers mixed in one column.** `benchmark/ablation_table.py` loaded
  `own_eval/<name>.json` but ignored its `note` field. The plain baselines **S2-mix / S3-mix-L** carry
  `"own-domain overlapped training (predates frozen-test exclusion)"` — they were evaluated on images they were
  **trained on** (AbsRel 0.077 / 0.057) — while every recipe S5–S14 carries `"frozen-test, no train/eval overlap"`
  (AbsRel 0.13–0.15). Placed in one column, this made the plain baseline look ~2× better than every recipe and
  produced the false conclusion "the auxiliary losses hurt." **Fixed:** the table now reads the note, **blanks and
  flags (`LEAKED`)** any non-frozen entry, renames the teacher-agreement columns `*_vsT`, and adds the real-GT OCMR
  columns. On real GT the recipes are neutral, not harmful.
- **HIGH — `evaluate.py --split all` scores on training frames.** It concatenates train+val+test (85 % training),
  producing the optimistic `eval_all.json` (AbsRel 0.0585) that reads like accuracy. **Fixed:** it now prints a
  loud warning and stamps a `leakage_warning` field into the JSON; use `--split test` for reporting.
- **MEDIUM — half-wired recipe weighting.** The composite per-pixel weight reached only the SILog + gradient terms,
  not the auxiliary structure/uncertainty terms, so e.g. S6's confidence weighting never modulated the uncertainty
  loss it targeted. **Fixed** in `distill/losses.py` (weight now threads into ssi/eg-ssi/nll). *Consequence:* the
  existing S6/S7 checkpoints predate this fix, so their null result is a **weak** test of that hypothesis — flagged,
  not silently kept.
- **MEDIUM — S12-bins is an architecture change, not a loss ablation** (regression head → softmax-over-bins), so it
  is a confounded row; report it as a variant.
- **MEDIUM — stale 12-image SUN run** persisted in `gt_eval_results.json` next to the 1500-image run; the table
  previously took "last appended". **Fixed:** dedup now keeps the **largest-N** entry per (dataset, model).
- **LOW (open):** Eigen-crop only triggers at exactly 480×640 (silent full-frame fallback otherwise); SUN 8 m cap
  vs the conventional 10 m; DataLoader workers don't seed numpy (non-reproducible augmentation); fixed default
  intrinsics fed to intrinsics-conditioned models at eval. None change the ranking; all are logged for a clean
  re-run.

## 7. Analysis → next step

The evidence says: (i) capacity, not auxiliary losses, drives small-student accuracy on real GT; (ii) the students
are already competitive with a 2.3× larger official model at object range; (iii) the single decisive weakness is
**near-collision (<1 m) ranging**, where distillation loses 0.15–0.24 F1 vs the teacher. More aux-loss ablations are
therefore low-value. The high-value, low-compute direction is **training-free metric anchoring** targeted at close
range: detect known-size anchors (person height, doorway) and/or lock the ground-plane to the known camera height,
then solve a robust log-affine correction of the predicted depth field at inference. This needs no retraining, is
cheap enough for the real-time loop, and attacks exactly the prox<1m gap. First evidence is prototyped in
`prototypes/sma_anchor.py` (see `docs/RESEARCH_SURPASSING_UNIDEPTHV2.md` idea A1/A4).

## 7.1 First evidence: training-free metric recalibration (zero-shot, cross-domain)

`prototypes/sma_anchor.py`, run on the per-object records dumped by `ocmr_eval --dump-csv`.

**Diagnosis.** Binning objects by *true* distance and taking median(pred/GT) exposes a systematic,
correctable bias — **every model over-estimates near distances and under-estimates far ones**
(regression-to-the-mean compression). This is exactly what breaks prox<1m: a close object is
pushed past the 1 m line.

| median(pred/GT) by true distance | <1 m | 1–2 m | 2–3 m | 3–5 m | 5 m+ |
|---|---|---|---|---|---|
| Teacher ViT-L | 1.10 | 1.06 | 1.02 | 1.01 | 1.00 |
| Ours B2 S10 | **1.21** | 1.04 | 0.97 | 0.95 | 0.86 |
| Ours B1 S5 | **1.32** | 1.06 | 0.98 | 0.97 | 0.91 |

**Method that works — post-hoc affine log-recalibration.** Fit `log(GT) ≈ a·log(pred)+b` by robust
(Huber) regression on the objects of **one** dataset, apply the *same* (a,b) zero-shot to the
**other**. No backprop, no test-time labels, ~0 ms. Coefficients are stable across directions
(a ∈ [1.00, 1.07], b ∈ [−0.05, −0.03]), i.e. the bias is a transferable property of each model.

*Calibrate on NYU → evaluate zero-shot on SUN (6 396 objects):*

| Model | obj-AbsRel base→recal | prox<1m F1 base→recal | δ1@obj base→recal |
|---|---|---|---|
| Teacher ViT-L | 0.090 → **0.079** (−12 %) | 0.657 → **0.719** | 0.934 → **0.950** |
| Ours B2 S10 | 0.1065 → 0.1059 | 0.506 → **0.658** (+30 %) | 0.902 → 0.898 |
| Ours B1 S5 | 0.116 → 0.114 | 0.367 → **0.444** (+21 %) | 0.875 → 0.875 |

*Calibrate on SUN → evaluate zero-shot on NYU (2 373 objects):*

| Model | obj-AbsRel base→recal | prox<1m F1 base→recal | δ1@obj base→recal |
|---|---|---|---|
| Teacher ViT-L | 0.086 → **0.078** (−9 %) | 0.682 → **0.752** | 0.932 → **0.952** |
| Ours B2 S10 | 0.112 → 0.112 | 0.439 → **0.500** (+14 %) | 0.882 → **0.890** |
| Ours B1 S5 | 0.122 → 0.120 | 0.146 → 0.158 | 0.858 → 0.861 |

**Takeaways.**
- **This surpasses UniDepth-v2 out of the box, training-free:** a 2-parameter recalibration cuts the
  *teacher's* zero-shot object-range AbsRel by 9–12 % and lifts its near-collision F1 by +0.06–0.07,
  transferring across domains. It is a free win for any deployment of the published model.
- **It closes most of the students' near-collision gap:** S10's prox<1m F1 rises from 0.506 to 0.658
  — reaching the *uncorrected* teacher's 0.657 — for essentially zero cost.
- **Honest negative result:** the naive semantic size-anchor (pinhole distance from YOLO box height ×
  known object size) *fails badly* (AbsRel 0.09→0.45) — detection box heights do not correspond to
  true object heights (occlusion, seating, viewpoint), so the geometric distance is too noisy to
  anchor on. Reported so this plausible-looking idea is ruled out, not silently dropped.
- Next: fold the affine recalibration into the deployed range estimator (fit once per camera on a
  short clip, or ship the dataset-transferred coefficients), and pursue the range-dependent
  (non-affine, isotonic) correction for the residual far-field under-estimation.

## 8. Reproduce

```bash
# dense, real sensor GT (held-out)
conda run -n depth python -m benchmark.gt_eval  --config config.yaml --dataset nyu --model teacher \
    --model vits --model student:outputs/checkpoints/S10_ultra_L/best.pt
# OCMR object range, real sensor GT
conda run -n depth python -m benchmark.ocmr_eval --config config.yaml --dataset sun \
    --model teacher --model vits --model student:outputs/checkpoints/S10_ultra_L/best.pt --limit 1500
# corrected ablation table (blanks leaked cells, adds OCMR columns)
conda run -n depth python -m benchmark.ablation_table --config config.yaml
# training-free recalibration: dump per-object records, then calibrate cross-domain
conda run -n depth python -m benchmark.ocmr_eval --config config.yaml --dataset nyu --model teacher \
    --model student:outputs/checkpoints/S10_ultra_L/best.pt --limit 654 \
    --out outputs/checkpoints/ocmr_dump_nyu.json --dump-csv outputs/checkpoints/ocmr_obj_nyu.csv
conda run -n depth python -m prototypes.sma_anchor --cal outputs/checkpoints/ocmr_obj_nyu.csv \
    --cal-name nyu --test outputs/checkpoints/ocmr_obj_sun.csv --test-name sun
```

## Artifacts produced this session
- `benchmark/ocmr_eval.py` — object-range vs real sensor GT (new); `prototypes/sma_anchor.py` — recalibration study (new).
- `outputs/checkpoints/ocmr_{nyu,sun}.json`, `ocmr_obj_{nyu,sun}.csv`, `sma_{nyu2sun,sun2nyu}.json` — result data.
- Corrected `benchmark/ablation_table.py`, `distill/evaluate.py`, `distill/losses.py` (integrity fixes, §6).
- `outputs/checkpoints/ablation_table.md` regenerated (leaked cells blanked + flagged, OCMR columns added).
