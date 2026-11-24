# Paper plan — OCMR: Object-Centric Metric Ranging (working title)

Target: CVPR 2027 (abstract ~Nov 2026). Lead claim (locked): OCMR benchmark +
training-free anchoring; efficient metric-preserving students as systems payload.
Every table cell below is marked [ours] (measured by our harness) or
[paper: UniDepthV2 Tab.N — only after docs/GATE_REPORT.md passes]. Numbers are never
hand-copied: results auto-quote from results_bench/ and results_ocmr/.

## Title candidates
1. "How Far Is That Chair? Object-Centric Metric Ranging as the Missing Evaluation
   Axis for Monocular Metric Depth"
2. "OCMR: Benchmarking Per-Object Metric Range from Monocular Depth Foundation Models"
3. "Metric Depth Meets the Object: A Benchmark and Training-Free Anchors for
   Monocular Range Estimation"

## Abstract — claims checklist (each must be backed by a table/figure)
- [ ] C1: Dense zero-shot depth metrics do not predict per-object ranging quality;
      we formalize OCMR with 57.7k object ranges on the official (NYU-filtered)
      SUN RGB-D test split + a measured egocentric gold set. [Tab.2, Fig.2]
- [ ] C2: Foundation models retain correctable residual metric-scale drift; a
      training-free ground-plane/camera-height lock + semantic size anchors
      reduce object-range error by X% (models with drift) at zero training cost,
      with identity fallback. [Tab.4]
- [ ] C3: Metric-preserving task-aligned distillation yields 5–15M students at
      59–79 FPS whose OBJECT-level ranging approaches/DENSE-beats the 34M official
      ViT-S; recipe grid of 15 ablations. [Tab.3, Tab.5]
- [ ] C4: Object-level uncertainty: per-object AUSE + conformal range intervals;
      the paper's own confidence output is error-like and collapses OOD (their
      Tab.VII), our calibrated student intervals reach nAUSE@obj <= Y. [Tab.6]

## 1. Introduction
- Hook: assistive/robotic decisions are per-OBJECT ("is the chair within 1 m?"),
  benchmarks are per-PIXEL. Dense δ1 saturates (>94 indoor) while proximity-F1@1m
  is far from solved [first OCMR numbers].
- Gap analysis grounded in the audit: UniDepthV2's ten-set suite has zero
  object-level evaluation (PROTOCOL_NOTES §1); the survey 2501.11841 names
  instance-aware aggregation an open problem; wildlife bench 2510.04723 is the
  nearest attempt (single class, outdoor, 93 images) — cite in para 2, not buried.
- Contributions list = C1..C4.

## 2. Related work (skeleton -> RELATED_WORK_DOSSIER.md tables)
2.1 Universal MMDE (UniDepth/V2, UniK3D, Metric3Dv2, DepthPro, MoGe-2, DA-V2/3).
2.2 Per-object distance estimation (MOM/CVIU'25, DisNet line, wildlife bench,
    MapAnything, VLM spatial benches SpatialRGPT/Q-Spatial).
2.3 Test-time metric scale correction (GVDepth, Metric-from-Human, RSA, sparse-
    point TTA, StableCamH; canonical-camera training-time line).
2.4 Efficient/distilled depth (Distill-Any-Depth, ZipDepth, DA-V2-S-metric,
    SharpDepth) — all relative or per-domain-FT; none metric zero-shot <30M.
2.5 Depth uncertainty & conformal (UQ-FM survey, gradient UQ, conformal bboxes).

## 3. OCMR: task, protocol, benchmark
- 3.1 Task: visible-surface range per object; why not 3D centroid (monocular
  rangers estimate the visible surface; safety needs nearest-surface).
- 3.2 GT construction: SUN alltest (NYU-overlap removed; 4,379 imgs), 2D instance
  polygons x RAW sensor depth (median + p10), quality gates; class stats;
  57,677 objects; range distribution (med 2.26 m, 39% < 2 m). Plural/synonym
  canonicalization pass [todo before release].
- 3.3 Metrics: objAbsRel, objD1, MAE(@5m), bias_log, proximity-P/R/F1@{1,2}m,
  range-band breakdown, per-class, AUSE@obj/nAUSE@obj/Spearman@obj
  (benchmark/ocmr_metrics.py; cite DepthPro boundary-F1 + evalmde 2510.19814 as
  metric-design precedents).
- 3.4 Protocols: box5 (deployed recipe) / maskmed (oracle mask) / center /
  det-box5 (YOLO-in-the-loop, benchmark/ocmr_eval.py) — the protocol ablation IS
  a finding (which aggregation is reliable for thin/occluded objects).
- 3.5 Gold egocentric subset: capture protocol (docs/GOLD_CAPTURE_PROTOCOL.md),
  target 200–500 laser-measured ranges, chest-height camera; consent rules.
- 3.6 Dense-suite context: our harness reproduces UniDepthV2's official metrics
  bit-exactly (protocol.py self-test) and their published rows within tolerance
  on SUN/IBims after protocol archaeology (GATE_REPORT.md): raw-δ1 semantics,
  squared-distance F_A quirk, confidence-is-error, res-level policy — these
  documented quirks are themselves a service to the community (subsection or
  supplementary "protocol notes").

## 4. Methods
- 4.1 Training-free anchoring (prototypes/anchor_lock.py):
  ground-lock (one extrinsic scalar; robust plane in predicted 3D) and SMA
  (size-prior votes, Huber-fused log-scale, identity fallback); post-hoc affine
  recalibration (prototypes/sma_anchor.py) as the calibration-transfer variant.
- 4.2 Metric-preserving task-aligned distillation (existing pipeline): SILog+grad
  (metric anchor), EG-SSI, ray/FiLM camera conditioning, per-object
  distance-consistency loss; NEW losses (Phase 4): anchor-consistency loss
  (A1-as-loss), proximity-weighted SILog (safety band <2 m up-weighted),
  calibrated beta-NLL + split-conformal per-object intervals.
- 4.3 Zero-shot-clean training mix (no SUN) for honest zero-shot student rows.

## 5. Experiments
- 5.1 Dense context table: NYU-654 official / SUN-raw alltest / IBims-1.
      Rows: V2-S/B/L [ours, res-9], V1-L [ours], students [ours];
      published UniDepthV2 rows quoted [paper] ONLY post-gate with checkpoint-drift
      footnote for V2-S. Metrics: d1, d1_ssi, ARel, SILog, F_A, rho_A.
- 5.2 OCMR main table [ours, all measured]: V2-S/B/L, V1-L, students,
      (stretch: Metric3Dv2, DA-V2-metric, DepthPro, MoGe-2 — add per availability),
      VLM row (Qwen-VL via HF fp16; Ollama-quantized only as deployed config).
      Protocols box5 + maskmed; per range-band; proximity-F1.
- 5.3 Anchoring results: base vs ground vs sma vs both, per model (V2-L should
      barely move = validity check; students should move a lot = the win);
      identity-fallback sanity (<=1% abs degradation without anchors).
- 5.4 Uncertainty: pixel AUSE/nAUSE (vs their Tab.VII post-gate) + object-level
      AUSE + conformal coverage/width for S6-unc student + teacher conf.
- 5.5 Efficiency: params/latency/VRAM on RTX 5070 laptop [ours] + A6000 rows
      [paper Tab.VIII post-gate]; FPS-vs-objAbsRel Pareto figure.
- 5.6 Ablations: (a) existing S0–S14 grid (registry.json + gt_eval history);
      (b) new losses on/off; (c) protocol ablation box5/mask/center/det;
      (d) anchor dictionary size sweep; (e) teacher-swap (UniDepthV2 -> UniK3D)
      backbone-agnosticism check [stretch].
- 5.7 Gold-set evaluation: all of the above on measured egocentric data.

## 6. Limitations & ethics
Indoor scope (outdoor = KITTI sanity only); pseudo-GT corpus only for distillation
claims; SUN annotation noise (polygon quality, plural classes) quantified; assistive
framing without clinical claims; consent protocol for person captures.

## Figures
F1 teaser: dense-δ1-saturated vs proximity-F1 gap. F2 GT construction pipeline.
F3 anchoring geometry (plane + size votes). F4 Pareto FPS vs objAbsRel.
F5 qualitative: box5 vs maskmed failures (thin objects); anchor corrections.

## Risk register
| Risk | Mitigation |
|---|---|
| V2-S published-row mismatch (checkpoint drift) | measured rows everywhere; footnote; gate report in supplement |
| Anchors don't help V2-L in-domain | expected & framed as validity; wins live on students/OOD/gold |
| Wildlife-bench/SpatialRGPT novelty attack | cited in intro; deltas table in related work |
| SUN class noise | canonicalization pass + per-class filtering (n>=25) + release cleaned list |
| Gold set slips | paper stands on SUN GT; gold becomes camera-ready addition |
| GPU limits (12GB laptop) | all trainings are <=16M students; teacher runs are inference-only |

## Timeline to CVPR 2027 (deadline ~Nov 2026)
- Jul: harness + gate [done-ish], OCMR v1 numbers [running], anchoring prototype.
- Aug: Phase 4 trainings (zero-shot-clean, new losses), uncertainty/conformal,
  gold capture (user), baseline breadth (Metric3Dv2/DA-V2/DepthPro on OCMR).
- Sep: ablation completion, figures, writing v1; teacher-swap stretch.
- Oct: internal review pass, supplementary (protocol notes), polish.
- Nov: submit.
