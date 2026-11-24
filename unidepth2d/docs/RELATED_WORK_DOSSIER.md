# Related-work dossier — positioning for the OCMR / anchoring / metric-distillation paper

Compiled 2026-07-11 from a fan-out web search (5 angles, deep-research harness) plus
paper-PDF verification of UniDepthV2 (arXiv 2502.20110v2). The harness's fetch/verify
phase was cut short by session limits, so each entry carries a **verification status**:
- [V] verified this session against the primary source (PDF/abstract read directly)
- [S] from search-agent snippet — verify the specific claim before citing numbers in the paper

Companion file: PROTOCOL_NOTES.md (verbatim UniDepthV2 tables and protocol facts, all [V]).

---

## Area 1 — Zero-shot metric monocular depth SOTA (2024–2026)

| Work | Venue | Output | Relevance to us | Status |
|---|---|---|---|---|
| UniDepthV2 (2502.20110) | TPAMI'25 | metric depth + camera + conf | Our backbone/target. Zero-shot ten-set suite; δ1^SSI/F_A/ρ_A protocol | [V] |
| UniK3D (Piccinelli et al.) | CVPR'25 | metric 3D points, any camera | Same authors' successor; owns the camera-generality axis — cite, use as 2nd frozen backbone if compute allows; do NOT lead on camera axis | [S] |
| Metric3D v2 (2404.15506) | TPAMI'24 | metric depth + normals (canonical camera) | Baseline; its canonical-camera trick is prior art adjacent to scale correction | [S] |
| Depth Pro (2410.02073) | ICLR'25 | metric depth + focal | Baseline; source of SI-boundary F1/Recall metrics (code: `eval/boundary_metrics.py` in their repo — port for exactness) | [S] |
| MoGe-2 (2507.02546) | NeurIPS'25 | metric point maps (MoGe-1 was affine-invariant) | Baseline for OCMR table | [S] |
| Depth Anything V2 (2406.09414) | NeurIPS'24 | relative; metric via per-domain fine-tune | DA-V2-S-metric (~25M) is the efficient-student reference point | [S] |
| **Depth Anything 3 (2511.10647)** | ICLR'26 | any-view geometry, depth-ray target | 2026 frontier; include as OCMR baseline if a metric single-view mode is exposed | [S] |
| MASt3R / VGGT line | CVPR'24/25 | pointmaps (pairwise/multi-view) | Context only; UniDepthV2 already compares vs MASt3R | [V paper table] |

Published zero-shot indoor rows to compare against: see PROTOCOL_NOTES §1 (Table I verbatim).

## Area 2 — Per-object / instance-level distance estimation (OCMR novelty kill-check)

| Work | Venue | Setting | Why OCMR still stands | Status |
|---|---|---|---|---|
| MOM: Masked Object Modeling per-object distance (2401.03191 + CVIU'25 journal) | CVIU'25 | KITTI/nuScenes/MOTSynth, trained regressors | Outdoor/automotive; no indoor; no zero-shot foundation-model evaluation; no uncertainty/proximity metrics | [S] |
| **Wildlife MMDE benchmark (2510.04723)** | arXiv'25 | camera-trap animals, 6 zero-shot MMDE baselines, protocol comparison incl. median extraction | **Biggest overlap red-flag**: anticipates our protocol-ablation idea. Mitigate: outdoor, single class, 93 images, no object-AbsRel/AUSE/proximity-F1, no SUN RGB-D, no efficiency axis. CITE PROMINENTLY | [S] |
| SpatialRGPT-Bench (2406.01584) | NeurIPS'24 | VLM QA incl. object-centroid distances (urban+indoor+sim) | VLM-answer tolerance framing, not a depth-model benchmark; no official-split rigor, no depth-style metrics | [S] |
| Q-Spatial-Bench (2409.09788) | arXiv'24 | 99 ScanNet images, expert-measured distances, VLM prompting | Tiny, ScanNet, VLM-side; its "reference object" reasoning is a VLM analogue of our anchors — cite in both areas | [S] |
| MapAnything urban-asset localization (2509.14839) | arXiv'25 | per-asset geolocalization, street-level | Outdoor mapping framing | [S] |
| BenchDepth (2507.15321) | arXiv'25 | re-benchmark depth FMs via 5 downstream tasks | No object ranging → supports OCMR; cite as the field's task-aligned-evaluation statement | [S] |
| DisNet & KITTI object-distance line (older) | – | outdoor regressors | classic citations for the task's history | [S, gather refs in writing pass] |

**Verdict**: no existing benchmark does indoor per-object METRIC ranging on an official
zero-shot split (SUN RGB-D) with foundation-model + efficient-student baselines, safety
(proximity-F1), object-level uncertainty (AUSE@obj), size stratification, and a
bbox-vs-mask protocol ablation. OCMR's framing sentence writes itself from this table —
but the wildlife benchmark and SpatialRGPT-Bench must be cited in the first paragraph,
not buried.

## Area 3 — Training-free / test-time metric scale correction (contribution b kill-check)

| Work | Venue | Mechanism | Delta of our SMA + ground-lock | Status |
|---|---|---|---|---|
| **GVDepth (2412.06080)** | arXiv'24 | fuses object-size + camera-height/ground-plane cues, uncertainty-weighted | **Nearest neighbor.** But: TRAINED network, fixed-height ground vehicles, road scenes. Ours: training-free closed-form robust fit on a FROZEN already-metric FM, indoor/egocentric, per-object eval | [S] |
| Metric from Human (NeurIPS'24, OpenReview GA8TVtxudf) | NeurIPS'24 | humans as known-size anchors, test-time optimization, lifts RELATIVE→metric | Single anchor class; optimization-based TTA; relative-depth regime. Ours: multi-class dictionary + plane lock, closed form, corrects RESIDUAL drift of metric FMs | [S] |
| RSA (NeurIPS'24) | NeurIPS'24 | language description → global linear scale for relative depth | trained text regressor, global scale only | [S] |
| TTA w/ sparse metric points (2412.14103) | arXiv'24 | global affine fit to external sensor points on frozen FM | anchors are sensors, not semantics; not uncertainty-weighted per-object | [S] |
| StableCamH (2312.04530) | arXiv'23 | camera-height aggregation from object heights | training-time, self-supervised driving | [S] |
| Language as Prior, Vision as Calibration (2601.01457) | arXiv'26 | captions → calibration envelope for relative backbone | 2026 competitor, sensor-free; our delta = explicit geometric anchors + robust log-affine + object-level eval | [S] |
| Metric3D/ZeroDepth canonical camera | TPAMI'24 etc. | train-time canonicalization | orthogonal (training-time), cite as the intrinsics-based scale mechanism we complement | [S] |

**Positioning sentence for (b)**: prior test-time scale recovery LIFTS relative depth to
metric (humans/language/sensors as the only anchor) or TRAINS fusion networks for fixed
rigs; we show that even *metric* foundation models retain a correctable low-order scale
drift OOD, and that a training-free, uncertainty-weighted robust log-affine fit over a
multi-class size dictionary + a single camera-height scalar recovers most of it —
evaluated per-object, where scale errors actually bite.

## Area 4 — Efficient / distilled metric depth (contribution c)

| Work | Venue | Params/speed | Metric? | Delta | Status |
|---|---|---|---|---|---|
| Distill Any Depth (2502.19204) | CVPR'25 | DAv2-S-based | **relative only** (SSI-normalized distill) | our SILog+grad keeps metres; object-aligned loss | [S] |
| **ZipDepth (2607.08771)** | arXiv'26 | 6.1M, 34 FPS Jetson | **relative only** (SSI objective) | the 2026 foil: no metric, no camera cond., no uncertainty | [S] |
| DA-V2-S metric fine-tunes | NeurIPS'24 | ~25M | metric via per-domain FT | not zero-shot-metric; benchmark against on NYU | [S] |
| SharpDepth (2411.18229) | CVPR'25 | large models | metric (sharpening) | refinement of big models, not compression | [S] |
| LMDepth (2505.00980) | arXiv'25 | edge Mamba, 84-122 FPS Jetson | per-dataset | efficiency reference row | [S] |
| MMDE survey (2501.11841) | arXiv'25 | – | – | landscape scaffold; confirms gap | [S] |

**Claim shape for (c)**: first ≤16M-param student that (i) preserves a metric foundation
teacher's absolute scale zero-shot (indoor suite), (ii) is camera-conditioned
(FiLM/ray), (iii) carries calibrated per-object uncertainty — at 59–79 FPS on a laptop
GPU. Registry ablations S0–S14 become the supporting grid.

## Area 5 — Depth uncertainty & conformal prediction

| Work | Venue | What it does | Delta | Status |
|---|---|---|---|---|
| UQ ∩ foundation-models synthesis (2501.08188) | arXiv'25 | systematic UQ eval on depth FMs (AUSE) | the must-cite map; no per-object intervals | [S] |
| Gradient-based UQ revisited (2502.05964) | arXiv'25 | post-hoc UQ for frozen depth nets, pixel AUSE | pixel-level only → object-level AUSE + conformal ranges are open | [S] |
| Two-step conformal bboxes (2403.07263) | ECCV'24 | conformal 2D box coordinates | the object-level conformal precedent; we conformalize metric RANGE | [S] |
| Conformalized distance fields (2607.00776) | arXiv'26 | functional conformal on predicted distance fields (planning) | robotics motion-planning regime; watchlist | [S] |
| UniDepthV2 Tab. VII | TPAMI'25 | pixel AUSE/nAUSE/ρ, zero-shot collapse (nAUSE .54–.65, ρ≈.29) | the weakness our B3/uncertainty work attacks | [V] |

## Area 6 — Evaluation-metric innovations to adopt/cite

- Depth Pro SI-boundary F1/Recall (port from their `eval/boundary_metrics.py`; our
  `benchmark/protocol.py:boundary_f1_si` is currently our own implementation — replace or
  cross-check before paper numbers). [S]
- **"How (Should One) Evaluate Monocular Depth Estimation?" (2510.19814, princeton-vl/evalmde)**
  — Oct'25 metric-sensitivity study, proposes relative-surface-normal + composite metrics;
  its taxonomy contains NO object-centric ranging metric → cite when introducing
  object-AbsRel/δ1@obj/proximity-F1/AUSE@obj. [S]
- UniDepthV2's F_A (F1-AUC, log-spaced thresholds min_depth→max_depth/20) and ρ_A
  (ray-angular AUC ≤15°) — implemented in protocol.py. [V]
- BenchDepth downstream-proxy evaluation (2507.15321) — cite as complementary. [S]
- d_AUC, τ@3%, median-log (official evaluation_depth.py). [V]

## Positioning matrix (contribution × nearest prior × concrete delta)

| Ours | Nearest prior | Concrete delta |
|---|---|---|
| OCMR benchmark | Wildlife MMDE bench (2510.04723); MOM (2401.03191); SpatialRGPT-Bench | indoor official zero-shot split (SUN, NYU-filtered), multi-class, foundation + efficient baselines, proximity-F1/AUSE@obj/size-strata, bbox-vs-mask ablation, egocentric measured gold slice |
| Training-free SMA + ground-lock | GVDepth; Metric from Human; RSA; 2412.14103 | training-free closed-form robust log-affine on FROZEN METRIC FM (residual-drift regime), multi-class dictionary + one extrinsic scalar, Σ-weighted, evaluated at object level |
| Metric-preserving student distillation | Distill Any Depth; ZipDepth; DA-V2-S-metric | keeps absolute scale zero-shot at 5–15M params, camera-conditioned, object-distance-consistency loss, β-NLL + conformal object intervals |
| Object-level conformal ranges | ECCV'24 conformal boxes; 2607.00776 | conformalize metric per-object range from monocular depth + uncertainty; safety framing (proximity alerts) |

## Red-flag list & mitigations

1. **Wildlife MMDE benchmark** anticipates protocol comparison for object distance from
   zero-shot MMDE. → Cite in intro; frame OCMR as the general indoor/egocentric,
   multi-class, safety-metric formalization; include their median-extraction protocol as
   one of our ablation arms.
2. **GVDepth** fuses the same two cues (size priors + camera height). → Emphasize
   training-free + frozen-metric-FM regime + indoor/egocentric + per-object eval; add
   GVDepth-style learned fusion as a baseline row if code available.
3. **UniK3D** (same authors) may be reviewers' "why not build on the newer one?" →
   position contributions as backbone-agnostic; run teacher-swap ablation (UniDepthV2 vs
   UniK3D) for at least one table.
4. **ZipDepth/Distill-Any-Depth** own "small distilled depth" mindshare → lead student
   claims with METRIC preservation + task alignment, never raw speed.
5. SpatialRGPT/Q-Spatial suggest VLMs already "do" object distance → add a VLM row
   (GPT-4o or Qwen-VL from the parent repo's VQA stack) to OCMR to show the gap
   quantitatively — turns the threat into a result.
6. Session-limit caveat: all [S] rows need primary-source verification before camera-ready
   citations (arXiv IDs with 26xx numbers especially — 2601.01457, 2607.00776, 2607.08771).

## To do in a later verification pass
- Fetch each [S] arXiv abstract; confirm venue/claims/numbers; upgrade to [V].
- Pull DepthPro boundary-metric code; cross-check `boundary_f1_si`.
- Check UniK3D repo for indoor zero-shot SUN/IBims rows usable as published baselines.
- Gather classic DisNet/SVR object-distance refs for the related-work section.
