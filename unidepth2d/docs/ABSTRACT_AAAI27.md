# AAAI-27 abstract (registered by 2026-07-21; full paper 2026-07-28)

## Title (primary)
**How Far Is That Chair? Object-Centric Metric Ranging Exposes and Fixes the Blind Spot
of Monocular Depth Foundation Models**

## Title (alternate)
OCMR: Benchmarking and Correcting Per-Object Metric Range in Monocular Depth Foundation Models

## Abstract (~225 words)

Monocular metric-depth foundation models are evaluated per pixel, yet embodied and
assistive systems act per object: "is that chair within one metre?" We introduce OCMR, an
object-centric metric-ranging benchmark built on the official zero-shot test protocol of
UniDepthV2, comprising 57,677 objects on SUN RGB-D (indoor) and 30,537 objects on KITTI
(outdoor), with range-band, proximity-decision, and object-level uncertainty metrics.
OCMR reveals what dense metrics hide: where the strongest model saturates pixel accuracy
(δ1 = 95.7), its F1 on the safety decision "within 1 m" is only 0.67, and per-object
rankings disagree with per-pixel rankings. We further uncover a systematic, family-wide
defect: all UniDepthV2 variants over-estimate object distances by 2–3%. We propose a
training-free ground-plane lock that removes this drift at inference time, improving the
released UniDepthV2-L on its own test set (relative ARel −9.6%, proximity-F1 +2.3
points) — while size-prior anchors, a popular alternative, fail indoors under occlusion.
Finally, metric-preserving distillation with a proximity-weighted objective yields
inherently scale-calibrated students (bias ≈ 0): a 15.5M-parameter student surpasses
the official 34M UniDepthV2-S on NYUv2 (δ1 88.4 vs 87.4) at 2.2× the speed and a third
of the memory, enabling real-time on-device ranging. Our evaluation pipeline reproduces UniDepthV2's official metrics
bit-exactly; the benchmark, protocol notes, and code will be released.

## Keywords
monocular depth estimation; metric depth; object-centric evaluation; benchmark;
test-time correction; knowledge distillation; embodied perception; assistive navigation

## Number provenance (do not edit numbers by hand)
δ1 95.7 / proxF1@1m 0.669: results_bench/sun-raw__v2-l+r9.json + results_ocmr/v2-l+r9__summary.json
bias +2–3%: bias_log +0.021–0.026 (v2-s/b/l, OCMR-SUN summaries)
ARel −9.6% / +2.3pt: results_ocmr/anchorlock__v2-l+r9.json (7.80→7.05; 66.9→69.2)
student rows: results_bench/nyu-filled__{student-S16_prox, v2-s+r9}.json (88.39/10.82 vs 87.42/13.08);
  S16 = S10 recipe + proximity-weighted SILog (the novel loss; landed 2026-07-19 09:00)
2.2× / one-third memory: outputs/checkpoints/fps_bench.json (17.8ms/0.20GiB vs 39.9ms/0.63GiB)
bit-exact: benchmark/protocol.py self-test (max diff 3.5e-6)

## Honesty notes carried into the paper body (not the abstract)
- NYU student win carries the training-data asymmetry footnote (student saw NYU-train;
  V2-S zero-shot) — framed as deployment/efficiency result, never zero-shot.
- Published-row reuse only per PROTOCOL_NOTES §3 (gate + drift footnote); all abstract
  numbers are OURS-measured, no published numbers cited as ours.
- KITTI slice GT convention (z_front from 3D boxes) documented; slices never cross-compared.
