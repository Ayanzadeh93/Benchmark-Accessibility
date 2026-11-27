# b20dws v2 — improved version (plan; nothing trained yet)

Branch: `b20dws-v2-improved`, started from tag `b20dws-v1` (commit 92f73d7).
v1 stays frozen on `b20dws-v1-indomain` (see `docs/B20DWS_V1_FREEZE.md`).

## Goal

All five evaluation sets zero-shot (IBims-1, NYU-v2, SUN RGB-D, DIODE indoors, ETH3D),
same 5.04M architecture and latency budget, reported as mean ± std over 3 seeds.

## Planned changes vs v1 (each one is kept only if it does not lower results; see rule below)

1. **Zero-shot-clean data:** no NYU-v2 and no SUN RGB-D in training. Remove the 50 own rows that
   match NYU-test ids. Replace their share with ARKitScenes / Hypersim (already on Z:/Datasets),
   preferably relabelled by the teacher (UniDepthV2-L) so the metric scale matches the own rows.
2. **Fixed seeds:** v1 sets no seed; rep1 showed raw δ1 moving up to 4.4 points between identical
   runs. Add a `--seed` and train 3 seeds.
3. **Recipe from the ablation:** drop any b20 component the leave-one-out ablation
   (`scripts/run_ablation_b20.sh`, results in `results_ablation/`) shows to be within run-to-run noise.
4. **Evaluation unchanged:** same harness and protocol as v1 so v1 and v2 rows are comparable.

## Rule

Any change that lowers a zero-shot metric beyond the seed noise is flagged and not kept.

## Status

- 2026-11-27: branch created. The v1 ablation queue is still running from this working tree
  (code identical to v1). No v2 training is started until the queue finishes and the plan is approved.
