# unidepth2d — UniDepth v2 teacher annotation + distillation to a small student

Self-contained pipeline, separate from the main AidGPT / InVLM code. Two stages:

1. **Teacher (`teacher/`)** — run the frozen **UniDepth v2** metric-depth foundation model over the
   dataset at `C:\Tim\nature\database\Fianl Version`, producing:
   - a per-pixel **metric** depth map per image (real meters, saved as 16-bit PNG in millimetres),
   - a per-pixel confidence map,
   - **per-object distances** at the frozen detection boxes, using the exact recipe from the
     *Distance Perception in VLMs for Blind Navigation* paper: sample 5 points (bbox centre + 4
     corners inset 20% from each edge), take the **median**, flag **uncertain** when the 5-sample
     **std > 1 m**. Distances in metres and feet, sorted nearest-first, `<1 m` proximity-flagged.
   - a manifest CSV + run summary.

2. **Distill (`distill/`)** — train a small, real-time **student** (timm encoder + light DPT-style
   decoder) to reproduce the teacher's **metric** depth. Loss is metric-preserving (SILog primary +
   multi-scale gradient matching), adapting the *Distill Any Depth* Local–Global idea WITHOUT its
   scale-shift-invariant normalization (which would throw away absolute scale — and absolute
   distance is the whole point here).

## Depth convention (read this)

Everything in this directory is **metric metres, larger = farther** (standard). This is the OPPOSITE
of the relative Depth Anything convention used inside `C:\Tim\AIDGPT\AidGPT` (larger = nearer). Do
not mix them.

## Order of operations

```
# 0. one-time install (see requirements.txt notes — UniDepth from git, no xformers on torch-2.11)
# 1. teacher pass over the dataset (long, GPU; produces distillation targets + per-object distances)
python -m teacher.run_teacher --config config.yaml
# 2. make train/val/test splits
python -m scripts.make_splits --config config.yaml
# 3. distill the student
python -m distill.train --config config.yaml
# 4. evaluate the student vs the teacher
python -m distill.evaluate --config config.yaml
```

## Status

- Teacher pipeline + per-object distance extraction: implemented.
- Student + distillation loss + train/eval loop: implemented.
- The full 7.7k-image teacher run is a multi-hour GPU job; distillation can only start once it
  finishes (it needs the cached teacher depth as targets).

## Honest caveats

- UniDepth v2 self-calibrates the camera, but "exact distance" is still a model *prediction*, not a
  measured ground truth. For paper-grade numbers you still need a small measured/gold subset to
  validate the teacher itself (the source paper validated on IBIMS-1 GT, reporting MAE 0.31 m /
  0.22 m within 5 m). `distill/evaluate.py` measures student-vs-teacher agreement, not absolute truth.
