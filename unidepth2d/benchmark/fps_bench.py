"""FPS / params / VRAM benchmark for the efficiency table — same-rig comparable.

Every row is measured on THIS machine with the same recipe (warmup + timed loop +
torch.cuda peak-memory), and carries the exact input policy so official models and
students are never silently compared under different conditions:

  officials  v2-s | v2-b | v2-l | v1-l (+ legacy aliases teacher/vits): end-to-end native
             `model.infer()` on a 640x480 uint8 frame (the model applies its own internal
             0.2-0.6MP budget; optional --res-level recorded in the row) — this measures
             the DEPLOYED cost, preprocessing included.
  students   student:<ckpt>: `infer_metres` on the checkpoint's train_size tensor
             (deploy mode, matching how students are evaluated in run_bench).

Environment metadata (GPU, torch, precision, warmup/iters, date) is stored in the output
so rows from different machines/sessions can never be mixed unnoticed. Paper Tab. VIII
rows (A6000) are a DIFFERENT rig — quote them only in a separate, clearly-marked block.

  python -m benchmark.fps_bench --config config.yaml \
      --model v2-l --model v2-s --model student:outputs/checkpoints/S10_ultra_L/best.pt

Writes outputs/checkpoints/fps_bench.json  {"env": {...}, "rows": [...]}.
(ablation_table.py reads legacy list-format files too; this writes the new dict format.)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time

import torch

from common.io_utils import load_config

WARMUP, ITERS = 10, 100
INPUT_HW = (480, 640)          # official-model benchmark frame (VGA, matches eval sets)


def _timed(fn, n=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
    return dt, peak_gib


def bench_student(cfg, ckpt, device):
    from distill.student import build_student, needs_intrinsics
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg_m = ck.get("cfg", cfg)
    m = build_student(cfg_m).to(device).eval()
    m.load_state_dict(ck["model"])
    s = cfg_m["student"]
    th, tw = int(s["train_size"][0]), int(s["train_size"][1])
    use_intr = needs_intrinsics(s)
    x = torch.randn(1, 3, th, tw, device=device)
    intr = torch.tensor([[0.8, 1.08, 0.5, 0.5]], device=device) if use_intr else None
    from distill.rep import fuse_with_parity
    # b20 rep4: time the fused deploy graph (no-op else). The parity check runs with TF32 off
    # (fuse_with_parity restores the flags), so the CUDA default allow_tf32=True cannot fail it.
    fuse_with_parity(m, x, intr)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        dt, peak = _timed(lambda: m.infer_metres(x, intr))
    return dict(model=f"student-{os.path.basename(os.path.dirname(ckpt))}",
                params_M=round(params, 2), input=f"{tw}x{th} tensor (deploy)",
                res_level=None, ms=round(dt * 1000, 1), fps=round(1 / dt, 1),
                peak_vram_GiB=round(peak, 2))


def bench_official(label, cls_name, hub, device, res_level=None):
    import unidepth.models as um
    m = getattr(um, cls_name).from_pretrained(hub).to(device).eval()
    if res_level is not None:
        m.resolution_level = int(res_level)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    img = torch.randint(0, 255, (3, *INPUT_HW), dtype=torch.uint8, device=device)

    def run():
        with torch.inference_mode():
            m.infer(img)

    dt, peak = _timed(run)
    name = label + (f"+r{res_level}" if res_level is not None else "")
    return dict(model=name, params_M=round(params, 2),
                input=f"{INPUT_HW[1]}x{INPUT_HW[0]} uint8, native infer()",
                res_level=res_level, ms=round(dt * 1000, 1), fps=round(1 / dt, 1),
                peak_vram_GiB=round(peak, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", action="append", required=True,
                    help="v2-s|v2-b|v2-l|v1-l|teacher|vits|student:<ckpt> (repeatable)")
    ap.add_argument("--res-level", type=int, default=None,
                    help="official models: resolution_level (recorded per row)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("FPS benchmark needs CUDA (peak-VRAM + sync timing)")

    from benchmark.run_bench import OFFICIAL
    alias = {"teacher": "v2-l", "vits": "v2-s"}      # legacy spec names
    rows = []
    fusion_failed = []
    for spec in args.model:
        spec = alias.get(spec, spec)
        try:
            if spec in OFFICIAL:
                cls_name, hub = OFFICIAL[spec]
                rows.append(bench_official(spec, cls_name, hub, device, args.res_level))
            elif spec.startswith("student:"):
                rows.append(bench_student(cfg, spec.split(":", 1)[1], device))
            else:
                print(f"[skip] unknown model spec: {spec}", flush=True)
                continue
            print(rows[-1], flush=True)
        except Exception as e:  # one broken model must not lose the whole table
            print(f"[skip] {spec}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if isinstance(e, AssertionError) and "fusion parity" in str(e):
                fusion_failed.append(spec)     # a missing rep4 row must not go unnoticed
        torch.cuda.empty_cache()

    out_obj = {
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "precision": "fp32 (model-native autocast where the model applies it)",
            "warmup": WARMUP, "iters": ITERS,
            "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "rows": rows,
    }
    print(json.dumps(out_obj, indent=2))
    out = os.path.join(cfg["distill"]["checkpoints_dir"], "fps_bench.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print("wrote", out)
    if fusion_failed:
        raise SystemExit(f"rep4 fusion parity FAILED for {fusion_failed}: rows missing from {out}")


if __name__ == "__main__":
    main()
