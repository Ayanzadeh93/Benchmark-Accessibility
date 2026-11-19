"""T1 ablation — isolate each training-free novelty (N1 IO-TTA, N2 SCMR, N3 camera search) against
vanilla V2 on one dataset, with the exact UniDepthV2 metrics. Fits SCMR once (on a calib set),
then evaluates a matrix of V2++ configurations in a single job (no redundant reloads).

  conda run -n depth python -m bench.ablate_t1 --dataset nyu --variant l --limit 300 \
      --calib-dataset sun --calib-limit 100
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from bench.datasets import get_loader
from bench.eval import HEADLINE, eval_model_on_dataset
from bench.refmodel import V2Model
from bench.v2pp import V2ppModel, fit_affine_pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="l", choices=["l", "s"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--calib-dataset", default="sun")
    ap.add_argument("--calib-limit", type=int, default=100)
    ap.add_argument("--levels", default="3,5,7")
    ap.add_argument("--out", default="outputs/checkpoints/t1_ablation.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "lpiccinelli/unidepth-v2-vitl14" if args.variant == "l" else "lpiccinelli/unidepth-v2-vits14"
    levels = tuple(int(x) for x in args.levels.split(","))
    loader = get_loader(args.dataset)

    # Fit SCMR affine once on the calib set (label-efficient, transferred zero-shot).
    cal = V2ppModel(ckpt, device, levels, True, "oracle", affine=None)
    a, b, ncal = fit_affine_pixels(cal, get_loader(args.calib_dataset), args.calib_limit, device)
    print(f"[SCMR] a={a:.4f} b={b:+.4f} (calib {args.calib_dataset} n={ncal})")
    del cal; torch.cuda.empty_cache()

    # Ablation matrix: each row adds one novelty so contributions are isolated.
    configs = [
        ("vanilla",        lambda: V2Model(ckpt, device, resolution_level=5)),
        ("N1_iotta",       lambda: V2ppModel(ckpt, device, levels, True, "oracle", affine=None)),
        ("N1_meanTTA",     lambda: V2ppModel(ckpt, device, levels, True, "mean", affine=None)),
        ("N2_scmr",        lambda: V2ppModel(ckpt, device, (5,), False, "mean", affine=(a, b))),
        ("N1+N2",          lambda: V2ppModel(ckpt, device, levels, True, "oracle", affine=(a, b))),
        ("N1+N2+N3_cam",   lambda: V2ppModel(ckpt, device, levels, True, "oracle", affine=(a, b),
                                             camera_search=True)),
    ]
    results = {}
    for tag, ctor in configs:
        m = ctor()
        print(f"[{args.dataset}] {tag}: {m.name} ...")
        met, n = eval_model_on_dataset(m, loader, args.limit, device)
        met["_num_images"] = n
        results[tag] = met
        del m; torch.cuda.empty_cache()

    print(f"\n=== T1 ablation on {args.dataset} (V2-{args.variant.upper()}, exact metrics) ===")
    print("config".ljust(16) + "".join(h.rjust(10) for h in HEADLINE))
    for tag, mt in results.items():
        print(tag.ljust(16) + "".join((f"{mt[h]:.4f}" if h in mt else "-").rjust(10) for h in HEADLINE))

    entry = {"dataset": args.dataset, "variant": args.variant, "scmr": [a, b], "configs": results}
    out = args.out if os.path.isabs(args.out) else os.path.join(os.getcwd(), args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allr = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    allr.append(entry)
    json.dump(allr, open(out, "w", encoding="utf-8"), indent=2)
    print("appended ->", out)


if __name__ == "__main__":
    main()
