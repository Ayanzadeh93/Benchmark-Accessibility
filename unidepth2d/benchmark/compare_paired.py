"""Paired comparison of two models on the same images, with bootstrap confidence intervals.

Every dataset mean in this project is an unweighted mean of per-image metrics, so two models
evaluated on the same images can be compared image by image. On small sets (IBims-1 has 100
images) a 1-2 point delta1 difference is often noise; this tool says whether a difference
is distinguishable from zero.

  python -m benchmark.compare_paired results_bench/ibims__A__pi.json results_bench/ibims__B__pi.json
  python -m benchmark.compare_paired A.json B.json --metric d1 --metric arel --json out.json

Both result files must have been written by run_bench with --save-per-image and must cover
the same images in the same order (same dataset, variant and limit).

Output per metric: mean(A), mean(B), mean difference B - A, a percentile bootstrap 95% CI
of the difference (image-level resampling, paired), and the fraction of images where B is
better. "significant" means the CI excludes zero.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

import numpy as np

# metrics where a LOWER value is better (everything else: higher is better)
LOWER_BETTER = {"arel", "sqrel", "rmse", "mae", "rmselog", "log10", "silog", "medianlog",
                "arel_ssi", "arel_si"}
DEFAULT_METRICS = ("d1", "arel", "d1_ssi", "rmse", "silog")


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                     seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Paired percentile bootstrap of mean(b - a) over images.

    Returns mean_a, mean_b, diff (= mean_b - mean_a), ci_lo, ci_hi, n, and
    significant (True iff the CI excludes 0). NaN pairs are dropped.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs differ in length: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n == 0:
        raise ValueError("no finite pairs to compare")
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def compare_rows(rows_a: List[dict], rows_b: List[dict], metrics: Sequence[str],
                 n_boot: int = 10000, seed: int = 0) -> Dict[str, dict]:
    """Compare two per-image row lists (run_bench `per_image`) on the given metrics."""
    if len(rows_a) != len(rows_b):
        raise ValueError(f"different image counts: {len(rows_a)} vs {len(rows_b)} "
                         "(results must come from the same dataset/variant/limit)")
    out = {}
    for m in metrics:
        if m not in rows_a[0] or m not in rows_b[0]:
            continue
        a = [r[m] for r in rows_a]
        b = [r[m] for r in rows_b]
        res = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
        d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
        better = (d < 0) if m in LOWER_BETTER else (d > 0)
        res["frac_images_b_better"] = float(np.mean(better))
        res["lower_is_better"] = m in LOWER_BETTER
        if res["significant"]:
            b_wins = (res["diff"] < 0) if m in LOWER_BETTER else (res["diff"] > 0)
            res["verdict"] = "B better" if b_wins else "A better"
        else:
            res["verdict"] = "no detectable difference"
        out[m] = res
    return out


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    if "per_image" not in r:
        raise SystemExit(f"{path} has no per-image rows; rerun run_bench with --save-per-image")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", help="result JSON of model A (reference)")
    ap.add_argument("b", help="result JSON of model B (candidate)")
    ap.add_argument("--metric", action="append", default=[])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="", help="also write the comparison to this file")
    args = ap.parse_args()

    ra, rb = _load(args.a), _load(args.b)
    for key in ("dataset", "variant"):
        if ra.get(key) != rb.get(key):
            raise SystemExit(f"cannot pair: {key} differs ({ra.get(key)} vs {rb.get(key)})")
    metrics = args.metric or list(DEFAULT_METRICS)
    res = compare_rows(ra["per_image"], rb["per_image"], metrics, args.n_boot, args.seed)
    print(f"A = {ra['model']}   B = {rb['model']}   dataset = {ra['dataset']} "
          f"({ra.get('variant')})   n = {len(ra['per_image'])}")
    print(f"{'metric':10s} {'A':>9s} {'B':>9s} {'B-A':>9s} {'95% CI':>21s}  {'B better on':>11s}  verdict")
    for m, r in res.items():
        print(f"{m:10s} {r['mean_a']:9.4f} {r['mean_b']:9.4f} {r['diff']:+9.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {100*r['frac_images_b_better']:9.0f}%  {r['verdict']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"a": ra["model"], "b": rb["model"], "dataset": ra["dataset"],
                       "variant": ra.get("variant"), "metrics": res}, f, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
