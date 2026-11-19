"""Assemble the surpass-campaign report from the harness result JSONs into a markdown doc:
  * Reproduction gate: vanilla V2-L / V2-S per dataset (bench_gate.json).
  * T1 training-free: vanilla V2 vs V2++ with per-metric delta (v2pp_results.json).
  * De-dup keeps the largest-N entry per (dataset, model). Every number is OUR harness, exact metrics.

  conda run -n depth python -m bench.report --out docs/BENCHMARK_SURPASS.md
"""

from __future__ import annotations

import argparse
import json
import os

CK = "outputs/checkpoints"
# metric: (label, lower_is_better)
COLS = [("d1_ssi", "δ1_ssi", False), ("arel", "AbsRel", True), ("arel_ssi", "AbsRel_ssi", True),
        ("silog", "SILog", True), ("rmse", "RMSE", True), ("f1_3d", "F_A", False),
        ("rho_a", "ρ_A", False)]


def _load(p):
    p = p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


def _dedup(entries):
    """entries: list of {dataset, models:{name:metrics}}. Keep largest-N per (dataset,name)."""
    best = {}
    for e in entries:
        ds = e["dataset"]
        for name, m in e.get("models", {}).items():
            n = int(m.get("_num_images", 0))
            key = (ds, name)
            if n >= best.get(key, (-1, None))[0]:
                best[key] = (n, m)
    out = {}
    for (ds, name), (n, m) in best.items():
        out.setdefault(ds, {})[name] = m
    return out


def _fmt(m, key):
    return f"{m[key]:.4f}" if key in m and m[key] == m[key] else "—"


def _table(per_ds):
    lines = []
    hdr = "| dataset | model | " + " | ".join(c[1] for c in COLS) + " | n |"
    sep = "|" + "---|" * (len(COLS) + 3)
    lines += [hdr, sep]
    for ds in sorted(per_ds):
        for name, m in per_ds[ds].items():
            row = f"| {ds} | {name} | " + " | ".join(_fmt(m, c[0]) for c in COLS)
            row += f" | {m.get('_num_images','?')} |"
            lines.append(row)
    return "\n".join(lines)


def _delta_table(v2pp_entries):
    """For each dataset with a vanilla+V2++ pair, show the improvement."""
    per = _dedup(v2pp_entries)
    lines = ["| dataset | metric | vanilla V2 | V2++ | Δ (better?) |", "|---|---|---|---|---|"]
    for ds in sorted(per):
        models = per[ds]
        van = next((m for n, m in models.items() if "V2++" not in n), None)
        pp = next((m for n, m in models.items() if "V2++" in n), None)
        if not (van and pp):
            continue
        for key, label, lower in COLS:
            if key in van and key in pp and van[key] == van[key] and pp[key] == pp[key]:
                d = pp[key] - van[key]
                better = (d < 0) if lower else (d > 0)
                mark = "✓" if better else "✗"
                lines.append(f"| {ds} | {label} | {van[key]:.4f} | {pp[key]:.4f} | {d:+.4f} {mark} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/BENCHMARK_SURPASS.md")
    args = ap.parse_args()

    gate = _dedup(_load(f"{CK}/bench_gate.json"))
    v2pp = _load(f"{CK}/v2pp_results.json")

    md = ["# Surpassing UniDepthV2 on its own benchmark — results\n",
          "*All numbers computed by our harness (`bench/`) with UniDepthV2's EXACT metrics "
          "(δ with SSI/SI rescale, F_A via validated pure-torch chamfer, ρ_A ray-angular AUC); "
          "no scale alignment on raw metrics. Vanilla V2 = official frozen checkpoint at "
          "resolution_level 5.*\n",
          "## Reproduction gate — vanilla V2-L / V2-S (baselines to beat)\n",
          _table(gate) if gate else "_(gate not run yet)_", ""]

    if v2pp:
        md += ["\n## T1 — training-free V2++ vs vanilla V2 (per-metric delta)\n",
               _delta_table(v2pp), "",
               "\n### Full V2++ numbers\n", _table(_dedup(v2pp))]

    out = args.out if os.path.isabs(args.out) else os.path.join(os.getcwd(), args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print("wrote", out)
    print("\n".join(md)[:2000])


if __name__ == "__main__":
    main()
