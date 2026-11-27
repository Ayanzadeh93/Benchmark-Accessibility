"""Original b20dws vs fine-tuned variants on IBims / NYU / SUN (same protocol, results_*/summary.csv).

  python -m scripts.compare_finetune            # prints a markdown table, writes docs/B20_FINETUNE_RESULTS.md
"""
from __future__ import annotations

import csv
import os

ROWS = [
    ("Original b20dws", "results_bench/summary.csv", "student-b20dws_last"),
    ("Control ft (lr 2e-5, orig data)", "results_finetune/summary.csv", "student-b20dws_ft2_ctrl_last"),
    ("All-data ft (lr 2e-5, +ARKit +Hypersim)", "results_finetune/summary.csv", "student-b20dws_ft2_all_last"),
    ("[dropped] ctrl ft lr 1e-4", "results_finetune/summary.csv", "student-b20dws_ft_ctrl_last"),
    ("[dropped] Hypersim-only ft lr 1e-4", "results_finetune/summary.csv", "student-b20dws_ft_hyper_last"),
]
COLS = [("ibims", "d1", "IBims d1 ^"), ("ibims", "rmse", "IBims RMSE v"), ("ibims", "arel", "IBims AbsRel v"),
        ("ibims", "boundary_f1", "IBims bF1 ^"), ("nyu", "d1", "NYU d1 ^"), ("nyu", "rmse", "NYU RMSE v"),
        ("sun", "d1", "SUN d1 ^"), ("sun", "rmse", "SUN RMSE v")]


def load(path, model):
    out = {}
    if not os.path.isfile(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["model"] == model:
            out[r["dataset"]] = r
    return out


def main():
    lines = ["| Model | " + " | ".join(c[2] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for name, path, model in ROWS:
        d = load(path, model)
        cells = []
        for ds, k, _ in COLS:
            v = d.get(ds, {}).get(k, "")
            cells.append(f"{float(v):.4f}" if v not in ("", None) else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs("docs", exist_ok=True)
    with open("docs/B20_FINETUNE_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# b20dws fine-tune comparison\n\nSame protocol for every row (IBims default K policy, NYU filled, SUN raw).\n\n" + txt + "\n")


if __name__ == "__main__":
    main()
