"""NYU-654 'clean' rescore: drop Eigen-test frames that a TRAINING row duplicates, from saved per-image rows.

Leaked ids = Eigen-test frames referenced by a CONFIRMED match (aligned pHash <= 8 or NCC >= 0.9,
the roles.yaml 'confirmed' rule) from a manifest training row:
  outputs/v2/roles/own_vs_nyu_labelled.csv         (own rows that are NYU frames)   -> 50 test ids
  outputs/v2/roles/nyu_vs_nyu_eigen_test_654.csv   (NYU-train rows near a test one) -> 3 test ids
No re-inference: every results_*/nyu-filled__*.json with 654 per-image rows is re-aggregated with
protocol.aggregate over the remaining frames (row order = splits.mat testNdxs order, as iter_nyu).

  python -m scripts.nyu_clean_rescore      -> prints + writes docs/NYU_CLEAN_RESCORE.md, outputs/nyu_leaked_test_ids.json
"""
from __future__ import annotations

import csv
import glob
import json
import os

from scipy.io import loadmat

from benchmark import protocol

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVID = os.path.join(ROOT, "outputs", "v2", "roles")


def leaked_ids():
    test = [f"nyu_{i - 1:04d}" for i in loadmat("Z:/Datasets/NYUv2/splits.mat")["testNdxs"].ravel()]
    man = {r["id"] for r in csv.DictReader(open(os.path.join(ROOT, "outputs", "data_gt", "manifest.csv"),
                                                encoding="utf-8"))}
    out = {}
    for name in ("own_vs_nyu_labelled", "nyu_vs_nyu_eigen_test_654"):
        hit = set()
        for r in csv.DictReader(open(os.path.join(EVID, name + ".csv"), encoding="utf-8")):
            if r["id"] not in man:
                continue
            if int(r["ham"]) <= 8:
                hit.add(r["ham_ref"])
            if float(r["ncc"]) >= 0.9:
                hit.add(r["ncc_ref"])
        out[name] = sorted(hit & set(test))
    return test, out


def main():
    test, by = leaked_ids()
    leaked = set().union(*by.values())
    keep = [i for i, t in enumerate(test) if t not in leaked]
    json.dump({"rule": "confirmed: aligned pHash<=8 or NCC>=0.9, training manifest rows", "by_source": by,
               "n_leaked": len(leaked), "n_clean": len(keep)},
              open(os.path.join(ROOT, "outputs", "nyu_leaked_test_ids.json"), "w"), indent=1)
    lines = [f"# NYU-654 clean rescore ({len(keep)} frames; {len(leaked)} leaked test ids removed)", "",
             f"Leaked: {len(by['own_vs_nyu_labelled'])} via own-data rows, "
             f"{len(by['nyu_vs_nyu_eigen_test_654'])} via NYU-train rows (overlap "
             f"{len(set(by['own_vs_nyu_labelled']) & set(by['nyu_vs_nyu_eigen_test_654']))}). "
             "Baselines never trained on our manifest, so their shift is pure subset effect.", "",
             "| Source | Model | d1 full | d1 clean | Δ | AbsRel full | AbsRel clean | RMSE full | RMSE clean | d1_si full | d1_si clean |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for d in ("results_bench", "results_finetune", "results_tta"):
        for p in sorted(glob.glob(os.path.join(ROOT, d, "nyu-filled__*.json"))):
            j = json.load(open(p, encoding="utf-8"))
            pi = j.get("per_image")
            if not pi or len(pi) != len(test):
                continue
            full, clean = protocol.aggregate(pi), protocol.aggregate([pi[i] for i in keep])
            rows.append((d.replace("results_", ""), os.path.basename(p)[len("nyu-filled__"):-5], full, clean))
    for d, tag, f, c in sorted(rows, key=lambda r: -r[3]["d1"]):
        lines.append(f"| {d} | {tag} | {100*f['d1']:.1f} | {100*c['d1']:.1f} | {100*(c['d1']-f['d1']):+.1f} | "
                     f"{f['arel']:.3f} | {c['arel']:.3f} | {f['rmse']:.3f} | {c['rmse']:.3f} | "
                     f"{100*f['d1_si']:.1f} | {100*c['d1_si']:.1f} |")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, "docs", "NYU_CLEAN_RESCORE.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
