"""A04/A05 comparison table from the rescoring runs (eval only; reads, never rewrites, results_bench).

  python -m scripts.rescore_compare [--dir results_rescore]

Reads <dir>/ibims__*.json written by `benchmark.run_bench --boundary --out-dir <dir>` and writes
<dir>/a04_a05_comparison.json and .md:
  * A04 boundary F1 per model: legacy (labelled legacy; plus the stored results_bench value),
    v2 (both_empty='zero') and DepthPro SI-bF1 (masked / unmasked), each with n_contributing;
  * the B20_SPEC dws4 gate restated as a paired bootstrap of bF1v2 b20core -> b20dws
    (pair_rows_by_id + paired_bootstrap_clustered, image resampling; accept if the 95% CI has 0);
  * A05 K-less student rows: legacy (FiLM off) vs default_k vs GT K.
Missing runs are listed as missing, never filled in.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark.protocol_v2 import BF_KEY, pair_rows_by_id, paired_bootstrap_clustered  # noqa: E402

KEYS = ("bf1_si_legacy", BF_KEY, "bf1_depthpro_masked", "bf1_depthpro_nomask")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results_rescore"))
    a = ap.parse_args()
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "ibims__*.json"))):
        r = _load(p)
        if "boundary_f1" in r:
            runs[os.path.basename(p)[:-5]] = r

    rows = []
    for key, r in runs.items():
        agg = r["boundary_f1"]["aggregate"]
        model = r["model"]
        stored_p = os.path.join(ROOT, "results_bench", f"ibims__{model}.json")
        stored = _load(stored_p).get("boundary_f1_si") if os.path.isfile(stored_p) else None
        rows.append({
            "run": key, "model": model, "n": r["num_images"],
            "k_passed": r.get("k_passed_to_model", True),
            "k_none_policy": r.get("k_none_policy", "legacy"),
            "d1": r["depth"]["d1"], "arel": r["depth"]["arel"],
            "stored_legacy_bf1": stored if r.get("k_passed_to_model", True) else None,
            **{f"{k}": agg[k]["bf1"] for k in KEYS},
            **{f"{k}_n": agg[k]["n_contributing"] for k in KEYS},
        })

    # dws4 gate restated: paired bootstrap of bF1v2, b20core -> b20dws (GT-K runs)
    gate = None
    a_key = next((k for k in runs if "student-b20core_last" in k and "noK" not in k), None)
    b_key = next((k for k in runs if "student-b20dws_last" in k and "noK" not in k), None)
    if a_key and b_key:
        ids, ra, rb, info = pair_rows_by_id(runs[a_key]["boundary_f1"]["per_image"],
                                            runs[b_key]["boundary_f1"]["per_image"])
        nan = float("nan")
        va = [nan if x[BF_KEY] is None else x[BF_KEY] for x in ra]
        vb = [nan if x[BF_KEY] is None else x[BF_KEY] for x in rb]
        bs = paired_bootstrap_clustered(va, vb, clusters=None, n_boot=10000, seed=0)
        gate = {"a": a_key, "b": b_key, "metric": BF_KEY, "n_paired": info["n_paired"],
                **bs, "accept_dws4": bool(bs["ci_lo"] <= 0.0 <= bs["ci_hi"])}

    out = {"source_dir": os.path.abspath(a.dir), "rows": rows, "dws4_gate_bf1v2": gate}
    with open(os.path.join(a.dir, "a04_a05_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md = ["# A04/A05 rescoring - IBims-1 (results_rescore)", "",
          "bF1 columns: legacy = protocol.boundary_f1_si (LEGACY, drops edgeless images); "
          "v2 = boundary_f1_si_v2 (both_empty='zero'); DP = DepthPro SI-bF1 "
          "(masked = valid-mask extension, the HEADLINE DepthPro number on IBims since its GT has "
          "holes; nomask = DepthPro reference, unmasked, GT holes counted as edges - not a "
          "headline number here). (n) = n_contributing; for legacy it counts every image not "
          "dropped (dropped = no GT edge OR no predicted edge). "
          "'stored' = legacy value in results_bench (2026-11-23 stack).", "",
          "| run | K to model | K=None policy | n | d1 | AbsRel | bF1 legacy (n) | stored legacy "
          "| bF1 v2 (n) | DP masked, headline (n) | DP nomask, holes=edges (n) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['run']} | {'GT K' if r['k_passed'] else 'none (assumed default K)'} | "
            f"{r['k_none_policy']} | {r['n']} | {_fmt(r['d1'])} | {_fmt(r['arel'])} | "
            f"{_fmt(r['bf1_si_legacy'])} ({r['bf1_si_legacy_n']}) | {_fmt(r['stored_legacy_bf1'])} | "
            f"{_fmt(r[BF_KEY])} ({r[BF_KEY + '_n']}) | "
            f"{_fmt(r['bf1_depthpro_masked'])} ({r['bf1_depthpro_masked_n']}) | "
            f"{_fmt(r['bf1_depthpro_nomask'])} ({r['bf1_depthpro_nomask_n']}) |")
    md.append("")
    if gate:
        md.append(f"dws4 gate (bF1v2, paired image bootstrap, b20core -> b20dws, n={gate['n']}): "
                  f"diff {gate['diff']:+.5f}, 95% CI [{gate['ci_lo']:+.5f}, {gate['ci_hi']:+.5f}] "
                  f"-> {'ACCEPT (CI includes 0)' if gate['accept_dws4'] else 'REJECT'}")
    else:
        md.append("dws4 gate: not computed (b20core/b20dws GT-K runs missing)")
    with open(os.path.join(a.dir, "a04_a05_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
