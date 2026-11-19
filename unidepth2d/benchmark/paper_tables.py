"""Assemble the paper-grade tables from the v2 result ecosystems (no hand-typed numbers).

Reads the JSON emitted by the paper-grade harness — NOT the legacy gt_eval ecosystem that
benchmark/ablation_table.py aggregates:

  results_bench/<dataset>[-variant]__<model>[__tag].json   (benchmark.run_bench)   -> Dense table
  results_bench/resmarg__<dataset>__<model>.json           (prototypes.res_marginalize)
  results_ocmr/<model>__summary.json                       (benchmark.ocmr_eval_ann) -> OCMR table
  results_ocmr/anchorlock__<model>.json                    (prototypes.anchor_lock)   -> Anchoring table

Every row is [ours] (measured by our harness). Published UniDepthV2 rows are NEVER injected
here — they enter a table only after docs/GATE_REPORT.md passes and are marked [paper] by hand
in the manuscript (PROTOCOL_NOTES §3). Dense/OCMR de-dup keeps the entry with the MOST
images/objects per (key) so a smoke run can't shadow a full run.

  python -m benchmark.paper_tables
Writes docs/PAPER_TABLES.md (+ results_bench/paper_dense.csv, results_ocmr/paper_ocmr.csv).
"""

from __future__ import annotations

import csv
import glob
import json
import os

HERE = os.path.dirname(__file__)
RESULTS_BENCH = os.path.join(HERE, "..", "results_bench")
RESULTS_OCMR = os.path.join(HERE, "..", "results_ocmr")
DOCS = os.path.join(HERE, "..", "docs")


def _load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _f(x, nd=4):
    return "" if x is None else (round(x, nd) if isinstance(x, (int, float)) else x)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


# ── Dense zero-shot table (results_bench) ────────────────────────────────────
def dense_rows():
    """Rows + comparability warnings. GT-cam-prompted runs are EXCLUDED (not zero-shot)."""
    best, best_n, warnings = {}, {}, []
    skipped_gtcam = []
    for p in sorted(glob.glob(os.path.join(RESULTS_BENCH, "*.json"))):
        if os.path.basename(p).startswith("resmarg__"):
            continue
        r = _load(p)
        if not r or "depth" not in r:
            continue
        if r.get("gt_cam_prompted"):
            skipped_gtcam.append(os.path.basename(p))
            continue
        key = (r.get("dataset"), r.get("variant", "default"), r.get("model"))
        n = int(r.get("num_images", 0))
        if n >= best_n.get(key, -1):
            best[key] = r
            best_n[key] = n
    if skipped_gtcam:
        warnings.append(f"excluded {len(skipped_gtcam)} GT-cam-prompted run(s) "
                        f"(not zero-shot): {', '.join(skipped_gtcam)}")

    # comparability checks within each (dataset, variant) group
    groups = {}
    for (ds, var, model), n in best_n.items():
        groups.setdefault((ds, var), {})[model] = n
    for (ds, var), models in sorted(groups.items()):
        if len(set(models.values())) > 1:
            warnings.append(f"dense {ds}-{var}: n DIFFERS across models {models} — "
                            "rows are not strictly comparable until reconciled")
        bases = {}
        for m in models:
            bases.setdefault(m.split("+")[0], []).append(m)
        for base, variants_ in sorted(bases.items()):
            if len(variants_) > 1:
                warnings.append(f"dense {ds}-{var}: MIXED resolution policy for {base}: "
                                f"{sorted(variants_)} — quote exactly one (policy: +r9) "
                                "in any paper table")

    rows = []
    for (ds, var, model), r in sorted(best.items()):
        d = r["depth"]
        unc = r.get("uncertainty") or {}
        rows.append([ds, var, model, best_n[(ds, var, model)],
                     _f(d.get("d1")), _f(d.get("d1_ssi")), _f(d.get("arel")),
                     _f(d.get("sqrel")), _f(d.get("rmse")), _f(d.get("silog"), 3),
                     _f(d.get("tau")),
                     _f((r.get("f3d") or {}).get("F1")), _f(r.get("rho_a")),
                     _f(unc.get("ause")), _f(unc.get("nause")), _f(unc.get("spearman")),
                     _f(r.get("boundary_f1_si"))])
    header = ["dataset", "variant", "model", "n", "d1", "d1_ssi", "arel", "sqrel",
              "rmse", "silog", "tau", "F_A", "rho_a", "ause", "nause", "spearman", "bF1"]
    return header, rows, warnings


# ── OCMR object-range table (results_ocmr) ───────────────────────────────────
def ocmr_rows(protocols=("box5", "maskmed", "center", "det_box5")):
    """Rows + comparability warnings (object-pool drift across models within a slice)."""
    rows, warnings = [], []
    pool = {}                                     # (slice, protocol) -> {model: n}
    for p in sorted(glob.glob(os.path.join(RESULTS_OCMR, "*__summary.json"))):
        r = _load(p)
        if not r or "protocols" not in r:
            continue
        # slice label: "sun" for the SUN track, "kitti"/"kitti-<classes>" for OCMR-KITTI
        base = os.path.basename(p)
        slice_lbl = base.split("__", 1)[0] if base.startswith("kitti") else "sun"
        for proto in protocols:
            s = r["protocols"].get(proto)
            if not s:
                continue
            pool.setdefault((slice_lbl, proto), {})[r.get("model")] = int(s.get("n", 0))
            # proximity thresholds differ per slice (indoor 1/2 m, outdoor 5/10 m)
            prox_keys = sorted((k for k in s if k.startswith("proxF1@")),
                               key=lambda k: float(k[len("proxF1@"):-1]))[:2]
            prox = [f"{k[len('proxF1@'):]}:{_f(s.get(k))}" for k in prox_keys]
            rows.append([slice_lbl, r.get("model"), proto, s.get("n"),
                         _f(s.get("objAbsRel")), _f(s.get("objD1")), _f(s.get("objD2")),
                         _f(s.get("MAE"), 3), _f(s.get("MAE@5m"), 3), _f(s.get("bias_log")),
                         "; ".join(prox),
                         _f(s.get("AUSE@obj")), _f(s.get("nAUSE@obj"))])
    for (slice_lbl, proto), models in sorted(pool.items()):
        ns = list(models.values())
        if ns and (max(ns) - min(ns)) > 0.005 * max(ns):
            warnings.append(f"OCMR {slice_lbl}/{proto}: object pool drifts across models "
                            f"{models} (>0.5%) — recompute on the common object subset "
                            "from the per-object CSVs before quoting side-by-side")
    header = ["slice", "model", "protocol", "n_obj", "objAbsRel", "objD1", "objD2", "MAE",
              "MAE@5m", "bias_log", "proxF1", "AUSE@obj", "nAUSE@obj"]
    return header, rows, warnings


# ── Training-free anchoring table (results_ocmr/anchorlock) ──────────────────
def anchor_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS_OCMR, "anchorlock__*.json"))):
        r = _load(p)
        if not r:
            continue
        for v in ("base", "ground", "sma", "both"):
            dense = (r.get("dense") or {}).get(v) or {}
            obj = (r.get("object") or {}).get(v) or {}
            if not dense and not obj:
                continue
            rows.append([r.get("model"), v, r.get("n_images"),
                         _f(dense.get("arel")), _f(dense.get("d1")),
                         _f(obj.get("objAbsRel")), _f(obj.get("objD1")),
                         _f(obj.get("MAE@5m"), 3), _f(obj.get("bias_log")),
                         _f(obj.get("proxF1@1m")), _f(obj.get("proxF1@2m"))])
    header = ["model", "variant", "n_img", "dense_arel", "dense_d1", "objAbsRel",
              "objD1", "objMAE@5m", "bias_log", "proxF1@1m", "proxF1@2m"]
    return header, rows


def resmarg_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS_BENCH, "resmarg__*.json"))):
        r = _load(p)
        if not r:
            continue
        comp = r.get("compute", {})
        for v, m in (r.get("metrics") or {}).items():
            rows.append([r.get("model"), r.get("dataset"), v, r.get("num_images"),
                         _f(m.get("arel")), _f(m.get("d1")), _f(m.get("silog"), 3),
                         _f(m.get("boundary_f1")), _f(comp.get("fused_ms") if "fuse" in v
                         else comp.get("per_level_ms", {}).get(v), 1)])
    header = ["model", "dataset", "variant", "n", "arel", "d1", "silog", "bF1", "ms"]
    return header, rows


def main():
    os.makedirs(DOCS, exist_ok=True)
    sections = []

    dh, dr, warn_d = dense_rows()
    sections.append(("Dense zero-shot (results_bench) [ours]", dh, dr))
    oh, orr, warn_o = ocmr_rows()
    sections.append(("OCMR object-range — box5 / maskmed / center (results_ocmr) [ours]", oh, orr))
    ah, ar = anchor_rows()
    sections.append(("Training-free anchoring (results_ocmr/anchorlock) [ours]", ah, ar))
    rh, rr = resmarg_rows()
    if rr:
        sections.append(("Resolution marginalization (results_bench/resmarg) [ours]", rh, rr))
    warnings = warn_d + warn_o

    md = ["# Paper tables (auto-generated — do not hand-edit)", "",
          "All rows [ours], zero-shot (GT-cam probe runs auto-excluded). Published "
          "UniDepthV2 numbers enter the manuscript only per PROTOCOL_NOTES §3 (gate + "
          "footnote) and are marked [paper] there. Legacy gt_eval/ablation_table numbers "
          "are a different protocol (pred-clip, HF repacks) — never mix them into these "
          "tables. Slices (sun / kitti / gold) are separate GT conventions — never compare "
          "absolute values across slices.", ""]
    md.append("## Comparability warnings")
    if warnings:
        md += [f"- ⚠️ {w}" for w in warnings]
    else:
        md.append("- none — all groups internally consistent")
    md.append("")
    for title, header, rows in sections:
        md.append(f"## {title}  ({len(rows)} rows)")
        md.append(_md_table(header, rows) if rows else "_no result files found yet_")
        md.append("")
    out_md = os.path.join(DOCS, "PAPER_TABLES.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    for w in warnings:
        print(f"WARNING: {w}")

    # machine-readable copies for the two headline tables
    with open(os.path.join(RESULTS_BENCH, "paper_dense.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(dh); w.writerows(dr)
    os.makedirs(RESULTS_OCMR, exist_ok=True)
    with open(os.path.join(RESULTS_OCMR, "paper_ocmr.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(oh); w.writerows(orr)

    print(f"wrote {out_md}")
    for title, _, rows in sections:
        print(f"  {title}: {len(rows)} rows")


if __name__ == "__main__":
    main()
