"""Reproduction gate: measured results vs published UniDepthV2 zero-shot rows.

Published targets (PROTOCOL_NOTES.md §1, Table I, %):
  SUN    v2-l 96.4 / ARel 6.8 / F_A 84.6 / rho_A 93.4 ; v2-s 90.8 / 10.5 / 74.2 / 87.7
  IBims  v2-l 94.5 / 7.8 / 70.9 / 74.1              ; v2-s 86.6 / 13.5 / 62.4 / 67.5

The published delta1 column is RAW metric delta1 for metric methods (PROTOCOL_NOTES
§2b discovery #1, verified 2026-07-13: V2-L IBims raw d1 94.2-94.6 == published 94.5,
whereas the SSI-aligned variant measures ~98.5). The gate therefore compares MEASURED
RAW d1 (not d1_ssi) against the target; d1_ssi is reported as an informational line.

HARD gate (must pass before any published-number reuse): |d1_raw| <= 0.5pt and
|ARel| <= 0.5pt on SUN + IBims for v2-l and v2-s.
SOFT checks (informational; our F_A/rho_A implementations have documented
interpretation freedom, and V2-S carries a known HF-checkpoint drift — §2b #5):
|F_A| <= 2pt, |rho_A| <= 2pt, and d1_ssi shown for reference.

  python -m benchmark.gate_check            # reads results_bench/*.json
Writes docs/GATE_REPORT.md.
"""

from __future__ import annotations

import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_bench")
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")

TARGETS = {  # (dataset, model) -> dict of published values in percent
    ("sun", "v2-l"): dict(d1_ssi=96.4, arel=6.8, F1=84.6, rho_a=93.4),
    ("sun", "v2-s"): dict(d1_ssi=90.8, arel=10.5, F1=74.2, rho_a=87.7),
    ("ibims", "v2-l"): dict(d1_ssi=94.5, arel=7.8, F1=70.9, rho_a=74.1),
    ("ibims", "v2-s"): dict(d1_ssi=86.6, arel=13.5, F1=62.4, rho_a=67.5),
}
HARD_TOL = 0.5   # percentage points
SOFT_TOL = 2.0


def latest_result(dataset: str, model: str):
    """Best matching results json (any variant) for (dataset, model).

    Matches on the JSON's own `model` field — the base name (`v2-l`) or a res-level
    suffixed run (`v2-l+r9`, our documented baseline policy §2b #4). GT-cam-prompted
    runs are excluded (not zero-shot). Among candidates, prefer LARGEST num_images
    (full run beats probe/smoke), newest mtime as tiebreak.
    """
    cands = []
    for p in glob.glob(os.path.join(RESULTS_DIR, f"{dataset}*__{model}*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        name = str(r.get("model", ""))
        if not (name == model or name.startswith(model + "+r")):
            continue                                   # rejects e.g. v2-l+gtcam
        if r.get("gt_cam_prompted"):
            continue
        cands.append((int(r.get("num_images", 0)), os.path.getmtime(p), r, os.path.basename(p)))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1]))
    _, _, r, fname = cands[-1]
    return r, fname


def main():
    lines = ["# Reproduction gate report", ""]
    all_hard_pass, n_checked = True, 0
    for (ds, model), tgt in TARGETS.items():
        found = latest_result(ds, model)
        if not found:
            lines.append(f"- **{ds} / {model}**: NOT RUN yet")
            all_hard_pass = False
            continue
        res, fname = found
        d = res["depth"]
        # The published d1 is RAW metric d1 for metric methods (§2b #1); compare raw d1.
        # d1_ssi is carried as an informational SOFT line against the SAME target (it will
        # read high for metric models — that is expected, not a failure).
        meas = dict(
            d1=100.0 * d["d1"],
            d1_ssi=100.0 * d.get("d1_ssi", float("nan")),
            arel=100.0 * d["arel"],
            F1=100.0 * res["f3d"]["F1"] if res.get("f3d") else None,
            rho_a=100.0 * res["rho_a"] if res.get("rho_a") is not None else None,
        )
        tgt = dict(tgt, d1=tgt["d1_ssi"])   # same published number, keyed for raw-d1 compare
        n_checked += 1
        lines.append(f"## {ds} / {model}  (n={res['num_images']}, "
                     f"variant={res.get('variant')}, file={fname})")
        for key, tol, kind in (("d1", HARD_TOL, "HARD"), ("arel", HARD_TOL, "HARD"),
                               ("d1_ssi", SOFT_TOL, "soft"), ("F1", SOFT_TOL, "soft"),
                               ("rho_a", SOFT_TOL, "soft")):
            if meas.get(key) is None or (isinstance(meas.get(key), float) and meas[key] != meas[key]):
                lines.append(f"- {key}: published {tgt[key]:.1f}, measured — (n/a)")
                if kind == "HARD":
                    all_hard_pass = False
                continue
            diff = meas[key] - tgt[key]
            ok = abs(diff) <= tol
            if kind == "HARD" and not ok:
                all_hard_pass = False
            note = ""
            if kind == "soft" and key == "d1_ssi":
                note = "  (info: SSI-aligned reads high for metric models — not a target)"
            if model == "v2-s" and not ok and kind == "HARD":
                note = "  (V2-S HF-checkpoint drift, PROTOCOL_NOTES §2b #5)"
            lines.append(f"- {key}: published {tgt[key]:.1f}, measured {meas[key]:.2f} "
                         f"(diff {diff:+.2f}) -> {'PASS' if ok else 'FAIL'} [{kind}]{note}")
        lines.append("")
    verdict = ("**GATE PASSED** — published UniDepthV2 rows may be quoted next to ours "
               "(mark rows [paper]/[ours])."
               if all_hard_pass and n_checked == len(TARGETS)
               else "**GATE NOT PASSED** — do not mix published numbers into our tables.")
    lines += ["---", verdict]
    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "GATE_REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
