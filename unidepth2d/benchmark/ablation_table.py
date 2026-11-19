"""Assemble the ablation table from the registry + all eval result files.

Pulls, per registered model:
  - NYU / SUN real-GT rows from outputs/checkpoints/gt_eval_results.json (latest entry per dataset
    that contains the model's ckpt-derived name)
  - OCMR object-range rows (real sensor GT) from outputs/checkpoints/ocmr_{nyu,sun}.json
  - own-test rows from outputs/checkpoints/own_eval/<name>.json — these are student-vs-TEACHER
    agreement (pseudo-GT), NOT real accuracy, so they are named *_vsT and any entry whose note is
    not "frozen-test, no train/eval overlap" is BLANKED and flagged LEAKED (train/test overlap).
  - FPS from outputs/checkpoints/fps_bench.json when present

Real-GT vs teacher-agreement are kept in clearly distinct columns. Real-GT dedup keeps the entry
with the LARGEST num_images per (dataset, model) so a stale smoke run can't overwrite a full run.

  python -m benchmark.ablation_table --config config.yaml
Writes outputs/checkpoints/ablation_table.md (+ .csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from common.io_utils import load_config


def _load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ck_dir = cfg["distill"]["checkpoints_dir"]

    registry = _load(os.path.join("benchmark", "registry.json")) or {}
    gt_results = _load(os.path.join(ck_dir, "gt_eval_results.json")) or []
    ocmr_results = _load(os.path.join(ck_dir, "ocmr_nyu.json")) or []
    ocmr_results += _load(os.path.join(ck_dir, "ocmr_sun.json")) or []
    # fps_bench.json: legacy format = list of rows; new format = {"env": ..., "rows": [...]}.
    # Row names: legacy students = "<parent dir>", new = "student-<parent dir>" — index both.
    fps_raw = _load(os.path.join(ck_dir, "fps_bench.json")) or []
    fps_list = fps_raw.get("rows", []) if isinstance(fps_raw, dict) else fps_raw
    fps_rows = {}
    for r in fps_list:
        fps_rows[r["model"]] = r
        if r["model"].startswith("student-"):
            fps_rows[r["model"][len("student-"):]] = r

    # Real-GT numbers per (dataset, model-label). De-dup by LARGEST num_images so a stale
    # 12-image smoke run can never silently overwrite the full 1500-image result (audit finding).
    gt, gt_n = {}, {}
    for entry in gt_results:
        n = int(entry.get("num_images", 0))
        for label, m in entry.get("models", {}).items():
            key = (entry["dataset"], label)
            if n >= gt_n.get(key, -1):
                gt[key] = m
                gt_n[key] = n
    # OCMR object-range numbers per (dataset, label), same largest-N de-dup.
    ocmr, ocmr_n = {}, {}
    for entry in ocmr_results:
        n = int(entry.get("num_images", 0))
        for label, m in entry.get("models", {}).items():
            key = (entry["dataset"], label)
            if n >= ocmr_n.get(key, -1):
                ocmr[key] = m
                ocmr_n[key] = n

    def gt_label_for(name, info):
        # gt_eval labels students as "student:<ckpt parent dir>" ("student:ckpt" for bare checkpoints/)
        parent = os.path.basename(os.path.dirname(info["ckpt"]))
        return f"student:{parent if parent != 'checkpoints' else 'ckpt'}"

    # own_eval is student-vs-TEACHER agreement (pseudo-GT), NOT real accuracy. Columns are named
    # *_vsT to prevent them being read as GT error, and any entry whose note is not the clean
    # frozen-test marker is BLANKED (its numbers are train/test-overlap leakage — audit critical).
    CLEAN_NOTE = "frozen-test, no train/eval overlap"
    header = ["model", "backbone", "params_M", "recipe",
              "NYU_AbsRel", "NYU_d1", "SUN_AbsRel", "SUN_d1",
              "NYU_objAbsRel", "NYU_objMAE5m", "NYU_prox1mF1",
              "ownAbsRel_vsT", "ownObjMAE5m_vsT", "own_leak", "FPS"]
    rows = []
    for name, info in registry.items():
        label = gt_label_for(name, info)
        nyu = gt.get(("nyu", label), {})
        sun = gt.get(("sun", label), {})
        oc = ocmr.get(("nyu", label), {})
        own = _load(os.path.join(ck_dir, "own_eval", f"{name}.json")) or {}
        note = str(own.get("note", ""))
        clean = note == CLEAN_NOTE
        dvt = own.get("dense_vs_teacher", {}) if clean else {}
        pot = own.get("per_object_vs_teacher", {}) if clean else {}
        leak_flag = "" if clean else ("LEAKED" if own else "")
        parent = os.path.basename(os.path.dirname(info["ckpt"]))
        fps = fps_rows.get(parent, {}).get("fps", "")
        rows.append([
            name, info.get("backbone", ""), info.get("params_M", ""), info.get("recipe", ""),
            nyu.get("AbsRel", ""), nyu.get("d1", ""), sun.get("AbsRel", ""), sun.get("d1", ""),
            oc.get("obj_AbsRel", ""), oc.get("obj_MAE_5m", ""), oc.get("prox1m_F1", ""),
            dvt.get("AbsRel", ""), pot.get("MAE_m_within_5m", ""), leak_flag, fps,
        ])

    md_path = os.path.join(ck_dir, "ablation_table.md")
    csv_path = os.path.join(ck_dir, "ablation_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"wrote {md_path} and .csv ({len(rows)} models)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
