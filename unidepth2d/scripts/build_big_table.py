"""Assemble the b10/b20 comparison table from results_bench/ + results_ocmr/ + fps_bench.json.

    python -m scripts.build_big_table            -> prints markdown, writes docs/B20_RESULTS.md
Missing cells print as '—'. Metrics: d1 (%, higher better), AbsRel, RMSE (m), MAE (m) (lower better).
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RB = os.path.join(ROOT, "results_bench")
RO = os.path.join(ROOT, "results_ocmr")

# (row label, results_bench model key, params M, group)
MODELS = [
    ("UniDepthV2-L (teacher)", "v2-l+r9", 354.0, "foundation"),
    ("UniDepthV2-B", "v2-b+r9", 97.0, "foundation"),
    ("UniDepthV2-S", "v2-s+r9", 34.0, "foundation"),
    ("MoGe-2-L", "moge2-l+r9", 326.0, "foundation"),
    ("MoGe-2-B", "moge2-b+r9", 104.0, "foundation"),
    ("MoGe-2-S", "moge2-s+r9", 35.0, "foundation"),
    ("UniK3D-B", "unik3d-b+r9", None, "foundation"),
    ("UniK3D-S", "unik3d-s+r9", None, "foundation"),
    ("SSDI (S21 + V2-L r0 scale)", "ssdi-S21_scale2+v2-l+r0", None, "hybrid"),
    ("S16-prox (B2)", "student-S16_prox", 15.5, "student 15M"),
    ("S21-scale2 (B2)", "student-S21_scale2", 15.5, "student 15M"),
    ("S10-ultra-L (B2)", "student-S10_ultra_L", 15.5, "student 15M"),
    ("b1_mix (B1, old)", "student-b1_mix", 5.1, "student 5M"),
    ("**b10** (B1, last)", "student-b10_last", 5.12, "student 5M"),
    ("b10 (B1, best own-val)", "student-b10_best", 5.12, "student 5M"),
    ("**b20core** (dsh+cstm+fov+ownar+trim)", "student-b20core_last", 5.18, "student 5M"),
    ("**b20dws** (b20core + dws4, -41% MACs, last)", "student-b20dws_last", 5.04, "student 5M"),
    ("b20dws (best gt-val)", "student-b20dws_best", 5.04, "student 5M"),
    ("b20 (+rep4, unstable run, last)", "student-b20_last", 5.04, "student 5M"),
    ("b20 (+rep4, best gt-val)", "student-b20_best", 5.04, "student 5M"),
]
DATASETS = [("IBims-1 (unseen)", "ibims__"), ("NYU-654", "nyu-filled__"), ("SUN-RGBD raw", "sun-raw__")]
OCMR_NAME = {"student-S16_prox": "student-S16_prox", "student-b10_last": "student-b10_last",
             "student-b20core_last": "student-b20core_last", "student-b20dws_last": "student-b20dws_last", "student-b20_last": "student-b20_last",
             "student-b1_mix": "student-b1_mix", "student-S10_ultra_L": "student-S10_ultra_L",
             "student-S21_scale2": "student-S21_scale2",
             "ssdi-S21_scale2+v2-l+r0": "ssdi-S21_scale2+v2-l+r0"}


def _load(prefix, key):
    p = os.path.join(RB, f"{prefix}{key}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))["depth"]
    except Exception:
        return None


def _ocmr(key):
    name = OCMR_NAME.get(key)
    for cand in ([name] if name else []) + [key]:
        p = os.path.join(RO, f"{cand}__summary.json")
        if os.path.exists(p):
            b = json.load(open(p, encoding="utf-8"))["protocols"]["box5"]
            return b
    return None


def _fps():
    p = os.path.join(ROOT, "outputs", "checkpoints", "fps_bench.json")
    out = {}
    for fp in (p, p.replace("fps_bench.json", "fps_bench_prev.json")):
        if os.path.exists(fp):
            d = json.load(open(fp, encoding="utf-8"))
            rows = d.get("rows", d.get("results", [])) if isinstance(d, dict) else d
            for r in rows:
                out.setdefault(r["model"], r)
    return out


def f(v, k, pct=False):
    if v is None or k not in v or v[k] is None:
        return "—"
    x = v[k]
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def main():
    fps = _fps()
    lines = ["# b10 / b20 results — full comparison", "",
             "Official v2 harness (no pred clip, per-image mean). δ1 in % (↑); AbsRel, RMSE (m), MAE (m) ↓. "
             "Students use GT intrinsics; foundation models at res-level 9, no GT-K. "
             "SUN is partly IN-DOMAIN for students (SUN is in the training mix). '—' = not measured.", ""]
    for title, prefix in DATASETS:
        lines += [f"## {title}", "", "| Model | Params (M) | δ1 ↑ | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 scale-aligned ↑ |",
                  "|---|---|---|---|---|---|---|"]
        for label, key, pm, _ in MODELS:
            d = _load(prefix, key)
            if d is None:
                continue
            lines.append(f"| {label} | {pm if pm else '—'} | {f(d,'d1',True)} | {f(d,'arel')} | {f(d,'rmse')} | "
                         f"{f(d,'mae')} | {f(d,'d1_ssi',True)} |")
        lines.append("")
    lines += ["## OCMR-SUN object distance (57,677 objects, box5)", "",
              "| Model | objAbsRel ↓ | obj δ1 ↑ | MAE (m) ↓ | RMSE (m) ↓ | bias_log |", "|---|---|---|---|---|---|"]
    for label, key, _, _ in MODELS:
        b = _ocmr(key)
        if b:
            lines.append(f"| {label} | {b['objAbsRel']:.3f} | {100*b['objD1']:.1f} | {b['MAE']:.3f} | {b['RMSE']:.3f} | "
                         f"{b['bias_log']:+.3f} |")
    lines += ["", "## Speed (RTX 5070 Ti laptop, fp32, 640x480, batch 1)", "",
              "| Model | Params (M) | ms ↓ | FPS ↑ | peak VRAM (GiB) |", "|---|---|---|---|---|"]
    for label, key, _, _ in MODELS:
        r = fps.get(key)
        if r:
            lines.append(f"| {label} | {r.get('params_M','—')} | {r['ms']} | {r['fps']} | {r.get('peak_vram_GiB','—')} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, "docs", "B20_RESULTS.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
