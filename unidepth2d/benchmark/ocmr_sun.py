"""Build the OCMR-SUN ground-truth cache: per-object visible-surface ranges from
SUN RGB-D 2D instance polygons x RAW sensor depth, on the NYU-filtered alltest split.

For every sample dir (same filtered split as datasets_tier1.iter_sun):
  - read annotation2Dfinal/index.json (instance polygons + object names)
  - drop stuff classes (floor/wall/ceiling/...) and degenerate polygons
  - rasterize each polygon; GT range r_vis = MEDIAN of valid RAW depth inside it
    (visible-surface convention — what a monocular ranger estimates); also p10
    (near-surface, safety), IQR spread, pixel area, tight bbox
  - quality gates: >=150 valid raw-depth px, area >= 0.05% of image
Cache: data_bench/ocmr_sun_gt.jsonl (one record per image) + printed stats.

  python -m benchmark.ocmr_sun build [--limit N]
  python -m benchmark.ocmr_sun stats
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from benchmark.datasets_tier1 import DATA_ROOT, _sun_alltest_paths, _sun_decode_depth, sun_extract

CACHE = os.path.join(DATA_ROOT, "ocmr_sun_gt.jsonl")

STUFF = {
    "floor", "wall", "walls", "ceiling", "room", "floors", "ceilings",
    "wall1", "wall2", "background",
}
MIN_VALID_PX = 150
MIN_AREA_FRAC = 0.0005


def _norm_name(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def build(limit: int = 0):
    base = sun_extract()
    rels = [r for r in _sun_alltest_paths() if "NYUdata" not in r]
    n_img, n_obj, skipped = 0, 0, {"no_ann": 0, "no_depth": 0, "small": 0, "stuff": 0}
    with open(CACHE, "w", encoding="utf-8") as out:
        for rel in rels:
            sdir = os.path.join(base, rel)
            ann_p = os.path.join(sdir, "annotation2Dfinal", "index.json")
            dep_dir = os.path.join(sdir, "depth")
            if not os.path.exists(ann_p) or not os.path.isdir(dep_dir):
                skipped["no_ann"] += 1
                continue
            try:
                ann = json.load(open(ann_p, encoding="utf-8"))
                dep_path = os.path.join(dep_dir, os.listdir(dep_dir)[0])
                png = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
                img_dir = os.path.join(sdir, "image")
                img_name = os.listdir(img_dir)[0]
                K = np.loadtxt(os.path.join(sdir, "intrinsics.txt")).reshape(3, 3)
            except (OSError, IndexError, ValueError, json.JSONDecodeError):
                skipped["no_ann"] += 1
                continue
            if png is None:
                skipped["no_depth"] += 1
                continue
            depth = _sun_decode_depth(png.astype(np.uint16))
            h, w = depth.shape
            objects_names = [
                (_norm_name(o.get("name", "")) if isinstance(o, dict) else _norm_name(o))
                for o in (ann.get("objects") or [])
            ]
            objs = []
            for fr in ann.get("frames", [])[:1]:
                for poly in fr.get("polygon", []):
                    try:
                        oi = int(poly.get("object", -1))
                        name = objects_names[oi] if 0 <= oi < len(objects_names) else ""
                        if not name or name in STUFF:
                            skipped["stuff"] += 1
                            continue
                        xs = np.asarray(poly["x"], dtype=np.float64)
                        ys = np.asarray(poly["y"], dtype=np.float64)
                        if xs.size < 3:
                            continue
                    except (TypeError, ValueError, KeyError):
                        continue
                    pts = np.stack([np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)], 1)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts.round().astype(np.int32)], 1)
                    area = int(mask.sum())
                    if area < MIN_AREA_FRAC * h * w:
                        skipped["small"] += 1
                        continue
                    dvals = depth[(mask > 0) & (depth > 0.005)]
                    if dvals.size < MIN_VALID_PX:
                        skipped["small"] += 1
                        continue
                    q10, med, q25, q75 = np.percentile(dvals, [10, 50, 25, 75])
                    x0, x1 = float(xs.min()), float(xs.max())
                    y0, y1 = float(ys.min()), float(ys.max())
                    objs.append({
                        "class": name,
                        "r_vis_med": round(float(med), 4),
                        "r_vis_p10": round(float(q10), 4),
                        "spread_iqr": round(float(q75 - q25), 4),
                        "n_valid_px": int(dvals.size),
                        "area_px": area,
                        "bbox_xyxy": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                        "poly_x": [round(float(v), 1) for v in xs],
                        "poly_y": [round(float(v), 1) for v in ys],
                    })
            if not objs:
                continue
            rec = {
                "rel": rel,
                "image": os.path.join(rel, "image", img_name).replace("\\", "/"),
                "w": w, "h": h,
                "K": [round(float(v), 4) for v in K.ravel()],
                "objects": objs,
            }
            out.write(json.dumps(rec) + "\n")
            n_img += 1
            n_obj += len(objs)
            if n_img % 500 == 0:
                print(f"  {n_img} images / {n_obj} objects", flush=True)
            if limit and n_img >= limit:
                break
    print(f"DONE images={n_img} objects={n_obj} skipped={skipped}")
    print("cache ->", CACHE)


def stats():
    from collections import Counter
    n_img, n_obj = 0, 0
    cls = Counter()
    ranges = []
    for line in open(CACHE, encoding="utf-8"):
        rec = json.loads(line)
        n_img += 1
        for o in rec["objects"]:
            n_obj += 1
            cls[o["class"]] += 1
            ranges.append(o["r_vis_med"])
    r = np.asarray(ranges)
    print(f"images={n_img} objects={n_obj} classes={len(cls)}")
    print(f"range med={np.median(r):.2f}m p10={np.percentile(r,10):.2f} p90={np.percentile(r,90):.2f}"
          f" | <1m: {(r<1).mean():.1%} | <2m: {(r<2).mean():.1%}")
    print("top-30 classes:", cls.most_common(30))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build(a.limit) if a.cmd == "build" else stats()
