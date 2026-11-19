"""Build the unified multi-source training corpus: real sensor GT (NYU, SUN RGB-D) + own
teacher-pseudo-labeled images, all in one layout + manifest.

Layout (images stored at 640x480 so training IO is cheap):
  outputs/data_gt/<source>/images/<id>.jpg
  outputs/data_gt/<source>/depth_mm/<id>.png     16-bit PNG, millimetres
  outputs/data_gt/manifest.csv                   id,source,img,depth,conf,is_gt,dmin,dmax,fx,fy,cx,cy

Sources:
  nyu   — HF vikhyatk/nyu_depth_v2 train, STREAMED (non-streaming blows the disk), every Nth frame.
          Depth is float32 metres. Known Kinect v1 intrinsics @640x480.
  sun   — HF wyrx/SUNRGBD_seg config 'default' split 'train' (test held out for eval).
          Depth is uint16; raw SUN encoding is bit-shifted (mm = (x>>3)|(x<<13) & 0xFFFF, i.e. x>>3
          for values in range). We auto-detect: if plain x/1000 gives implausible indoor medians,
          apply the shift. Assert plausible median (0.5–6 m) afterwards.
  own   — no copying; manifest rows point at existing images/ + outputs/depth_mm/ + conf/.
          is_gt=0; intrinsics from outputs/intrinsics.json (teacher-predicted, normalized).

Usage:
  python -m scripts.build_gt_dataset --config config.yaml --source nyu --limit 30    # spot check
  python -m scripts.build_gt_dataset --config config.yaml --source all
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np

from common.io_utils import load_config, list_images

OUT_W, OUT_H = 640, 480
MANIFEST_HEADER = ["id", "source", "img", "depth", "conf", "is_gt", "dmin", "dmax", "fx", "fy", "cx", "cy"]

# Kinect v1 intrinsics @ 640x480 (standard NYU constants), resolution-normalized.
NYU_INTR = (518.86 / 640.0, 519.47 / 480.0, 325.58 / 640.0, 253.74 / 480.0)


def _dirs(root, source):
    img_d = os.path.join(root, source, "images")
    dep_d = os.path.join(root, source, "depth_mm")
    os.makedirs(img_d, exist_ok=True)
    os.makedirs(dep_d, exist_ok=True)
    return img_d, dep_d


def _save_pair(img_d, dep_d, sid, rgb, depth_m, clip_m):
    """rgb HxWx3 uint8 (any res) -> 640x480 jpg; depth_m float32 metres -> 16-bit mm png."""
    rgb_r = cv2.resize(rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d, 0.0, min(clip_m, 65.535))
    d_r = cv2.resize(d, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    png16 = (d_r * 1000.0).astype(np.uint16)
    ip = os.path.join(img_d, sid + ".jpg")
    dp = os.path.join(dep_d, sid + ".png")
    cv2.imwrite(ip, cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(dp, png16)
    return ip, dp


def build_nyu(cfg, rows, limit=0, stride=3):
    from datasets import load_dataset
    root = cfg["data_gt"]["root"]
    img_d, dep_d = _dirs(root, "nyu")
    print(f"[nyu] streaming train split, keeping every {stride}rd frame ...")
    ds = load_dataset("vikhyatk/nyu_depth_v2", split="train", streaming=True)
    kept = 0
    fx, fy, cx, cy = NYU_INTR
    for i, s in enumerate(ds):
        if i % stride:
            continue
        rgb = np.array(s["image"])
        depth_m = np.array(s["depth_map"]).astype(np.float32)
        sid = f"nyu_{i:06d}"
        ip, dp = _save_pair(img_d, dep_d, sid, rgb, depth_m, clip_m=10.0)
        rows.append([sid, "nyu", ip, dp, "", 1, 0.2, 10.0, fx, fy, cx, cy])
        kept += 1
        if kept % 1000 == 0:
            print(f"  [nyu] kept {kept}")
        if limit and kept >= limit:
            break
    print(f"[nyu] DONE kept={kept}")


def _decode_sun_depth(arr_u16: np.ndarray) -> np.ndarray:
    """SUN RGB-D raw depth is bit-shifted uint16. Auto-detect whether the repack undid it."""
    plain_m = arr_u16.astype(np.float32) / 1000.0
    med = float(np.median(plain_m[plain_m > 0])) if (plain_m > 0).any() else 0.0
    if 0.4 <= med <= 8.0:          # plain mm already plausible
        return plain_m
    shifted = np.bitwise_or(np.right_shift(arr_u16, 3), np.left_shift(arr_u16, 13)).astype(np.float32) / 1000.0
    return shifted


def build_sun(cfg, rows, limit=0):
    from datasets import load_dataset
    root = cfg["data_gt"]["root"]
    img_d, dep_d = _dirs(root, "sun")
    print("[sun] streaming wyrx/SUNRGBD_seg default/train ...")
    ds = load_dataset("wyrx/SUNRGBD_seg", "default", split="train", streaming=True)
    kept = 0
    medians = []
    for i, s in enumerate(ds):
        rgb = np.array(s["image"].convert("RGB"))
        depth_u16 = np.array(s["depth"])
        if depth_u16.dtype != np.uint16:
            depth_u16 = depth_u16.astype(np.uint16)
        depth_m = _decode_sun_depth(depth_u16)
        depth_m[depth_m > 8.0] = 0.0            # Kinect-family range cap; beyond = invalid
        v = depth_m[depth_m > 0]
        if v.size < 1000:
            continue
        medians.append(float(np.median(v)))
        sid = f"sun_{i:06d}"
        ip, dp = _save_pair(img_d, dep_d, sid, rgb, depth_m, clip_m=8.0)
        # per-image intrinsics unavailable in this repack -> blank; filled by extract_intrinsics later
        rows.append([sid, "sun", ip, dp, "", 1, 0.2, 8.0, "", "", "", ""])
        kept += 1
        if kept % 1000 == 0:
            print(f"  [sun] kept {kept} (running median {np.median(medians):.2f} m)")
        if limit and kept >= limit:
            break
    med_all = float(np.median(medians)) if medians else 0.0
    print(f"[sun] DONE kept={kept} overall-median={med_all:.2f} m")
    assert 0.5 <= med_all <= 6.0, f"SUN depth decode implausible (median {med_all:.2f} m) — check bit-shift"


def build_own(cfg, rows, limit=0):
    """Cache 640x480 copies of the teacher-pseudo-labeled own dataset (kills the training IO
    bottleneck — originals are up to 2560x1440). Conf maps cached alongside."""
    images_dir = cfg["data"]["images_dir"]
    depth_dir = cfg["output"]["depth_dir"]
    conf_dir = cfg["output"]["conf_dir"]
    root = cfg["data_gt"]["root"]
    img_d, dep_d = _dirs(root, "own")
    cnf_d = os.path.join(root, "own", "conf")
    os.makedirs(cnf_d, exist_ok=True)
    ipath = os.path.join(cfg["output"]["root"], "intrinsics.json")
    intr = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
    n = 0
    for ip in list_images(images_dir, cfg["data"]["image_exts"]):
        stem = os.path.splitext(os.path.basename(ip))[0]
        dp = os.path.join(depth_dir, stem + ".png")
        if not os.path.exists(dp):
            continue
        out_ip = os.path.join(img_d, stem + ".jpg")
        out_dp = os.path.join(dep_d, stem + ".png")
        out_cp = os.path.join(cnf_d, stem + ".png")
        if not (os.path.exists(out_ip) and os.path.exists(out_dp)):   # resumable
            bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
            d16 = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
            if bgr is None or d16 is None:
                continue
            cv2.imwrite(out_ip, cv2.resize(bgr, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(out_dp, cv2.resize(d16, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST))
            c8 = cv2.imread(os.path.join(conf_dir, stem + ".png"), cv2.IMREAD_UNCHANGED)
            if c8 is not None:
                cv2.imwrite(out_cp, cv2.resize(c8, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST))
        fx, fy, cx, cy = intr.get(stem, ["", "", "", ""])
        rows.append([stem, "own", out_ip, out_dp, out_cp if os.path.exists(out_cp) else "",
                     0, 0.2, 30.0, fx, fy, cx, cy])
        n += 1
        if n % 1000 == 0:
            print(f"  [own] cached {n}")
        if limit and n >= limit:
            break
    print(f"[own] DONE rows={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default="all", choices=["all", "nyu", "sun", "own"])
    ap.add_argument("--limit", type=int, default=0, help="cap kept samples per source (spot check)")
    ap.add_argument("--nyu-stride", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("data_gt", {"root": os.path.join(cfg["output"]["root"], "data_gt")})
    root = cfg["data_gt"]["root"]
    os.makedirs(root, exist_ok=True)
    manifest = os.path.join(root, "manifest.csv")

    # load existing rows so sources can be built incrementally without losing others
    rows = []
    done_sources = set()
    if os.path.exists(manifest):
        with open(manifest, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append([r[k] for k in MANIFEST_HEADER])
                done_sources.add(r["source"])

    def rebuild(source):
        nonlocal rows
        rows = [r for r in rows if r[1] != source]

    if args.source in ("all", "nyu"):
        rebuild("nyu"); build_nyu(cfg, rows, limit=args.limit, stride=args.nyu_stride)
    if args.source in ("all", "sun"):
        rebuild("sun"); build_sun(cfg, rows, limit=args.limit)
    if args.source in ("all", "own"):
        rebuild("own"); build_own(cfg, rows, limit=args.limit)

    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        w.writerows(rows)
    from collections import Counter
    print("manifest:", manifest, Counter(r[1] for r in rows))


if __name__ == "__main__":
    main()
