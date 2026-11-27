"""Download a diverse subset of the ARKitScenes depth-upsampling TRAINING split to the external SSD.

One video per visit (scene), up to --n videos, deterministic (seeded). For each video it fetches
  upsampling/Training/<vid>.zip           (color 1920x1440, highres_depth (laser, mm uint16),
                                           lowres_depth, confidence)
  raw/Training/<vid>/wide_intrinsics.zip  (per-frame .pincam: w h fx fy cx cy)
unzips into <root>/upsampling/Training/<vid>/ and <root>/raw/Training/<vid>/wide_intrinsics/,
deletes the zips, and is resumable (skips finished videos). Stops if free space < --min-free-gb.
The Validation split is NOT downloaded (it is used as a test set by other papers).

  python -m scripts.download_arkitscenes_subset --root Z:/Datasets/ARKitScenes --n 450
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import time
import urllib.request
import zipfile

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"


def fetch(url: str, dst: str, tries: int = 4) -> bool:
    tmp = dst + ".part"
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, dst)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  retry {k + 1}/{tries} {url}: {e}", flush=True)
            time.sleep(5 * (k + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def dir_gb(root: str) -> float:
    total = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / 1e9


def unzip_and_delete(zp: str, dst: str) -> bool:
    try:
        with zipfile.ZipFile(zp) as z:
            z.extractall(dst)
        ok = True
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  bad zip {zp}: {e}", flush=True)
        ok = False
    os.remove(zp)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/Datasets/ARKitScenes")
    ap.add_argument("--csv", default="Z:/Datasets/ARKitScenes/repo/depth_upsampling/upsampling_train_val_splits.csv")
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--max-gb", type=float, default=0.0, help="stop once <root> holds this many GB (0 = no cap)")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if r["fold"] == "Training"]
    by_visit: dict[str, list[str]] = {}
    for r in rows:
        by_visit.setdefault(r["visit_id"], []).append(r["video_id"])
    rng = random.Random(a.seed)
    visits = sorted(by_visit)
    rng.shuffle(visits)
    chosen = [sorted(by_visit[v])[rng.randrange(len(by_visit[v]))] for v in visits][: a.n]
    os.makedirs(a.root, exist_ok=True)
    with open(os.path.join(a.root, "subset_video_ids.txt"), "w") as f:
        f.write("\n".join(chosen) + "\n")
    print(f"{len(rows)} training videos / {len(by_visit)} visits -> downloading {len(chosen)}", flush=True)

    up_dir = os.path.join(a.root, "upsampling", "Training")
    raw_dir = os.path.join(a.root, "raw", "Training")
    os.makedirs(up_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    done = 0
    for i, vid in enumerate(chosen):
        free = shutil.disk_usage(a.root).free / 1e9
        if free < a.min_free_gb:
            print(f"STOP: only {free:.1f} GB free", flush=True)
            break
        if a.max_gb > 0 and dir_gb(a.root) >= a.max_gb:
            print(f"STOP: size cap {a.max_gb} GB reached", flush=True)
            break
        vdir = os.path.join(up_dir, vid)
        idir = os.path.join(raw_dir, vid, "wide_intrinsics")
        if not (os.path.isdir(os.path.join(vdir, "highres_depth")) and os.path.isdir(idir)):
            if not os.path.isdir(os.path.join(vdir, "highres_depth")):
                zp = os.path.join(up_dir, f"{vid}.zip")
                if not fetch(f"{BASE}/upsampling/Training/{vid}.zip", zp):
                    print(f"[{i}] FAIL {vid} upsampling", flush=True)
                    continue
                unzip_and_delete(zp, up_dir)
            if not os.path.isdir(idir):
                os.makedirs(os.path.join(raw_dir, vid), exist_ok=True)
                zp = os.path.join(raw_dir, vid, "wide_intrinsics.zip")
                if fetch(f"{BASE}/raw/Training/{vid}/wide_intrinsics.zip", zp):
                    unzip_and_delete(zp, os.path.join(raw_dir, vid))
                else:
                    print(f"[{i}] FAIL {vid} intrinsics", flush=True)
        done += 1
        if i % 10 == 0:
            print(f"[{i}/{len(chosen)}] {vid} ok, free {free:.1f} GB", flush=True)
    print(f"DONE {done}/{len(chosen)}", flush=True)


if __name__ == "__main__":
    main()
