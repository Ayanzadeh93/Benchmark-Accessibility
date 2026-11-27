"""Download a Hypersim TRAIN-split subset to the external SSD, file by file (HTTP range reads into the
official per-scene zips, via contrib/99991's WebFile) - no full 1-20 GB zips are stored.

Keeps, for --n-scenes scenes sampled from the official scene-level 'train' partition and only
frames with included_in_public_release=True:
  images/scene_cam_XX_final_preview/frame.IIII.tonemap.jpg   (tone-mapped RGB, 1024x768)
  images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5  (Euclidean distance to the
                                                                  camera centre, NOT planar depth)
  _detail/metadata_scene.csv, _detail/cam_XX/* camera files
Frames are subsampled with --frame-stride. Resumable (existing files are skipped).

  python -m scripts.download_hypersim_subset --root Z:/Datasets/Hypersim --n-scenes 120
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import zipfile

import requests

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/Datasets/Hypersim")
    ap.add_argument("--repo", default="Z:/Datasets/Hypersim/repo")
    ap.add_argument("--n-scenes", type=int, default=120)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--max-gb", type=float, default=0.0, help="stop once <root>/scenes holds this many GB (0 = no cap)")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.repo, "contrib", "99991"))
    from download import WebFile  # noqa: E402  (official contrib helper)

    split = os.path.join(a.repo, "evermotion_dataset", "analysis", "metadata_images_split_scene_v1.csv")
    keep: dict[str, set[tuple[str, int]]] = {}
    for r in csv.DictReader(open(split)):
        if r["split_partition_name"] == "train" and r["included_in_public_release"] == "True":
            keep.setdefault(r["scene_name"], set()).add((r["camera_name"], int(r["frame_id"])))
    scenes = sorted(keep)
    random.Random(a.seed).shuffle(scenes)
    scenes = sorted(scenes[: a.n_scenes])
    out = os.path.join(a.root, "scenes")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(a.root, "subset_scenes.txt"), "w") as f:
        f.write("\n".join(scenes) + "\n")
    print(f"{len(keep)} train scenes -> downloading {len(scenes)}", flush=True)

    sess = requests.session()
    for si, sc in enumerate(scenes):
        free = shutil.disk_usage(a.root).free / 1e9
        if free < a.min_free_gb:
            print(f"STOP: only {free:.1f} GB free", flush=True)
            break
        used = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs) / 1e9
        if a.max_gb > 0 and used >= a.max_gb:
            print(f"STOP: size cap {a.max_gb} GB reached ({used:.1f} GB)", flush=True)
            break
        allowed = {(c, fi) for (c, fi) in keep[sc] if fi % a.frame_stride == 0}
        try:
            z = zipfile.ZipFile(WebFile(f"{BASE}/{sc}.zip", sess))
        except Exception as e:  # noqa: BLE001
            print(f"[{si}] FAIL open {sc}: {e}", flush=True)
            continue
        n = 0
        for e in z.infolist():
            fn = e.filename
            if e.is_dir():
                continue
            want = False
            if "/_detail/" in fn and (fn.endswith(".csv") or "camera_keyframe" in fn):
                want = True
            else:
                for kind, suf in (("_final_preview/", ".tonemap.jpg"), ("_geometry_hdf5/", ".depth_meters.hdf5")):
                    if kind in fn and fn.endswith(suf):
                        cam = "cam_" + fn.split("scene_cam_")[1][:2]
                        fi = int(fn.rsplit("frame.", 1)[1].split(".")[0])
                        want = (cam, fi) in allowed
            if not want or os.path.isfile(os.path.join(out, fn)):
                continue
            for k in range(3):
                try:
                    z.extract(fn, out)
                    n += 1
                    break
                except Exception as ex:  # noqa: BLE001
                    print(f"  retry {fn}: {ex}", flush=True)
        print(f"[{si + 1}/{len(scenes)}] {sc}: {n} files, free {free:.1f} GB", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
