"""Create train/val/test splits over images that already have teacher depth output.

Only images with a saved teacher depth map are eligible (so distillation never references a missing
target). Deterministic given the seed.

  python -m scripts.make_splits --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random

from common.io_utils import list_images, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    depth_dir = cfg["output"]["depth_dir"]
    images = list_images(cfg["data"]["images_dir"], cfg["data"]["image_exts"])

    eligible = []
    for ip in images:
        stem = os.path.splitext(os.path.basename(ip))[0]
        if os.path.exists(os.path.join(depth_dir, stem + ".png")):
            eligible.append(stem)

    if not eligible:
        raise SystemExit(
            f"No teacher depth maps found in {depth_dir}. Run teacher.run_teacher first."
        )

    scfg = cfg["splits"]
    rng = random.Random(int(scfg.get("seed", 1234)))
    rng.shuffle(eligible)

    n = len(eligible)
    n_val = int(round(n * float(scfg.get("val_frac", 0.10))))
    n_test = int(round(n * float(scfg.get("test_frac", 0.05))))
    val = eligible[:n_val]
    test = eligible[n_val:n_val + n_test]
    train = eligible[n_val + n_test:]

    out = {"train": train, "val": val, "test": test, "counts": {
        "train": len(train), "val": len(val), "test": len(test), "total": n}}
    os.makedirs(os.path.dirname(scfg["file"]), exist_ok=True)
    with open(scfg["file"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote splits to {scfg['file']}: {out['counts']}")


if __name__ == "__main__":
    main()
