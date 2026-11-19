"""Extract UniDepth v2's predicted camera intrinsics per image (for the conditioned-student experiment).

The teacher gets its metric scale from a camera module; a plain RGB student lacks that. To test
whether handing the student the intrinsics closes the far-object gap, we first cache UniDepth's
predicted K per image. Intrinsics are stored RESOLUTION-NORMALIZED (fx/W, fy/H, cx/W, cy/H) so they
are independent of the (varying) source resolutions and of the student's train size.

  python -m teacher.extract_intrinsics --config config.yaml            # all images
  python -m teacher.extract_intrinsics --config config.yaml --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from common.io_utils import list_images, load_config

INFER_LONGEST = 640  # intrinsics are global; a smaller resolution is faster and plenty accurate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images-dir", default="", help="override: extract for this dir instead of data.images_dir")
    ap.add_argument("--out", default="", help="override output json path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.images_dir:
        cfg["data"]["images_dir"] = args.images_dir

    from unidepth.models import UniDepthV2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = cfg["teacher"]["checkpoints"]
    model = None
    for c in ckpts:
        try:
            model = UniDepthV2.from_pretrained(c).to(device).eval()
            print("loaded", c)
            break
        except Exception as e:
            print("failed", c, str(e)[:120])
    if model is None:
        raise SystemExit("no UniDepth checkpoint loaded")

    images = list_images(cfg["data"]["images_dir"], cfg["data"]["image_exts"])
    if args.limit:
        images = images[: args.limit]

    out_path = args.out or os.path.join(cfg["output"]["root"], "intrinsics.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result = {}
    if os.path.exists(out_path):  # resume
        result = json.load(open(out_path, encoding="utf-8"))

    t0 = time.time()
    done = 0
    for i, ip in enumerate(images):
        stem = os.path.splitext(os.path.basename(ip))[0]
        if stem in result:
            continue
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > INFER_LONGEST:
            s = INFER_LONGEST / max(h, w)
            rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        ih, iw = rgb.shape[:2]
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        with torch.inference_mode():
            pred = model.infer(t.to(device))
        K = pred["intrinsics"].squeeze().float().cpu().numpy()  # 3x3 at the fed resolution
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        # resolution-normalized
        result[stem] = [round(fx / iw, 5), round(fy / ih, 5), round(cx / iw, 5), round(cy / ih, 5)]
        done += 1
        if done % 200 == 0:
            json.dump(result, open(out_path, "w", encoding="utf-8"))
            print(f"[{i+1}/{len(images)}] {done} done  ({done/max(1e-6,time.time()-t0):.1f}/s)")

    json.dump(result, open(out_path, "w", encoding="utf-8"))
    print(f"DONE {len(result)} intrinsics -> {out_path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
