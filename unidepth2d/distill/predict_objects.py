"""Demo: image + detection boxes -> per-object metric distance from the DISTILLED STUDENT.

This is the actual deployment path: one student forward pass -> metric depth map -> sample the paper's
5-point recipe at each box. If a teacher objects JSON exists for the image, prints student vs teacher
side by side so you can see the student measuring real distances on its own.

  python -m distill.predict_objects --config config.yaml --stem <image_stem>
  python -m distill.predict_objects --config config.yaml --split test --n 3   # demo N test images
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from common import depth_ops as D
from common.io_utils import load_config, parse_yolo_txt
from distill.dataset import IMAGENET_MEAN, IMAGENET_STD
from distill.student import build_student


def load_student(cfg, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_model = ck.get("cfg", cfg)
    model = build_student(cfg_model).to(device).eval()
    model.load_state_dict(ck["model"])
    use_intr = bool(cfg_model["student"].get("use_intrinsics", False))
    intr_map = {}
    if use_intr:
        p = os.path.join(cfg["output"]["root"], "intrinsics.json")
        if os.path.exists(p):
            intr_map = json.load(open(p, encoding="utf-8"))
    return model, use_intr, intr_map


def student_depth(model, img_path, cfg, device, use_intr, intr_map, stem):
    s = cfg["student"]
    th, tw = int(s["train_size"][0]), int(s["train_size"][1])
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    oh, ow = rgb.shape[:2]
    rz = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rz = (rz - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(np.ascontiguousarray(np.transpose(rz, (2, 0, 1)))).float()[None].to(device)
    intr = None
    if use_intr:
        intr = torch.tensor([intr_map.get(stem, [0.7, 0.9, 0.5, 0.5])], dtype=torch.float32, device=device)
    with torch.no_grad():
        dm = model.infer_metres(t, intr)[0, 0].cpu().numpy()
    return cv2.resize(dm, (ow, oh), interpolation=cv2.INTER_LINEAR), (ow, oh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stem", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(cfg["distill"]["checkpoints_dir"], "best.pt")
    model, use_intr, intr_map = load_student(cfg, ckpt, device)

    if args.stem:
        stems = [args.stem]
    else:
        stems = json.load(open(cfg["splits"]["file"], encoding="utf-8"))[args.split]
        # prefer images that actually have objects, for a meaningful demo
        chosen, objdir = [], cfg["output"]["objects_dir"]
        for st in stems:
            jp = os.path.join(objdir, st + ".json")
            if os.path.exists(jp) and json.load(open(jp, encoding="utf-8")).get("num_objects", 0) >= 2:
                chosen.append(st)
            if len(chosen) >= args.n:
                break
        stems = chosen

    images_dir = cfg["data"]["images_dir"]
    exts = cfg["data"]["image_exts"]
    objects_dir = cfg["output"]["objects_dir"]

    for stem in stems:
        img_path = next((os.path.join(images_dir, stem + e)
                         for e in exts if os.path.exists(os.path.join(images_dir, stem + e))), None)
        if img_path is None:
            print(f"[skip] no image for {stem}")
            continue
        depth_m, (ow, oh) = student_depth(model, img_path, cfg, device, use_intr, intr_map, stem)

        # teacher objects JSON gives us the boxes (+ teacher's own distance to compare)
        teacher = {}
        jp = os.path.join(objects_dir, stem + ".json")
        boxes = []
        if os.path.exists(jp):
            oj = json.load(open(jp, encoding="utf-8"))
            gw = oj.get("depth_grid_w", oj.get("image_w", ow))
            gh = oj.get("depth_grid_h", oj.get("image_h", oh))
            sx, sy = ow / float(gw), oh / float(gh)   # map stored box coords onto original res
            for o in oj["objects"]:
                x1, y1, x2, y2 = o["bbox_xyxy"]
                boxes.append((o.get("class_name") or f"class{o.get('class_id')}",
                              (x1 * sx, y1 * sy, x2 * sx, y2 * sy)))
                teacher[(round(x1 * sx), round(y1 * sy))] = o["distance_m"]

        print(f"\n=== {stem[:40]}  ({ow}x{oh}) ===")
        print(f"{'object':<22}{'STUDENT':>12}{'(feet)':>9}{'TEACHER':>12}{'flag':>10}")
        rows = []
        for name, box in boxes:
            od = D.sample_object_distance(depth_m, box, inset_frac=0.20,
                                          uncertainty_std_m=1.0, proximity_warn_m=1.0)
            if od is None:
                continue
            tval = teacher.get((round(box[0]), round(box[1])))
            rows.append((name, od, tval))
        for name, od, tval in sorted(rows, key=lambda r: r[1].distance_m):
            flag = "NEAR!" if od.proximity_warn else ("~unsure" if od.uncertain else "")
            tstr = f"{tval:.2f} m" if tval is not None else "  -"
            print(f"{name[:22]:<22}{od.distance_m:>9.2f} m{od.distance_ft:>8.1f}{tstr:>12}{flag:>10}")


if __name__ == "__main__":
    main()
