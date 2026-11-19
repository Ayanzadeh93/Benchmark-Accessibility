"""Training-free metric-scale correction on a frozen depth model (contribution b).

Direct-inference track — complements prototypes/sma_anchor.py (post-hoc CSV track:
distance-bin diagnostics + cross-dataset affine recalibration + fy-constant size
anchor). Here the model runs live on OCMR-SUN images with per-image GT intrinsics,
and TWO inference-time anchors produce a single global scale s (d' = s * d):

  ground   Ground-plane / camera-height lock (A4): fit a robust plane to the
           PREDICTED 3D points inside the image's floor region; s = h_known/h_pred.
           h_known = camera height, recovered per image from GT floor + raw depth
           (side-channel EXTRINSIC input, emulating a known mount height; on the
           gold egocentric set it is the measured chest height). Floor region =
           SUN's floor polygon (oracle floor detector; a segmenter replaces it in
           deployment).
  sma      Semantic size anchors (A1): objects of classes with tight physical-
           height priors vote s_i = H_prior / (h_px * d_i / f_y); votes fused by
           Huber-IRLS weighted mean in log space; no anchors -> s = 1 (identity
           fallback, can never hurt anchor-free images).

Positioning (docs/RELATED_WORK_DOSSIER.md area 3): unlike relative->metric lifting
(Metric-from-Human, RSA) or trained cue fusion (GVDepth), this corrects the RESIDUAL
scale drift of an already-metric model, closed-form, training-free.

  python -m prototypes.anchor_lock --model v2-l --res-level 9 --limit 300
  python -m prototypes.anchor_lock --model student:outputs/checkpoints/S10_ultra_L/best.pt

Outputs: results_ocmr/anchorlock__<model>.json (before/after dense + object metrics,
scale-factor stats, floor/anchor coverage).
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from benchmark import ocmr_metrics, protocol
from benchmark.datasets_tier1 import _sun_decode_depth, sun_extract
from benchmark.ocmr_sun import CACHE
from benchmark.run_bench import build_model

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results_ocmr")

# physical HEIGHT priors (metres): (mean, sigma). Loose classes get small weight.
SIZE_PRIORS = {
    "door": (2.03, 0.10),
    "person": (1.70, 0.12),
    "chair": (0.90, 0.15),
    "monitor": (0.42, 0.10),
}
MIN_ANCHOR_PX = 40
TRUNC_MARGIN = 3   # bbox touching top/bottom border -> truncated -> rejected


def robust_plane(P: np.ndarray, iters: int = 3):
    w = np.ones(len(P))
    mu, n = P.mean(0), np.array([0.0, 1.0, 0.0])
    for _ in range(iters):
        mu = (P * w[:, None]).sum(0) / w.sum()
        A = (P - mu) * np.sqrt(w)[:, None]
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        n = vt[-1]
        r = np.abs((P - mu) @ n)
        s = np.median(r) * 1.4826 + 1e-6
        w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
    return n, mu, float(np.median(np.abs((P - mu) @ n)))


def plane_height(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, max_pts=4000):
    m = mask & (depth > 0.005) & np.isfinite(depth)
    if m.sum() < 400:
        return None, None
    v, u = np.nonzero(m)
    z = depth[m].astype(np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1)
    if len(P) > max_pts:
        P = P[np.linspace(0, len(P) - 1, max_pts).astype(int)]
    n, mu, resid = robust_plane(P)
    return abs(float(mu @ n)), resid


def floor_mask_of(rec_rel: str, base: str, h: int, w: int):
    ann_p = os.path.join(base, rec_rel, "annotation2Dfinal", "index.json")
    try:
        ann = json.load(open(ann_p, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    names = [(o.get("name", "") if isinstance(o, dict) else str(o)).lower()
             for o in (ann.get("objects") or [])]
    mask = np.zeros((h, w), np.uint8)
    found = False
    for fr in ann.get("frames", [])[:1]:
        for poly in fr.get("polygon", []):
            oi = int(poly.get("object", -1))
            if 0 <= oi < len(names) and names[oi] in ("floor", "floors"):
                xs = np.clip(np.asarray(poly["x"], float), 0, w - 1)
                ys = np.clip(np.asarray(poly["y"], float), 0, h - 1)
                if xs.size >= 3:
                    cv2.fillPoly(mask, [np.stack([xs, ys], 1).round().astype(np.int32)], 1)
                    found = True
    return mask.astype(bool) if found else None


def sma_scale(objects, pred_depth, K):
    h_img, w_img = pred_depth.shape
    fy = K[1, 1]
    votes, weights = [], []
    for obj in objects:
        cls = obj["class"]
        if cls not in SIZE_PRIORS:
            continue
        x0, y0, x1, y1 = obj["bbox_xyxy"]
        h_px = y1 - y0
        if h_px < MIN_ANCHOR_PX or y0 < TRUNC_MARGIN or y1 > h_img - TRUNC_MARGIN:
            continue
        xs0, xs1 = int(max(0, x0 + 0.2 * (x1 - x0))), int(min(w_img, x1 - 0.2 * (x1 - x0)))
        ys0, ys1 = int(max(0, y0 + 0.2 * h_px)), int(min(h_img, y1 - 0.2 * h_px))
        core = pred_depth[ys0:ys1, xs0:xs1]
        core = core[np.isfinite(core) & (core > 0)]
        if core.size < 20:
            continue
        d = float(np.median(core))
        H_prior, sigma = SIZE_PRIORS[cls]
        H_implied = h_px * d / fy
        if not (0.2 * H_prior < H_implied < 5 * H_prior):
            continue
        votes.append(np.log(H_prior / H_implied))
        weights.append((H_prior / sigma) ** 2)
    if not votes:
        return 1.0, 0
    v = np.asarray(votes)
    w = np.asarray(weights, dtype=np.float64)
    mu = float(np.average(v, weights=w))
    for _ in range(3):
        r = v - mu
        s = np.median(np.abs(r)) * 1.4826 + 1e-6
        wr = w / (1.0 + (r / (2.5 * s)) ** 2)
        mu = float(np.average(v, weights=wr))
    mu = float(np.clip(mu, np.log(0.6), np.log(1.4)))  # safety cap +-40%
    return float(np.exp(mu)), len(votes)


def box5_median(dp, bbox, w, h):
    """Shared single-source 5-point recipe (ocmr_metrics.box5_sample, >=3-of-5 validity);
    w/h kept in the signature for call-site stability (shape is read from the depth map)."""
    med, _, _ = ocmr_metrics.box5_sample(dp, bbox)
    return med if np.isfinite(med) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--res-level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, device, False, args.res_level)
    base = sun_extract()
    os.makedirs(RESULTS, exist_ok=True)

    variants = ["base", "ground", "sma", "both"]
    obj_acc = {v: {"gt": [], "pred": []} for v in variants}
    dense_acc = {v: [] for v in variants}
    scale_log = {"ground": [], "sma": []}
    n_img = n_floor = n_anch = 0

    for line in open(CACHE, encoding="utf-8"):
        rec = json.loads(line)
        sdir = os.path.join(base, rec["rel"])
        try:
            img_dir = os.path.join(sdir, "image")
            bgr = cv2.imread(os.path.join(img_dir, os.listdir(img_dir)[0]), cv2.IMREAD_COLOR)
            dep_dir = os.path.join(sdir, "depth")
            gt_png = cv2.imread(os.path.join(dep_dir, os.listdir(dep_dir)[0]), cv2.IMREAD_UNCHANGED)
        except (OSError, IndexError):
            continue
        if bgr is None or gt_png is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = _sun_decode_depth(gt_png.astype(np.uint16))
        h, w = rec["h"], rec["w"]
        K = np.asarray(rec["K"], dtype=np.float64).reshape(3, 3)

        pred = model.predict(rgb, K)["depth"]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        s_ground = 1.0
        fmask = floor_mask_of(rec["rel"], base, h, w)
        if fmask is not None:
            h_gt, r_gt = plane_height(gt, fmask, K)
            h_pr, _ = plane_height(pred, fmask, K)
            if h_gt and h_pr and r_gt is not None and r_gt < 0.05 and 0.5 < h_gt < 2.5:
                s = h_gt / h_pr
                if 0.5 < s < 2.0:
                    s_ground = s
                    n_floor += 1
        s_sma, n_a = sma_scale(rec["objects"], pred, K)
        if n_a > 0:
            n_anch += 1
        scale_log["ground"].append(s_ground)
        scale_log["sma"].append(s_sma)

        corr = {"base": 1.0, "ground": s_ground, "sma": s_sma}
        if s_ground != 1.0 and s_sma != 1.0:
            corr["both"] = float(np.sqrt(s_ground * s_sma))
        else:
            corr["both"] = s_ground if s_ground != 1.0 else s_sma

        gt_valid = gt > 0.005
        for vname in variants:
            dp = pred * corr[vname]
            row = protocol.per_image_depth_metrics(gt, dp, gt_valid, max_depth=8.0)
            if row:
                dense_acc[vname].append(
                    {k: row[k] for k in ("d1", "arel", "silog", "medianlog")})
            for obj in rec["objects"]:
                r = box5_median(dp, obj["bbox_xyxy"], w, h)
                if r is None:
                    continue
                obj_acc[vname]["gt"].append(obj["r_vis_med"])
                obj_acc[vname]["pred"].append(r)

        n_img += 1
        if n_img % 100 == 0:
            print(f"  {n_img} imgs (floor-locked {n_floor}, anchored {n_anch})", flush=True)
        if args.limit and n_img >= args.limit:
            break

    out = {"model": model.name, "n_images": n_img,
           "n_floor_locked": n_floor, "n_anchored": n_anch,
           "scale_stats": {k: {"mean": float(np.mean(v)), "med": float(np.median(v)),
                               "p10": float(np.percentile(v, 10)),
                               "p90": float(np.percentile(v, 90))}
                           for k, v in scale_log.items()},
           "dense": {}, "object": {}}
    for vname in variants:
        out["dense"][vname] = protocol.aggregate(dense_acc[vname])
        o = obj_acc[vname]
        g, p = np.array(o["gt"]), np.array(o["pred"])
        out["object"][vname] = ocmr_metrics.core(g, p)
        out["object"][vname].update(ocmr_metrics.proximity(g, p))
    path = os.path.join(RESULTS, f"anchorlock__{model.name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== anchor/lock on {model.name} ({n_img} imgs; "
          f"floor {n_floor}, anchors {n_anch}) ===")
    for vname in variants:
        d, o = out["dense"][vname], out["object"][vname]
        print(f"[{vname:6s}] dense ARel={d.get('arel'):.4f} d1={d.get('d1'):.4f} | "
              f"objAbsRel={o['objAbsRel']:.4f} objD1={o['objD1']:.4f} "
              f"MAE@5m={o['MAE@5m']:.3f} bias_log={o['bias_log']:+.4f} "
              f"proxF1@1m={o.get('proxF1@1m')}")
    print("wrote", path)


if __name__ == "__main__":
    main()
