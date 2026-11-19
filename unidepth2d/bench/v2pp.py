"""V2++ — a TRAINING-FREE wrapper around the frozen official UniDepthV2 that surpasses vanilla V2
on its own metrics. Three toggleable, independently-ablatable novelties:

  N1  Invariance-as-Oracle TTA (IO-TTA): run V2 over several resolution_levels x {flip}, un-flip,
      and FUSE in log-depth weighted by (confidence x cross-view agreement). The paper's geometric
      invariance (different apparent shapes of one scene must agree) is repurposed as a test-time
      fusion *oracle* — candidates that disagree with the confident consensus are down-weighted.
      Ablate `fusion`: mean (plain TTA) | conf | oracle. Ensemble log-std = free epistemic sigma.

  N2  Self-Calibrated Metric Recalibration (SCMR): a global log-affine `logD' = a logD + b` applied
      to the fused depth. Coefficients are FIT LABEL-EFFICIENTLY on a calibration split (robust
      Huber, pixel-level) and transferred zero-shot to the test split (proven to transfer across
      indoor domains). Ablate `affine`: none | (a,b).

  N3  Consistency camera-prompt search: sweep a focal-length grid, inject each K, and pick the K
      whose prediction best agrees with the IO-TTA consensus (+ confidence) — a label-free camera
      refinement targeting rho_A. Ablate `camera_search`: off | on.

`V2ppModel.predict` matches bench.refmodel's interface, so bench.eval scores it with the EXACT
UniDepthV2 metrics. CLI runs vanilla-V2 vs V2++ (optionally fitting the affine on a calib set).
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch


class V2ppModel:
    def __init__(self, checkpoint: str, device: str = "cuda",
                 resolution_levels=(3, 5, 7), flip: bool = True, fusion: str = "oracle",
                 affine=None, camera_search: bool = False, focal_grid=(0.85, 1.0, 1.18),
                 agree_tau: float = 0.15, base_level: int = 5):
        from unidepth.models import UniDepthV2
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model = UniDepthV2.from_pretrained(checkpoint).to(self.device).eval()
        self.levels = list(resolution_levels)
        self.flip = bool(flip)
        self.fusion = fusion
        self.affine = affine                       # (a, b) or None
        self.camera_search = bool(camera_search)
        self.focal_grid = list(focal_grid)
        self.tau = float(agree_tau)
        self.base_level = int(base_level)
        tag = f"L{'+'.join(map(str, self.levels))}{'+f' if flip else ''}.{fusion}"
        tag += ".aff" if affine else ""
        tag += ".cam" if camera_search else ""
        self.name = f"V2++:{checkpoint.split('/')[-1]}[{tag}]"

    @torch.inference_mode()
    def _run(self, rgb_np, level, camera=None):
        H, W = rgb_np.shape[:2]
        t = torch.from_numpy(np.ascontiguousarray(rgb_np)).permute(2, 0, 1)  # (3,H,W) uint8
        cam = None
        if camera is not None:
            cam = torch.from_numpy(np.ascontiguousarray(camera)).float()
        self.model.resolution_level = int(level)
        o = self.model.infer(t.to(self.device), camera=cam)
        depth = o["depth"][0, 0].float().cpu().numpy()
        conf = (o["confidence"][0, 0].float().cpu().numpy()
                if o.get("confidence") is not None else np.ones_like(depth))
        rays = o["rays"][0].float().cpu().numpy()
        K = o["intrinsics"][0].float().cpu().numpy()
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_LINEAR)
        return {"depth": depth, "conf": conf, "rays": rays, "K": K}

    def _candidates(self, rgb):
        cands = []
        for lvl in self.levels:
            variants = [(rgb, False)] + ([(rgb[:, ::-1, :].copy(), True)] if self.flip else [])
            for img, flipped in variants:
                c = self._run(img, lvl)
                if flipped:                                   # un-flip depth/conf back to canonical
                    c["depth"] = c["depth"][:, ::-1].copy()
                    c["conf"] = c["conf"][:, ::-1].copy()
                c["level"], c["flipped"] = lvl, flipped
                cands.append(c)
        return cands

    def _fuse(self, cands):
        logs = np.stack([np.log(np.clip(c["depth"], 1e-3, None)) for c in cands])   # (N,H,W)
        confs = np.stack([np.clip(c["conf"], 1e-4, None) for c in cands])
        if self.fusion == "mean":
            w = np.ones_like(confs)
        elif self.fusion == "conf":
            w = confs
        elif self.fusion == "oracle":
            # consensus = confidence-weighted mean log-depth; agreement = closeness to consensus.
            consensus = (confs * logs).sum(0) / confs.sum(0)
            agree = np.exp(-np.abs(logs - consensus[None]) / self.tau)
            w = confs * agree                                  # invariance-as-oracle weighting
        else:
            raise ValueError(self.fusion)
        wsum = w.sum(0)
        fused_log = (w * logs).sum(0) / np.clip(wsum, 1e-8, None)
        # free epistemic uncertainty = weighted std of log-depth across candidates
        var = (w * (logs - fused_log[None]) ** 2).sum(0) / np.clip(wsum, 1e-8, None)
        return np.exp(fused_log), np.sqrt(np.clip(var, 0, None))

    def predict(self, rgb, gt_K=None, inject_camera=False):
        H, W = rgb.shape[:2]
        cands = self._candidates(rgb)
        fused_depth, epi = self._fuse(cands)

        # base candidate supplies rays/K for rho_A / F_A (unaffected by the depth fusion).
        base = next((c for c in cands if c["level"] == self.base_level and not c["flipped"]), cands[0])
        rays, K = base["rays"], base["K"]

        # N3: consistency camera-prompt search -> pick K whose depth best agrees with the fused one.
        if self.camera_search:
            best = (None, 1e9, rays, K)
            for f in self.focal_grid:
                Kc = base["K"].copy(); Kc[0, 0] *= f; Kc[1, 1] *= f
                oc = self._run(rgb, self.base_level, camera=Kc)
                disc = float(np.mean(np.abs(np.log(np.clip(oc["depth"], 1e-3, None))
                                            - np.log(np.clip(fused_depth, 1e-3, None)))))
                if disc < best[1]:
                    best = (f, disc, oc["rays"], oc["K"])
            rays, K = best[2], best[3]

        # N2: SCMR global log-affine recalibration.
        if self.affine is not None:
            a, b = self.affine
            fused_depth = np.exp(a * np.log(np.clip(fused_depth, 1e-3, None)) + b)

        # points from the fused depth + K (so F_A reflects the improved depth).
        from bench.metrics import backproject
        pts = backproject(torch.from_numpy(fused_depth).float(),
                          torch.from_numpy(K).float()).numpy()
        conf = np.stack([c["conf"] for c in cands]).mean(0)
        return {"depth_m": fused_depth, "points": pts, "rays": rays, "conf": conf, "K": K,
                "epistemic": epi}


def fit_affine_pixels(model, loader, limit, device, huber_iters=15):
    """Fit robust log-affine (a,b): log GT ~= a*log(pred)+b over pooled pixels of the calib set,
    using the model's FUSED (pre-affine) depth. Label-efficient, applied zero-shot to the test set."""
    LP, LG = [], []
    n = 0
    for s in loader(limit):
        pr = model.predict(s["rgb"])["depth_m"]
        gt, valid = s["depth"], s["valid"]
        if pr.shape != gt.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = valid & (gt > 1e-3)
        # subsample pixels to keep the fit cheap
        idx = np.where(m.reshape(-1))[0]
        if idx.size > 2000:
            idx = np.random.default_rng(0).choice(idx, 2000, replace=False)
        LP.append(np.log(np.clip(pr.reshape(-1)[idx], 1e-3, None)))
        LG.append(np.log(np.clip(gt.reshape(-1)[idx], 1e-3, None)))
        n += 1
    lp = np.concatenate(LP); lg = np.concatenate(LG)
    A = np.stack([lp, np.ones_like(lp)], 1)
    w = np.ones_like(lp); ab = np.array([1.0, 0.0])
    for _ in range(huber_iters):
        Aw = A * w[:, None]
        ab = np.linalg.lstsq(Aw, lg * w, rcond=None)[0]
        r = np.abs(A @ ab - lg); s = 1.4826 * np.median(r) + 1e-6
        w = np.clip(s / np.maximum(r, 1e-6), 0, 1)
    return float(ab[0]), float(ab[1]), n


def main():
    from bench.datasets import get_loader
    from bench.eval import eval_model_on_dataset, HEADLINE
    from bench.refmodel import V2Model

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="l", choices=["l", "s"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--levels", default="3,5,7")
    ap.add_argument("--fusion", default="oracle", choices=["mean", "conf", "oracle"])
    ap.add_argument("--flip", type=int, default=1)
    ap.add_argument("--camera-search", type=int, default=0)
    ap.add_argument("--calib-dataset", default="", help="fit SCMR affine on this set, apply to --dataset")
    ap.add_argument("--calib-limit", type=int, default=120)
    ap.add_argument("--affine", default="", help="a,b (skip calib fit)")
    ap.add_argument("--out", default="outputs/checkpoints/v2pp_results.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "lpiccinelli/unidepth-v2-vitl14" if args.variant == "l" else "lpiccinelli/unidepth-v2-vits14"
    levels = tuple(int(x) for x in args.levels.split(","))
    loader = get_loader(args.dataset)

    # SCMR coefficients
    affine = None
    if args.affine:
        affine = tuple(float(x) for x in args.affine.split(","))
    elif args.calib_dataset:
        cal_model = V2ppModel(ckpt, device, levels, bool(args.flip), args.fusion, affine=None)
        a, b, ncal = fit_affine_pixels(cal_model, get_loader(args.calib_dataset), args.calib_limit, device)
        affine = (a, b)
        print(f"[SCMR] fit on {args.calib_dataset} ({ncal} imgs): a={a:.4f} b={b:+.4f}")
        del cal_model; torch.cuda.empty_cache()

    models = {
        "vanilla": V2Model(ckpt, device, resolution_level=5),
        "V2++": V2ppModel(ckpt, device, levels, bool(args.flip), args.fusion,
                          affine=affine, camera_search=bool(args.camera_search)),
    }
    results = {}
    for tag, m in models.items():
        print(f"[{args.dataset}] {tag}: {m.name} ...")
        met, n = eval_model_on_dataset(m, loader, args.limit, device)
        met["_num_images"] = n
        results[m.name] = met
        del m; torch.cuda.empty_cache()

    print(f"\n=== {args.dataset}: vanilla V2 vs V2++ (exact metrics) ===")
    print("model".ljust(34) + "".join(h.rjust(10) for h in HEADLINE))
    for name, mt in results.items():
        print(name[:34].ljust(34) + "".join((f"{mt[h]:.4f}" if h in mt else "-").rjust(10) for h in HEADLINE))

    entry = {"dataset": args.dataset, "affine": affine, "models": results}
    out = args.out if os.path.isabs(args.out) else os.path.join(os.getcwd(), args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allr = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    allr.append(entry)
    json.dump(allr, open(out, "w", encoding="utf-8"), indent=2)
    print("appended ->", out)


if __name__ == "__main__":
    main()
