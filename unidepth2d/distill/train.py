"""Distill UniDepth v2 -> small student. Single-GPU AMP loop with grad accumulation.

  python -m distill.train --config config.yaml

Requires: teacher outputs (teacher.run_teacher) + splits (scripts.make_splits).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from common.io_utils import load_config
from distill.dataset import DistillDataset, MultiSourceDataset
from distill.losses import (DistillLoss, cam_eq_loss, dsh_scale_target, icd_consistency_loss,
                            scale_head_loss, trim_mask_own)
from distill.student import build_student

CSTM_FC_PX = 518.9          # b20 C2: canonical focal (NYU Kinect at 640x480)
FOV_UNCROP = {"p": 0.25, "zmin": 0.6, "zmax": 0.95}   # b20 C3b
B20_TAGS = ("dsh", "dsh_detach", "dshft", "cstm", "fov", "ownar", "trim", "rep4", "dws4", "selgt")


class EMA:
    """Exponential moving average of model weights; validated/saved weights use the EMA copy."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def state_dict_like(self, model):
        return {k: self.shadow[k].to(v.dtype) for k, v in model.state_dict().items()}


def _abs_rel(pred_m, target_m, valid):
    v = valid & (target_m > 1e-3)
    if v.sum() == 0:
        return torch.tensor(0.0)
    return ((pred_m[v] - target_m[v]).abs() / target_m[v]).mean()


def select_value(sel_metric: str, val_mixed: float, val_per: dict) -> float:
    """Checkpoint-selection scalar. 'mixed' | a source name | 'gt' (b20 selgt: mean of the
    nyu and sun val AbsRel — real-GT domains, comparable across own-data pipelines)."""
    if sel_metric == "mixed":
        return val_mixed
    if sel_metric == "gt":
        vals = [val_per[s] for s in ("nyu", "sun") if s in val_per]
        return sum(vals) / len(vals) if vals else val_mixed
    return val_per.get(sel_metric, val_mixed)


B20_K_TAGS = ("cstm", "dsh", "fov", "ownar")
# Terms that call model(...) directly (not return_parts): under dsh their output contains s_hat,
# so the pixel-space loss would reach scale_mlp / k_embed (spec: dL_pix/d scale_mlp = 0), and their
# targets would mix the s* (p_al) and s_hat scale definitions.
DSH_INCOMPATIBLE = ("bins", "anch", "scale", "scale2", "ssionly", "sced", "icd")


def check_recipe_compat(recipes, cfg: dict) -> None:
    """Assert the b20 recipe-tag combinations are valid (b10 tag sets are never affected)."""
    if "dsh" in recipes:
        bad = [t for t in DSH_INCOMPATIBLE if t in recipes]
        assert not bad, (f"dsh is incompatible with {bad} (scale_bias / bins / affine must be off; "
                         "sced / icd would back-propagate pixel losses into the scale head)")
    if "dsh_detach" in recipes or "dshft" in recipes:
        assert "dsh" in recipes, "dsh_detach / dshft require dsh"
    if "fov" in recipes:
        assert "cstm" in recipes and "ray" in recipes, "fov requires cstm and ray"
    if any(t in recipes for t in B20_K_TAGS):
        # the dataset emits the 'intrinsics' key only with student.use_intrinsics; without it the
        # model silently uses _DEFAULT_K (constant c / e(k), fov/ownar K updates dropped)
        assert cfg["student"].get("use_intrinsics"), \
            f"b20 tags {[t for t in B20_K_TAGS if t in recipes]} need --use-intrinsics (K must reach the model)"


@torch.no_grad()
def validate(model, loader, device, log_min, log_max, max_batches: int = 0):
    """Returns (mixed AbsRel, {source: AbsRel}) so checkpoint selection can target a domain.
    max_batches > 0 caps the number of val batches (smoke tests only)."""
    model.eval()
    sums, counts = {}, {}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img = batch["image"].to(device)
        target_m = batch["depth_m"].to(device)
        valid = batch["valid"].to(device)
        intr = batch["intrinsics"].to(device) if "intrinsics" in batch else None
        pred_log = model(img, intr)[:, :1].clamp(log_min, log_max)
        pred_m = torch.exp(pred_log)
        srcs = batch.get("source", ["all"] * img.shape[0])
        for k in range(img.shape[0]):
            a = float(_abs_rel(pred_m[k:k + 1], target_m[k:k + 1], valid[k:k + 1]))
            s = srcs[k]
            sums[s] = sums.get(s, 0.0) + a
            counts[s] = counts.get(s, 0) + 1
    per = {s: round(sums[s] / counts[s], 4) for s in sorted(sums)}
    mixed = sum(sums.values()) / max(1, sum(counts.values()))
    return mixed, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=0, help="override epochs (0 = use config)")
    ap.add_argument("--use-intrinsics", action="store_true", help="FiLM-condition on camera intrinsics")
    ap.add_argument("--ckpt-subdir", default="", help="subdir under checkpoints_dir (isolate experiments)")
    ap.add_argument("--data", default="legacy", choices=["legacy", "gt"],
                    help="legacy = own-only splits.json; gt = multi-source data_gt manifest")
    ap.add_argument("--backbone", default="", help="override student backbone (timm name)")
    ap.add_argument("--recipe", default="", help="comma list of technique tags: s4,s5,s6,s7,s8")
    ap.add_argument("--name", default="", help="registry name for this model (e.g. S4-edge)")
    ap.add_argument("--init-from", default="", help="warm-start student weights from this ckpt (strict=False)")
    ap.add_argument("--lr", type=float, default=0.0, help="override LR (0 = use config; use lower for fine-tune)")
    ap.add_argument("--batch-size", type=int, default=0, help="override batch size (0 = use config)")
    ap.add_argument("--grad-accum", type=int, default=0, help="override grad accumulation (0 = use config)")
    ap.add_argument("--mix", default="", help="override data_gt.mix, e.g. 'own:0.5,nyu:0.5,sun:0' "
                                              "(sun:0 = zero-shot-clean training, no SUN)")
    ap.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases logging")
    ap.add_argument("--wandb-project", default="UnidepthV2+", help="W&B project name")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="smoke test: end each epoch's loop after N iterations (0 = off); the "
                         "epoch is still validated/checkpointed. Combine with --epochs 2 so the "
                         "trim epoch (>=1) is reached")
    ap.add_argument("--max-val-batches", type=int, default=0,
                    help="smoke test: cap validation at N batches (0 = full val split)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["distill"]
    if args.epochs > 0:
        dcfg["epochs"] = args.epochs
    if args.lr > 0:
        dcfg["lr"] = args.lr
    if args.batch_size > 0:
        dcfg["batch_size"] = args.batch_size
    if args.grad_accum > 0:
        dcfg["grad_accum"] = args.grad_accum
    if args.use_intrinsics:
        cfg["student"]["use_intrinsics"] = True
    if args.backbone:
        cfg["student"]["backbone"] = args.backbone
    if args.ckpt_subdir:
        dcfg["checkpoints_dir"] = os.path.join(dcfg["checkpoints_dir"], args.ckpt_subdir)
    if args.mix:
        mix = {}
        for part in args.mix.split(","):
            k, v = part.split(":")
            mix[k.strip()] = float(v)
        mix = {k: v for k, v in mix.items() if v > 0}
        cfg["data_gt"]["mix"] = mix
        print(f"data mix override: {mix}")
        # zero-shot-clean guard: with sun excluded, assert no SUN rows can be sampled
        # (sampler_weights gives weight 0 to sources missing from the mix; belt-and-braces)
        if "sun" not in mix:
            print("zero-shot-clean mode: SUN weight 0 -> no SUN rows in training")

    # ── recipe tags → config overrides (each individually ablatable) ──
    recipes = [t.strip() for t in args.recipe.split(",") if t.strip()]
    nll_warmup_epochs = 0
    for t in recipes:
        if t == "s4":                                   # edge-guided SSI aux
            dcfg["loss"]["eg_ssi_weight"] = 0.3
        elif t == "s5":                                 # cross-context crop distillation
            dcfg["crop_context"] = {"p": 0.5, "min_scale": 0.5}
        elif t == "s6":                                 # uncertainty head + rank-norm conf
            cfg["student"]["uncertainty_head"] = True
            dcfg["loss"]["nll_weight"] = 0.5
            dcfg["conf_mode"] = "rank"
            nll_warmup_epochs = 2
        elif t == "s7":                                 # task-weighted distillation (novelty)
            dcfg["task_weight_maps"] = True
            dcfg["loss"]["obj_dist_weight"] = 0.3
        elif t == "s8":                                 # assistant-agreement multi-teacher
            dcfg["assistant"] = True
        elif t == "ray":                                # dense ray-map input conditioning
            cfg["student"]["ray_map"] = True
        elif t == "bins":                               # hybrid log-bin head
            cfg["student"]["bins"] = 64
            dcfg["loss"]["bins_weight"] = 1.0
        elif t == "norm":                               # normal-consistency loss
            dcfg["loss"]["normal_weight"] = 0.3
        elif t == "range":                              # range-band reweighting (range-first)
            dcfg["range_weight_alpha"] = 1.0
        elif t == "anch":                               # N4 AnchDistill: learned per-image affine head
            cfg["student"]["affine_head"] = True
        elif t == "icd":                                # N5 Invariance-Consistency Distillation
            dcfg["icd_weight"] = float(dcfg.get("icd_weight", 0.3))
        elif t == "prox":                               # proximity-weighted SILog (safety band <2m up-weighted)
            dcfg["loss"]["prox_silog_weight"] = float(dcfg["loss"].get("prox_silog_weight", 0.5))
        elif t == "anchloss":                           # anchor-consistency loss (A1-as-loss, known-size geometry)
            dcfg["loss"]["anchor_weight"] = float(dcfg["loss"].get("anchor_weight", 0.3))
        elif t == "scale":                              # S19: per-image scale-bias loss on real-GT rows
            dcfg["loss"]["scale_bias_weight"] = float(dcfg["loss"].get("scale_bias_weight", 0.3))
        elif t == "scale2":                             # S21: scale-bias toward target on ALL rows (own = teacher scale)
            dcfg["loss"]["scale_bias_weight"] = float(dcfg["loss"].get("scale_bias_weight", 1.0))
            dcfg["loss"]["scale_bias_all"] = True
        elif t == "ssionly":                            # P1: SSI-only distillation (Distill-Any-Depth style
            dcfg["loss"]["silog_weight"] = 0.0          # baseline — discards metric scale by construction)
            dcfg["loss"]["ssi_weight"] = 1.0
        elif t == "csd":                                # P3: consensus-scale-refined pseudo-labels
            dcfg["csd"] = True                          # (needs scripts/build_consensus_scale.py output)
        elif t == "rel":                                # P4: teacher scale-reliability weighting of the scale loss
            dcfg["loss"]["scale_rel"] = True
        elif t == "sced":                               # S19: camera-equivariant crop consistency (virtual camera)
            dcfg["sced_weight"] = float(dcfg.get("sced_weight", 0.5))
            dcfg["sced_p"] = float(dcfg.get("sced_p", 0.5))
        # ── b20 "CanonScale-Rep" tags (docs/B20_SPEC.md section 2) ──
        elif t == "dsh":                                # C1 decoupled global scale head
            cfg["student"]["scale_head"] = True
            dcfg["loss"]["scale_head_weight"] = float(dcfg["loss"].get("scale_head_weight", 1.0))
            dcfg["loss"]["scale_head_beta"] = float(dcfg["loss"].get("scale_head_beta", 0.05))
        elif t == "dsh_detach":                         # C1 ablation: L_s does not reach the trunk
            cfg["student"]["scale_head_detach"] = True
        elif t == "dshft":                              # C1 fallback: retrain the scale head only
            dcfg["dshft"] = True
        elif t == "cstm":                               # C2 canonical-focal depth space
            cfg["student"]["cstm_fc"] = CSTM_FC_PX
        elif t == "fov":                                # C3b zoom-out uncrop on top of s5
            dcfg["fov_uncrop"] = dict(FOV_UNCROP)
        elif t == "ownar":                              # C3a aspect-true own rows (v2 cache)
            dcfg["own_source"] = "v2cache"
            dcfg["v2cache_manifest"] = os.path.join(
                cfg["output"]["root"], "v2", "cache", "own", "manifest.csv").replace("\\", "/")
        elif t == "trim":                               # C4 top-10% residual trim, own rows only
            dcfg["loss"]["trim_frac_own"] = 0.10
            dcfg["loss"]["trim_start_epoch"] = 1
        elif t == "dws4":                               # C5' plain depthwise-separable stride-4 blocks
            cfg["student"]["dws4"] = True
        elif t == "rep4":                               # C5 re-parameterised stride-4 blocks
            cfg["student"]["rep4"] = True
        elif t == "selgt":                              # select checkpoints on mean(nyu, sun) val
            dcfg["select_metric"] = "gt"
        else:
            raise SystemExit(f"unknown recipe tag: {t}")
    check_recipe_compat(recipes, cfg)
    if recipes:
        print(f"recipes active: {recipes}")
    dsh_on = bool(cfg["student"].get("scale_head", False))
    dsh_w = float(dcfg["loss"].get("scale_head_weight", 0.0)) if dsh_on else 0.0
    dsh_beta = float(dcfg["loss"].get("scale_head_beta", 0.05))
    trim_frac = float(dcfg["loss"].get("trim_frac_own", 0.0))
    trim_start = int(dcfg["loss"].get("trim_start_epoch", 1))
    dshft = bool(dcfg.get("dshft", False))
    device = dcfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = build_student(cfg).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device, weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # new affine head (N4) is expected-missing -> keeps its identity-init; warn on anything else
        miss = [k for k in missing if not k.startswith(("affine.", "k_embed.", "scale_mlp."))]
        print(f"warm-start from {args.init_from}: kept_new(affine/scale-head)={len(missing)-len(miss)} "
              f"other_missing={miss} unexpected={list(unexpected)}")
    if dshft:   # C1 fallback: only the scale head (scale_mlp, k_embed) trains; BN stats frozen below
        n_tr = 0
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(("scale_mlp.", "k_embed.")))
            n_tr += p.numel() if p.requires_grad else 0
        print(f"dshft: training {n_tr} scale-head params only")
    criterion = DistillLoss(cfg).to(device)

    if args.data == "gt":
        train_ds = MultiSourceDataset(cfg, "train", augment=True)
        val_ds = MultiSourceDataset(cfg, "val", augment=False)
        mix = cfg["data_gt"].get("mix", {})
        weights = train_ds.sampler_weights(mix)
        n_per_epoch = int(cfg["data_gt"].get("samples_per_epoch", 0)) or len(train_ds)
        sampler = WeightedRandomSampler(weights, num_samples=n_per_epoch, replacement=True)
        train_ld = DataLoader(train_ds, batch_size=int(dcfg["batch_size"]), sampler=sampler,
                              num_workers=int(dcfg.get("num_workers", 4)), drop_last=True, pin_memory=True)
        from collections import Counter
        print(f"multi-source: {Counter(train_ds.sources)} mix={mix}")
    else:
        train_ds = DistillDataset(cfg, "train", augment=True)
        val_ds = DistillDataset(cfg, "val", augment=False)
        train_ld = DataLoader(train_ds, batch_size=int(dcfg["batch_size"]), shuffle=True,
                              num_workers=int(dcfg.get("num_workers", 4)), drop_last=True, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=int(dcfg["batch_size"]), shuffle=False,
                        num_workers=int(dcfg.get("num_workers", 4)), pin_memory=True)
    print(f"train={len(train_ds)} val={len(val_ds)} batches/epoch={len(train_ld)}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg["lr"]),
                            weight_decay=float(dcfg["weight_decay"]))
    epochs = int(dcfg["epochs"])
    accum = int(dcfg.get("grad_accum", 1))
    icd_w = float(dcfg.get("icd_weight", 0.0))              # N5 flip-consistency weight (0 = off)
    sced_w = float(dcfg.get("sced_weight", 0.0))            # S19 camera-equivariance weight (0 = off)
    sced_p = float(dcfg.get("sced_p", 0.5))                 # fraction of steps carrying the extra view
    steps_per_epoch = max(1, len(train_ld) // accum)
    if args.max_iters:                                       # smoke test: capped epochs
        steps_per_epoch = max(1, min(len(train_ld), args.max_iters) // accum)
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(500, max(1, total_steps // 20))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=warmup_steps),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))],
        milestones=[warmup_steps],
    )
    ema = EMA(model, decay=float(dcfg.get("ema_decay", 0.999)))
    use_amp = bool(dcfg.get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Weights & Biases — credentials come from the user's machine-wide netrc (wandb login);
    # never from the repo. Fails soft: training must not die because logging can't reach W&B.
    wb = None
    if not args.no_wandb:
        try:
            import wandb
            params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            wb = wandb.init(
                project=args.wandb_project,
                name=args.name or args.ckpt_subdir or "unnamed",
                tags=recipes + ([f"mix:{k}" for k in cfg["data_gt"].get("mix", {})] if args.data == "gt" else []),
                config={"recipes": recipes, "mix": cfg["data_gt"].get("mix", {}),
                        "backbone": cfg["student"]["backbone"], "params_M": params_m,
                        "epochs": epochs, "lr": float(dcfg["lr"]),
                        "batch_size": int(dcfg["batch_size"]), "grad_accum": accum,
                        "samples_per_epoch": int(cfg["data_gt"].get("samples_per_epoch", 0)),
                        "select_metric": str(dcfg.get("select_metric", "mixed")),
                        "loss": dcfg.get("loss", {}), "init_from": args.init_from},
            )
        except Exception as e:                                   # offline, quota, etc.
            print(f"wandb disabled ({e})")
            wb = None

    ckpt_dir = dcfg["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    log_min, log_max = model.log_min, model.log_max

    best = math.inf
    for epoch in range(epochs):
        model.train()
        if dshft:   # frozen trunk: keep its BN running statistics fixed too
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        criterion.nll_active = epoch >= nll_warmup_epochs
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        running = {}
        for it, batch in enumerate(train_ld):
            if args.max_iters and it >= args.max_iters:
                break
            img = batch["image"].to(device, non_blocking=True)
            target_log = batch["log_depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True) if "weight" in batch else None
            intr = batch["intrinsics"].to(device, non_blocking=True) if "intrinsics" in batch else None
            boxes = batch["boxes"].to(device, non_blocking=True) if "boxes" in batch else None
            box_dists = batch["box_dists"].to(device, non_blocking=True) if "box_dists" in batch else None
            box_counts = batch["box_count"].to(device, non_blocking=True) if "box_count" in batch else None
            box_sizes = batch["box_sizes"].to(device, non_blocking=True) if "box_sizes" in batch else None
            is_gt = batch["is_gt"].to(device, non_blocking=True) if "is_gt" in batch else None
            scale_sigma = batch["scale_sigma"].to(device, non_blocking=True) if "scale_sigma" in batch else None
            pad = batch["pad"].to(device, non_blocking=True).float() if "pad" in batch else None
            with torch.amp.autocast("cuda", enabled=use_amp):
                if dsh_on:
                    # C1: decoupled scale. s* = stopgrad median(logD - U(r_c) - c) on valid pixels
                    # (fp32); every pixel loss sees p_al = U(r_c) + c + s*, so a scale error never
                    # sends gradient into the structure branch; s_hat learns s* via L_s only.
                    parts = model(img, intr, return_parts=True)
                    s_star, s_ok = dsh_scale_target(parts["r_up"], parts["c"], target_log, valid,
                                                    fallback=parts["s_hat"])
                    shift = (parts["c"].float() + s_star).to(parts["r_up"].dtype)
                    p_al = parts["r_up"] + shift[:, :, None, None]
                    out = torch.cat([p_al, parts["extra"]], dim=1) if parts["extra"] is not None else p_al
                else:
                    out = model(img, intr)
                # output layout: [mu | logsigma iff uncertainty_head | bin logits iff bins>0]
                pred_log = out[:, :1]
                # C4 trim: own (is_gt=0) rows, from trim_start epoch; s* above is pre-trim
                valid_loss = valid
                if trim_frac > 0 and epoch >= trim_start:
                    valid_loss = trim_mask_own(pred_log.detach(), target_log, valid, is_gt, trim_frac)
                has_sig = getattr(model, "uncertainty_head", False)
                pred_logsigma = out[:, 1:2] if has_sig else None
                n_bins = getattr(model, "bins", 0)
                bin_logits = out[:, (2 if has_sig else 1):] if n_bins > 0 else None
                bin_centres = getattr(model, "bin_centres_log", None) if n_bins > 0 else None
                loss, terms = criterion(pred_log, target_log, valid_loss, conf, weight=weight,
                                        pred_logsigma=pred_logsigma, boxes=boxes,
                                        box_dists=box_dists, box_counts=box_counts,
                                        bin_logits=bin_logits, bin_centres=bin_centres,
                                        intrinsics=intr, box_sizes=box_sizes, is_gt=is_gt,
                                        scale_sigma=scale_sigma, pad=pad)
                if dsh_on:
                    l_s = scale_head_loss(parts["s_hat"], s_star, s_ok, dsh_beta)
                    loss = loss + dsh_w * l_s
                    terms["scale_head"] = l_s.detach()
                    terms["total"] = loss.detach()
                    okf = s_ok.float()
                    terms["s_hat_err"] = (((parts["s_hat"].float().view(-1) - s_star.view(-1)).abs()
                                           * okf).sum() / okf.sum().clamp(min=1.0)).detach()
                if valid_loss is not valid:
                    own = (is_gt.view(-1, 1, 1, 1) == 0) if is_gt is not None else torch.ones_like(valid)
                    n_v = (valid & own).sum().clamp(min=1)
                    terms["trim_frac"] = ((valid & own).sum() - (valid_loss & own).sum()).float() / n_v
                if pad is not None:
                    terms["pad_frac"] = pad.mean().detach()
                # S19 SCED: a crop-and-resize is a VIRTUAL CAMERA (f'=f/s, shifted
                # principal point) under which true metric depth is unchanged. Pull the
                # crop-view prediction toward the detached full-view prediction at
                # corresponding pixels: only the camera-conditioning pathway can satisfy
                # this, so the per-image scale latent becomes K-causal and view-stable.
                # Needs intrinsics (without K the constraint is unsatisfiable in principle).
                if sced_w > 0 and intr is not None and torch.rand(()) < sced_p:
                    B, _, H, W = img.shape
                    s = float(torch.empty(()).uniform_(0.55, 0.85))
                    ch, cw = max(64, int(H * s)), max(64, int(W * s))
                    oy = int(torch.randint(0, H - ch + 1, ()))
                    ox = int(torch.randint(0, W - cw + 1, ()))
                    img_c = torch.nn.functional.interpolate(
                        img[:, :, oy:oy + ch, ox:ox + cw], size=(H, W),
                        mode="bilinear", align_corners=False)
                    sx, sy = cw / W, ch / H
                    intr_c = intr.clone()
                    intr_c[:, 0] = intr[:, 0] / sx                       # fx/W
                    intr_c[:, 1] = intr[:, 1] / sy                       # fy/H
                    intr_c[:, 2] = (intr[:, 2] - ox / W) / sx            # cx/W
                    intr_c[:, 3] = (intr[:, 3] - oy / H) / sy            # cy/H
                    mu_crop = model(img_c, intr_c)[:, :1]
                    with torch.no_grad():
                        tgt_c = torch.nn.functional.interpolate(
                            pred_log.detach()[:, :, oy:oy + ch, ox:ox + cw],
                            size=(H, W), mode="bilinear", align_corners=False)
                    l_sced = cam_eq_loss(mu_crop, tgt_c)
                    loss = loss + sced_w * l_sced
                    terms["sced"] = l_sced.detach()
                # N5 ICD: pull the prediction toward its flip-consistent view (invariance distillation).
                # The reference view is computed detached (mean-teacher style) so this adds one forward
                # pass but no second backward graph — cheap and memory-safe.
                if icd_w > 0:
                    with torch.no_grad():
                        intr_f = None
                        if intr is not None:
                            intr_f = intr.clone()
                            intr_f[:, 2] = 1.0 - intr_f[:, 2]     # flip mirrors the principal point
                        mu_ref = torch.flip(model(torch.flip(img, dims=[3]), intr_f)[:, :1], dims=[3])
                    l_icd = icd_consistency_loss(pred_log, mu_ref, valid)
                    loss = loss + icd_w * l_icd
                    terms["icd"] = l_icd.detach()
                loss = loss / accum
            scaler.scale(loss).backward()
            if (it + 1) % accum == 0:
                # unscale -> clip -> step. Gradient clipping caps the finite-but-huge updates that
                # precede an fp16 overflow; GradScaler.step still skips any inf/nan-grad step.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)
            for k, v in terms.items():
                running[k] = running.get(k, 0.0) + float(v)
            if (it + 1) % (accum * 50) == 0 or (args.max_iters and (it + 1) % accum == 0):
                avg = {k: round(v / (it + 1), 4) for k, v in running.items()}
                last = {k: round(float(v), 4) for k, v in terms.items()} if args.max_iters else {}
                print(f"  e{epoch} it{it+1}/{len(train_ld)} {avg} lr={sched.get_last_lr()[0]:.2e}"
                      + (f" last={last}" if last else ""), flush=True)

        # validate + checkpoint the EMA weights (they eval/deploy better than the raw ones)
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict_like(model))
        val_mixed, val_per = validate(model, val_ld, device, log_min, log_max,
                                      max_batches=args.max_val_batches)
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(raw_state)
        sel_metric = str(dcfg.get("select_metric", "mixed"))
        val_absrel = select_value(sel_metric, val_mixed, val_per)
        dt = time.time() - t0
        print(f"[epoch {epoch}] val AbsRel={val_mixed:.4f} (EMA) per-source={val_per} "
              f"select[{sel_metric}]={val_absrel:.4f}  ({dt:.0f}s)")
        if wb is not None:
            log = {"epoch": epoch, "val/absrel_mixed": val_mixed,
                   f"val/absrel_select_{sel_metric}": val_absrel,
                   "epoch_seconds": dt, "lr": sched.get_last_lr()[0]}
            log.update({f"val/absrel_{s}": v for s, v in val_per.items()})
            log.update({f"train/{k}": v / max(1, len(train_ld)) for k, v in running.items()})
            wb.log(log, step=epoch)

        torch.save({"model": ema_state, "cfg": cfg, "epoch": epoch,
                    "val_absrel": val_absrel}, os.path.join(ckpt_dir, "last.pt"))
        if val_absrel < best:
            best = val_absrel
            torch.save({"model": ema_state, "cfg": cfg, "epoch": epoch,
                        "val_absrel": val_absrel}, os.path.join(ckpt_dir, "best.pt"))
            print(f"  ** new best AbsRel={best:.4f} -> best.pt")

    print(f"Done. best val AbsRel={best:.4f}")
    with open(os.path.join(ckpt_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_absrel": best, "epochs": epochs, "recipes": recipes}, f, indent=2)
    if wb is not None:
        wb.summary["best_val_absrel"] = best
        wb.finish()

    if args.name:   # register the model for the ablation table
        reg_path = os.path.join("benchmark", "registry.json")
        reg = json.load(open(reg_path, encoding="utf-8")) if os.path.exists(reg_path) else {}
        params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        rel_ckpt = os.path.join(ckpt_dir, "best.pt").replace("\\", "/")
        # Record the TRAINING MIX, not just the recipe tags. Without this a zero-shot-clean run
        # is indistinguishable from a leaky one in the registry (S18-zsclean and S10 carry
        # identical recipe strings despite completely different training data) — and the
        # training data is exactly what decides whether a comparison is fair.
        train_mix = cfg["data_gt"].get("mix", {}) if args.data == "gt" else {"own(legacy)": 1.0}
        eval_domains = {"nyu", "sun", "kitti", "ibims", "tum"}
        reg[args.name] = {"ckpt": rel_ckpt, "backbone": cfg["student"]["backbone"],
                          "recipe": ','.join(recipes) or 'none',
                          "train_mix": train_mix,
                          "zero_shot_clean": not (set(train_mix) & eval_domains),
                          "select_metric": str(dcfg.get("select_metric", "mixed")),
                          "params_M": params_m, "best_val_absrel": best}
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
        print(f"registered {args.name} -> {rel_ckpt}")


if __name__ == "__main__":
    main()
