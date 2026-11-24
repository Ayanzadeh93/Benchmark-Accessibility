"""b20 C4 trim: scale-aligned top-10% residual trim on own rows only. CPU only."""
import pytest
import torch

from distill.losses import trim_mask_own


def _batch(seed=0, B=4, H=120, W=160):
    g = torch.Generator().manual_seed(seed)
    tgt = 0.3 + 0.5 * torch.rand(B, 1, H, W, generator=g)
    pred = tgt + 0.2 + 0.05 * torch.randn(B, 1, H, W, generator=g)
    pred = pred + (torch.rand(B, 1, H, W, generator=g) < 0.05).float() * 1.5 * torch.randn(B, 1, H, W, generator=g)
    valid = torch.rand(B, 1, H, W, generator=g) > 0.2
    is_gt = torch.tensor([0, 1, 0, 1])
    return pred, tgt, valid, is_gt


def test_trims_ten_percent_of_own_and_never_gt():
    pred, tgt, valid, is_gt = _batch()
    new = trim_mask_own(pred, tgt, valid, is_gt, frac=0.10)
    assert new.dtype == torch.bool and new.shape == valid.shape
    assert not (new & ~valid).any()                        # only ever removes pixels
    for b in range(4):
        if int(is_gt[b]) == 1:
            assert torch.equal(new[b], valid[b])
        else:
            removed = float((valid[b] & ~new[b]).sum()) / float(valid[b].sum())
            assert abs(removed - 0.10) <= 0.005, removed


def test_global_shift_invariance():
    pred, tgt, valid, is_gt = _batch(seed=1)
    a = trim_mask_own(pred, tgt, valid, is_gt, frac=0.10)
    b = trim_mask_own(pred + 1.0, tgt, valid, is_gt, frac=0.10)
    assert torch.equal(a, b)


def test_off_and_no_gt_info():
    pred, tgt, valid, _ = _batch(seed=2)
    assert trim_mask_own(pred, tgt, valid, None, frac=0.0) is valid
    new = trim_mask_own(pred, tgt, valid, None, frac=0.10)   # is_gt unknown -> treated as own
    removed = float((valid & ~new).sum()) / float(valid.sum())
    assert abs(removed - 0.10) <= 0.005


def test_selgt_select_value():
    from distill.train import select_value
    per = {"own": 0.20, "nyu": 0.10, "sun": 0.14}
    assert abs(select_value("gt", 0.3, per) - 0.12) < 1e-12           # mean(nyu, sun)
    assert select_value("own", 0.3, per) == 0.20                       # legacy paths unchanged
    assert select_value("mixed", 0.3, per) == 0.3
    assert select_value("gt", 0.3, {"own": 0.2}) == 0.3                # no GT rows -> mixed


# ── NaN / degenerate-batch robustness of the dsh + trim step (folded from reviewer scratch) ──
from distill.losses import DistillLoss, dsh_scale_target, scale_head_loss  # noqa: E402


def _b20_model(seed=0):
    from _orig import b10_student_cfg
    from distill.student import build_student
    cfg = b10_student_cfg()
    cfg["student"].update(scale_head=True, cstm_fc=518.9, rep4=True)
    cfg["distill"] = {"use_teacher_confidence": False,
                      "loss": {"silog_weight": 1.0, "silog_lambda": 0.85, "grad_weight": 0.5,
                               "grad_scales": 4, "eg_ssi_weight": 0.3, "prox_silog_weight": 0.5,
                               "range_weight_alpha": 1.0}}
    torch.manual_seed(seed)
    return build_student(cfg), cfg


def _train_step(model, crit, img, tgt, valid, conf, weight, intr, is_gt, pad, trim_frac=0.1,
                autocast_dtype=None):
    """Mirror of the dsh+trim branch of distill/train.py's step."""
    ctx = (torch.autocast("cpu", dtype=autocast_dtype) if autocast_dtype is not None
           else torch.autocast("cpu", enabled=False))
    with ctx:
        parts = model(img, intr, return_parts=True)
        s_star, s_ok = dsh_scale_target(parts["r_up"], parts["c"], tgt, valid, fallback=parts["s_hat"])
        shift = (parts["c"].float() + s_star).to(parts["r_up"].dtype)
        p_al = parts["r_up"] + shift[:, :, None, None]
        pred_log = p_al[:, :1]
        valid_loss = trim_mask_own(pred_log.detach(), tgt, valid, is_gt, trim_frac)
        loss, terms = crit(pred_log, tgt, valid_loss, conf, weight=weight, intrinsics=intr,
                           is_gt=is_gt, pad=pad)
        l_s = scale_head_loss(parts["s_hat"], s_star, s_ok, 0.05)
        loss = loss + l_s
    model.zero_grad(set_to_none=True)
    with torch.backends.mkldnn.flags(enabled=autocast_dtype is None):
        loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    return loss, terms, l_s, s_star, s_ok, bad


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16])
def test_b20_step_no_nan_degenerate(dtype):
    model, cfg = _b20_model()
    model.train()
    crit = DistillLoss(cfg)
    B, H, W = 4, 480, 640
    torch.manual_seed(3)
    img = torch.randn(B, 3, H, W)
    tgt = torch.zeros(B, 1, H, W)
    valid = torch.zeros(B, 1, H, W, dtype=torch.bool)
    pad = torch.zeros(B, 1, H, W)
    # row 0: all pad, no valid pixel.  row 1: < 500 valid.  row 2: half pad.  row 3: normal GT row.
    pad[0] = 1
    valid[1, :, 10:20, 10:30] = True
    pad[2, :, :, : W // 2] = 1
    valid[2, :, :, W // 2:] = True
    valid[3] = True
    tgt[valid] = torch.log(torch.empty(int(valid.sum())).uniform_(0.3, 12.0))
    conf = torch.ones(B, 1, H, W)
    weight = torch.ones(B, 1, H, W)
    intr = torch.tensor([[0.81, 1.08, 0.5, 0.53], [0.5, 0.66, 0.5, 0.5],
                         [1.4, 1.9, 0.3, 0.6], [0.81, 1.08, 0.5, 0.53]])
    is_gt = torch.tensor([0, 0, 0, 1])
    loss, terms, l_s, s_star, s_ok, bad = _train_step(model, crit, img, tgt, valid, conf, weight,
                                                      intr, is_gt, pad, autocast_dtype=dtype)
    assert torch.isfinite(loss) and torch.isfinite(l_s)
    assert torch.isfinite(s_star).all()
    assert s_ok.tolist() == [False, False, True, True]
    assert not bad


def test_b20_step_all_rows_empty():
    """Every row empty (e.g. a batch of all-pad fov crops of invalid-depth images)."""
    model, cfg = _b20_model()
    model.train()
    crit = DistillLoss(cfg)
    B, H, W = 2, 480, 640
    img = torch.randn(B, 3, H, W)
    tgt = torch.zeros(B, 1, H, W)
    valid = torch.zeros(B, 1, H, W, dtype=torch.bool)
    pad = torch.ones(B, 1, H, W)
    intr = torch.tensor([[0.81, 1.08, 0.5, 0.53]] * B)
    loss, terms, l_s, s_star, s_ok, bad = _train_step(model, crit, img, tgt, valid, torch.ones_like(tgt),
                                                      torch.ones_like(tgt), intr,
                                                      torch.tensor([0, 0]), pad)
    assert torch.isfinite(loss) and not bad


def test_dsh_target_fp16_inputs_extreme():
    B, H, W = 3, 60, 80
    r = (torch.randn(B, 1, H, W) * 3).half()
    c = torch.zeros(B, 1).half()
    tgt = torch.full((B, 1, H, W), 3.4)
    v = torch.ones(B, 1, H, W, dtype=torch.bool)
    v[1] = False
    s, ok = dsh_scale_target(r, c, tgt, v, fallback=torch.full((B, 1), 65000.0).half())
    assert torch.isfinite(s).all() and s.dtype == torch.float32
    l = scale_head_loss(torch.full((B, 1), 60000.0).half(), s, ok)
    assert torch.isfinite(l)


def test_trim_degenerate():
    B, H, W = 3, 40, 40
    p = torch.randn(B, 1, H, W)
    t = torch.randn(B, 1, H, W)
    v = torch.zeros(B, 1, H, W, dtype=torch.bool)
    v[1, :, :3, :3] = True                       # 9 px -> < min_valid on the ::2 grid
    v[2] = True
    out = trim_mask_own(p.half(), t, v, torch.tensor([0, 0, 0]), 0.1)
    assert out.dtype == torch.bool and not out[0].any() and torch.equal(out[1], v[1])
    kept = float(out[2].float().mean())
    assert 0.85 < kept < 0.95, kept
    # constant residuals: tau = 0 -> nothing dropped (rho <= tau everywhere)
    out2 = trim_mask_own(t.clone(), t, v, torch.tensor([0, 0, 0]), 0.1)
    assert torch.equal(out2, v)
