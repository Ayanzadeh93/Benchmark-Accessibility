"""b20 C1 dsh: decoupled global scale head (docs/B20_SPEC.md section 2, C1). CPU only."""
import math
import os

import pytest
import torch

from distill.losses import DistillLoss, dsh_scale_target, scale_head_loss
from distill.student import _cstm_term, build_student
from _orig import CKPTS, b10_student_cfg, load_orig

MID = math.log(math.sqrt(0.2 * 30.0))           # 0.8959
H, W = 128, 160                                  # small multiple of 32 for speed


def _cfg(**kw):
    c = b10_student_cfg()
    c["student"].update(kw)
    return c


def _loss_cfg():
    return {"student": {"depth_min_m": 0.2, "depth_max_m": 30.0},
            "distill": {"use_teacher_confidence": False,
                        "loss": {"silog_weight": 1.0, "silog_lambda": 0.85, "grad_weight": 0.5,
                                 "grad_scales": 4, "eg_ssi_weight": 0.3, "prox_silog_weight": 0.5}}}


def _k(B):
    return torch.tensor([[0.81, 1.08, 0.51, 0.53]] * B)


def test_init_output_is_constant():
    torch.manual_seed(0)
    for cstm in (0.0, 518.9):
        m = build_student(_cfg(scale_head=True, cstm_fc=cstm)).eval()
        x, k = torch.randn(2, 3, H, W), _k(2)
        with torch.no_grad():
            out = m(x, k)
        c = _cstm_term(k, H, W, cstm) if cstm > 0 else torch.zeros(2, 1)
        assert out.shape == (2, 1, H, W)
        for b in range(2):
            assert float(out[b].max() - out[b].min()) == 0.0
            assert abs(float(out[b, 0, 0, 0]) - (MID + float(c[b]))) < 1e-5


def test_s_star_recovers_offset():
    g = torch.Generator().manual_seed(1)
    B = 3
    r = torch.randn(B, 1, 96, 128, generator=g)
    r = r - r.mean(dim=(2, 3), keepdim=True)
    c = torch.tensor([[0.0], [0.2], [-0.1]])
    tgt = r + c.view(B, 1, 1, 1) + 0.7 + 0.01 * torch.randn(B, 1, 96, 128, generator=g)
    valid = torch.rand(B, 1, 96, 128, generator=g) > 0.3
    tgt = torch.where(valid, tgt, torch.zeros_like(tgt))     # invalid pixels hold 0 like the dataset
    s, ok = dsh_scale_target(r, c, tgt, valid)
    assert ok.all()
    assert torch.allclose(s, torch.full((B, 1), 0.7), atol=1e-3)
    # too few valid pixels -> weight 0 and the fallback value
    v2 = torch.zeros_like(valid)
    v2[:, :, :10, :10] = True                               # 100 < 500
    s2, ok2 = dsh_scale_target(r, c, tgt, v2, fallback=torch.full((B, 1), 9.0))
    assert not ok2.any() and torch.all(s2 == 9.0)
    assert float(scale_head_loss(torch.zeros(B, 1), s2, ok2)) == 0.0


def _decoupled_losses(m, x, k, tgt, valid):
    parts = m(x, k, return_parts=True)
    s_star, ok = dsh_scale_target(parts["r_up"], parts["c"], tgt, valid, fallback=parts["s_hat"])
    p_al = parts["r_up"] + (parts["c"] + s_star)[:, :, None, None]
    l_pix, _ = DistillLoss(_loss_cfg())(p_al, tgt, valid, None, weight=torch.ones_like(tgt))
    l_s = scale_head_loss(parts["s_hat"], s_star, ok, 0.05)
    return l_pix, l_s


def _grads(m):
    return {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.named_parameters()}


@pytest.mark.parametrize("detach", [False, True])
def test_gradient_decoupling(detach):
    torch.manual_seed(2)
    m = build_student(_cfg(scale_head=True, cstm_fc=518.9, scale_head_detach=detach)).train()
    with torch.no_grad():                       # move off the degenerate init so all paths are live
        m.head[-1].weight.normal_(0, 0.1)
        m.scale_mlp[-1].weight.normal_(0, 0.1)
    x, k = torch.randn(2, 3, H, W), _k(2)
    tgt = 0.5 + 0.3 * torch.randn(2, 1, H, W)
    valid = torch.ones(2, 1, H, W, dtype=torch.bool)

    m.zero_grad(set_to_none=True)
    l_pix, l_s = _decoupled_losses(m, x, k, tgt, valid)
    l_s.backward()
    g = _grads(m)
    assert g["head.1.weight"] is None or float(g["head.1.weight"].abs().max()) == 0.0
    assert g["scale_mlp.2.weight"] is not None and float(g["scale_mlp.2.weight"].abs().max()) > 0
    enc = [v for n, v in g.items() if n.startswith("encoder.") and v is not None and float(v.abs().max()) > 0]
    assert (len(enc) == 0) if detach else (len(enc) > 0)    # L_s reaches the trunk unless dsh_detach

    m.zero_grad(set_to_none=True)
    l_pix, l_s = _decoupled_losses(m, x, k, tgt, valid)
    l_pix.backward()
    g = _grads(m)
    for n, v in g.items():
        if n.startswith(("scale_mlp.", "k_embed.")):
            assert v is None or float(v.abs().max()) == 0.0, n
    assert g["head.1.weight"] is not None and float(g["head.1.weight"].abs().max()) > 0


def test_dsh_incompatible_heads():
    with pytest.raises(ValueError):
        build_student(_cfg(scale_head=True, bins=64))
    m = build_student(_cfg())
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 64, 64), _k(1), return_parts=True)


@pytest.mark.parametrize("name", ["S16_prox", "b1_mix"])
def test_old_checkpoint_loads_identically(name):
    p = os.path.join(CKPTS, name, "best.pt")
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    orig = load_orig("distill/student.py", "orig_b10_student")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["student"]["pretrained_backbone"] = False
    new_m = build_student(cfg).eval()
    old_m = orig.build_student(cfg).eval()
    new_m.load_state_dict(ck["model"], strict=True)          # no dsh keys -> strict load works
    old_m.load_state_dict(ck["model"], strict=True)
    torch.manual_seed(3)
    x = torch.randn(1, 3, 480, 640)
    k = _k(1)
    with torch.no_grad():
        assert torch.equal(new_m(x, k), old_m(x, k))


def test_scale_head_stays_float_under_w8():
    from edge.quant_sim import quantize_weights_
    torch.manual_seed(4)
    m = build_student(_cfg(scale_head=True, cstm_fc=518.9))
    with torch.no_grad():
        m.scale_mlp[-1].weight.normal_(0, 0.1)
    keep = {n: p.detach().clone() for n, p in m.named_parameters() if n.startswith(("scale_mlp.", "k_embed."))}
    film_w = m.film[0].weight.detach().clone()
    st = quantize_weights_(m, bits=8)
    for n, p in m.named_parameters():
        if n in keep:
            assert torch.equal(p, keep[n]), n
    assert not torch.equal(m.film[0].weight, film_w)          # everything else is still quantised
    assert st["params_float"] >= sum(v.numel() for v in keep.values())


# ── full train.py dsh + trim step on rep4 + dsh + cstm (folded from reviewer scratch) ─────────
def _step_batch():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 3, H, W, generator=g)
    k = torch.tensor([[0.81, 1.08, 0.51, 0.53], [0.6, 0.8, 0.45, 0.5],
                      [1.2, 1.6, 0.55, 0.48], [0.81, 1.08, 0.49, 0.52]])
    tgt = 1.0 + 0.4 * torch.randn(4, 1, H, W, generator=g)
    valid = torch.rand(4, 1, H, W, generator=g) > 0.25
    pad = torch.zeros(4, 1, H, W)
    pad[1, :, :, :30] = 1
    valid &= pad == 0
    tgt = torch.where(valid, tgt, torch.zeros_like(tgt))
    valid[3] = False
    valid[3, :, :10, :20] = True                     # 200 < 500 valid -> ok=False
    return x, k, tgt, valid, pad, torch.tensor([0, 0, 1, 0])


def _full_step(m, x, k, tgt, valid, pad, is_gt):
    from distill.losses import trim_mask_own
    parts = m(x, k, return_parts=True)
    s_star, s_ok = dsh_scale_target(parts["r_up"], parts["c"], tgt, valid, fallback=parts["s_hat"])
    p_al = parts["r_up"] + (parts["c"].float() + s_star).to(parts["r_up"].dtype)[:, :, None, None]
    valid_loss = trim_mask_own(p_al.detach(), tgt, valid, is_gt, 0.10)
    l_pix, _ = DistillLoss(_loss_cfg())(p_al, tgt, valid_loss, None, weight=torch.ones_like(tgt),
                                        is_gt=is_gt, pad=pad)
    return parts, s_star, s_ok, valid_loss, l_pix, scale_head_loss(parts["s_hat"], s_star, s_ok, 0.05)


def test_full_step_isolation_and_target():
    torch.manual_seed(0)
    m = build_student(_cfg(rep4=True, scale_head=True, cstm_fc=518.9)).train()
    with torch.no_grad():
        m.head[-1].weight.normal_(0, 0.1)
        m.scale_mlp[-1].weight.normal_(0, 0.1)
    x, k, tgt, valid, pad, is_gt = _step_batch()

    m.zero_grad(set_to_none=True)
    parts, s_star, s_ok, valid_loss, l_pix, l_s = _full_step(m, x, k, tgt, valid, pad, is_gt)
    l_s.backward()                                   # L_s never reaches head.* (incl. RepDWSep)
    assert [n for n, p in m.named_parameters() if n.startswith("head.") and p.grad is not None
            and float(p.grad.abs().max()) != 0.0] == []

    m.zero_grad(set_to_none=True)
    parts, s_star, s_ok, valid_loss, l_pix, l_s = _full_step(m, x, k, tgt, valid, pad, is_gt)
    l_pix.backward()                                 # trimmed, padded L_pix never reaches the head
    assert [n for n, p in m.named_parameters() if n.startswith(("scale_mlp.", "k_embed."))
            and p.grad is not None and float(p.grad.abs().max()) != 0.0] == []
    assert float(m.head[0].dw3[0].weight.grad.abs().max()) > 0

    # s* = fp64 lower median over valid of logD - r_up - c; fallback row keeps s_hat
    assert s_ok.tolist() == [True, True, True, False]
    r_up = parts["r_up"].detach().double()
    c = _cstm_term(k.double(), H, W, 518.9)
    for b in range(3):
        srt = torch.sort((tgt[b].double() - r_up[b] - c[b]).flatten()[valid[b].flatten()]).values
        assert abs(float(s_star[b]) - float(srt[(srt.numel() - 1) // 2])) < 1e-5
    assert torch.equal(s_star[3], parts["s_hat"][3].detach())
    # trim: GT row untouched, own rows lose ~10%, s* from the untrimmed mask
    assert torch.equal(valid_loss[2], valid[2])
    for b in (0, 1):
        frac = float((valid[b] & ~valid_loss[b]).sum()) / float(valid[b].sum())
        assert abs(frac - 0.10) < 0.01, frac
    s_trim, _ = dsh_scale_target(parts["r_up"], parts["c"], tgt, valid_loss)
    assert not torch.allclose(s_trim[:2], s_star[:2])


# ── recipe guards in distill/train.py ───────────────────────────────────────────────────────
B20 = ["s4", "s5", "ray", "range", "prox", "ownar", "fov", "cstm", "dsh", "trim", "rep4", "selgt"]


def _guard(recipes, use_intrinsics=True):
    from distill.train import check_recipe_compat
    check_recipe_compat(recipes, {"student": {"use_intrinsics": use_intrinsics}})


def test_recipe_guards_accept_b10_and_b20():
    _guard(["s4", "s5", "ray", "range", "prox"])                    # b10 (with --use-intrinsics)
    _guard(["s4", "s5", "ray", "range", "prox"], use_intrinsics=False)   # b10 tags never need it
    _guard(B20)
    _guard(["s4", "s5", "ray", "range", "prox", "cstm", "dsh", "selgt"])   # b20core


@pytest.mark.parametrize("extra", ["sced", "icd", "bins", "anch", "scale", "scale2", "ssionly"])
def test_recipe_guard_dsh_incompatible(extra):
    with pytest.raises(AssertionError, match="dsh is incompatible"):
        _guard(B20 + [extra])
    _guard(["s4", "s5", "ray", extra])                               # fine without dsh


@pytest.mark.parametrize("tag", ["cstm", "dsh", "fov", "ownar"])
def test_recipe_guard_needs_use_intrinsics(tag):
    tags = ["ray", "cstm", tag] if tag == "fov" else [tag]
    _guard(tags)
    with pytest.raises(AssertionError, match="--use-intrinsics"):
        _guard(tags, use_intrinsics=False)


def test_sced_icd_path_would_leak_into_scale_head():
    """Why sced / icd are excluded under dsh: they call model(...) directly, whose dsh output
    contains s_hat, so a pixel loss on it back-propagates into scale_mlp / k_embed."""
    torch.manual_seed(5)
    m = build_student(_cfg(scale_head=True, cstm_fc=518.9)).train()
    with torch.no_grad():
        m.scale_mlp[-1].weight.normal_(0, 0.1)
    out = m(torch.randn(2, 3, H, W), _k(2))[:, :1]
    out.square().mean().backward()
    assert float(m.scale_mlp[-1].weight.grad.abs().max()) > 0


# ── Core ML --w8: only the scale_mlp.0 matrix is kept in fp16 ──────────────────────────────
def test_coreml_w8_keeps_only_scale_mlp_matrix():
    from types import SimpleNamespace as NS
    from edge.export_coreml import scale_head_ops
    cfg = _cfg(rep4=True, scale_head=True, cstm_fc=518.9)
    m = build_student(cfg).eval()
    m.reparameterize()
    # fake get_weights_metadata(): every Conv / Linear weight and bias (BN folded -> conv bias)
    meta = {}
    for name, mod in m.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            meta[name + ".weight"] = NS(val=mod.weight.detach().numpy(), child_ops=[NS(name=name)])
            cout = mod.weight.shape[0]
            meta[name + ".bias"] = NS(val=torch.zeros(cout).numpy(), child_ops=[NS(name=name)])
    keep = scale_head_ops(meta)
    assert keep == ["scale_mlp.0"], keep
    # the old shape set would also have kept encoder convs with 128 / 16 output channels
    old = {(128, 496), (1, 128), (16, 4), (128,), (1,), (16,)}
    old_keep = {md.child_ops[0].name for md in meta.values() if tuple(md.val.shape) in old}
    assert any(n.startswith("encoder.") for n in old_keep)
