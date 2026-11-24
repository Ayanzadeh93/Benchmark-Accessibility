"""b20 C5 rep4: RepDWSep exact BN fusion. CPU only."""
import copy
import os

import pytest
import torch
import torch.nn as nn

from distill.rep import RepDWSep, fuse_with_parity, reparameterize_model
from distill.student import build_student
from _orig import b10_student_cfg


def _randomize_bn(m: nn.Module, g: torch.Generator):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            c = mod.num_features
            mod.running_mean.copy_(torch.randn(c, generator=g, dtype=mod.running_mean.dtype) * 0.5)
            mod.running_var.copy_(torch.rand(c, generator=g, dtype=mod.running_var.dtype) * 2 + 0.1)
            mod.weight.data.copy_(torch.randn(c, generator=g, dtype=mod.weight.dtype))
            mod.bias.data.copy_(torch.randn(c, generator=g, dtype=mod.bias.dtype) * 0.3)


def test_fusion_exact_fp64():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    blk = RepDWSep(96).double().eval()
    with torch.no_grad():
        _randomize_bn(blk, g)
    x = torch.randn(2, 96, 30, 40, dtype=torch.float64)
    with torch.no_grad():
        ref = blk(x)
        blk.reparameterize()
        got = blk(x)
    assert blk.deployed
    assert float((ref - got).abs().max()) < 1e-10


def test_param_count_analytic():
    blk = RepDWSep(96)
    blk.reparameterize()
    n = sum(p.numel() for p in blk.parameters())
    assert n == RepDWSep.deploy_param_count(96) == 96 * 9 + 96 + 96 * 96 + 96 == 10272
    # whole model: two dense ConvBNAct(96,96,3) (82944 + 192 each) -> two fused blocks
    m10 = build_student(b10_student_cfg())
    cfg = b10_student_cfg()
    cfg["student"]["rep4"] = True
    m20 = build_student(cfg)
    m20.eval()
    assert m20.reparameterize() == 2
    d = sum(p.numel() for p in m20.parameters()) - sum(p.numel() for p in m10.parameters())
    assert d == -2 * ((96 * 96 * 9 + 2 * 96) - 10272) == -145728


def test_full_model_fusion_parity():
    torch.manual_seed(1)
    cfg = b10_student_cfg()
    cfg["student"].update(rep4=True, scale_head=True, cstm_fc=518.9)
    m = build_student(cfg).eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        _randomize_bn(m.smooths[0], g)
        _randomize_bn(m.head[0], g)
        m.head[-1].weight.normal_(0, 0.1)
    x = torch.randn(1, 3, 128, 160)
    k = torch.tensor([[0.81, 1.08, 0.5, 0.53]])
    assert fuse_with_parity(m, x, k, tol=1e-4) == 2
    assert reparameterize_model(m) == 0                       # idempotent
    m10 = build_student(b10_student_cfg()).eval()
    assert fuse_with_parity(m10, x, k) == 0                   # no-op for b10


def test_onnx_ops_within_allowed_set(tmp_path):
    onnx = pytest.importorskip("onnx")
    blk = nn.Sequential(RepDWSep(96), nn.Conv2d(96, 1, 1)).eval()
    with torch.no_grad():     # non-trivial BN: identical all-zero biases would be de-duplicated
        _randomize_bn(blk, torch.Generator().manual_seed(2))   # by the exporter into an Identity
    blk[0].reparameterize()
    p = os.path.join(tmp_path, "rep.onnx")
    x = torch.randn(1, 96, 30, 40)
    torch.onnx.export(blk, (x,), p, input_names=["x"], output_names=["y"], opset_version=17,
                      dynamo=False)
    ops = {n.op_type for n in onnx.load(p).graph.node}
    allowed = {"Conv", "Relu", "Add", "Mul", "Resize", "ReduceMean", "Gemm", "Log", "Clip", "Concat"}
    assert ops <= allowed, ops


# ── fusion uses each BN's own eps; 1-3 DW branches (folded from reviewer scratch) ───────────
def test_fusion_eps_and_branches():
    for nb in (1, 2, 3):
        for eps in (1e-5, 1e-3, 0.1):
            g = torch.Generator().manual_seed(nb)
            blk = RepDWSep(16, n_branches=nb).double().eval()
            with torch.no_grad():
                for mod in blk.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        c = mod.num_features
                        mod.eps = eps
                        mod.running_mean.copy_(torch.randn(c, generator=g, dtype=torch.float64))
                        mod.running_var.copy_(torch.rand(c, generator=g, dtype=torch.float64) * 1e-3)
                        mod.weight.data.copy_(torch.randn(c, generator=g, dtype=torch.float64))
                        mod.bias.data.copy_(torch.randn(c, generator=g, dtype=torch.float64))
                    elif isinstance(mod, nn.Conv2d):
                        mod.weight.data.copy_(torch.randn(mod.weight.shape, generator=g, dtype=torch.float64))
                x = torch.randn(2, 16, 9, 11, generator=g, dtype=torch.float64)
                ref = blk(x)
                blk.reparameterize()
                got = blk(x)
            err = float((ref - got).abs().max())
            assert err < 1e-9 * max(1.0, float(ref.abs().max())), (nb, eps, err)


# ── fuse_with_parity runs with TF32 off and restores the caller's flags ─────────────────────
def test_fuse_with_parity_disables_tf32_and_restores():
    seen = []

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = RepDWSep(8)

        def forward(self, x, intr=None):
            seen.append((torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))
            return self.blk(x)

    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.allow_tf32 = True               # PyTorch's cuDNN default
        torch.backends.cuda.matmul.allow_tf32 = True
        m = Probe().eval()
        assert fuse_with_parity(m, torch.randn(1, 8, 6, 7), None) == 1
        assert len(seen) == 4 and all(f == (False, False) for f in seen)  # fp64 + device passes, no TF32
        assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == (True, True)
        # restored when the parity check fails too (tol < 0 forces the failure)
        m2 = Probe().eval()
        with pytest.raises(AssertionError):
            fuse_with_parity(m2, torch.randn(1, 8, 6, 7), None, tol=-1.0)
        assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == (True, True)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


# ── run_bench evaluates (and q:<bits> quantises) the FUSED deploy graph ─────────────────────
def _rep4_ckpt(tmp_path, rep4=True):
    cfg = b10_student_cfg()
    cfg["student"].update(rep4=rep4, scale_head=rep4, cstm_fc=518.9 if rep4 else 0.0)
    torch.manual_seed(7)
    m = build_student(cfg).eval()
    if rep4:
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            _randomize_bn(m.smooths[0], g)
            _randomize_bn(m.head[0], g)
            m.head[-1].weight.normal_(0, 0.1)
    p = os.path.join(tmp_path, "b20x" if rep4 else "b10x", "best.pt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"cfg": cfg, "model": m.state_dict()}, p)
    return p, cfg, m


def test_run_bench_fuses_and_quantises_deploy_graph(tmp_path):
    from benchmark.run_bench import QuantWeightStudent, StudentModelRunner
    from distill.rep import has_unfused_rep
    from edge.quant_sim import quantize_weights_
    p, cfg, m_unfused = _rep4_ckpt(str(tmp_path))
    r = StudentModelRunner(p, "cpu")
    assert not has_unfused_rep(r.model) and r.use_intr
    x, k = torch.randn(1, 3, 480, 640), torch.tensor([[0.81, 1.08, 0.5, 0.53]])
    with torch.no_grad():
        assert float((r.model(x, k) - m_unfused(x, k)).abs().max()) < 1e-4
    q = QuantWeightStudent("8:" + p, "cpu")
    ref = build_student(cfg).eval()
    ref.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
    ref.reparameterize()
    st = quantize_weights_(ref, bits=8)                       # deploy W8: fuse, then quantise
    assert st == q.quant_stats
    sd_q, sd_r = q.model.state_dict(), ref.state_dict()
    assert sd_q.keys() == sd_r.keys() and all(torch.equal(sd_q[n], sd_r[n]) for n in sd_q)
    assert "head.0.dw_fused.weight" in sd_q


def test_run_bench_b10_ckpt_untouched(tmp_path, monkeypatch):
    import distill.rep
    from benchmark.run_bench import StudentModelRunner
    p, cfg, m = _rep4_ckpt(str(tmp_path), rep4=False)

    def _boom(*a, **k):
        raise AssertionError("fuse_with_parity must not run for a non-rep4 checkpoint")
    monkeypatch.setattr(distill.rep, "fuse_with_parity", _boom)
    torch.manual_seed(123)
    r = StudentModelRunner(p, "cpu")
    st_new = torch.get_rng_state()
    torch.manual_seed(123)                                    # what the runner did before b20
    build_student(torch.load(p, map_location="cpu", weights_only=False)["cfg"]).eval()
    assert torch.equal(st_new, torch.get_rng_state())         # no extra RNG draw
    sd = m.state_dict()
    assert all(torch.equal(v, sd[n]) for n, v in r.model.state_dict().items())


def test_quant_sim_matches_weight_int8_on_fused_block(tmp_path):
    """q:8 (quant_sim on the fused torch convs) == edge.weight_int8 on the exported fused ONNX."""
    pytest.importorskip("onnx")
    import numpy as np
    from edge.quant_sim import quantize_weights_
    from edge.weight_int8 import dequantize_weights, quantize_weights_int8
    blk = nn.Sequential(RepDWSep(96), nn.Conv2d(96, 1, 1)).eval()
    with torch.no_grad():
        _randomize_bn(blk, torch.Generator().manual_seed(3))
    blk[0].reparameterize()
    p = os.path.join(tmp_path, "rep.onnx")
    torch.onnx.export(blk, (torch.randn(1, 96, 30, 40),), p, input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)
    w8, _ = quantize_weights_int8(p)
    dq = dequantize_weights(w8)
    sim = copy.deepcopy(blk)
    quantize_weights_(sim, bits=8)
    convs = {n: mod.weight.detach().numpy() for n, mod in sim.named_modules() if isinstance(mod, nn.Conv2d)}
    assert set(convs) == {"0.dw_fused", "0.pw_fused", "1"}
    for name, w in convs.items():
        got = dq[name + ".weight"]
        assert got.shape == w.shape
        assert np.allclose(got, w, rtol=0, atol=1e-6 * float(np.abs(w).max())), name
