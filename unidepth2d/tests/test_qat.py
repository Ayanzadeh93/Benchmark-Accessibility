"""Tests for distill/v2/qat.py (eager QAT). Env: conda "depth" (torch only). CPU, offline, < 60 s.

The integration tests at the end run on distill.v2.model.build_v2_model (pretrained=False, so no
download; skipped if timm is missing). The ONNX export/mapping test needs onnx and runs only in
env depth_cmp (pytest.importorskip); the name-parsing test runs everywhere.
"""
import copy
import io
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from distill.v2 import qat as Q


# ── a tiny CoDeQ-shaped net: Conv/BN/ReLU blocks, linear projections + skip Add, bilinear x2
#    Resize, PixelShuffle residual, attribute-style (conv/bn/act) and Sequential-style CBRs ──────
class Add(nn.Module):
    def forward(self, a, b):
        return a + b


def cbr(a, b, k=3, g=1):
    return nn.Sequential(nn.Conv2d(a, b, k, 1, k // 2, groups=g, bias=False), nn.BatchNorm2d(b),
                         nn.ReLU(inplace=True))


class CBA(nn.Module):
    def __init__(self, a, b, k, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(a, b, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(b)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Block(nn.Module):
    def __init__(self, c, e=2):
        super().__init__()
        self.body = nn.Sequential(CBA(c, c * e, 1), CBA(c * e, c * e, 3, g=c * e),
                                  CBA(c * e, c, 1, act=False))
        self.add = Add()

    def forward(self, x):
        return self.add(self.body(x), x)


def up2():
    return nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)


class TinyCoDeQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = CBA(3, 8, 3, s=2)
        self.s1 = nn.Sequential(CBA(8, 16, 3, s=2), Block(16))
        self.s2 = nn.Sequential(CBA(16, 24, 3, s=2), Block(24))
        self.lat1, self.lat2 = cbr(16, 16, 1), cbr(24, 16, 1)
        self.up, self.add = up2(), Add()
        self.fuse = nn.Sequential(cbr(16, 16, 3, 16), cbr(16, 16, 1))
        self.l0, self.to0, self.up0, self.add0 = cbr(8, 8, 1), cbr(16, 8, 1), up2(), Add()
        self.f0 = nn.Sequential(cbr(8, 8, 3, 8), cbr(8, 8, 1))
        self.head = nn.Sequential(cbr(8, 8, 3, 8), cbr(8, 8, 1), nn.Conv2d(8, 1, 1))
        self.upc = up2()
        self.r_conv, self.ps, self.add_out = nn.Conv2d(8, 4, 3, padding=1), nn.PixelShuffle(2), Add()
        nn.init.constant_(self.head[-1].bias, math.log(math.sqrt(0.2 * 30)))
        nn.init.zeros_(self.r_conv.weight)
        nn.init.zeros_(self.r_conv.bias)
        self.log_min, self.log_max = math.log(0.2), math.log(30.0)

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.s1(f0)
        f2 = self.s2(f1)
        y = self.fuse(self.add(self.lat1(f1), self.up(self.lat2(f2))))
        y0 = self.f0(self.add0(self.l0(f0), self.up0(self.to0(y))))
        return self.add_out(self.upc(self.head(y0)), self.ps(self.r_conv(y0)))


N_BN, N_CONV = 19, 21
N_RELU, N_ADD, N_RESIZE, N_CONV_OUT = 17, 5, 3, 4


def randomize_bn_(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.3)
            m.running_var.copy_(torch.rand(n, generator=g) * 1.5 + 0.2)
            if m.affine:
                m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
                m.bias.data.copy_(torch.randn(n, generator=g) * 0.2)
    return model


@pytest.fixture()
def float_model():
    torch.manual_seed(0)
    return randomize_bn_(TinyCoDeQ()).eval()


def x_in(seed=1, b=2, hw=(32, 48)):
    return torch.randn(b, 3, *hw, generator=torch.Generator().manual_seed(seed))


def prepared(float_model, **cfg):
    qm, rep = Q.prepare_qat(float_model, Q.QATConfig(**cfg), x_in(0, 1))
    return qm, rep


# ── BN folding ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("groups,bias,affine", [(1, False, True), (8, False, True),
                                                (1, True, True), (2, True, False)])
def test_folded_conv_equals_conv_bn_in_eval(groups, bias, affine):
    torch.manual_seed(groups + 3 * bias)
    conv = nn.Conv2d(8, 16, 3, 2, 1, groups=groups, bias=bias)
    bn = randomize_bn_(nn.Sequential(nn.BatchNorm2d(16, affine=affine)))[0].eval()
    fused = Q.fold_conv_bn(conv, bn)
    x = torch.randn(3, 8, 17, 23)
    with torch.no_grad():
        ref, got = bn(conv(x)), fused(x)
    assert fused.bias is not None
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), float((ref - got).abs().max())


def test_every_conv_bn_pair_folded(float_model):
    x = x_in()
    with torch.no_grad():
        ref = float_model(x)
    m = copy.deepcopy(float_model)
    rep = Q.fold_bn_(m, x_in(0, 1))
    assert rep["bn_total"] == N_BN and rep["bn_folded"] == N_BN
    assert rep["bn_frozen_unfolded"] == []
    assert not any(isinstance(mm, nn.BatchNorm2d) for mm in m.modules())
    assert sum(isinstance(mm, nn.Conv2d) for mm in m.modules()) == N_CONV
    assert rep["fold_verify_dtype"] == "float64" and rep["fold_verify_max_abs"] < 1e-6
    with torch.no_grad():
        assert torch.allclose(m(x), ref, atol=1e-5)


class BNAfterAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 4, 1), nn.Conv2d(3, 4, 1)
        self.add, self.bn = Add(), nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.add(self.c1(x), self.c2(x)))


class ConvTwoConsumers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv, self.bn, self.add = nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), Add()

    def forward(self, x):
        y = self.conv(x)
        return self.add(self.bn(y), y)


@pytest.mark.parametrize("cls", [BNAfterAdd, ConvTwoConsumers])
def test_unfoldable_bn_raises_strict_and_freezes_otherwise(cls):
    torch.manual_seed(0)
    model = randomize_bn_(cls()).eval()
    x = torch.randn(2, 3, 8, 8)
    with pytest.raises(RuntimeError, match="cannot be folded"):
        Q.fold_bn_(copy.deepcopy(model), x, strict=True)
    m = copy.deepcopy(model)
    rep = Q.fold_bn_(m, x, strict=False)
    assert rep["bn_frozen_unfolded"] == ["bn"]
    assert isinstance(m.bn, Q.FrozenBN)
    m.train()                                    # train mode must not matter any more
    with torch.no_grad():
        assert torch.allclose(m(x), model(x), atol=1e-5)


def test_bn_frozen_no_buffer_changes_in_train_mode(float_model):
    qm, _ = prepared(float_model)
    assert not any(isinstance(mm, nn.BatchNorm2d) for mm in qm.modules())
    qm.train()
    with torch.no_grad():
        qm(x_in(1))                              # initialises observers
    Q.freeze_observers(qm)
    before = {k: v.clone() for k, v in qm.state_dict().items() if torch.is_tensor(v)}
    with torch.no_grad():
        qm(x_in(2) * 3)
        qm(x_in(3))
    after = qm.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), k


# ── fake-quant primitives ─────────────────────────────────────────────────────────────────────
def test_weight_fake_quant_idempotent_and_matches_quant_sim():
    from edge.quant_sim import quantize_tensor
    torch.manual_seed(0)
    w = torch.randn(16, 8, 3, 3) * torch.rand(16, 1, 1, 1)
    w[3] = 0.0                                   # all-zero channel stays zero
    q1 = Q.fake_quant_weight(w)
    q2 = Q.fake_quant_weight(q1)
    assert torch.allclose(q1, q2, rtol=1e-6, atol=1e-9)
    scale = w.abs().amax(dim=(1, 2, 3)) / 127
    codes = torch.round(q1 / scale.clamp_min(1e-12).view(-1, 1, 1, 1))
    assert codes.abs().max() <= 127 and torch.equal(q1[3], torch.zeros_like(q1[3]))
    assert all(len(torch.unique(codes[c])) <= 255 for c in range(16))
    ref = quantize_tensor(w, 8, per_channel=True)
    step = scale.view(-1, 1, 1, 1)
    assert ((q1 - ref).abs() <= step * 1.0001 + 1e-12).all()
    assert (q1 == ref).float().mean() > 0.99


def test_act_fake_quant_idempotent_uint8_grid():
    fq = Q.ActFakeQuant(8).train()
    x = torch.randn(4, 3, 16, 16) * 2 + 0.5
    fq(x)                                        # observe
    fq.eval()
    y1 = fq(x)
    y2 = fq(y1)
    assert torch.equal(y1, y2)
    assert len(torch.unique(y1)) <= 256
    scale, zp = fq.qparams()
    assert 0 <= int(zp) <= 255
    assert float(fq(torch.zeros(1))) == 0.0     # zero exactly representable
    pos = Q.ActFakeQuant(8).train()
    pos(torch.relu(x))
    assert int(pos.qparams()[1]) == 0            # ReLU range starts at 0


def test_act_fake_quant_passthrough_before_init():
    fq = Q.ActFakeQuant(8).eval()
    x = torch.randn(5)
    assert torch.equal(fq(x), x) and not fq.initialized


# ── observers ─────────────────────────────────────────────────────────────────────────────────
def test_observer_updates_in_train_freezes_and_never_updates_in_eval():
    fq = Q.ActFakeQuant(8, momentum=0.5).train()
    fq(torch.tensor([-1.0, 1.0]))
    assert (float(fq.min_val), float(fq.max_val)) == (-1.0, 1.0)
    fq(torch.tensor([-3.0, 3.0]))
    assert float(fq.max_val) == pytest.approx(2.0)       # EMA with momentum .5
    fq.eval()
    fq(torch.tensor([-100.0, 100.0]))
    assert float(fq.max_val) == pytest.approx(2.0)       # eval never calibrates
    fq.train()
    Q.freeze_observers(nn.Sequential(fq))
    fq(torch.tensor([-50.0, 50.0]))
    assert float(fq.max_val) == pytest.approx(2.0)       # frozen
    assert fq.fq_on                                      # fake-quant keeps running


def test_qat_epoch_start_freezes_from_epoch_4(float_model):
    qm, _ = prepared(float_model)
    states = [Q.qat_epoch_start(qm, e, Q.QATConfig())["observers_frozen"] for e in range(6)]
    assert states == [False, False, False, False, True, True]
    assert Q.observers_frozen(qm)


def _rng(qm):
    return {k: (r["rmin"], r["rmax"]) for k, r in Q.activation_ranges(qm).items()}


def test_observers_frozen_in_model_forward(float_model):
    qm, _ = prepared(float_model)
    qm.train()
    with torch.no_grad():
        qm(x_in(1))
        r0 = _rng(qm)
        qm(x_in(2) * 5)
        r1 = _rng(qm)
        assert r0 != r1                                  # observing changes ranges
        Q.freeze_observers(qm)
        qm(x_in(3) * 9)
        assert _rng(qm) == r1
        assert all(r["frozen"] for r in Q.activation_ranges(qm).values())


def test_state_dict_roundtrip_keeps_ranges_and_freeze(float_model):
    qm, _ = prepared(float_model)
    qm.train()
    with torch.no_grad():
        qm(x_in(1))
    Q.freeze_observers(qm)
    sd = qm.state_dict()
    qm2, _ = prepared(float_model)
    qm2.load_state_dict(sd)
    assert Q.observers_frozen(qm2)
    qm.eval(), qm2.eval()
    with torch.no_grad():
        assert torch.equal(qm(x_in(4)), qm2(x_in(4)))


# ── instrumentation coverage ──────────────────────────────────────────────────────────────────
def test_quant_points_cover_relu_add_resize_conv_outputs(float_model):
    qm, rep = prepared(float_model)
    assert rep["conv_weight_quantized"] == N_CONV
    assert sum(isinstance(m, Q.QConv2d) for m in qm.modules()) == N_CONV
    assert rep["act_points"] == {"relu": N_RELU, "add": N_ADD, "resize": N_RESIZE,
                                 "conv_out": N_CONV_OUT, "input": 1}
    assert rep["final_output_producer"] == "add_out"
    assert rep["unhandled_leaf_types"] == {}
    ranges = Q.activation_ranges(qm)
    assert len(ranges) == N_RELU + N_ADD + N_RESIZE + N_CONV_OUT + 1


def test_activation_ranges_valid_after_observing(float_model):
    qm, _ = prepared(float_model)
    qm.train()
    with torch.no_grad():
        qm(x_in(1))
    for name, r in Q.activation_ranges(qm).items():
        assert r["initialized"], name
        assert r["rmin"] <= 0.0 <= r["rmax"], name
        assert 0 <= r["zero_point"] <= 255 and r["scale"] > 0, name
        if r["kind"] == "relu":
            assert r["rmin"] == 0.0 and r["zero_point"] == 0, name


def test_exclude_keeps_layers_float(float_model):
    qm, rep = prepared(float_model, exclude=("lat1",))
    assert type(qm.lat1[0]) is nn.Conv2d
    assert not hasattr(qm.lat1[2], "qat_out")
    assert set(rep["excluded"]) == {"lat1.0", "lat1.2"}
    assert rep["conv_weight_quantized"] == N_CONV - 1


def test_quant_output_flag_and_output_bits(float_model):
    _, rep = prepared(float_model, quant_output=False)
    assert "add_out:out" not in rep["act_point_names"]
    qm, _ = prepared(float_model, output_bits=16)
    assert qm.add_out.qat_out.bits == 16 and qm.add.qat_out.bits == 8


# ── Resize output shares its producer's qparams (ORT QDQResize) ───────────────────────────────
def test_resize_shares_producer_qparams(float_model):
    qm, rep = prepared(float_model)
    assert rep["resize_shares_qparams_with"] == {"up:out": "lat2.2:out", "up0:out": "to0.2:out",
                                                 "upc:out": "head.2:out"}
    for r in ("up", "up0", "upc"):
        fq = getattr(qm, r).qat_out
        assert isinstance(fq, Q.SharedFakeQuant)
        assert not list(fq.buffers())                     # no observer state of its own
    assert qm.up.qat_out.source is qm.lat2[2].qat_out
    qm.train()
    with torch.no_grad():
        qm(x_in(1))
    rng = Q.activation_ranges(qm)
    assert rng["up:out"]["scale"] == rng["lat2.2:out"]["scale"]
    assert rng["up:out"]["shares_qparams_with"] == "lat2.2:out"
    # the resize output lies on the producer's uint8 grid
    qm.eval()
    seen = {}
    h = qm.up.register_forward_hook(lambda m, a, o: seen.setdefault("y", o.detach()))
    with torch.no_grad():
        qm(x_in(2))
    h.remove()
    s, z = rng["lat2.2:out"]["scale"], rng["lat2.2:out"]["zero_point"]
    codes = seen["y"] / s + z
    assert torch.allclose(codes, codes.round(), atol=1e-3)
    # EMA / deepcopy keep the link inside the copy
    ema = Q.make_ema(qm)
    assert ema.up.qat_out.source is ema.lat2[2].qat_out
    assert ema.up.qat_out.source is not qm.lat2[2].qat_out


def test_resize_own_observer_when_sharing_off(float_model):
    qm, rep = prepared(float_model, resize_share_input_qparams=False)
    assert rep["resize_shares_qparams_with"] == {}
    assert isinstance(qm.up.qat_out, Q.ActFakeQuant)
    assert "up.qat_out.min_val" in Q.observer_state_names(qm)


# ── functional ops are reported (they would get no quantization point) ────────────────────────
class InlineAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2, self.act = nn.Conv2d(3, 4, 1), nn.Conv2d(3, 4, 1), nn.ReLU()

    def forward(self, x):
        return self.act(self.c1(x) + self.c2(x))


def test_functional_ops_reported_and_strict_graph_raises():
    m = InlineAdd().eval()
    x = torch.randn(1, 3, 8, 8)
    _, rep = Q.prepare_qat(m, Q.QATConfig(), x)
    assert rep["functional_inputs"] == ["act"]
    with pytest.raises(RuntimeError, match="functional ops"):
        Q.prepare_qat(m, Q.QATConfig(strict_graph=True), x)


def test_weight_fq_flag_persists_in_state_dict(float_model):
    qm, _ = prepared(float_model)
    qm.train()
    with torch.no_grad():
        qm(x_in(1))
    Q.set_fake_quant(qm, False)
    qm2, _ = prepared(float_model)
    qm2.load_state_dict(qm.state_dict())
    assert all(not m.w_fq_on for m in qm2.modules() if isinstance(m, Q.QConv2d))
    assert all(not m.fq_on for m in qm2.modules()
               if isinstance(m, (Q.ActFakeQuant, Q.SharedFakeQuant)))
    qm3, _ = prepared(float_model, weight_bits=4)
    with pytest.raises(ValueError, match="bits mismatch"):
        qm3.load_state_dict(qm.state_dict())


# ── ONNX tensor-name mapping (TensorQuantOverrides) ───────────────────────────────────────────
def test_onnx_module_path_parsing():
    f = Q._onnx_module_path
    assert f("/decoder/coarse/coarse.0/coarse.0.1/act/Relu") == "decoder.coarse.0.1.act"
    assert f("/decoder/lat/lat.2/conv/Conv_output_0") == "decoder.lat.2.conv"
    assert f("/model/decoder/up_c/Resize") == "model.decoder.up_c"
    assert f("/Add") == "" and f("Identity_5") is None and f("log_depth") is None


def _fake_graph(nodes, inputs=("image",)):
    n = [SimpleNamespace(name=a, op_type=b, output=list(c)) for a, b, c in nodes]
    return SimpleNamespace(graph=SimpleNamespace(
        node=n, input=[SimpleNamespace(name=i) for i in inputs], initializer=[]))


def test_onnx_tensor_ranges_on_fake_graph():
    ranges = {":input": {"rmin": -2.0, "rmax": 2.5, "initialized": True},
              "a.act:out": {"rmin": 0.0, "rmax": 3.0, "initialized": True},
              "up:out": {"rmin": 0.0, "rmax": 3.0, "initialized": True},
              "add:out": {"rmin": -1.0, "rmax": 4.0, "initialized": True}}
    g = _fake_graph([("/model/a/conv/Conv", "Conv", ["/model/a/conv/Conv_output_0"]),
                     ("/model/a/act/Relu", "Relu", ["/model/a/act/Relu_output_0"]),
                     ("/model/up/Constant", "Constant", ["/model/up/Constant_output_0"]),
                     ("/model/up/Resize", "Resize", ["/model/up/Resize_output_0"]),
                     ("/model/add/Add", "Add", ["log_depth"])])
    got = Q.onnx_tensor_ranges(g, ranges)                # prefix "model." auto-detected
    assert got == {"image": (-2.0, 2.5), "/model/a/act/Relu_output_0": (0.0, 3.0),
                   "/model/up/Resize_output_0": (0.0, 3.0), "log_depth": (-1.0, 4.0)}
    with pytest.raises(KeyError, match="not found"):
        Q.onnx_tensor_ranges(g, ranges, scope_prefix="")


def test_onnx_export_and_mapping_cover_every_point(float_model):
    onnx = pytest.importorskip("onnx")
    qm, _ = prepared(float_model)
    qm.train()
    with torch.no_grad():
        qm(x_in(1, hw=(32, 32)))
    plain = Q.strip_qat(qm).eval()
    x = x_in(0, 1, hw=(32, 32))
    f = io.BytesIO()
    torch.onnx.export(plain, (x,), f, input_names=["image"], output_names=["log_depth"],
                      opset_version=18, dynamo=False)
    om = onnx.load_from_string(f.getvalue())
    ranges = Q.activation_ranges(qm)
    got = Q.onnx_tensor_ranges(om, ranges)
    assert len(got) == len(ranges)                       # every point has its own tensor
    assert got["image"] == (ranges[":input"]["rmin"], ranges[":input"]["rmax"])
    assert got["log_depth"] == (ranges["add_out:out"]["rmin"], ranges["add_out:out"]["rmax"])
    produced = {o: n.op_type for n in om.graph.node for o in n.output}
    kinds = {"relu": "Relu", "add": "Add", "resize": "Resize", "conv_out": "Conv"}
    for p, r in ranges.items():
        if p == ":input":
            continue
        t = Q.onnx_tensor_ranges(om, {p: r})
        (name,) = t
        assert produced[name] == kinds[r["kind"]], (p, name)


# ── gradients, strip, EMA, loss, schedule ─────────────────────────────────────────────────────
def test_gradients_flow_through_ste(float_model):
    qm, _ = prepared(float_model)
    qm.train()
    x = x_in(1).requires_grad_(True)
    out = qm(x)                                  # first batch sets every range: nothing clipped
    (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, m in qm.named_modules():
        if isinstance(m, Q.QConv2d):
            g = m.weight.grad
            assert g is not None and torch.isfinite(g).all(), name
            assert g.abs().sum() > 0, name


def test_strip_qat_equals_fake_quant_disabled_equals_float(float_model):
    qm, _ = prepared(float_model)
    qm.train()
    with torch.no_grad():
        qm(x_in(1))
    plain = Q.strip_qat(qm).eval()
    assert not any(isinstance(m, (Q.ActFakeQuant, Q.QConv2d)) for m in plain.modules())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in plain.modules())
    qm.eval()
    x = x_in(5)
    with torch.no_grad():
        quant = qm(x)
        Q.set_fake_quant(qm, False)
        nofq = qm(x)
        ref = float_model(x)
        got = plain(x)
    assert torch.allclose(got, nofq, atol=1e-6)
    assert torch.allclose(got, ref, atol=1e-5)
    assert not torch.allclose(quant, ref, atol=1e-6)     # fake-quant really changed something


def test_ema_excludes_observers(float_model):
    qm, _ = prepared(float_model)
    qm.train()
    with torch.no_grad():
        qm(x_in(1))
    ema = Q.make_ema(qm)
    w0 = ema.lat1[0].weight.clone()
    with torch.no_grad():
        qm.lat1[0].weight.add_(1.0)
        qm.add.qat_out.max_val.fill_(99.0)
    Q.freeze_observers(qm)
    Q.ema_update_(ema, qm, decay=0.5)
    assert torch.allclose(ema.lat1[0].weight, w0 + 0.5, atol=1e-6)      # averaged
    assert float(ema.add.qat_out.max_val) == 99.0                       # copied, not averaged
    assert Q.observers_frozen(ema)
    names = Q.observer_state_names(qm)
    assert "add.qat_out.max_val" in names and "qat_in.min_val" in names
    assert not any(n.endswith("weight") for n in names)


def test_qat_loss_formula():
    base = lambda p, b: (torch.tensor(1.0), {"base": 1.0})            # noqa: E731
    pf = torch.zeros(2, 1, 4, 4)
    pq = pf + 0.1
    loss, terms = Q.QATLoss(base)(pq, pf, {})
    assert float(loss) == pytest.approx(1.0 + 0.1 + 10 * 0.01, rel=1e-5)
    assert terms["base"] == 1.0 and terms["qat_twin_abs"] == pytest.approx(0.1, rel=1e-5)
    pq2 = pf.clone()
    pq2[0] += 0.1
    pq2[1] -= 0.1
    lp, _ = Q.QATLoss(base, mean_mode="per_image")(pq2, pf, {})
    lb, _ = Q.QATLoss(base, mean_mode="batch")(pq2, pf, {})
    assert float(lp) == pytest.approx(1.2, rel=1e-5) and float(lb) == pytest.approx(1.1, rel=1e-5)
    pq3 = pf.clone().requires_grad_(True)
    pf3 = (pf + 1.0).requires_grad_(True)
    l3, _ = Q.QATLoss(base)(pq3 + 0.0, pf3, {})
    l3.backward()
    assert pf3.grad is None                                            # twin is detached
    with pytest.raises(ValueError):
        Q.QATLoss(base, mean_mode="pixel")


def test_qat_lr_cosine():
    lrs = [Q.qat_lr(s, 10, 1e-4, 1e-5) for s in range(10)]
    assert lrs[0] == pytest.approx(1e-4) and lrs[-1] == pytest.approx(1e-5)
    assert all(a > b for a, b in zip(lrs, lrs[1:]))


def test_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        Q.QATConfig.from_dict({"weight_bit": 8})
    c = Q.QATConfig.from_dict({"exclude": ["head.*"], "epochs": 3})
    assert c.exclude == ("head.*",) and c.epochs == 3


def test_qat_finetune_smoke(float_model):
    g = torch.Generator().manual_seed(7)
    data = [{"image": torch.randn(2, 3, 32, 32, generator=g),
             "target_log": torch.randn(2, 1, 32, 32, generator=g) * 0.3 + 1.0} for _ in range(3)]

    def base(pred, batch):
        loss = (pred - batch["target_log"]).abs().mean()
        return loss, {"l1": float(loss.detach())}

    cfg = Q.QATConfig(epochs=3, freeze_observers_after=1, example_hw=(32, 32))
    out = Q.qat_finetune(float_model, lambda ep: iter(data), base, cfg, steps_per_epoch=3)
    h = out["history"]
    assert [r["observers_frozen"] for r in h] == [False, True, True]
    assert h[1]["range_checksum"] == h[0]["range_checksum"] == h[2]["range_checksum"]
    assert all(math.isfinite(r["qat_total"]) and r["steps"] == 3 for r in h)
    assert h[-1]["lr_last"] == pytest.approx(1e-5)
    assert Q.observers_frozen(out["model"]) and Q.observers_frozen(out["ema"])
    assert not out["ema"].training
    assert Q.activation_ranges(out["ema"]) == Q.activation_ranges(out["model"])
    assert all(not p.requires_grad for p in out["twin"].parameters())
    assert not any(isinstance(m, nn.BatchNorm2d) for m in out["model"].modules())


# ── optional: the real CoDeQ from distill/v2/model.py (other agent's module) ──────────────────
def test_v2_model_integration():
    mod = pytest.importorskip("distill.v2.model")
    pytest.importorskip("timm")
    model = mod.build_v2_model({"variant": "P", "decoder_mode": "ps", "pretrained": False}).eval()
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    qm, rep = Q.prepare_qat(model, Q.QATConfig(strict_graph=True), x)
    assert rep["fold"]["bn_frozen_unfolded"] == []
    assert rep["fold"]["bn_total"] == rep["n_folded_pairs"] > 0
    assert not any(isinstance(m, nn.BatchNorm2d) for m in qm.modules())
    assert rep["functional_inputs"] == [] and rep["unhandled_leaf_types"] == {}
    assert rep["act_points"].get("relu", 0) > 0 and rep["act_points"].get("add", 0) > 0
    assert rep["act_points"]["resize"] == 5 == len(rep["resize_shares_qparams_with"])
    assert rep["final_output_producer"] == "decoder.add_out"
    qm.train()
    out = qm(x)
    assert out.shape == (1, 1, 64, 64) and torch.isfinite(out).all()
    out.mean().backward()
    assert all(m.weight.grad is not None for m in qm.modules() if isinstance(m, Q.QConv2d))
    assert Q.uninitialized_points(qm) == []


def test_v2_model_n_counts_match_plan_e3():
    """CoDeQ-N: 80 Conv, 47 Relu, 18 Add, 5 Resize (plan E3 ONNX op counts)."""
    mod = pytest.importorskip("distill.v2.model")
    pytest.importorskip("timm")
    model = mod.build_v2_model({"variant": "N", "decoder_mode": "ps", "pretrained": False}).eval()
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    _, rep = Q.prepare_qat(model, Q.QATConfig(strict_graph=True), x)
    assert rep["conv_weight_quantized"] == 80
    assert rep["act_points"]["relu"] == 47 and rep["act_points"]["add"] == 18
    assert rep["act_points"]["resize"] == 5 and rep["fold"]["bn_frozen_unfolded"] == []
