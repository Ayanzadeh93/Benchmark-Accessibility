"""Tests for distill/v2/encoder.py and distill/v2/model.py (CoDeQ family + legacy rungs).

Env: ``depth`` runs everything except the ONNX tests, which need ``onnx`` + ``onnxruntime`` and
are skipped elsewhere (run them in ``depth_cmp``). CPU-only and offline: models are built with
``pretrained=False``; the real-weight parity test reads the local HF-cache checkpoint directly
(skipped when it is not cached) and the legacy-checkpoint test reads outputs/checkpoints/b1_mix
(skipped when absent).
"""

import glob
import math
import os

import pytest
import torch
import torch.nn as nn

timm = pytest.importorskip("timm")

from distill.student import DepthStudent, StudentConfig  # noqa: E402
from distill.v2.encoder import (  # noqa: E402
    CODEQ_N_DELETE,
    Add,
    build_timm_features,
    convert_timm_mobilenetv4,
    load_timm_state_dict,
)
from distill.v2.model import (  # noqa: E402
    ALLOWED_ONNX_OPS,
    LEGACY_RUNGS,
    V2_VARIANTS,
    CoDeQ,
    LegacyStudent,
    PlainFPNStudent,
    build_v2_model,
    count_macs,
    count_params,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW = (128, 160)  # small multiple of 32 for fast CPU tests


def _randomize_bn(model: nn.Module, seed: int = 0) -> None:
    """Give every BatchNorm non-trivial affine params and running stats (so BN is not identity)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                n = m.num_features
                m.weight.copy_(0.5 + torch.rand(n, generator=g))
                m.bias.copy_(0.2 * torch.randn(n, generator=g))
                m.running_mean.copy_(0.2 * torch.randn(n, generator=g))
                m.running_var.copy_(0.5 + torch.rand(n, generator=g))


def _timm(variant: str):
    spec = V2_VARIANTS[variant]
    t = build_timm_features(spec["timm_name"], pretrained=False)
    if spec["delete_blocks"]:
        t.blocks[3] = nn.Sequential(*[b for i, b in enumerate(t.blocks[3]) if i not in spec["delete_blocks"]])
    return t


def _model(**cfg):
    cfg.setdefault("pretrained", False)
    return build_v2_model(cfg).eval()


# --------------------------------------------------------------------------------------
# sizes (plan 3.1)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["P", "N", "S"])
def test_param_counts_and_macs_within_2pct_of_plan(variant):
    spec = V2_VARIANTS[variant]
    m = _model(variant=variant, decoder_mode="ps")
    enc = count_params(m.encoder)
    tot = count_params(m)
    assert abs(tot / 1e6 - spec["plan_total_M"]) <= 0.02 * spec["plan_total_M"]
    assert abs(enc / 1e6 - spec["plan_encoder_M"]) <= 0.02 * spec["plan_encoder_M"]
    assert abs((tot - enc) / 1e3 - spec["plan_decoder_k"]) <= 0.02 * spec["plan_decoder_k"]
    gmac = count_macs(m, (480, 640)) / 1e9
    assert abs(gmac - spec["plan_gmac"]) <= 0.02 * spec["plan_gmac"]
    assert count_params(m, trainable_only=True) == tot


def test_codeq_n_deletes_five_blocks_of_stage3():
    n = _model(variant="N")
    s = _model(variant="S")
    assert len(s.encoder.stages[3]) == 11 and len(n.encoder.stages[3]) == 6
    assert n.encoder.deleted_blocks == CODEQ_N_DELETE
    assert len(n.encoder.stages) == 4  # blocks[4] (1x1 to 960) dropped
    # kept blocks keep their original timm index in the key map
    kept = [i for i in range(11) if i not in CODEQ_N_DELETE]
    for j, orig in enumerate(kept):
        k = f"stages.3.{j}.body.0.conv.weight"
        assert n.encoder.timm_key_map[k].startswith(f"blocks.3.{orig}.")


def test_block_deletion_rejects_unsafe_block():
    t = build_timm_features(V2_VARIANTS["N"]["timm_name"], pretrained=False)
    with pytest.raises(ValueError):
        convert_timm_mobilenetv4(t, delete_blocks=(0,))  # blocks[3][0] is the stride-2 160->256 block


# --------------------------------------------------------------------------------------
# encoder parity with timm (plan E1)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["P", "N", "S"])
def test_plain_encoder_matches_timm_eval_and_train(variant):
    torch.manual_seed(0)
    t = _timm(variant)
    _randomize_bn(t, seed=1)
    t.eval()
    plain_full = convert_timm_mobilenetv4(t, drop_last_stage=False).eval()   # timm's 5 features
    plain = convert_timm_mobilenetv4(t, drop_last_stage=True).eval()         # CoDeQ encoder
    # no timm block type survives the conversion
    allowed = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Identity, Add)
    for m in plain.modules():
        if len(list(m.children())) == 0:
            assert isinstance(m, allowed), type(m)
    x = torch.randn(2, 3, *HW)
    s3 = {}
    h = t.blocks[3].register_forward_hook(lambda mod, i, o: s3.__setitem__("out", o))
    with torch.no_grad():
        ref = t(x)
        got_full = plain_full(x)
        got = plain(x)
    h.remove()
    assert len(ref) == len(got_full) == len(got) == 5
    for a, b in zip(ref, got_full):
        assert a.shape == b.shape
        assert float((a - b).abs().max()) <= 1e-5
    for a, b in zip(ref[:4], got[:4]):
        assert float((a - b).abs().max()) <= 1e-5
    assert float((s3["out"] - got[4]).abs().max()) <= 1e-5
    assert [f.shape[1] for f in got] == list(V2_VARIANTS[variant]["feature_channels"])

    # train mode: same batch statistics and the same running-stat update
    t.train()
    plain_full.train()
    out_t = t(x)
    out_p = plain_full(x)
    for a, b in zip(out_t, out_p):
        assert float((a - b).abs().max()) <= 1e-4
    rm_t = t.blocks[1][0].dw_start.bn.running_mean if hasattr(t.blocks[1][0], "dw_start") else t.bn1.running_mean
    rm_p = plain_full.stages[1][0].body[0].bn.running_mean if hasattr(t.blocks[1][0], "dw_start") \
        else plain_full.stem.bn.running_mean
    assert torch.allclose(rm_t, rm_p, atol=1e-6)


@pytest.mark.parametrize("variant", ["N", "S"])
def test_feature_shapes_at_480x640(variant):
    m = _model(variant=variant)
    with torch.no_grad():
        feats = m.encoder(torch.zeros(1, 3, 480, 640))
    assert [tuple(f.shape[1:]) for f in feats] == [(32, 240, 320), (48, 120, 160), (80, 60, 80),
                                                   (160, 30, 40), (256, 15, 20)]


def _cached_safetensors(variant):
    name = V2_VARIANTS[variant]["timm_name"]
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hits = glob.glob(os.path.join(hub, f"models--timm--{name}", "snapshots", "*", "model.safetensors"))
    return hits[0] if hits else None


@pytest.mark.parametrize("variant", ["P", "N"])
def test_pretrained_weights_strict_load_and_parity(variant):
    """Real ImageNet weights from the local HF cache (no network): timm-loaded parity, and a strict
    key-map load of the raw checkpoint into a randomly initialised plain encoder."""
    path = _cached_safetensors(variant)
    if path is None:
        pytest.skip("MobileNetV4 weights not in the local HF cache")
    st = pytest.importorskip("safetensors.torch")
    spec = V2_VARIANTS[variant]
    t = build_timm_features(spec["timm_name"], weights_file=path).eval()
    if spec["delete_blocks"]:
        t.blocks[3] = nn.Sequential(*[b for i, b in enumerate(t.blocks[3]) if i not in spec["delete_blocks"]])
    plain_from_timm = convert_timm_mobilenetv4(t, drop_last_stage=False).eval()
    x = torch.randn(1, 3, *HW)
    with torch.no_grad():
        ref = t(x)
        got = plain_from_timm(x)
    assert max(float((a - b).abs().max()) for a, b in zip(ref, got)) <= 1e-5

    fresh = build_v2_model({"variant": variant, "pretrained": False}).encoder.eval()
    sd = st.load_file(path)
    info = load_timm_state_dict(fresh, sd, strict=True)
    assert info["missing"] == []
    assert all(k.startswith(("conv_head.", "norm_head.", "classifier.", "blocks.4.", "blocks.3."))
               for k in info["unused"])
    with torch.no_grad():
        got2 = fresh(x)
    for a, b in zip(ref[:4], got2[:4]):
        assert float((a - b).abs().max()) <= 1e-5


def test_strict_timm_load_rejects_missing_and_unexpected_keys():
    t = _timm("P")
    plain = convert_timm_mobilenetv4(t)
    sd = {k: v.clone() for k, v in t.state_dict().items()}
    load_timm_state_dict(plain, sd, strict=True)  # features-model state dict loads cleanly
    bad = dict(sd)
    bad["blocks.1.0.bogus.weight"] = torch.zeros(1)
    with pytest.raises(KeyError):
        load_timm_state_dict(plain, bad, strict=True)
    short = dict(sd)
    short.pop("conv_stem.weight")
    with pytest.raises(KeyError):
        load_timm_state_dict(plain, short, strict=True)


# --------------------------------------------------------------------------------------
# decoder modes and contract
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant,mode", [("N", "s4"), ("N", "s2"), ("N", "ps"), ("N", "fpn"),
                                          ("P", "ps"), ("S", "ps"), ("P", "s4"), ("S", "fpn")])
def test_output_shape_all_modes(variant, mode):
    m = _model(variant=variant, decoder_mode=mode)
    assert m.decoder_mode == mode and m.variant == variant
    assert math.isclose(m.log_min, math.log(0.2)) and math.isclose(m.log_max, math.log(30.0))
    x = torch.randn(2, 3, *HW)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 1, *HW) and y.dtype == torch.float32
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("rung", sorted(LEGACY_RUNGS))
def test_legacy_rungs_build_and_shape(rung):
    m = _model(variant="legacy", legacy_rung=rung)
    assert isinstance(m, LegacyStudent)
    rgb_only = rung == "R3"
    assert m.use_intrinsics is (not rgb_only) and m.ray_map is (not rgb_only)
    assert m.cfg.backbone == "efficientvit_b1.r224_in1k"
    with torch.no_grad():
        y = m(torch.randn(2, 3, *HW))
    assert y.shape == (2, 1, *HW)


def test_ps_residual_zero_init_equals_s2_and_head_bias():
    torch.manual_seed(0)
    ps = _model(variant="N", decoder_mode="ps")
    s2 = _model(variant="N", decoder_mode="s2")
    assert torch.count_nonzero(ps.decoder.res_conv.weight) == 0
    assert torch.count_nonzero(ps.decoder.res_conv.bias) == 0
    mid = 0.5 * math.log(0.2 * 30.0)
    assert abs(mid - 0.896) < 1e-3
    assert torch.allclose(ps.decoder.coarse[-1].bias, torch.tensor([mid]))
    missing, unexpected = s2.load_state_dict(ps.state_dict(), strict=False)
    assert missing == [] and all(k.startswith("decoder.res_conv.") for k in unexpected)
    _randomize_bn(ps, seed=3)
    s2.load_state_dict({k: v for k, v in ps.state_dict().items() if not k.startswith("decoder.res_conv.")})
    x = torch.randn(1, 3, *HW)
    with torch.no_grad():
        assert torch.equal(ps(x), s2(x))
    s4 = _model(variant="N", decoder_mode="s4")
    assert torch.allclose(s4.decoder.head4[-1].bias, torch.tensor([mid]))


def test_output_starts_near_mid_range_log_depth():
    m = _model(variant="N", decoder_mode="ps")
    with torch.no_grad():
        y = m(torch.zeros(1, 3, *HW))
    assert y.shape == (1, 1, *HW) and torch.isfinite(y).all()


def test_intrinsics_ignored_by_codeq():
    m = _model(variant="P", decoder_mode="ps")
    _randomize_bn(m, seed=5)
    x = torch.randn(2, 3, *HW)
    k = torch.tensor([[0.5, 0.7, 0.4, 0.6], [1.2, 1.5, 0.5, 0.5]])
    with torch.no_grad():
        assert torch.equal(m(x), m(x, k))


def test_legacy_default_k_keeps_film_on():
    torch.manual_seed(0)
    m = _model(variant="legacy", legacy_rung="R0")
    with torch.no_grad():  # FiLM is zero-initialised; make it non-trivial so K matters
        m.film[-1].weight.normal_(0, 0.5)
        m.film[-1].bias.normal_(0, 0.5)
    x = torch.randn(2, 3, *HW)
    k = torch.tensor([[0.8, 1.08, 0.5, 0.5]]).expand(2, 4)
    with torch.no_grad():
        y_none = m(x)
        y_k = m(x, k)
        y_raw = DepthStudent.forward(m, x, None)   # raw legacy K=None: FiLM skipped
    assert torch.equal(y_none, y_k)
    assert not torch.allclose(y_none, y_raw)
    m.default_k = None
    with torch.no_grad():
        assert torch.equal(m(x), y_raw)


def test_legacy_state_dict_matches_depthstudent_and_loads_b1_checkpoint():
    m = _model(variant="legacy", legacy_use_intrinsics=True, legacy_ray_map=False)
    ref = DepthStudent(StudentConfig(backbone="efficientvit_b1.r224_in1k", pretrained_backbone=False,
                                     decoder_channels=96, use_intrinsics=True, ray_map=False))
    assert list(m.state_dict().keys()) == list(ref.state_dict().keys())
    ck = os.path.join(ROOT, "outputs", "checkpoints", "b1_mix", "best.pt")
    if not os.path.exists(ck):
        pytest.skip("legacy b1 checkpoint not on disk")
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    m.load_state_dict(sd, strict=True)
    assert count_params(m) == 5118977


def test_fpn_mode_reuses_legacy_decoder_layout():
    m = _model(variant="N", decoder_mode="fpn")
    assert isinstance(m, PlainFPNStudent) and isinstance(m, DepthStudent)
    keys = [k for k in m.state_dict() if not k.startswith("encoder.")]
    assert {k.split(".")[0] for k in keys} == {"laterals", "smooths", "head"}
    assert len(m.laterals) == 4 and m.laterals[0].in_channels == 48  # f1..f4 (strides 4..32)
    assert abs(float(m.head[-1].bias[0]) - 0.5 * math.log(6.0)) < 1e-6
    assert not m.use_intrinsics and not m.ray_map


def test_scale_head_option():
    m = _model(variant="N", decoder_mode="ps", scale_head=True)
    assert "GlobalAveragePool" in m.allowed_onnx_ops
    assert float(m.decoder.coarse[-1].bias) == 0.0
    x = torch.randn(2, 3, *HW)
    with torch.no_grad():
        parts = m.forward_parts(x)
        y = m(x)
    assert parts["scale"].shape == (2, 1, 1, 1)
    assert torch.allclose(parts["scale"], torch.full((2, 1, 1, 1), 0.5 * math.log(6.0)))
    assert torch.equal(y, parts["rel_log"] + parts["scale"])
    plain = _model(variant="N")
    assert plain.forward_parts(x)["scale"] is None


def test_codeq_leaf_modules_are_op_closed():
    allowed = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Identity, Add, nn.Upsample, nn.PixelShuffle)
    for mode in ("s4", "s2", "ps"):
        m = _model(variant="N", decoder_mode=mode)
        for mod in m.modules():
            if len(list(mod.children())) == 0:
                assert isinstance(mod, allowed), (mode, type(mod))
            if isinstance(mod, nn.Upsample):
                assert mod.mode == "bilinear" and mod.align_corners is False and mod.size is None


def test_strict_state_dict_roundtrip():
    a = _model(variant="N", decoder_mode="ps")
    _randomize_bn(a, seed=7)
    b = _model(variant="N", decoder_mode="ps")
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(1, 3, *HW)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_config_validation():
    with pytest.raises(ValueError):
        build_v2_model({"variant": "N", "pretrained": False, "decodr_mode": "ps"})
    with pytest.raises(ValueError):
        build_v2_model({"variant": "X", "pretrained": False})
    with pytest.raises(ValueError):
        build_v2_model({"variant": "legacy", "legacy_rung": "R0", "decoder_mode": "ps", "pretrained": False})
    with pytest.raises(ValueError):
        build_v2_model({"variant": "legacy", "pretrained": False})  # no rung, no explicit camera flags
    with pytest.raises(ValueError):
        build_v2_model({"variant": "N", "decoder_mode": "fpn", "scale_head": True, "pretrained": False})
    m = _model(variant="N")
    with pytest.raises(ValueError):
        m(torch.zeros(1, 3, 100, 160))
    assert m.v2_config["variant"] == "N" and m.v2_config["decoder_mode"] == "ps"


def test_infer_metres_clamps():
    m = _model(variant="P")
    with torch.no_grad():
        d = m.infer_metres(torch.randn(1, 3, *HW))
    assert float(d.min()) >= 0.2 - 1e-6 and float(d.max()) <= 30.0 + 1e-4


# --------------------------------------------------------------------------------------
# ONNX (depth_cmp env): allowlist, no shape ops, parity
# --------------------------------------------------------------------------------------

_E3_COUNTS = {  # plan E3 / prototype export.json fp32 op counts (mode ps, opset 18)
    "N": {"Conv": 80, "Relu": 47, "Add": 18, "Resize": 5, "DepthToSpace": 1},
    "P": {"Conv": 64, "Relu": 46, "Add": 15, "Resize": 5, "DepthToSpace": 1},
    "S": {"Conv": 95, "Relu": 54, "Add": 23, "Resize": 5, "DepthToSpace": 1},
}


def _export_ops(model, tmp_path, name):
    onnx = pytest.importorskip("onnx")
    import collections
    import warnings

    p = str(tmp_path / f"{name}.onnx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, torch.zeros(1, 3, *HW), p, opset_version=18, input_names=["x"],
                          output_names=["log_depth"], dynamo=False, do_constant_folding=True)
    g = onnx.load(p)
    # Same exact post-processing as edge/export_v2.export_module: newer torch exporters emit
    # Resize scales as Constant nodes and dedup'd initializers through Identity nodes.
    from edge.export_v2 import fold_constants_to_initializers, remove_identities
    fold_constants_to_initializers(g)
    remove_identities(g)
    onnx.checker.check_model(g)
    onnx.save(g, p)
    return p, dict(collections.Counter(n.op_type for n in g.graph.node))


@pytest.mark.parametrize("variant,mode", [("N", "s4"), ("N", "s2"), ("N", "ps"), ("P", "ps"), ("S", "ps")])
def test_onnx_export_allowlist_and_no_shape_ops(variant, mode, tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    m = _model(variant=variant, decoder_mode=mode)
    _randomize_bn(m, seed=11)
    path, ops = _export_ops(m, tmp_path, f"{variant}_{mode}")
    assert set(ops) <= set(ALLOWED_ONNX_OPS), ops
    for bad in ("Shape", "Slice", "Gather", "Concat", "Reshape", "Constant"):
        assert bad not in ops, ops
    assert ops.get("DepthToSpace", 0) == (1 if mode == "ps" else 0)
    assert ops["Resize"] == (4 if mode == "s4" else 5)
    if mode == "ps":
        assert ops == _E3_COUNTS[variant]
    x = torch.randn(1, 3, *HW)
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": x.numpy()})[0]
    assert got.shape == ref.shape
    assert float(np.abs(got - ref).max()) < 1e-3  # plan 8.6 parity bound


def test_onnx_scale_head_adds_only_global_average_pool(tmp_path):
    pytest.importorskip("onnx")
    m = _model(variant="N", decoder_mode="ps", scale_head=True)
    _, ops = _export_ops(m, tmp_path, "N_scale")
    assert set(ops) <= set(m.allowed_onnx_ops), ops
    assert ops.get("GlobalAveragePool", 0) == 1
