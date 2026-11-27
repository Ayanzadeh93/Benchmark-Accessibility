"""Read-only access to the ORIGINAL b10 code in C:/Tim/nature/unidepth2d for parity tests.

Modules are loaded straight from their file paths under private names, with bytecode writing
disabled, so nothing is ever written under the live training tree (a b10 run is using it).

Shared helper (not a test module; the leading underscore keeps pytest from collecting it).
Imported by test_b10_parity, test_cstm, test_dsh, test_fov_ownar, test_rep and test_trim,
so it must not be deleted or renamed without updating those imports.
"""
import importlib.util
import os
import sys

ORIG_ROOT = os.environ.get("B10_ORIG_ROOT", r"C:\Tim\nature\unidepth2d")
OUTPUTS = os.path.join(ORIG_ROOT, "outputs")
CKPTS = os.path.join(OUTPUTS, "checkpoints")


def load_orig(relpath: str, name: str):
    path = os.path.join(ORIG_ROOT, relpath)
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"original code not found: {path}")
    if name in sys.modules:
        return sys.modules[name]
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # dataclasses look the module up while decorating
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def b10_student_cfg(pretrained: bool = False) -> dict:
    """Student section of the b10 recipe (s4,s5,ray,range,prox + --use-intrinsics, B1)."""
    return {"student": {"backbone": "efficientvit_b1.r224_in1k", "pretrained_backbone": pretrained,
                        "decoder_channels": 96, "train_size": [480, 640], "depth_min_m": 0.2,
                        "depth_max_m": 30.0, "use_intrinsics": True, "ray_map": True}}
