"""Dataset loaders in UniDepthV2's zero-shot protocol. Each exposes iter_samples(limit) yielding
dicts: {rgb (H,W,3 uint8), depth (H,W float32 metres), K (3,3 float32) or None, valid (H,W bool),
max_depth float, name str}. K is the GT camera (needed for rho_A and F_A back-projection)."""

from importlib import import_module

REGISTRY = {
    "nyu": ("bench.datasets.nyu", "iter_samples"),
    "sun": ("bench.datasets.sun", "iter_samples"),
    "ibims": ("bench.datasets.ibims", "iter_samples"),
    "diode": ("bench.datasets.diode", "iter_samples"),
    "eth3d": ("bench.datasets.eth3d", "iter_samples"),
    "sintel": ("bench.datasets.sintel", "iter_samples"),
    "tum": ("bench.datasets.tum", "iter_samples"),
}


def get_loader(name):
    mod, fn = REGISTRY[name]
    return getattr(import_module(mod), fn)
