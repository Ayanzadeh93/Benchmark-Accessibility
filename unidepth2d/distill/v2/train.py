"""V2 (CoDeQ) trainer: one ladder rung per run (docs/V2_PLAN.md sections 4, 6.6, 7; amendments A6, A7,
A9, A12, A19).

CLI (env ``depth``; run from the repo root with PYTHONUTF8=1)::

    python -m distill.v2.train --config configs/v2/tier1/R0.yaml
    python -m distill.v2.train --config configs/v2/tier1/R1a.yaml --parent-run outputs/v2/runs/<R0 run id>
    python -m distill.v2.train --resume outputs/v2/runs/<run id>          # continue after a crash
    python -m distill.v2.train --config configs/v2/tier1/R0.yaml --smoke  # 3 CPU steps, synthetic data
    python -m distill.v2.train --config configs/v2/tier1/R0.yaml --timing-steps 200   # T1-00

Contract
--------
Config (YAML). ``extends: <path>`` (relative to the file) is resolved recursively and deep-merged
(:func:`load_config`); a dict holding ``__replace__: true`` replaces instead of merging, and
``data.draw_weights`` is always replaced as a whole (a distribution, never a key-wise merge).
Sections:

* identity (never compared): ``schema rung parent record_type factor_changed factor_keys components
  notes extends``.
* comparable (one-factor checks, config hash, registry ``config_resolved``): ``seed model loss data
  train eval``. :func:`comparable_view` replaces ``model`` by the resolved ``build_v2_model`` cfg and
  ``loss`` by ``V2Loss.resolved_config()`` so loss weights and masks are compared, not preset names
  (A7).
* ``runtime`` (never compared): device, workers, logging and save cadence, eval batch size, the
  420 s epoch abort.

Training: AdamW (lr 3e-4, wd .01), linear warm-up (``min(warmup_steps, warmup_frac_max*total)``
steps from ``warmup_start_factor``) then cosine to ``min_lr_ratio``; AMP fp16 + GradScaler on CUDA;
grad-norm clip 1.0; EMA .999 of parameters and float buffers, QAT observer state COPIED, never
averaged (``distill.v2.qat.observer_state_names``); batch 12 with NO gradient accumulation
(``train.grad_accum`` other than 1 is refused); ``draws_per_epoch`` (12,000) weighted draws per
epoch from ``make_sampler(ds, draw_weights, n_draws, seed=seed*1000+epoch)``. Seeds python, numpy
and torch (CPU + CUDA).

Run folder ``<runs_root>/<run_id>/`` (unique: created with ``exist_ok=False``):

* ``config.yaml`` the fully merged config (authoritative on ``--resume``), ``manifest.json``
  (``benchmark.provenance.run_manifest``), ``train.log``.
* ``last.pt`` (atomic, every ``runtime.save_every_steps`` and at every epoch end): ``model`` (live
  weights), ``ema``, ``optim``, ``sched``, ``scaler``, ``rng`` (python/numpy/torch/cuda), ``step``,
  ``epoch``, ``step_in_epoch``, ``history``, ``comparable_hash``. Resume refuses a changed config.
* ``final_ema.pt`` = the EMA after the last step of the final epoch, never selected on any dataset:
  ``{"model": ema_state_dict, "model_cfg": resolved build_v2_model cfg, "cfg", "kind": "final_ema",
  "selection": "none", "run_id", "epoch", "step"}`` (``edge.export_v2.load_v2_checkpoint`` layout).
* ``dev_eval.json``: D_SUN dev evaluation of the final EMA (``V2Dataset(roles, eval.split,
  eval.dataset)``): per-image rows ``{id, source, d1, arel, rmse, d1_ssi, bf1_si_v2, b_shift,
  n_valid}`` and a summary ``{D1 (%), AbsRel, RMSE, d1_ssi (%), BF (corrected boundary F1,
  protocol_v2), BF_n_contributing, b_mean, b_std, n_images, nan_free, camera_mode}``; with a parent
  run, ``vs_parent`` holds the paired bootstrap (``protocol_v2.paired_bootstrap_clustered`` over the
  roles.yaml D_SUN near-duplicate clusters, rows joined on image id) for d1, arel, d1_ssi, bf1_si_v2.
* ``run.json``: status (running|interrupted|trained|complete|failed|aborted_slow|timing|smoke),
  stages, timings (seconds per epoch, data-wait share), params, checkpoint sha256, dev summary,
  registry record ids.

Registry (``benchmark.registry_v2.append_record``; guarded import): one trained record (type
``record_type``: root|ablation|finalist, status ``trained``; parent = the parent's run_id;
``factor_keys`` = the concrete resolved keys that differ from the parent) and one ``eval`` record
for dev_eval.json. ``--smoke`` writes to ``<run_dir>/registry_smoke.jsonl`` as a root record with
status ``smoke`` so smoke runs never enter the real chain.

GPU lock: :class:`GpuLock` on ``outputs/v2/gpu.lock`` (O_CREAT|O_EXCL, pid liveness for stale
locks). A child started by the lock holder passes through when its env ``V2_GPU_LOCK_TOKEN`` equals
the lock's token (that is how scripts/v2/run_ladder.py runs the trainer). CPU runs take no lock.

Exit codes: 0 ok, 2 config/setup error, 3 training failed (non-finite loss), 4 trained but the
registry refused the record, 5 epoch slower than ``runtime.max_epoch_seconds`` (last.pt saved,
resumable with ``--ignore-epoch-limit``), 6 GPU lock busy.

Data contract (``distill/v2/data.py``): ``V2Dataset(roles, split, cfg)`` items ``{"image" (3,H,W)
float ImageNet-normalised, "target_log" (1,H,W), "valid" (1,H,W) bool, "is_gt" bool, "dmax" float,
"source" str, "id" str}`` (+ optional ``"intrinsics"`` (4,)); ``make_sampler(ds, draw_weights,
n_draws, seed)``. ``--smoke`` uses :class:`SyntheticV2Dataset` (same item contract, no files).

Tests: tests/test_v2_train.py (env depth, CPU, offline).
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import fnmatch
import json
import math
import os
import random
import socket
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "outputs" / "v2" / "runs"
DEFAULT_GPU_LOCK = REPO_ROOT / "outputs" / "v2" / "gpu.lock"
LOCK_TOKEN_ENV = "V2_GPU_LOCK_TOKEN"

IDENTITY_KEYS = ("schema", "rung", "parent", "record_type", "factor_changed", "factor_keys",
                 "components", "notes", "extends")
COMPARABLE_SECTIONS = ("seed", "model", "loss", "data", "train", "eval")
ATOMIC_PATHS = frozenset({"data.draw_weights"})
REPLACE_MARKER = "__replace__"
RECORD_TYPES_TRAINED = ("root", "ablation", "finalist")

EXIT_OK, EXIT_CONFIG, EXIT_FAILED, EXIT_REGISTRY, EXIT_SLOW, EXIT_LOCK = 0, 2, 3, 4, 5, 6

TEACHER_IDS = {
    "v2l": ["lpiccinelli/unidepth-v2-vitl14"],
    "composite": ["lpiccinelli/unidepth-v2-vitl14", "Ruicheng/moge-2-vitl-normal"],
}

# Defaults under every config (base.yaml repeats them so the YAML is self-describing).
DEFAULTS: Dict[str, Any] = {
    "schema": "v2-train/1",
    "rung": "unnamed",
    "parent": None,
    "record_type": "root",
    "factor_changed": None,
    "factor_keys": [],
    "components": [],
    "notes": "",
    "seed": 2027,
    "model": {"variant": "N", "decoder_mode": "ps", "pretrained": True},
    "loss": {"preset": "base"},
    "data": {
        "roles": "configs/v2/roles.yaml",
        "train_split": "train",
        "draw_weights": {"own": 0.5, "nyu": 0.5},
        "dataset": {},
    },
    "train": {
        "epochs": 10, "draws_per_epoch": 12000, "batch_size": 12, "grad_accum": 1,
        "optimizer": "adamw", "lr": 3.0e-4, "weight_decay": 0.01, "betas": [0.9, 0.999],
        "warmup_steps": 500, "warmup_frac_max": 0.05, "warmup_start_factor": 0.05,
        "min_lr_ratio": 0.0, "amp": True, "grad_clip": 1.0, "ema_decay": 0.999,
        "intrinsics": "none", "max_nonfinite_steps": 20,
    },
    "eval": {
        "enabled": True, "split": "dev", "dataset": {}, "camera_mode": "none", "min_depth_m": 0.2,
        "n_boot": 10000, "boot_seed": 0, "compare_metrics": ["d1", "arel", "d1_ssi", "bf1_si_v2"],
    },
    "runtime": {
        "device": "cuda", "num_workers": 4, "pin_memory": True, "log_every": 50,
        "save_every_steps": 250, "eval_batch_size": 8, "metric_workers": 8,
        "max_epoch_seconds": 420, "cudnn_benchmark": True,
    },
}


class ConfigError(ValueError):
    """Invalid or inconsistent trainer config."""


class GpuLockBusy(RuntimeError):
    """Another live process holds the GPU lock."""


# ═════════════════════════════════════════════════════════════════════════════════════════
# config
# ═════════════════════════════════════════════════════════════════════════════════════════

def _is_replace(d: Any) -> bool:
    return isinstance(d, dict) and bool(d.get(REPLACE_MARKER, False))


def _strip_markers(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_markers(v) for k, v in d.items() if k != REPLACE_MARKER}
    if isinstance(d, list):
        return [_strip_markers(v) for v in d]
    return d


def deep_merge(base: Any, over: Any, path: str = "") -> Any:
    """Recursive merge of ``over`` onto ``base`` (neither is modified).

    dict + dict merges key by key, except at :data:`ATOMIC_PATHS` and when ``over`` carries
    ``__replace__: true`` (then ``over`` replaces ``base``). Any other value (lists, scalars, None)
    replaces. Markers are removed from the result."""
    if isinstance(base, dict) and isinstance(over, dict) and not _is_replace(over) \
            and path not in ATOMIC_PATHS:
        out = {k: copy.deepcopy(v) for k, v in base.items()}
        for k, v in over.items():
            if k == REPLACE_MARKER:
                continue
            sub = f"{path}.{k}" if path else str(k)
            out[k] = deep_merge(out[k], v, sub) if k in out else _strip_markers(copy.deepcopy(v))
        return out
    return _strip_markers(copy.deepcopy(over))


def load_yaml(path: os.PathLike) -> dict:
    import yaml
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return d


def own_keys(path: os.PathLike) -> dict:
    """The file's own keys (its delta), without ``extends``."""
    d = load_yaml(path)
    d.pop("extends", None)
    return d


def _load_chain(path: Path, seen: Tuple[str, ...]) -> dict:
    key = str(path.resolve()).lower()
    if key in seen:
        raise ConfigError(f"extends cycle: {' -> '.join(seen + (key,))}")
    d = load_yaml(path)
    ext = d.pop("extends", None)
    if ext is None:
        return d
    parent = (path.parent / ext) if not Path(ext).is_absolute() else Path(ext)
    return deep_merge(_load_chain(parent, seen + (key,)), d)


def load_config(path: os.PathLike, with_defaults: bool = True) -> dict:
    """Resolve the ``extends`` chain of ``path`` and merge it over :data:`DEFAULTS`."""
    merged = _load_chain(Path(path), ())
    return deep_merge(DEFAULTS, merged) if with_defaults else merged


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """``deep_merge(cfg, overrides)`` (identity keys in ``overrides`` replace the parent's)."""
    return deep_merge(cfg, overrides)


def resolved_model_cfg(model_cfg: dict) -> dict:
    """Resolved ``build_v2_model`` cfg without building the network."""
    from distill.v2.model import _resolve  # noqa: WPS450 - resolution without weights
    r = dict(_resolve(dict(model_cfg)))
    if r.get("legacy_default_k") is not None:
        r["legacy_default_k"] = [float(v) for v in r["legacy_default_k"]]
    return r


def resolved_loss_cfg(loss_cfg: dict) -> dict:
    from distill.v2.losses import V2Loss
    return V2Loss(dict(loss_cfg)).resolved_config()


def comparable_view(cfg: dict) -> dict:
    """The sections compared by the one-factor rule, with model and loss RESOLVED (A7)."""
    out: Dict[str, Any] = {}
    for k in COMPARABLE_SECTIONS:
        out[k] = copy.deepcopy(cfg.get(k))
    out["model"] = resolved_model_cfg(cfg.get("model") or {})
    out["loss"] = resolved_loss_cfg(cfg.get("loss") or {})
    return _jsonable(out)


def comparable_hash(cfg: dict) -> str:
    from benchmark.provenance import config_hash
    return config_hash(comparable_view(cfg))


def flatten(d: Any, prefix: str = "") -> Dict[str, str]:
    """Nested dict -> {dotted key: canonical JSON leaf}. Lists and empty dicts are leaves."""
    out: Dict[str, str] = {}
    if isinstance(d, dict) and d:
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = json.dumps(_jsonable(d), sort_keys=True)
    return out


def config_diff(parent_cfg: dict, child_cfg: dict) -> List[str]:
    """Dotted keys whose resolved values differ between two FULL configs (comparable view)."""
    a, b = flatten(comparable_view(parent_cfg)), flatten(comparable_view(child_cfg))
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def key_matches(key: str, pattern: str) -> bool:
    """``pattern`` covers ``key`` when equal, a dotted prefix of it, or an fnmatch glob of it.
    A parent-level key (e.g. ``data.dataset.zoom_crop`` that became a leaf) is also covered by a
    pattern below it."""
    return (key == pattern or key.startswith(pattern + ".") or pattern.startswith(key + ".")
            or fnmatch.fnmatchcase(key, pattern))


def check_one_factor(parent_cfg: dict, child_cfg: dict,
                     factor_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """One-factor rule on RESOLVED configs (plan 7, A7).

    Returns ``{"ok", "diff", "outside", "problems"}``: ``diff`` = every resolved key that differs,
    ``outside`` = differing keys not covered by ``factor_keys`` (default: the child's
    ``factor_keys``). ok iff ``diff`` is non-empty and ``outside`` is empty and the child declares
    exactly one ``factor_changed``."""
    keys = list(child_cfg.get("factor_keys") or []) if factor_keys is None else list(factor_keys)
    diff = config_diff(parent_cfg, child_cfg)
    outside = [k for k in diff if not any(key_matches(k, p) for p in keys)]
    problems = []
    fc = child_cfg.get("factor_changed")
    if isinstance(fc, (list, tuple)):
        if len(fc) != 1:
            problems.append(f"factor_changed must name exactly one factor, got {list(fc)}")
    elif not fc:
        problems.append("factor_changed is empty")
    if not keys:
        problems.append("factor_keys is empty: declare which resolved keys the factor may change")
    if not diff:
        problems.append("resolved config is identical to the parent: no factor changed")
    if outside:
        problems.append(f"resolved config differs outside the declared factor: {outside}")
    return {"ok": not problems, "diff": diff, "outside": outside, "problems": problems}


def validate_config(cfg: dict, smoke: bool = False) -> List[str]:
    """Raise ConfigError on fatal problems; return warnings."""
    warn = []
    tr = cfg["train"]
    if int(tr.get("grad_accum", 1)) != 1:
        raise ConfigError("gradient accumulation is not supported (plan: batch 12 without accumulation)")
    bs, draws = int(tr["batch_size"]), int(tr["draws_per_epoch"])
    if bs <= 0 or draws <= 0 or int(tr["epochs"]) <= 0:
        raise ConfigError("batch_size, draws_per_epoch and epochs must be positive")
    if draws % bs:
        raise ConfigError(f"draws_per_epoch {draws} is not a multiple of batch_size {bs}")
    if not smoke and bs != 12:
        warn.append(f"batch_size {bs} differs from the plan's 12")
    if cfg.get("record_type") not in RECORD_TYPES_TRAINED:
        raise ConfigError(f"record_type must be one of {RECORD_TYPES_TRAINED}")
    if cfg["record_type"] == "ablation" and not cfg.get("parent"):
        raise ConfigError("an ablation config needs a parent rung")
    if cfg["record_type"] == "root" and cfg.get("parent"):
        raise ConfigError("a root config has no parent (use record_type ablation or finalist)")
    if tr.get("intrinsics") not in ("none", "batch"):
        raise ConfigError("train.intrinsics must be 'none' or 'batch'")
    if cfg["eval"].get("camera_mode") not in ("none", "default-k", "gt-k"):
        raise ConfigError("eval.camera_mode must be none, default-k or gt-k")
    dw = cfg["data"].get("draw_weights") or {}
    if not any(float(v) > 0 for v in dw.values()):
        raise ConfigError("data.draw_weights has no positive weight")
    resolved_model_cfg(cfg["model"])       # raises on unknown model keys
    resolved_loss_cfg(cfg["loss"])         # raises on unknown loss keys
    return warn


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def dump_yaml(obj: dict, path: os.PathLike) -> None:
    import yaml
    text = yaml.safe_dump(_jsonable(obj), sort_keys=False, default_flow_style=False, width=120)
    atomic_write_text(path, text)


# ═════════════════════════════════════════════════════════════════════════════════════════
# atomic IO, seeding, logging
# ═════════════════════════════════════════════════════════════════════════════════════════

def atomic_write_text(path: os.PathLike, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(_jsonable(obj), indent=2, sort_keys=False) + "\n")


def atomic_torch_save(obj: Any, path: os.PathLike) -> None:
    """torch.save to a temp file in the same folder, fsync, then os.replace (never a torn file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:                                      # noqa: BLE001 - device count changed
            pass


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunLog:
    """Line logger to stdout and <run_dir>/train.log (append, flushed per line)."""

    def __init__(self, path: Optional[os.PathLike] = None):
        self.path = Path(path) if path else None

    def __call__(self, msg: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ═════════════════════════════════════════════════════════════════════════════════════════
# GPU lock
# ═════════════════════════════════════════════════════════════════════════════════════════

def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists on this machine."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        h = k32.OpenProcess(0x1000, False, pid)               # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5                # access denied: it exists
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
            return bool(ok) and code.value == 259              # STILL_ACTIVE
        finally:
            k32.CloseHandle(ctypes.c_void_p(h))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GpuLock:
    """Exclusive GPU lock file so two runs never overlap.

    ``acquire()`` creates ``path`` with O_CREAT|O_EXCL and writes ``{pid, token, owner, host,
    created}``. An existing lock whose pid is dead (same host) is stale and removed. A process whose
    env ``V2_GPU_LOCK_TOKEN`` equals the lock's token is a child of the holder and passes through
    without owning the lock. ``release()`` removes the file only if this object created it."""

    def __init__(self, path: os.PathLike = DEFAULT_GPU_LOCK, owner: str = "", wait_s: float = 0.0,
                 poll_s: float = 1.0):
        self.path = Path(path)
        self.owner = owner
        self.wait_s = float(wait_s)
        self.poll_s = float(poll_s)
        self.token: Optional[str] = None
        self.owned = False
        self.passthrough = False

    def read(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return {}

    def acquire(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        env_token = os.environ.get(LOCK_TOKEN_ENV)
        while True:
            token = uuid.uuid4().hex
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = self.read()
                if info is None:
                    continue                                   # removed meanwhile: retry
                if info == {}:                                 # being written: retry shortly
                    if time.monotonic() - t0 > max(self.wait_s, 5.0):
                        raise GpuLockBusy(f"GPU lock {self.path} is unreadable; delete it if no run is active")
                    time.sleep(0.05)
                    continue
                if env_token and info.get("token") == env_token:
                    self.token, self.passthrough = env_token, True
                    return self
                same_host = info.get("host") in (None, socket.gethostname())
                if same_host and not pid_alive(info.get("pid", -1)):
                    try:
                        self.path.unlink()
                        print(f"[gpu-lock] removed stale lock {self.path} (pid {info.get('pid')} is gone)",
                              file=sys.stderr, flush=True)
                    except OSError:
                        pass
                    continue
                if time.monotonic() - t0 >= self.wait_s:
                    raise GpuLockBusy(f"GPU lock {self.path} is held by pid {info.get('pid')} "
                                      f"({info.get('owner')}, since {info.get('created')})")
                time.sleep(self.poll_s)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "token": token, "owner": self.owner,
                           "host": socket.gethostname(), "created": utc_now()}, f)
            self.token, self.owned = token, True
            return self

    def env(self) -> Dict[str, str]:
        """Environment for child processes that may run under this lock."""
        return {LOCK_TOKEN_ENV: self.token} if self.token else {}

    def release(self) -> None:
        if not self.owned:
            return
        info = self.read()
        if info and info.get("token") == self.token:
            try:
                self.path.unlink()
            except OSError:
                pass
        self.owned = False

    def __enter__(self) -> "GpuLock":
        return self.acquire()

    def __exit__(self, *exc) -> bool:
        self.release()
        return False


# ═════════════════════════════════════════════════════════════════════════════════════════
# schedule and EMA
# ═════════════════════════════════════════════════════════════════════════════════════════

def warmup_steps_for(train_cfg: dict, total_steps: int) -> int:
    w = min(int(train_cfg.get("warmup_steps", 500)),
            int(float(train_cfg.get("warmup_frac_max", 0.05)) * total_steps))
    return max(1, w)


def lr_factor(step: int, total_steps: int, warmup_steps: int, start_factor: float = 0.05,
              min_ratio: float = 0.0) -> float:
    """LR multiplier at optimizer step ``step`` (0-based): linear from ``start_factor`` to 1 over
    ``warmup_steps``, then cosine from 1 to ``min_ratio`` at ``total_steps``."""
    if step < warmup_steps:
        return start_factor + (1.0 - start_factor) * step / float(warmup_steps)
    span = max(1, total_steps - warmup_steps)
    t = min(1.0, (step - warmup_steps) / float(span))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def _observer_names(model: nn.Module) -> set:
    try:
        from distill.v2.qat import observer_state_names
    except Exception:                                          # noqa: BLE001 - qat optional
        return set()
    return set(observer_state_names(model))


class ModelEMA:
    """EMA copy of a model. Float parameters and buffers are averaged; integer buffers and QAT
    observer state (``distill.v2.qat.observer_state_names``) are copied, never averaged."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.observer_keys = _observer_names(model)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        d = self.decay
        for k, v in self.module.state_dict().items():
            if not torch.is_tensor(v):
                continue
            src = msd[k].detach()
            if k in self.observer_keys or not v.dtype.is_floating_point:
                v.copy_(src)
            else:
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd, strict=True)


# ═════════════════════════════════════════════════════════════════════════════════════════
# data
# ═════════════════════════════════════════════════════════════════════════════════════════

_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SyntheticV2Dataset(torch.utils.data.Dataset):
    """Deterministic random rows following the V2Dataset item contract (smoke tests only).

    Item i depends only on (seed, split, i): smooth positive depth in roughly [0.4, 9] m, an RGB
    image correlated with it, ~95% valid pixels, dmax 10, sources alternating own/nyu."""

    def __init__(self, n: int = 16, hw: Sequence[int] = (64, 96), split: str = "train", seed: int = 0,
                 sources: Sequence[str] = ("own", "nyu"), intrinsics: bool = False):
        self.n, self.hw, self.split, self.seed = int(n), (int(hw[0]), int(hw[1])), split, int(seed)
        self.sources = [sources[i % len(sources)] for i in range(self.n)]
        self.ids = [f"syn_{split}_{i:04d}" for i in range(self.n)]
        self.intrinsics = bool(intrinsics)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        h, w = self.hw
        rng = np.random.default_rng([self.seed, abs(hash(self.split)) % (2 ** 31), int(i)])
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = rng.uniform(0.3, 1.6)
        logd = base + rng.uniform(-0.6, 0.6) * (yy / h) + rng.uniform(-0.4, 0.4) * (xx / w)
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        blob = np.exp(-(((xx - cx) / (0.15 * w)) ** 2 + ((yy - cy) / (0.15 * h)) ** 2))
        logd = (logd - 0.8 * blob).astype(np.float32)
        rgb = np.stack([np.clip(0.5 + 0.3 * np.sin(logd * (k + 2)) + 0.05 * rng.standard_normal((h, w)), 0, 1)
                        for k in range(3)], axis=-1).astype(np.float32)
        img = ((rgb - _IMNET_MEAN) / _IMNET_STD).transpose(2, 0, 1)
        valid = rng.uniform(size=(h, w)) > 0.05
        src = self.sources[i]
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(img)),
            "target_log": torch.from_numpy(logd[None]),
            "valid": torch.from_numpy(valid[None]),
            "is_gt": src == "nyu",
            "dmax": 10.0,
            "source": src,
            "id": self.ids[i],
        }
        if self.intrinsics:
            item["intrinsics"] = torch.tensor([0.8, 1.08, 0.5, 0.5], dtype=torch.float32)
        return item


def _group_members(sources: Sequence[str], key: str) -> List[int]:
    names = {s.strip() for s in str(key).split("+") if s.strip()}
    return [i for i, s in enumerate(sources) if s in names]


def simple_weighted_indices(sources: Sequence[str], draw_weights: Dict[str, float], n_draws: int,
                            seed: int) -> List[int]:
    """Fallback sampler (synthetic smoke data): pick a group with probability proportional to its
    weight ("own+x_indoor" = the union of those sources), then a row uniformly inside it."""
    groups = [(k, float(v), _group_members(sources, k)) for k, v in draw_weights.items() if float(v) > 0]
    groups = [g for g in groups if g[2]]
    if not groups:
        raise ConfigError(f"no rows for draw_weights {draw_weights}")
    p = np.array([g[1] for g in groups], dtype=np.float64)
    p /= p.sum()
    rng = np.random.default_rng(int(seed))
    gi = rng.choice(len(groups), size=int(n_draws), p=p)
    out = []
    for g in gi:
        members = groups[g][2]
        out.append(int(members[int(rng.integers(0, len(members)))]))
    return out


def _import_data():
    try:
        from distill.v2 import data as data_mod                # noqa: WPS433
    except ImportError as e:
        raise ConfigError(f"distill/v2/data.py is not importable ({e}); real-data training needs it "
                          "(use --smoke for synthetic data)") from e
    return data_mod


def positive_weights(draw_weights: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in (draw_weights or {}).items() if float(v) > 0}


def build_train_dataset(cfg: dict, synthetic: bool):
    d = cfg["data"]
    if synthetic:
        hw = d["dataset"].get("hw", (64, 96))
        return SyntheticV2Dataset(n=24, hw=hw, split="train", seed=int(cfg["seed"]),
                                  intrinsics=bool(d["dataset"].get("intrinsics", False)))
    data_mod = _import_data()
    return data_mod.V2Dataset(str(_repo_path(d["roles"])), d.get("train_split", "train"), dict(d["dataset"]))


def build_dev_dataset(cfg: dict, synthetic: bool):
    e = cfg["eval"]
    if synthetic:
        hw = e["dataset"].get("hw", cfg["data"]["dataset"].get("hw", (64, 96)))
        return SyntheticV2Dataset(n=6, hw=hw, split="dev", seed=12345,
                                  intrinsics=bool(e["dataset"].get("intrinsics", False)))
    data_mod = _import_data()
    return data_mod.V2Dataset(str(_repo_path(cfg["data"]["roles"])), e.get("split", "dev"), dict(e["dataset"]))


def epoch_indices(ds, cfg: dict, epoch: int, synthetic: bool) -> List[int]:
    """The deterministic draw order of one epoch (seed*1000 + epoch), as a list of row indices."""
    tr = cfg["train"]
    n_draws = int(tr["draws_per_epoch"])
    seed = int(cfg["seed"]) * 1000 + int(epoch)
    weights = positive_weights(cfg["data"]["draw_weights"])
    if synthetic:
        return simple_weighted_indices(ds.sources, weights, n_draws, seed)
    data_mod = _import_data()
    sampler = data_mod.make_sampler(ds, weights, n_draws, seed)
    idx = [int(i) for i in iter(sampler)]
    if len(idx) != n_draws:
        raise ConfigError(f"make_sampler returned {len(idx)} draws, expected {n_draws}")
    return idx


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def _repo_path(p: os.PathLike) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def load_roles(path: os.PathLike) -> Optional[dict]:
    p = _repo_path(path)
    if not p.is_file():
        return None
    import yaml
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(p, encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def cluster_labels(ids: Sequence[str], roles: Optional[dict]) -> Optional[List[str]]:
    """Cluster label per id from roles.yaml dev.d_sun_clusters (list of id lists; ids not listed
    are singletons). None when the roles carry no clusters (image-level bootstrap)."""
    clusters = ((roles or {}).get("dev") or {}).get("d_sun_clusters")
    if not clusters:
        return None
    lab = {}
    for ci, members in enumerate(clusters):
        for m in members:
            lab[str(m)] = f"c{ci}"
    return [lab.get(str(i), f"s:{i}") for i in ids]


# ═════════════════════════════════════════════════════════════════════════════════════════
# D_SUN dev evaluation
# ═════════════════════════════════════════════════════════════════════════════════════════

def per_image_dev_metrics(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> Optional[dict]:
    """Per-image dev metrics on (H,W) metres. Definitions are reused, not redefined:
    d1 = protocol._delta, d1_ssi = protocol.ssi_align then _delta, bf1_si_v2 =
    protocol_v2.boundary_f1_si_v2 (corrected), b_shift = protocol_v2.per_image_log_shift."""
    from benchmark import protocol, protocol_v2
    m = valid.astype(bool) & np.isfinite(gt) & (gt > 0)
    if not m.any():
        return None
    g = gt[m].astype(np.float64)
    p = np.clip(pred[m].astype(np.float64), 1e-6, None)
    pa = np.clip(protocol.ssi_align(g, p), 1e-6, None)
    pred_full = np.clip(np.nan_to_num(pred.astype(np.float64), nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6, None)
    return {
        "d1": protocol._delta(g, p, 1.0),
        "arel": float((np.abs(g - p) / g).mean()),
        "rmse": float(np.sqrt(((g - p) ** 2).mean())),
        "d1_ssi": protocol._delta(g, pa, 1.0),
        protocol_v2.BF_KEY: protocol_v2.boundary_f1_si_v2(np.where(m, gt, 0.0).astype(np.float64),
                                                          pred_full, m),
        "b_shift": protocol_v2.per_image_log_shift(gt, pred, m),
        "n_valid": int(m.sum()),
    }


def summarize_dev(rows: Sequence[dict], nan_free: bool = True, camera_mode: str = "none") -> dict:
    from benchmark import protocol_v2
    if not rows:
        return {"n_images": 0, "nan_free": bool(nan_free), "camera_mode": camera_mode}
    bf = protocol_v2.aggregate_boundary_f1([r.get(protocol_v2.BF_KEY) for r in rows])
    sc = protocol_v2.dataset_scale_summary([r.get("b_shift") for r in rows])
    return {
        "n_images": len(rows),
        "D1": 100.0 * float(np.mean([r["d1"] for r in rows])),
        "AbsRel": float(np.mean([r["arel"] for r in rows])),
        "RMSE": float(np.mean([r["rmse"] for r in rows])),
        "d1_ssi": 100.0 * float(np.mean([r["d1_ssi"] for r in rows])),
        "BF": bf["bf1"], "BF_n_contributing": bf["n_contributing"],
        "BF_n_no_gt_boundary": bf["n_no_gt_boundary"],
        "b_mean": sc["b_mean"], "b_std": sc["b_std"],
        "nan_free": bool(nan_free), "camera_mode": camera_mode,
        "metric_defs": {"D1": "100*mean raw delta1 (protocol._delta)", "BF": "protocol_v2.boundary_f1_si_v2",
                        "d1_ssi": "100*mean delta1 after protocol.ssi_align", "b": "protocol_v2 per_image_log_shift"},
    }


@torch.no_grad()
def evaluate_dev(model: nn.Module, ds, cfg: dict, device: torch.device, log: Callable = print) -> dict:
    """Final-EMA evaluation on the dev split. Returns ``{"rows", "summary", "seconds"}``."""
    t0 = time.time()
    e, rt = cfg["eval"], cfg["runtime"]
    cam = e.get("camera_mode", "none")
    loader = torch.utils.data.DataLoader(ds, batch_size=int(rt.get("eval_batch_size", 8)), shuffle=False,
                                         num_workers=int(rt.get("num_workers", 0)),
                                         pin_memory=device.type == "cuda")
    model.eval()
    lmin, lmax = float(model.log_min), float(model.log_max)
    min_d = float(e.get("min_depth_m", 0.2))
    rows: List[dict] = []
    futures = []
    nan_free = True
    n_nonfinite = 0
    with ThreadPoolExecutor(max_workers=max(1, int(rt.get("metric_workers", 4)))) as ex:
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            intr = None
            if cam == "gt-k":
                if "intrinsics" not in batch:
                    raise ConfigError("eval.camera_mode gt-k needs items with 'intrinsics'")
                intr = batch["intrinsics"].to(device)
            out = model(img, intr)
            out = out[:, :1].float()
            fin = torch.isfinite(out)
            if not bool(fin.all()):
                nan_free = False
                n_nonfinite += int((~fin).sum())
            pred = torch.exp(torch.nan_to_num(out, nan=lmin).clamp(lmin, lmax)).cpu().numpy()
            tlog = batch["target_log"].float()
            valid = batch["valid"].bool() & torch.isfinite(tlog)
            gt = torch.exp(torch.where(valid, tlog, torch.zeros_like(tlog))).numpy()
            dmax = batch.get("dmax")
            vnp = valid.numpy()
            for k in range(img.shape[0]):
                dm = float(dmax[k]) if dmax is not None else float("inf")
                v = vnp[k, 0] & (gt[k, 0] >= min_d) & (gt[k, 0] <= dm)
                futures.append((batch["id"][k], batch["source"][k],
                                ex.submit(per_image_dev_metrics, gt[k, 0], pred[k, 0], v)))
        for sid, src, fut in futures:
            r = fut.result()
            if r is None:
                continue
            r = {"id": str(sid), "source": str(src), **r}
            rows.append(r)
    summary = summarize_dev(rows, nan_free=nan_free, camera_mode=cam)
    summary["n_nonfinite_pred_pixels"] = n_nonfinite
    summary["n_dataset_rows"] = len(ds)
    dt = time.time() - t0
    log(f"[eval] dev n={summary['n_images']} D1={summary.get('D1', float('nan')):.2f} "
        f"AbsRel={summary.get('AbsRel', float('nan')):.4f} BF={summary.get('BF')} "
        f"b_std={summary.get('b_std')} ({dt:.0f}s)")
    return {"rows": rows, "summary": summary, "seconds": dt}


def compare_to_parent(rows: Sequence[dict], parent_rows: Sequence[dict], roles: Optional[dict],
                      metrics: Sequence[str], n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired (cluster) bootstrap of child - parent per metric, rows joined on image id."""
    from benchmark.protocol_v2 import pair_rows_by_id, paired_bootstrap_clustered
    out: Dict[str, Any] = {}
    try:
        ids, a_rows, b_rows, info = pair_rows_by_id(parent_rows, rows)
    except ValueError as e:
        ids, a_rows, b_rows, info = pair_rows_by_id(parent_rows, rows, allow_intersection=True)
        out["pairing_warning"] = str(e)
    out["pairing"] = {k: (v if not isinstance(v, list) else len(v)) for k, v in info.items()}
    labels = cluster_labels(ids, roles)
    out["resampling"] = "cluster" if labels is not None else "image"
    for m in metrics:
        a = [np.nan if r.get(m) is None else float(r[m]) for r in a_rows]
        b = [np.nan if r.get(m) is None else float(r[m]) for r in b_rows]
        try:
            res = paired_bootstrap_clustered(a, b, labels, n_boot=n_boot, seed=seed)
        except ValueError as e:
            out[m] = {"error": str(e)}
            continue
        if m in ("d1", "d1_ssi"):
            res["diff_points"] = 100.0 * res["diff"]
            res["ci_points"] = [100.0 * res["ci_lo"], 100.0 * res["ci_hi"]]
        out[m] = res
    return out


# ═════════════════════════════════════════════════════════════════════════════════════════
# registry
# ═════════════════════════════════════════════════════════════════════════════════════════

def train_sources(cfg: dict) -> List[str]:
    srcs = set()
    for k in positive_weights(cfg["data"]["draw_weights"]):
        srcs.update(s.strip() for s in k.split("+") if s.strip())
    return sorted(srcs)


def teachers_for(cfg: dict) -> List[str]:
    labels = str(cfg["data"]["dataset"].get("labels", "v2l"))
    t = list(TEACHER_IDS.get(labels, [f"unknown:{labels}"]))
    if "nyu" in train_sources(cfg):
        t.append("sensor:nyu")
    return t


def _rel(p: os.PathLike) -> str:
    from benchmark.provenance import repo_relpath
    return repo_relpath(p)


def build_trained_record(cfg: dict, run: dict, run_dir: Path, parent_run: Optional[dict],
                         smoke: bool) -> dict:
    from benchmark.provenance import config_hash, sha256_file
    comp = comparable_view(cfg)
    rtype = "root" if smoke else cfg["record_type"]
    rec: Dict[str, Any] = {
        "type": rtype, "run_id": run["run_id"], "rung": cfg.get("rung"),
        "parent_rung": cfg.get("parent"),
        "factor_changed": cfg.get("factor_changed"),
        "config_hash": config_hash(comp), "config_resolved": comp,
        "code_tree_hash": run.get("code_tree_hash"), "seed": int(cfg["seed"]),
        "draw_weights": positive_weights(cfg["data"]["draw_weights"]),
        "train_sources": train_sources(cfg), "teachers": teachers_for(cfg),
        "checkpoint_path": _rel(run_dir / "final_ema.pt"),
        "checkpoint_sha256": run["checkpoint_sha256"], "checkpoint_kind": "final_ema",
        "selection": "none", "camera_mode": cfg["eval"]["camera_mode"],
        "train_camera_mode": run.get("train_camera_mode"),
        "status": "smoke" if smoke else "trained",
        "metrics_dev": run.get("dev_summary"),
        "params": run.get("params"), "epochs": int(cfg["train"]["epochs"]),
        "manifest_file": _rel(run_dir / "manifest.json"),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "split_hashes": run.get("split_hashes"),
    }
    if (run_dir / "dev_eval.json").is_file():
        rec["results"] = [{"dataset": "d_sun", "split": "dev", "result_file": _rel(run_dir / "dev_eval.json"),
                           "result_sha256": sha256_file(run_dir / "dev_eval.json")}]
    if smoke:
        rec["smoke_of"] = {"rung": cfg.get("rung"), "record_type": cfg.get("record_type"),
                           "parent_run": parent_run.get("run_id") if parent_run else None}
    elif rtype == "ablation":
        if not parent_run:
            raise ConfigError("ablation run has no parent run: pass --parent-run")
        rec["parent"] = parent_run["run_id"]
        rec["factor_keys"] = run.get("one_factor", {}).get("diff") or list(cfg.get("factor_keys") or [])
    elif rtype == "finalist":
        rec["components"] = list(cfg.get("components") or [])
        if parent_run:
            rec["parent_run_id"] = parent_run["run_id"]
        rec["factor_keys"] = run.get("one_factor", {}).get("diff") or list(cfg.get("factor_keys") or [])
    return rec


def register_run(cfg: dict, run: dict, run_dir: Path, parent_run: Optional[dict], smoke: bool,
                 registry_path: Optional[os.PathLike]) -> dict:
    """Append the trained record and the dev eval record. Returns {"trained": rec, "eval": rec}."""
    from benchmark.registry_v2 import append_record          # guarded by the caller
    rec = build_trained_record(cfg, run, run_dir, parent_run, smoke)
    stored = append_record(rec, registry_path)
    out = {"trained": {"id": stored["id"], "record_hash": stored["record_hash"], "seq": stored["seq"]}}
    if (run_dir / "dev_eval.json").is_file():
        ev = {"type": "eval", "model_ref": stored["id"], "dataset": "d_sun", "variant": "dev",
              "split": "dev", "camera_mode": cfg["eval"]["camera_mode"],
              "result_file": _rel(run_dir / "dev_eval.json"), "artefact": "pytorch-final-ema",
              "artefact_path": _rel(run_dir / "final_ema.pt"),
              "artefact_sha256": run["checkpoint_sha256"],
              "status": "smoke" if smoke else "registered",
              "note": "development result (D_SUN); never zero-shot (plan 5.1)"}
        es = append_record(ev, registry_path)
        out["eval"] = {"id": es["id"], "record_hash": es["record_hash"], "seq": es["seq"]}
    return out


# ═════════════════════════════════════════════════════════════════════════════════════════
# run folder helpers
# ═════════════════════════════════════════════════════════════════════════════════════════

def new_run_id(cfg: dict, smoke: bool = False) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        h = comparable_hash(cfg)[:8]
    except Exception:                                          # noqa: BLE001
        h = uuid.uuid4().hex[:8]
    rung = str(cfg.get("rung") or "run").replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{'smoke-' if smoke else ''}{rung}-{stamp}-{h}"


def create_run_dir(runs_root: Path, run_id: str) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    d = runs_root / run_id
    d.mkdir(parents=False, exist_ok=False)                     # unique run folder or failure
    return d


def read_run(run_dir: os.PathLike) -> dict:
    p = Path(run_dir) / "run.json"
    if not p.is_file():
        raise FileNotFoundError(f"no run.json in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def find_parent_run(runs_root: Path, rung: str) -> Optional[Path]:
    """Latest COMPLETE (non-smoke) run of ``rung`` under ``runs_root``."""
    best = None
    if not runs_root.is_dir():
        return None
    for d in runs_root.iterdir():
        try:
            r = read_run(d)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if r.get("rung") == rung and r.get("status") == "complete" and not r.get("smoke"):
            if best is None or r.get("finished_utc", "") > best[0]:
                best = (r.get("finished_utc", ""), d)
    return best[1] if best else None


def resolve_parent_run(arg: Optional[str], runs_root: Path) -> Optional[Path]:
    if not arg:
        return None
    p = Path(arg)
    if p.is_dir():
        return p
    q = runs_root / arg
    if q.is_dir():
        return q
    raise ConfigError(f"--parent-run {arg!r} is neither a run folder nor a run id under {runs_root}")


def apply_smoke(cfg: dict, synthetic: bool, device: Optional[str]) -> dict:
    """Tiny CPU settings: 3 optimizer steps, batch 2, no AMP, no pretrained download, small images."""
    c = copy.deepcopy(cfg)
    c["model"]["pretrained"] = False
    c["train"].update({"epochs": 1, "batch_size": 2, "draws_per_epoch": 6, "amp": False,
                       "warmup_steps": 1})
    c["runtime"].update({"device": device or "cpu", "num_workers": 0, "pin_memory": False,
                         "log_every": 1, "save_every_steps": 1, "eval_batch_size": 3, "metric_workers": 2,
                         "max_epoch_seconds": None})
    c["eval"]["n_boot"] = 200
    if synthetic:
        hw = [64, 96]
        c["data"]["dataset"] = dict(c["data"]["dataset"], hw=hw)
        c["eval"]["dataset"] = dict(c["eval"]["dataset"], hw=hw)
        dw = positive_weights(c["data"]["draw_weights"])
        known = {"own", "nyu"}
        if not any(set(k.split("+")) & known for k in dw):
            c["data"]["draw_weights"] = {"own": 1.0}
    return c


# ═════════════════════════════════════════════════════════════════════════════════════════
# trainer
# ═════════════════════════════════════════════════════════════════════════════════════════

class TrainingFailed(RuntimeError):
    pass


class EpochTooSlow(RuntimeError):
    pass


class Trainer:
    """One run: build, train (resumable), final EMA, dev eval, paired comparison, registry."""

    def __init__(self, cfg: dict, run_dir: Path, run: dict, *, smoke: bool, synthetic: bool,
                 parent_dir: Optional[Path], registry_path: Optional[os.PathLike], register: bool,
                 stop_after_steps: Optional[int] = None, ignore_epoch_limit: bool = False):
        self.cfg, self.run_dir, self.run = cfg, run_dir, run
        self.smoke, self.synthetic = smoke, synthetic
        self.parent_dir = parent_dir
        self.registry_path = registry_path
        self.register = register
        self.stop_after_steps = stop_after_steps
        self.ignore_epoch_limit = ignore_epoch_limit
        self.log = RunLog(run_dir / "train.log")
        dev = str(cfg["runtime"].get("device", "cuda"))
        if dev.startswith("cuda") and not torch.cuda.is_available():
            raise ConfigError("runtime.device is cuda but CUDA is not available (pass --device cpu)")
        self.device = torch.device(dev)

    # -- persistence -----------------------------------------------------------------------
    def save_run(self, **fields) -> None:
        self.run.update(fields)
        self.run["updated_utc"] = utc_now()
        atomic_write_json(self.run_dir / "run.json", self.run)

    # -- build -------------------------------------------------------------------------------
    def build(self):
        from distill.v2.losses import V2Loss
        from distill.v2.model import build_v2_model, count_params
        cfg = self.cfg
        tr = cfg["train"]
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(cfg["runtime"].get("cudnn_benchmark", True))
        seed_everything(int(cfg["seed"]))
        self.model = build_v2_model(dict(cfg["model"])).to(self.device)
        self.model_cfg = _jsonable(dict(self.model.v2_config))
        self.criterion = V2Loss(dict(cfg["loss"]))
        self.needs_aux = bool(self.criterion.cfg.get("decoupled_scale"))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(tr["lr"]),
                                     weight_decay=float(tr["weight_decay"]),
                                     betas=tuple(float(b) for b in tr.get("betas", (0.9, 0.999))))
        self.steps_per_epoch = int(tr["draws_per_epoch"]) // int(tr["batch_size"])
        self.total_steps = self.steps_per_epoch * int(tr["epochs"])
        self.warmup = warmup_steps_for(tr, self.total_steps)
        sf, mr = float(tr.get("warmup_start_factor", 0.05)), float(tr.get("min_lr_ratio", 0.0))
        total, warm = self.total_steps, self.warmup
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, total, warm, sf, mr))
        self.use_amp = bool(tr.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = ModelEMA(self.model, float(tr.get("ema_decay", 0.999)))
        self.state = {"step": 0, "epoch": 0, "step_in_epoch": 0, "nonfinite": 0, "consecutive_nonfinite": 0,
                      "history": [], "epoch_seconds": [], "train_camera_mode": None}
        self.params = int(count_params(self.model))

    def save_last(self) -> None:
        ck = {
            "model": self.model.state_dict(), "ema": self.ema.state_dict(),
            "optim": self.opt.state_dict(), "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(), "rng": rng_state(),
            "step": self.state["step"], "epoch": self.state["epoch"],
            "step_in_epoch": self.state["step_in_epoch"], "state": self.state,
            "comparable_hash": self.run["comparable_hash"], "run_id": self.run["run_id"],
            "model_cfg": self.model_cfg, "cfg": self.cfg, "kind": "last",
        }
        atomic_torch_save(ck, self.run_dir / "last.pt")

    def load_last(self) -> bool:
        p = self.run_dir / "last.pt"
        if not p.is_file():
            return False
        ck = torch.load(p, map_location=self.device, weights_only=False)
        if ck.get("comparable_hash") != self.run["comparable_hash"]:
            raise ConfigError("last.pt was written with a different config; refusing to resume")
        self.model.load_state_dict(ck["model"], strict=True)
        self.ema.load_state_dict(ck["ema"])
        self.opt.load_state_dict(ck["optim"])
        self.sched.load_state_dict(ck["sched"])
        if self.scaler.is_enabled() and ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        self.state = dict(ck["state"])
        self.log(f"[resume] step {self.state['step']} (epoch {self.state['epoch']}, "
                 f"step_in_epoch {self.state['step_in_epoch']})")
        return True

    # -- training ---------------------------------------------------------------------------
    def _loader(self, ds, indices: List[int], epoch: int):
        rt = self.cfg["runtime"]
        g = torch.Generator()
        g.manual_seed(int(self.cfg["seed"]) * 1000 + int(epoch))
        nw = int(rt.get("num_workers", 0))
        return torch.utils.data.DataLoader(
            ds, batch_size=int(self.cfg["train"]["batch_size"]), sampler=indices, drop_last=True,
            num_workers=nw, pin_memory=bool(rt.get("pin_memory", True)) and self.device.type == "cuda",
            worker_init_fn=_seed_worker if nw > 0 else None, generator=g, persistent_workers=False)

    def train_step(self, batch: dict) -> Tuple[bool, dict]:
        cfg = self.cfg
        img = batch["image"].to(self.device, non_blocking=True)
        b = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        intr = None
        if cfg["train"].get("intrinsics") == "batch" and "intrinsics" in b:
            intr = b["intrinsics"].float()
            self.state["train_camera_mode"] = "batch-k"
        elif self.state["train_camera_mode"] is None:
            self.state["train_camera_mode"] = "default-k" if cfg["model"].get("variant") == "legacy" else "none"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            if self.needs_aux:
                parts = self.model.forward_parts(img)
                pred, aux = parts["log_depth"], parts
            else:
                pred, aux = self.model(img, intr), None
        pred = pred[:, :1]
        loss, terms = self.criterion(pred, b, aux) if aux is not None else self.criterion(pred, b)
        if not torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            return False, terms
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(cfg["train"]["grad_clip"]))
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.ema.update(self.model)
        terms = dict(terms)
        terms["grad_norm"] = float(gn)
        return True, terms

    def fit(self, ds) -> str:
        """Train until the last step. Returns 'done' or 'interrupted' (stop_after_steps)."""
        cfg, rt = self.cfg, self.cfg["runtime"]
        epochs = int(cfg["train"]["epochs"])
        bs = int(cfg["train"]["batch_size"])
        session_steps = 0
        max_ep_s = rt.get("max_epoch_seconds")
        max_bad = int(cfg["train"].get("max_nonfinite_steps", 20))
        while self.state["epoch"] < epochs:
            ep = self.state["epoch"]
            idx = epoch_indices(ds, cfg, ep, self.synthetic)
            start = self.state["step_in_epoch"]
            fresh_epoch = start == 0
            loader = self._loader(ds, idx[start * bs:], ep)
            self.model.train()
            t_ep = time.time()
            t_wait = 0.0
            t_last = time.time()
            running: Dict[str, float] = {}
            n_run = 0
            for batch in loader:
                t_wait += time.time() - t_last
                ok, terms = self.train_step(batch)
                self.sched.step()
                self.state["step"] += 1
                self.state["step_in_epoch"] += 1
                session_steps += 1
                if not ok:
                    self.state["nonfinite"] += 1
                    self.state["consecutive_nonfinite"] += 1
                    self.log(f"[warn] non-finite loss at step {self.state['step']} "
                             f"({self.state['nonfinite']} total)")
                    if self.state["consecutive_nonfinite"] > max_bad or self.state["nonfinite"] > 5 * max_bad:
                        self.save_last()
                        raise TrainingFailed(f"non-finite loss on {self.state['nonfinite']} steps "
                                             f"({self.state['consecutive_nonfinite']} consecutive)")
                else:
                    self.state["consecutive_nonfinite"] = 0
                    for k, v in terms.items():
                        running[k] = running.get(k, 0.0) + float(v)
                    n_run += 1
                step = self.state["step"]
                if step % int(rt.get("log_every", 50)) == 0 and n_run:
                    avg = {k: round(v / n_run, 4) for k, v in running.items()}
                    self.log(f"  e{ep} step {step}/{self.total_steps} lr={self.sched.get_last_lr()[0]:.2e} {avg}")
                if step % int(rt.get("save_every_steps", 250)) == 0 and self.state["step_in_epoch"] < self.steps_per_epoch:
                    self.save_last()
                if self.stop_after_steps is not None and session_steps >= self.stop_after_steps \
                        and self.state["step_in_epoch"] < self.steps_per_epoch:
                    self.save_last()
                    self.log(f"[stop] stop_after_steps={self.stop_after_steps} reached at step {step}")
                    return "interrupted"
                t_last = time.time()
            if self.state["step_in_epoch"] != self.steps_per_epoch:
                raise TrainingFailed(f"epoch {ep} ended after {self.state['step_in_epoch']} of "
                                     f"{self.steps_per_epoch} steps (sampler/loader too short)")
            dt = time.time() - t_ep
            rec = {"epoch": ep, "seconds": round(dt, 2), "fresh": fresh_epoch,
                   "data_wait_frac": round(t_wait / dt, 4) if dt > 0 else None,
                   "loss": {k: round(v / max(1, n_run), 5) for k, v in running.items()},
                   "lr_end": self.sched.get_last_lr()[0]}
            self.state["history"].append(rec)
            if fresh_epoch:
                self.state["epoch_seconds"].append(round(dt, 2))
            self.state["epoch"] += 1
            self.state["step_in_epoch"] = 0
            self.save_last()
            self.save_run(progress={"epoch": self.state["epoch"], "step": self.state["step"]},
                          epoch_seconds=self.state["epoch_seconds"])
            self.log(f"[epoch {ep}] {dt:.0f}s data-wait {rec['data_wait_frac']} loss {rec['loss'].get('total')}")
            if self.stop_after_steps is not None and session_steps >= self.stop_after_steps \
                    and self.state["epoch"] < epochs:
                self.log(f"[stop] stop_after_steps={self.stop_after_steps} reached at epoch end")
                return "interrupted"
            if fresh_epoch and max_ep_s and not self.ignore_epoch_limit and dt > float(max_ep_s) \
                    and self.state["epoch"] < epochs:
                raise EpochTooSlow(f"epoch {ep} took {dt:.0f}s > {max_ep_s}s (plan 7.1: stop and diagnose)")
        return "done"

    def timing(self, ds, n_steps: int) -> dict:
        """T1-00: n_steps real optimizer steps, steady-state seconds per step and projected epoch."""
        cfg = self.cfg
        bs = int(cfg["train"]["batch_size"])
        idx = epoch_indices(ds, cfg, 0, self.synthetic)
        need = n_steps * bs
        while len(idx) < need:
            idx = idx + idx
        loader = self._loader(ds, idx[:need], 0)
        self.model.train()
        warm = min(20, max(1, n_steps // 5))
        times, waits = [], []
        t_last = time.time()
        for i, batch in enumerate(loader):
            t_w = time.time() - t_last
            t0 = time.time()
            self.train_step(batch)
            self.sched.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            if i >= warm:
                times.append(t1 - t_last)
                waits.append(t_w)
            t_last = t1
            _ = t0
        if not times:
            raise ConfigError("timing run too short to measure")
        step_s = float(np.mean(times))
        out = {"n_steps": n_steps, "warmup_steps_excluded": warm, "seconds_per_step": step_s,
               "seconds_per_step_median": float(np.median(times)),
               "data_wait_frac": float(np.sum(waits) / np.sum(times)),
               "steps_per_epoch": self.steps_per_epoch,
               "epoch_seconds": step_s * self.steps_per_epoch,
               "batch_size": bs, "device": str(self.device),
               "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if self.device.type == "cuda" else None}
        return out


# ═════════════════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════════════════

def _parse(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="V2 (CoDeQ) trainer: one ladder rung per run")
    ap.add_argument("--config", default="", help="rung YAML (extends chain resolved)")
    ap.add_argument("--resume", default="", help="run folder to continue (uses its config.yaml)")
    ap.add_argument("--run-id", default="", help="explicit unique run id (the ladder passes one)")
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ap.add_argument("--parent-run", default="", help="parent run folder or run id (paired bootstrap, registry parent)")
    ap.add_argument("--smoke", action="store_true", help="3 CPU steps, tiny images, smoke registry")
    ap.add_argument("--smoke-data", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--timing-steps", type=int, default=0, help="T1-00: time N steps, no checkpoint/eval")
    ap.add_argument("--device", default="", help="override runtime.device")
    ap.add_argument("--registry", default="", help="registry path (default benchmark/registry_v2.jsonl)")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--gpu-lock", default=str(DEFAULT_GPU_LOCK))
    ap.add_argument("--stop-after-steps", type=int, default=0, help="testing: save last.pt and exit after N steps")
    ap.add_argument("--ignore-epoch-limit", action="store_true")
    return ap.parse_args(argv)


def _prepare(args) -> Tuple[dict, Path, dict, bool]:
    """Returns (cfg, run_dir, run, resumed)."""
    runs_root = Path(args.runs_root)
    if args.resume:
        run_dir = Path(args.resume)
        if not (run_dir / "config.yaml").is_file():
            raise ConfigError(f"--resume {run_dir}: no config.yaml")
        cfg = load_config(run_dir / "config.yaml")
        if args.config:
            other = load_config(args.config)
            if args.smoke:
                other = apply_smoke(other, args.smoke_data == "synthetic", args.device or None)
            if comparable_hash(other) != comparable_hash(cfg):
                raise ConfigError("--config differs from the run's config.yaml; resume uses the saved config")
        run = read_run(run_dir)
        if args.device:
            cfg["runtime"]["device"] = args.device
        return cfg, run_dir, run, True
    if not args.config:
        raise ConfigError("--config or --resume is required")
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg, args.smoke_data == "synthetic", args.device or None)
    elif args.device:
        cfg["runtime"]["device"] = args.device
    run_id = args.run_id or new_run_id(cfg, smoke=args.smoke)
    if args.timing_steps and not args.run_id:
        run_id = "timing-" + run_id
    run_dir = create_run_dir(runs_root, run_id)
    dump_yaml(cfg, run_dir / "config.yaml")
    run = {"schema": "v2-run/1", "run_id": run_id, "rung": cfg.get("rung"), "parent_rung": cfg.get("parent"),
           "record_type": cfg.get("record_type"), "status": "created", "smoke": bool(args.smoke),
           "smoke_data": args.smoke_data if args.smoke else None, "created_utc": utc_now(),
           "comparable_hash": comparable_hash(cfg), "config_file": "config.yaml",
           "source_config": str(args.config), "stages": {}}
    atomic_write_json(run_dir / "run.json", run)
    return cfg, run_dir, run, False


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    try:
        cfg, run_dir, run, resumed = _prepare(args)
    except (ConfigError, FileExistsError, FileNotFoundError) as e:
        print(f"[train] config error: {e}", file=sys.stderr, flush=True)
        return EXIT_CONFIG
    smoke = bool(run.get("smoke"))
    synthetic = smoke and run.get("smoke_data", "synthetic") == "synthetic"
    log = RunLog(run_dir / "train.log")
    try:
        warnings = validate_config(cfg, smoke=smoke)
    except ConfigError as e:
        log(f"[train] config error: {e}")
        run.update(status="failed", error=f"config: {e}")
        atomic_write_json(run_dir / "run.json", run)
        return EXIT_CONFIG
    for w in warnings:
        log(f"[warn] {w}")

    # parent run + one-factor check (belt and braces: the ladder checks before launching)
    runs_root = Path(args.runs_root)
    parent_dir: Optional[Path] = None
    parent_run: Optional[dict] = None
    try:
        parent_dir = resolve_parent_run(args.parent_run or run.get("parent_run_dir"), runs_root)
        if parent_dir is None and cfg.get("parent") and not smoke and not args.timing_steps:
            parent_dir = find_parent_run(runs_root, str(cfg["parent"]))
        if parent_dir is not None:
            parent_run = read_run(parent_dir)
            pcfg = load_config(parent_dir / "config.yaml")
            if not smoke and cfg["record_type"] in ("ablation", "finalist"):
                of = check_one_factor(pcfg, cfg)
                run["one_factor"] = of
                if not of["ok"]:
                    raise ConfigError("one-factor rule violated vs parent "
                                      f"{parent_run['run_id']}: {of['problems']}")
            run["parent_run_id"] = parent_run["run_id"]
            run["parent_run_dir"] = str(parent_dir)
    except (ConfigError, FileNotFoundError) as e:
        log(f"[train] {e}")
        run.update(status="failed", error=str(e))
        atomic_write_json(run_dir / "run.json", run)
        return EXIT_CONFIG

    device = str(cfg["runtime"].get("device", "cuda"))
    lock = None
    if device.startswith("cuda") and not args.no_gpu_lock:
        lock = GpuLock(args.gpu_lock, owner=f"distill.v2.train {run['run_id']}")
        try:
            lock.acquire()
        except GpuLockBusy as e:
            log(f"[train] {e}")
            return EXIT_LOCK
    registry_path = args.registry or None
    if smoke and not registry_path:
        registry_path = str(run_dir / "registry_smoke.jsonl")
    try:
        trainer = Trainer(cfg, run_dir, run, smoke=smoke, synthetic=synthetic, parent_dir=parent_dir,
                          registry_path=registry_path, register=not args.no_register,
                          stop_after_steps=args.stop_after_steps or None,
                          ignore_epoch_limit=args.ignore_epoch_limit)
        return _run(trainer, args, resumed, parent_run, log)
    except ConfigError as e:
        log(f"[train] config error: {e}")
        run.update(status="failed", error=f"config: {e}")
        atomic_write_json(run_dir / "run.json", run)
        return EXIT_CONFIG
    finally:
        if lock is not None:
            lock.release()


def _run(tr: Trainer, args, resumed: bool, parent_run: Optional[dict], log: RunLog) -> int:
    from benchmark.provenance import run_manifest, sha256_file
    cfg, run_dir, run = tr.cfg, tr.run_dir, tr.run
    stages = run.setdefault("stages", {})
    if not (run_dir / "manifest.json").is_file():
        roles = None if tr.synthetic else load_roles(cfg["data"]["roles"])
        hash_files = {}
        rp = _repo_path(cfg["data"]["roles"])
        if rp.is_file():
            hash_files["roles"] = str(rp)
        man = run_manifest(write_to=run_dir / "manifest.json", config=comparable_view(cfg),
                           hash_files=hash_files or None, run_id=run["run_id"], rung=cfg.get("rung"),
                           smoke=bool(run.get("smoke")), roles_hashes=(roles or {}).get("hashes"),
                           gpu=tr.device.type == "cuda")
        run["code_tree_hash"] = man["code_tree_hash"]
        run["split_hashes"] = ((roles or {}).get("hashes") or {}).get("roles")
    tr.build()
    run["params"] = tr.params
    run["steps_per_epoch"], run["total_steps"], run["warmup_steps"] = tr.steps_per_epoch, tr.total_steps, tr.warmup
    log(f"[train] run {run['run_id']} rung={cfg.get('rung')} parent={run.get('parent_run_id')} "
        f"model={cfg['model'].get('variant')}/{tr.model.decoder_mode} params={tr.params / 1e6:.3f}M "
        f"steps={tr.total_steps} device={tr.device} amp={tr.use_amp} smoke={run.get('smoke')}")

    # ---- timing (T1-00) --------------------------------------------------------------------
    if args.timing_steps:
        ds = build_train_dataset(cfg, tr.synthetic)
        t = tr.timing(ds, int(args.timing_steps))
        atomic_write_json(run_dir / "timing.json", t)
        tr.save_run(status="timing", timing=t, finished_utc=utc_now())
        log(f"[timing] {t['seconds_per_step']:.4f}s/step -> {t['epoch_seconds']:.1f}s/epoch "
            f"(data-wait {t['data_wait_frac']:.2%})")
        return EXIT_OK

    # ---- training ----------------------------------------------------------------------------
    if stages.get("train") != "done":
        ds = build_train_dataset(cfg, tr.synthetic)
        run["n_train_rows"] = len(ds)
        if resumed:
            tr.load_last()
        tr.save_run(status="running", started_utc=run.get("started_utc") or utc_now())
        try:
            res = tr.fit(ds)
        except TrainingFailed as e:
            log(f"[train] FAILED: {e}")
            tr.save_run(status="failed", error=str(e), nonfinite=tr.state["nonfinite"])
            return EXIT_FAILED
        except EpochTooSlow as e:
            log(f"[train] STOP: {e}")
            tr.save_run(status="aborted_slow", error=str(e), epoch_seconds=tr.state["epoch_seconds"])
            return EXIT_SLOW
        if res == "interrupted":
            tr.save_run(status="interrupted", progress={"epoch": tr.state["epoch"], "step": tr.state["step"]})
            return EXIT_OK
        final = {"model": tr.ema.state_dict(), "model_cfg": tr.model_cfg, "cfg": cfg, "kind": "final_ema",
                 "selection": "none", "run_id": run["run_id"], "epoch": tr.state["epoch"],
                 "step": tr.state["step"], "rung": cfg.get("rung"),
                 "note": "EMA after the last step of the final epoch; no dataset selected this checkpoint"}
        atomic_torch_save(final, run_dir / "final_ema.pt")
        stages["train"] = "done"
        es = tr.state["epoch_seconds"]
        tr.save_run(status="trained", checkpoint="final_ema.pt",
                    checkpoint_sha256=sha256_file(run_dir / "final_ema.pt"),
                    epoch_seconds=es, mean_epoch_seconds=float(np.mean(es)) if es else None,
                    nonfinite=tr.state["nonfinite"], train_camera_mode=tr.state["train_camera_mode"],
                    history=tr.state["history"], trained_utc=utc_now())
    else:
        ck = torch.load(run_dir / "final_ema.pt", map_location="cpu", weights_only=False)
        if ck.get("kind") != "final_ema":
            raise ConfigError("final_ema.pt is not a final-EMA checkpoint")
        tr.ema.load_state_dict(ck["model"])

    # ---- dev evaluation ------------------------------------------------------------------
    if cfg["eval"].get("enabled", True) and stages.get("eval") != "done":
        dev_ds = build_dev_dataset(cfg, tr.synthetic)
        ema_model = tr.ema.module.to(tr.device).eval()
        ev = evaluate_dev(ema_model, dev_ds, cfg, tr.device, log)
        out = {"schema": "v2-dev-eval/1", "run_id": run["run_id"], "rung": cfg.get("rung"),
               "dataset": "d_sun" if not tr.synthetic else "synthetic", "split": cfg["eval"].get("split", "dev"),
               "camera_mode": cfg["eval"]["camera_mode"], "checkpoint": "final_ema.pt",
               "checkpoint_sha256": run.get("checkpoint_sha256"), "summary": ev["summary"],
               "seconds": ev["seconds"], "per_image": ev["rows"]}
        if parent_run is not None:
            pe = Path(run["parent_run_dir"]) / "dev_eval.json"
            if pe.is_file():
                prow = json.loads(pe.read_text(encoding="utf-8"))
                roles = None if tr.synthetic else load_roles(cfg["data"]["roles"])
                out["vs_parent"] = {"parent_run_id": parent_run["run_id"],
                                    **compare_to_parent(ev["rows"], prow["per_image"], roles,
                                                        cfg["eval"]["compare_metrics"],
                                                        int(cfg["eval"]["n_boot"]), int(cfg["eval"]["boot_seed"]))}
            else:
                out["vs_parent"] = {"parent_run_id": parent_run["run_id"], "error": "parent has no dev_eval.json"}
        atomic_write_json(run_dir / "dev_eval.json", out)
        stages["eval"] = "done"
        tr.save_run(dev_summary=ev["summary"], eval_seconds=ev["seconds"],
                    vs_parent_summary={k: {kk: v.get(kk) for kk in ("diff", "ci_lo", "ci_hi", "significant")}
                                       for k, v in out.get("vs_parent", {}).items() if isinstance(v, dict)
                                       and "diff" in v})

    # ---- registry ----------------------------------------------------------------------------
    code = EXIT_OK
    if tr.register and stages.get("register") != "done":
        try:
            ids = register_run(cfg, run, run_dir, parent_run, bool(run.get("smoke")), tr.registry_path)
            stages["register"] = "done"
            tr.save_run(registry=ids, registry_path=str(tr.registry_path or "default"))
        except ImportError as e:
            log(f"[registry] benchmark.registry_v2 unavailable: {e}")
            tr.save_run(registry_error=f"import: {e}")
            code = EXIT_REGISTRY
        except Exception as e:                                 # noqa: BLE001 - recorded, run kept
            log(f"[registry] append refused: {e}")
            tr.save_run(registry_error=str(e))
            code = EXIT_REGISTRY
    final_status = ("smoke" if run.get("smoke") else "complete") if code == EXIT_OK else "trained"
    tr.save_run(status=final_status, finished_utc=utc_now())
    log(f"[train] {final_status}: {run_dir}")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
