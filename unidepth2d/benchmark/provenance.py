"""Provenance helpers for the V2 (CoDeQ) experiment registry: content hashes, env stamp, manifests.

Contract (docs/V2_PLAN.md section 9.1 "Registry", task 3):

    sha256_file(path, chunk_bytes=1 MiB) -> str
        Hex SHA-256 of a file's raw bytes. Raises FileNotFoundError with the path in the message.

    code_tree_hash(paths=None, *, root=REPO_ROOT, exts=(".py", ".yaml", ".yml"),
                   exclude_dirs=DEFAULT_EXCLUDE_DIRS, return_files=False) -> str | (str, dict)
        Content hash of the code tree. A content hash is needed because many harness files are
        not tracked by git. `paths` is a list of files and/or directories (default: the whole
        repo root, minus data/output folders). Every matching file is hashed with sha256_file,
        and the tree hash is SHA-256 over the sorted lines "<relative posix path>\\0<file sha>\\n",
        so it does not depend on walk order or on the OS path separator. Raw bytes are hashed, so
        a CRLF/LF conversion counts as a change. With return_files=True it also returns
        {relative_path: sha256} so two trees can be diffed.

    canonical_json(obj) -> str
        Deterministic JSON (sorted keys, compact separators, UTF-8). NaN/inf floats become the
        strings "nan", "inf", "-inf"; numpy scalars/arrays, tuples, sets and Paths are converted.

    config_hash(obj) -> str
        SHA-256 of canonical_json(obj). Key order does not matter; any value change does.

    env_stamp(*, gpu=True, packages=DEFAULT_PACKAGES) -> dict
        python version/executable/implementation, platform, conda env name, installed versions
        of the listed packages (read from package metadata, nothing imported), and, when torch
        is importable and gpu=True, torch/CUDA/cuDNN versions and the visible GPU names.

    run_manifest(**fields) -> dict
        {"manifest_version", "created_utc", "env", "code_tree_hash", **fields}. Special keyword
        arguments: checkpoint_path -> adds checkpoint_sha256 (+ size); config -> adds
        config_hash; hash_files (list of paths or {name: path}) -> adds file_hashes;
        code_paths -> paths for code_tree_hash; write_to -> also writes the manifest
        atomically as JSON to that path. Every other field is copied verbatim (JSON-converted).

    atomic_write_json(path, obj) / atomic_write_text(path, text)
        Write to a temporary file in the same folder, fsync, then os.replace.

    repo_relpath(path, root=REPO_ROOT) -> str
        Repo-relative POSIX path when the file is inside the repo, else the absolute POSIX path.

    roles_fingerprint(path=DEFAULT_ROLES) -> dict
        {"roles_path", "roles_sha256", "roles_hashes"} for configs/v2/roles.yaml (the id-list
        hashes written by scripts/v2/build_roles.py). Trained-run records store it so
        benchmark/policy_check.py can prove the roles did not change after a run. Raises
        FileNotFoundError when the file is missing (never a silent default).

No function here imports torch at module import time; this module runs in every env
(depth, depth_cmp, depth_yolo) and in CPU-only tests.
Tests: tests/test_registry.py (env depth or depth_cmp; CPU, offline).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLES = REPO_ROOT / "configs" / "v2" / "roles.yaml"

# Folders that hold data, results, caches or vendored environments, not harness code.
DEFAULT_EXCLUDE_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints", ".idea",
    ".vscode", "outputs", "wandb", "results_bench", "results_ocmr", "data_bench", "docs",
    "node_modules", ".venv", "venv",
})
DEFAULT_CODE_EXTS = (".py", ".yaml", ".yml")

DEFAULT_PACKAGES = (
    "torch", "torchvision", "timm", "numpy", "scipy", "opencv-python", "opencv-python-headless",
    "pillow", "pyyaml", "onnx", "onnxruntime", "onnxscript", "ultralytics", "coremltools",
    "moge", "unik3d", "unidepth", "xformers", "pytest",
)

PathLike = Union[str, os.PathLike]


# ----------------------------------------------------------------------------- hashing
def sha256_file(path: PathLike, chunk_bytes: int = 1 << 20) -> str:
    """Hex SHA-256 of the file's raw bytes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sha256_file: no such file: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_code_files(paths: Sequence[PathLike], exts: Sequence[str],
                     exclude_dirs: Iterable[str]) -> Iterable[Path]:
    exclude = set(exclude_dirs)
    exts_l = tuple(e.lower() for e in exts)
    for base in paths:
        b = Path(base)
        if b.is_file():
            if b.suffix.lower() in exts_l:
                yield b
            continue
        if not b.is_dir():
            raise FileNotFoundError(f"code_tree_hash: path does not exist: {b}")
        for dirpath, dirnames, filenames in os.walk(b):
            dirnames[:] = sorted(d for d in dirnames if d not in exclude)
            for fn in sorted(filenames):
                if fn.lower().endswith(exts_l):
                    yield Path(dirpath) / fn


def code_tree_hash(paths: Optional[Sequence[PathLike]] = None, *, root: PathLike = REPO_ROOT,
                   exts: Sequence[str] = DEFAULT_CODE_EXTS,
                   exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
                   return_files: bool = False):
    """Order- and separator-independent content hash over the code files under `paths`."""
    root_p = Path(root).resolve()
    if paths is None:
        paths = [root_p]
    files: Dict[str, str] = {}
    for f in _iter_code_files(list(paths), exts, exclude_dirs):
        fr = f.resolve()
        try:
            rel = fr.relative_to(root_p).as_posix()
        except ValueError:
            rel = fr.as_posix()
        files[rel] = sha256_file(fr)
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8") + b"\0" + files[rel].encode("ascii") + b"\n")
    digest = h.hexdigest()
    return (digest, files) if return_files else digest


# ----------------------------------------------------------------------------- JSON
def to_jsonable(obj: Any) -> Any:
    """Convert to plain JSON types. NaN/inf -> 'nan'/'inf'/'-inf' strings (never silently None)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return float(obj)
    if isinstance(obj, os.PathLike):
        return Path(obj).as_posix()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_jsonable(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    # numpy without importing it
    mod = type(obj).__module__
    if mod == "numpy" or mod.startswith("numpy."):
        if hasattr(obj, "tolist"):
            return to_jsonable(obj.tolist())
        if hasattr(obj, "item"):
            return to_jsonable(obj.item())
    if hasattr(obj, "item") and callable(obj.item):          # torch scalar tensors, etc.
        try:
            return to_jsonable(obj.item())
        except Exception:                                      # noqa: BLE001
            pass
    raise TypeError(f"value of type {type(obj).__name__} is not JSON-serialisable: {obj!r}")


def canonical_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def config_hash(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def atomic_write_text(path: PathLike, text: str) -> None:
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


def atomic_write_json(path: PathLike, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(obj), indent=indent, sort_keys=True,
                                       ensure_ascii=False, allow_nan=False) + "\n")


def repo_relpath(path: PathLike, root: PathLike = REPO_ROOT) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = Path(root) / p
    p = p.resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_repo_path(path: str, root: PathLike = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ----------------------------------------------------------------------------- environment
def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:                                          # noqa: BLE001
        return None


def _conda_env_name() -> Optional[str]:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = Path(sys.prefix)
    # Running an env's python by full path does not set CONDA_DEFAULT_ENV; trust sys.prefix.
    if prefix.parent.name == "envs":
        return prefix.name
    return env


def env_stamp(*, gpu: bool = True, packages: Sequence[str] = DEFAULT_PACKAGES) -> Dict[str, Any]:
    """Python, platform, conda env, package versions and (optionally) torch/GPU details."""
    stamp: Dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).as_posix(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_env": _conda_env_name(),
        "packages": {p: v for p in packages if (v := _pkg_version(p)) is not None},
        "pythonutf8": os.environ.get("PYTHONUTF8"),
    }
    if gpu:
        torch_info: Dict[str, Any] = {}
        try:
            import torch                                       # noqa: WPS433 (lazy on purpose)
            torch_info["version"] = torch.__version__
            torch_info["cuda_build"] = getattr(torch.version, "cuda", None)
            try:
                torch_info["cudnn"] = torch.backends.cudnn.version()
            except Exception:                                  # noqa: BLE001
                torch_info["cudnn"] = None
            avail = bool(torch.cuda.is_available())
            torch_info["cuda_available"] = avail
            if avail:
                torch_info["gpus"] = [torch.cuda.get_device_name(i)
                                      for i in range(torch.cuda.device_count())]
        except Exception as e:                                 # noqa: BLE001
            torch_info["error"] = f"torch not importable: {type(e).__name__}"
        stamp["torch"] = torch_info
    return stamp


# ----------------------------------------------------------------------------- manifests
def run_manifest(**fields: Any) -> Dict[str, Any]:
    """Build (and optionally write) a run manifest; see module docstring for special keys."""
    write_to = fields.pop("write_to", None)
    checkpoint_path = fields.pop("checkpoint_path", None)
    config = fields.pop("config", None)
    hash_files = fields.pop("hash_files", None)
    code_paths = fields.pop("code_paths", None)
    gpu = fields.pop("gpu", True)

    man: Dict[str, Any] = {
        "manifest_version": 1,
        "created_utc": utc_now(),
        "env": env_stamp(gpu=bool(gpu)),
        "code_tree_hash": code_tree_hash(code_paths),
    }
    if checkpoint_path is not None:
        man["checkpoint_path"] = repo_relpath(checkpoint_path)
        man["checkpoint_sha256"] = sha256_file(checkpoint_path)
        man["checkpoint_bytes"] = Path(checkpoint_path).stat().st_size
    if config is not None:
        man["config"] = to_jsonable(config)
        man["config_hash"] = config_hash(config)
    if hash_files is not None:
        items: List[Tuple[str, PathLike]]
        if isinstance(hash_files, dict):
            items = list(hash_files.items())
        else:
            items = [(repo_relpath(p), p) for p in hash_files]
        man["file_hashes"] = {name: sha256_file(p) for name, p in items}
    for k, v in fields.items():
        if k in man:
            raise ValueError(f"run_manifest: field '{k}' collides with a computed field")
        man[k] = to_jsonable(v)
    if write_to is not None:
        atomic_write_json(write_to, man)
    return man


def load_yaml(path: PathLike) -> Any:
    """yaml.safe_load (C loader when available). PyYAML is imported lazily."""
    import yaml                                                # noqa: WPS433
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def roles_fingerprint(path: PathLike = DEFAULT_ROLES) -> Dict[str, Any]:
    """Content hash of roles.yaml plus its per-role id-list hashes (see module docstring)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"roles file not found: {p} (run scripts/v2/build_roles.py)")
    roles = load_yaml(p) or {}
    return {"roles_path": repo_relpath(p), "roles_sha256": sha256_file(p),
            "roles_hashes": ((roles.get("hashes") or {}).get("roles") or {})}


if __name__ == "__main__":                                     # quick manual check
    print(json.dumps({"code_tree_hash": code_tree_hash(), "env": env_stamp()}, indent=2))
