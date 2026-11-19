"""Append-only, hash-chained experiment registry for V2 (CoDeQ): benchmark/registry_v2.jsonl.

Contract (docs/V2_PLAN.md 9.1 "Registry", amendment A19 record types, task 4). Runs in every env
(depth, depth_cmp, depth_yolo); imports neither torch nor numpy at module import time.

    append_record(rec, path=None, *, verify=True, lock_timeout=60.0) -> dict
        Validates `rec` (validate_record), then, under an exclusive lock file
        (<registry>.lock), appends ONE line of canonical JSON with these fields added:
        id (derived when absent, see record_id), seq (0-based line index), created_utc (when
        absent), prev_hash (record_hash of the previous line, 64 zeros for the first) and
        record_hash = sha256(canonical_json(record without record_hash)). The line is written
        with one write() on an O_APPEND descriptor and fsync'd; the head file <registry>.head
        is then replaced atomically. By default the chain is verified first and nothing is
        appended to a broken chain. Returns the stored record. Raises ValueError on an invalid
        record, RegistryIntegrityError on a broken chain, TimeoutError when the lock is held.

    load_records(path=None, verify=False) -> list[dict]       every line, in chain order
    verify_chain(path=None, anchor=None) -> (ok: bool, problems: list[str])
        Detects edited records (record_hash mismatch), deleted / inserted / reordered lines (seq
        and prev_hash mismatch), a torn last line (no trailing newline), deletion of the tail
        (head file disagrees), unparseable lines, unknown record types and duplicate ids without
        a valid `supersedes`. `anchor` = (seq, record_hash) published elsewhere (independent
        audit, A10) also detects a full rewrite of file + head, which nothing inside the repo
        can detect on its own.

Record types (A19; `type` field):
    root          trained run with no parent (R0, T1-00, F0 ...)                key: run_id
    ablation      trained run changing exactly ONE factor vs `parent`            key: run_id
                  (the one-factor rule applies to this type only; check_one_factor)
    finalist      trained run on an accepted recipe; lists `components`          key: run_id
    baseline      third-party model (UniDepthV2, MoGe-2, UniK3D, YOLO26 ...)     key: run_id
    legacy-import pre-registry student (b1_mix, S16_prox), legacy=True           key: run_id
    eval          one result file: model_ref, dataset, camera_mode,              key: derived
                  result_file, result_sha256
    freeze        recipe freeze (A9): `covers` = run ids and/or `recipe_hash`    key: run_id

Fields read by benchmark/policy_check.py (writers should fill them):
    trained (root|ablation|finalist): run_id, parent, factor_changed (str or [str]),
        factor_keys (config keys of the factor), config_hash, config_resolved (nested dict; the
        one-factor rule diffs it against the parent's, A7), code_tree_hash, seed,
        train_sources ([str]) or draw_weights ({source: weight}) or
        config_resolved.data.draw_weights, dev_sources (default ["sun"] = D_SUN),
        selection_sources (default = dev_sources), teachers ([ids] or {name: [ids]}),
        checkpoint_path, checkpoint_sha256, checkpoint_kind ("final_ema"), selection ("none"),
        recipe_hash, roles_sha256 / roles_hashes, gates, status
        (trained|accepted|rejected|smoke|failed|withdrawn).
    eval: model_ref (a model record id) or checkpoint_sha256, model_name, dataset, variant,
        benchmark, split (dev|proxy|test), camera_mode (none|default-k|gt-k), result_file,
        result_sha256, source_file, source_sha256, artefact (torch-fp32, onnx-int8, coreml-...),
        artefact_path, artefact_sha256, calibration ({sources, ids | ids_file + ids_sha256, n,
        seed}), claim_level (Core ML), claim ("zero-shot" when a row is claimed as such),
        status (registered|withdrawn).
    baseline / legacy-import: model_key, aliases, train_sources, dev_sources,
        selection_sources, teachers, lineage {status, declared_excluded, evidence},
        legacy_provenance (True = hashes computed at import time, not at run time).
        lineage.status is one of LINEAGE_STATUSES:
          complete            our own run: train/dev/selection lists are the whole truth
          published-list      the full training list is published and recorded in
                              train_sources (opaque entries limit it to declared_excluded)
          declared-exclusions only the listed exclusions are declared (others unverified)
          partial             composition unknown; declared_excluded are vendor/paper claims
          unverified          nothing is verified
    freeze: covers ([run ids]), recipe_hash, nyu_leak_rule_confirmed (bool, plan 5.1 step 3).
    A later record replaces an earlier one with the same id only by setting
    `supersedes` = the record_hash of the latest record with that id (append-only updates,
    e.g. a gate decision turning status "trained" into "accepted").

Result snapshots: register_result / backfill_legacy copy the result JSON into
outputs/v2/registry_store/results/<sha16>__<name> and register the COPY (result_file), keeping
the origin in source_file/source_sha256. A later run_bench run that overwrites results_bench/
therefore cannot silently change a registry-linked number; editing the snapshot is detected by
policy_check (P4).

Other helpers: latest_by_id, active_records, record_id, check_one_factor, validate_record,
eval_record_from_result, register_result, model_ref_for, resolve_model, withdraw, repair_head,
canonical_dataset, dataset_split, normalize_camera_mode, normalize_teachers, trained_sources,
snapshot_result, backfill_legacy.

CLI (any env; no torch needed):
    python -m benchmark.registry_v2 verify   [--registry PATH] [--anchor SEQ:HASH]
    python -m benchmark.registry_v2 list     [--type eval]
    python -m benchmark.registry_v2 head
    python -m benchmark.registry_v2 backfill-legacy [--dry-run]
    python -m benchmark.registry_v2 register-result results_bench/X.json --camera-mode none
    python -m benchmark.registry_v2 repair-head      (only for a crash between append and head)
The registry path defaults to benchmark/registry_v2.jsonl; env UNIDEPTH_REGISTRY_V2 overrides it
(use a scratch path for smoke runs so they never enter the real chain).

Tests: tests/test_registry.py (env depth or depth_cmp; CPU, offline).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark.provenance import (REPO_ROOT, atomic_write_json, canonical_json, code_tree_hash,
                                  repo_relpath, resolve_repo_path, sha256_file, sha256_text,
                                  to_jsonable, utc_now)

GENESIS = "0" * 64
RECORD_TYPES = ("root", "ablation", "finalist", "baseline", "legacy-import", "eval", "freeze")
TRAINED_TYPES = ("root", "ablation", "finalist")
MODEL_TYPES = TRAINED_TYPES + ("baseline", "legacy-import")
INACTIVE_STATUSES = frozenset({"withdrawn", "superseded", "smoke", "failed"})
RESERVED_FIELDS = ("seq", "prev_hash", "record_hash")
CAMERA_MODES = ("none", "default-k", "gt-k")
CORE_ML_CLAIM_LEVELS = ("converted-not-executed", "executed-parity", "validated-on-device")
LINEAGE_STATUSES = ("complete", "published-list", "declared-exclusions", "partial", "unverified")
DEFAULT_STORE = "outputs/v2/registry_store/results"


class RegistryIntegrityError(RuntimeError):
    """The registry hash chain is broken; nothing may be appended until it is repaired."""


# ----------------------------------------------------------------------------- vocabulary
_DATASET_ALIASES = {
    # benchmarks
    "nyu": "nyu", "nyuv2": "nyu", "nyu_v2": "nyu", "nyu_depth_v2": "nyu", "nyud": "nyu",
    "nyu-depth-v2": "nyu",
    "sun": "sun", "sunrgbd": "sun", "sun_rgbd": "sun", "sun-rgbd": "sun", "d_sun": "sun",
    "dsun": "sun", "d-sun": "sun", "sun_dev": "sun", "sun-dev": "sun",
    "ibims": "ibims", "ibims1": "ibims", "ibims-1": "ibims", "ibims_1": "ibims",
    "eth3d": "eth3d", "eth3d_indoor": "eth3d", "eth3d-indoor": "eth3d",
    "diode": "diode", "diode_indoor": "diode", "diode-indoor": "diode", "diode_val": "diode",
    "diode_indoor_val": "diode",
    "tum": "tum", "tum_rgbd": "tum", "tum-rgbd": "tum", "tumrgbd": "tum",
    "diml": "diml", "diml_cvl": "diml", "diml-cvl": "diml", "dimlcvl": "diml",
    "kitti": "kitti", "kitti_obj": "kitti", "kitti-obj": "kitti",
    "make3d": "make3d",
    # training pools of this project
    "own": "own", "x_indoor": "x_indoor", "x-indoor": "x_indoor", "xindoor": "x_indoor",
    "mit_indoor67": "mit_indoor67", "mit-indoor67": "mit_indoor67", "indoor67": "mit_indoor67",
    "mitindoor67": "mit_indoor67", "mit67": "mit_indoor67",
    "imagenet": "imagenet", "imagenet100": "imagenet", "imagenet-100": "imagenet",
    "x_in": "imagenet", "x-in": "imagenet", "imagenet_pseudo": "imagenet",
    "imagenet-pseudo": "imagenet",
    # third-party training sources (UniDepthV2 23-dataset list, YOLO26 mix)
    "a2d2": "a2d2", "argoverse2": "argoverse2", "arkitscenes": "arkitscenes",
    "arkit-scenes": "arkitscenes", "bedlam": "bedlam", "blendedmvs": "blendedmvs",
    "dl3dv": "dl3dv", "drivingstereo": "drivingstereo", "dynamicreplica": "dynamicreplica",
    "eden": "eden", "hoi4d": "hoi4d", "hm3d": "hm3d", "matterport3d": "matterport3d",
    "mapillary_psd": "mapillary_psd", "mapillary-psd": "mapillary_psd",
    "matrixcity": "matrixcity", "megadepth": "megadepth", "nianticmapfree": "nianticmapfree",
    "pointodyssey": "pointodyssey", "scannet": "scannet", "scannetpp": "scannetpp",
    "scannet++": "scannetpp", "tartanair": "tartanair", "taskonomy": "taskonomy",
    "waymo": "waymo", "wildrgbd": "wildrgbd", "hypersim": "hypersim",
    "vkitti2": "vkitti2", "virtual_kitti_2": "vkitti2",
}
KNOWN_SOURCES = frozenset(_DATASET_ALIASES.values())
_DEV_SPLIT_KEYS = {"d_sun", "dsun", "d-sun", "sun_dev", "sun-dev"}
KNOWN_BENCHMARKS = ("ibims", "eth3d", "diode", "tum", "diml", "nyu", "sun", "kitti", "make3d")
# Dataset A's images contain dataset B's images (SUN RGB-D pools NYU Depth v2's labelled frames).
POOLS = {"sun": frozenset({"nyu"})}


def canonical_dataset(name: Optional[str]) -> Optional[str]:
    """Dataset family key ('nyu-filled' -> 'nyu', 'd_sun' -> 'sun', 'ETH3D_indoor' -> 'eth3d').

    Names outside the vocabulary are returned lower-cased (they are 'opaque' sources)."""
    if name is None:
        return None
    n = str(name).strip().lower().replace(" ", "_")
    if n in _DATASET_ALIASES:
        return _DATASET_ALIASES[n]
    for sep in ("__", ":", "/", "-", "_"):
        head = n.split(sep, 1)[0]
        if head in _DATASET_ALIASES:
            return _DATASET_ALIASES[head]
    return n


def pooled_with(benchmark: str) -> frozenset:
    """Datasets whose images overlap `benchmark` (either direction of POOLS)."""
    b = canonical_dataset(benchmark)
    out = set(POOLS.get(b, ()))
    out |= {a for a, inner in POOLS.items() if b in inner}
    return frozenset(out)


def dataset_split(name: Optional[str], variant: Optional[str] = None) -> str:
    """'dev' for D_SUN keys, 'proxy' for local stand-ins of a benchmark, else 'test'."""
    n = str(name or "").strip().lower()
    v = str(variant or "").strip().lower()
    if n in _DEV_SPLIT_KEYS or v in {"dev", "d_sun", "dsun"}:
        return "dev"
    if "proxy" in n or "local" in n or v in {"proxy", "local"}:
        return "proxy"
    return "test"


def normalize_camera_mode(mode: Any) -> Optional[str]:
    if mode is None:
        return None
    m = str(mode).strip().lower().replace("_", "-")
    aliases = {"none": "none", "no-k": "none", "default": "default-k", "default-k": "default-k",
               "defaultk": "default-k", "gt": "gt-k", "gt-k": "gt-k", "gtk": "gt-k"}
    return aliases.get(m, m)


def normalize_factors(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def normalize_teachers(value: Any) -> List[str]:
    """Teacher ids from a list, a string or {name: [checkpoint ids]} (keys used if values empty)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        out: List[str] = []
        for k, v in value.items():
            vals = normalize_teachers(v)
            out.extend(vals if vals else [str(k)])
        return sorted(set(out))
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(normalize_teachers(v))
        return sorted(set(out))
    return [str(value)]


def trained_sources(rec: Dict[str, Any]) -> List[str]:
    """Canonical training sources of a record: train_sources, else draw_weights > 0, else
    config_resolved.data.draw_weights > 0. Empty list when none is recorded."""
    srcs = rec.get("train_sources")
    if srcs:
        return sorted({canonical_dataset(s) for s in normalize_factors(srcs)})
    dw = rec.get("draw_weights")
    if not isinstance(dw, dict):
        cfg = rec.get("config_resolved") or {}
        dw = ((cfg.get("data") or {}).get("draw_weights") if isinstance(cfg, dict) else None)
    if isinstance(dw, dict):
        out = set()
        for k, w in dw.items():
            try:
                if float(w) > 0:
                    out.add(canonical_dataset(k))
            except (TypeError, ValueError):
                out.add(canonical_dataset(k))
        return sorted(out)
    return []


# ----------------------------------------------------------------------------- paths & lock
def default_registry_path() -> Path:
    env = os.environ.get("UNIDEPTH_REGISTRY_V2")
    return Path(env) if env else REPO_ROOT / "benchmark" / "registry_v2.jsonl"


def _paths(path: Optional[os.PathLike]) -> Tuple[Path, Path, Path]:
    p = Path(path) if path is not None else default_registry_path()
    return p, Path(str(p) + ".head"), Path(str(p) + ".lock")


class RegistryLock:
    """Exclusive lock via O_CREAT|O_EXCL on <registry>.lock (works across processes, any OS).

    A lock older than `stale_after` seconds is treated as left by a crashed writer and removed
    (appends take milliseconds). Otherwise waits up to `timeout` seconds, then TimeoutError.
    """

    def __init__(self, lock_path: Path, timeout: float = 60.0, stale_after: float = 600.0,
                 poll: float = 0.02):
        self.lock_path, self.timeout, self.stale_after, self.poll = (
            Path(lock_path), timeout, stale_after, poll)

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"pid": os.getpid(), "utc": utc_now()}))
                return self
            except (FileExistsError, PermissionError):
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_after:
                    try:
                        self.lock_path.unlink()
                        print(f"[registry_v2] removed stale lock {self.lock_path} "
                              f"({age:.0f}s old)", file=sys.stderr)
                    except OSError:
                        pass
                    continue
                if time.monotonic() - t0 > self.timeout:
                    raise TimeoutError(
                        f"registry lock {self.lock_path} is held ({age:.0f}s old). If no other "
                        "process is writing the registry, delete the lock file.")
                time.sleep(self.poll)

    def __exit__(self, *exc):
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        return False


# ----------------------------------------------------------------------------- hashing / io
def record_hash(rec: Dict[str, Any]) -> str:
    return sha256_text(canonical_json({k: v for k, v in rec.items() if k != "record_hash"}))


def _read_raw(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def _split_lines(raw: bytes) -> List[str]:
    lines = raw.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def load_records(path: Optional[os.PathLike] = None, verify: bool = False) -> List[Dict[str, Any]]:
    """All records in chain order. With verify=True a broken chain raises."""
    p, _, _ = _paths(path)
    if verify:
        ok, problems = verify_chain(p)
        if not ok:
            raise RegistryIntegrityError("registry chain broken:\n  " + "\n  ".join(problems))
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(_split_lines(_read_raw(p))):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise RegistryIntegrityError(f"{p}: line {i + 1} is not JSON: {e}") from e
        out.append(rec)
    return out


def read_head(path: Optional[os.PathLike] = None) -> Optional[Dict[str, Any]]:
    _, head_p, _ = _paths(path)
    if not head_p.exists():
        return None
    return json.loads(head_p.read_text(encoding="utf-8"))


def verify_chain(path: Optional[os.PathLike] = None,
                 anchor: Optional[Tuple[int, str]] = None) -> Tuple[bool, List[str]]:
    """(ok, problems) for the whole chain; see the module docstring for what is detected."""
    p, head_p, _ = _paths(path)
    problems: List[str] = []
    if not p.exists():
        if head_p.exists():
            problems.append(f"registry file {p} is missing but its head file exists "
                            "(registry deleted?)")
        if anchor is not None:
            problems.append(f"anchor seq {anchor[0]} given but the registry does not exist")
        return (not problems, problems)
    raw = _read_raw(p)
    if raw and not raw.endswith(b"\n"):
        problems.append("torn tail: the last line has no trailing newline (interrupted write)")
    try:
        lines = _split_lines(raw)
    except UnicodeDecodeError as e:
        return False, [f"registry is not valid UTF-8: {e}"]
    prev: Optional[str] = GENESIS
    latest_hash_by_id: Dict[str, str] = {}
    hash_by_seq: Dict[int, str] = {}
    last_hash, last_seq = None, -1
    for i, line in enumerate(lines):
        if not line.strip():
            problems.append(f"line {i + 1}: blank line inside the chain")
            prev = None
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {i + 1}: not JSON ({e})")
            prev = None
            continue
        if not isinstance(rec, dict):
            problems.append(f"line {i + 1}: not a JSON object")
            prev = None
            continue
        rid = rec.get("id")
        if rec.get("type") not in RECORD_TYPES:
            problems.append(f"line {i + 1} (id {rid}): unknown record type {rec.get('type')!r}")
        if rec.get("seq") != i:
            problems.append(f"line {i + 1} (id {rid}): seq {rec.get('seq')} != position {i} "
                            "(a record was deleted, inserted or reordered)")
        if prev is not None and rec.get("prev_hash") != prev:
            problems.append(f"line {i + 1} (id {rid}): prev_hash does not match the previous "
                            "record (chain broken by deletion, insertion or reordering)")
        if record_hash(rec) != rec.get("record_hash"):
            problems.append(f"line {i + 1} (id {rid}): record_hash mismatch "
                            "(record edited after it was appended)")
        if rid is not None:
            if rid in latest_hash_by_id and rec.get("supersedes") != latest_hash_by_id[rid]:
                problems.append(f"line {i + 1}: duplicate id {rid} without a valid 'supersedes'")
            latest_hash_by_id[rid] = rec.get("record_hash")
        prev = rec.get("record_hash")
        hash_by_seq[i] = rec.get("record_hash")
        last_hash, last_seq = rec.get("record_hash"), i
    if head_p.exists():
        try:
            head = json.loads(head_p.read_text(encoding="utf-8"))
            if head.get("seq") != last_seq or head.get("record_hash") != last_hash:
                problems.append(f"head file says seq {head.get('seq')} hash "
                                f"{str(head.get('record_hash'))[:12]}..., chain ends at seq "
                                f"{last_seq} hash {str(last_hash)[:12]}... (tail deleted, "
                                "appended outside append_record, or crash before head update)")
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"head file unreadable: {e}")
    elif lines:
        problems.append("head file missing for a non-empty registry")
    if anchor is not None:
        a_seq, a_hash = int(anchor[0]), str(anchor[1])
        got = hash_by_seq.get(a_seq)
        if len(a_hash) < 12:
            problems.append(f"anchor hash '{a_hash}' is too short (give at least 12 hex chars)")
        elif got is None:
            problems.append(f"anchor seq {a_seq} is beyond the end of the chain (truncated?)")
        elif not got.startswith(a_hash):
            problems.append(f"anchor mismatch at seq {a_seq}: published {a_hash[:16]}..., "
                            f"chain has {got[:16]}... (history rewritten)")
    return (not problems, problems)


# ----------------------------------------------------------------------------- record logic
def record_id(rec: Dict[str, Any]) -> str:
    if rec.get("id"):
        return str(rec["id"])
    t = rec.get("type")
    if t == "eval":
        ref = rec.get("model_ref") or ("sha:" + str(rec.get("checkpoint_sha256", ""))[:12])
        ds = str(rec.get("dataset"))
        var = rec.get("variant")
        if var and var != "default":
            ds = f"{ds}-{var}"
        art = rec.get("artefact")
        tail = f"{art}:" if art else ""
        return f"eval:{ref}:{ds}:{tail}{str(rec.get('result_sha256', ''))[:12]}"
    if rec.get("run_id"):
        return str(rec["run_id"])
    raise ValueError(f"record of type {t!r} needs a 'run_id' (or an explicit 'id')")


def latest_by_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in records:
        rid = r.get("id")
        if rid is not None:
            out[rid] = r
    return out


def is_active(rec: Dict[str, Any]) -> bool:
    return str(rec.get("status", "")).lower() not in INACTIVE_STATUSES


def active_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in latest_by_id(records).items() if is_active(v)}


_IGNORED_CONFIG_KEYS = frozenset({
    "run_id", "run_name", "name", "out_dir", "output_dir", "run_dir", "outdir", "notes", "note",
    "description", "parent", "factor_changed", "factor", "factor_keys", "tags", "wandb",
    "created", "created_utc", "date", "smoke", "resume", "device", "num_workers", "log_every",
    "eval_every", "save_every", "registry", "type", "status",
})


def flatten_config(d: Any, prefix: str = "") -> Dict[str, str]:
    """Nested dict -> {dotted.key: canonical-json leaf}. Lists are leaves."""
    out: Dict[str, str] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out.update(flatten_config(v, key))
            else:
                out[key] = canonical_json(v)
    else:
        out[prefix] = canonical_json(d)
    return out


def config_diff_keys(parent_cfg: Dict[str, Any], child_cfg: Dict[str, Any]) -> List[str]:
    a, b = flatten_config(parent_cfg), flatten_config(child_cfg)
    keys = []
    for k in sorted(set(a) | set(b)):
        parts = k.split(".")
        if parts[0] in _IGNORED_CONFIG_KEYS or parts[-1] in _IGNORED_CONFIG_KEYS:
            continue
        if a.get(k) != b.get(k):
            keys.append(k)
    return keys


def _key_in_factor(key: str, factor: str, factor_keys: Sequence[str]) -> bool:
    for f in [factor, *factor_keys]:
        if key == f or key.startswith(f + ".") or f.startswith(key + "."):
            return True
    return False


def check_one_factor(rec: Dict[str, Any], latest: Dict[str, Dict[str, Any]]) -> List[str]:
    """One-factor rule for ablation records (A7/A19). Returns problems (empty = pass).

    Parent must exist, be a trained or legacy-import record registered before the child, and
    exactly one factor may be declared. When both records carry config_resolved, every key
    whose resolved value differs must belong to that factor (its name as a dotted prefix, or
    one of factor_keys), so a run cannot hide a second change behind one declared name."""
    probs: List[str] = []
    parent = rec.get("parent")
    par = None
    if not parent:
        probs.append("ablation record has no parent")
    else:
        par = latest.get(str(parent))
        if par is None:
            probs.append(f"parent '{parent}' is not in the registry")
        elif par.get("type") not in TRAINED_TYPES + ("legacy-import",):
            probs.append(f"parent '{parent}' is a {par.get('type')} record, not a trained run")
        elif rec.get("seq") is not None and par.get("seq") is not None \
                and par["seq"] >= rec["seq"]:
            probs.append(f"parent '{parent}' was registered after its child")
    factors = normalize_factors(rec.get("factor_changed"))
    if len(factors) != 1:
        probs.append(f"exactly one factor must change vs the parent; declared {len(factors)}: "
                     f"{factors}")
    pc = par.get("config_resolved") if par else None
    cc = rec.get("config_resolved")
    if isinstance(pc, dict) and isinstance(cc, dict):
        diff = config_diff_keys(pc, cc)
        if not diff:
            probs.append("config_resolved is identical to the parent's: no factor changed")
        elif len(factors) == 1:
            fkeys = [str(k) for k in (rec.get("factor_keys") or [])]
            outside = [k for k in diff if not _key_in_factor(k, factors[0], fkeys)]
            if outside:
                probs.append(f"resolved config differs from the parent outside the declared "
                             f"factor '{factors[0]}': {outside[:12]}")
    return probs


def validate_record(rec: Dict[str, Any], records: Optional[List[Dict[str, Any]]] = None
                    ) -> List[str]:
    """Structural problems that make append_record refuse `rec` (empty list = appendable).

    Content rules (lineage, teachers, final-EMA, calibration, zero-shot) are policy, checked by
    benchmark/policy_check.py, so a writer is never blocked from recording what happened."""
    if not isinstance(rec, dict):
        return ["record must be a dict"]
    probs: List[str] = []
    t = rec.get("type")
    if t not in RECORD_TYPES:
        return [f"type must be one of {RECORD_TYPES}, got {t!r}"]
    for k in RESERVED_FIELDS:
        if k in rec:
            probs.append(f"field '{k}' is set by the registry and must not be supplied")
    try:
        to_jsonable(rec)
    except TypeError as e:
        probs.append(str(e))
    latest = latest_by_id(records or [])
    try:
        rid = record_id(rec)
    except ValueError as e:
        return probs + [str(e)]
    if rid in latest:
        if rec.get("supersedes") != latest[rid].get("record_hash"):
            probs.append(f"id '{rid}' already exists; to update it set supersedes="
                         f"{latest[rid].get('record_hash')}")
        elif latest[rid].get("type") != t:
            probs.append(f"id '{rid}' is a {latest[rid].get('type')} record; a superseding "
                         f"record must keep the type")
    elif rec.get("supersedes"):
        probs.append(f"supersedes is set but id '{rid}' does not exist yet")
    smoke = str(rec.get("status", "")).lower() == "smoke"
    if t == "ablation" and not smoke:
        probs += [f"one-factor rule: {p}" for p in check_one_factor(rec, latest)]
    elif t == "root":
        if rec.get("parent"):
            probs.append("root records have no parent; a run with a parent is an ablation "
                         "(one factor) or a finalist (components)")
    elif t == "finalist" and not smoke:
        comps = normalize_factors(rec.get("components"))
        if not comps:
            probs.append("finalist records must list their accepted component runs "
                         "('components')")
        for c in comps:
            if c not in latest:
                probs.append(f"finalist component '{c}' is not in the registry")
    elif t in ("baseline", "legacy-import"):
        lin = rec.get("lineage")
        if not isinstance(lin, dict) or lin.get("status") not in LINEAGE_STATUSES:
            probs.append(f"{t} records need lineage.status in {LINEAGE_STATUSES} (P8)")
        if not isinstance(rec.get("train_sources"), list):
            probs.append(f"{t} records need train_sources (a list, [] if unknown)")
    elif t == "freeze":
        covers = normalize_factors(rec.get("covers"))
        if not covers and not rec.get("recipe_hash"):
            probs.append("freeze records need 'covers' (run ids) and/or 'recipe_hash'")
        for c in covers:
            if c not in latest:
                probs.append(f"freeze covers unknown run '{c}'")
    elif t == "eval":
        if not rec.get("model_ref") and not rec.get("checkpoint_sha256"):
            probs.append("eval records need model_ref (a model record id, e.g. 'baseline:v2-l') "
                         "or checkpoint_sha256")
        elif rec.get("model_ref") and rec["model_ref"] not in latest:
            probs.append(f"eval model_ref '{rec['model_ref']}' is not in the registry")
        if not rec.get("dataset"):
            probs.append("eval records need 'dataset'")
        cm = normalize_camera_mode(rec.get("camera_mode"))
        if cm not in CAMERA_MODES:
            probs.append(f"eval camera_mode must be one of {CAMERA_MODES}, got "
                         f"{rec.get('camera_mode')!r}")
        if not rec.get("result_file"):
            probs.append("eval records need 'result_file'")
        else:
            rp = resolve_repo_path(rec["result_file"])
            if not rp.is_file():
                probs.append(f"result_file does not exist: {rec['result_file']}")
            elif rec.get("result_sha256") and sha256_file(rp) != rec["result_sha256"]:
                probs.append("result_sha256 does not match the result file on disk")
        if str(rec.get("artefact", "")).startswith("coreml") \
                and rec.get("claim_level") not in CORE_ML_CLAIM_LEVELS:
            probs.append(f"Core ML eval records need claim_level in {CORE_ML_CLAIM_LEVELS}")
    return probs


def append_record(rec: Dict[str, Any], path: Optional[os.PathLike] = None, *,
                  verify: bool = True, lock_timeout: float = 60.0) -> Dict[str, Any]:
    """Validate and append one record; returns the stored record (with seq/hashes)."""
    p, head_p, lock_p = _paths(path)
    rec = dict(rec)
    if rec.get("type") == "eval" and rec.get("result_file") and not rec.get("result_sha256"):
        rp = resolve_repo_path(rec["result_file"])
        if rp.is_file():
            rec["result_sha256"] = sha256_file(rp)
    if "camera_mode" in rec:
        rec["camera_mode"] = normalize_camera_mode(rec["camera_mode"])
    with RegistryLock(lock_p, timeout=lock_timeout):
        if verify:
            ok, problems = verify_chain(p)
            if not ok:
                raise RegistryIntegrityError(
                    "refusing to append to a broken registry chain:\n  " + "\n  ".join(problems))
        if p.exists():
            raw = _read_raw(p)
            if raw and not raw.endswith(b"\n"):
                raise RegistryIntegrityError("registry ends with a torn line; repair it first")
        records = load_records(p) if p.exists() else []
        problems = validate_record(rec, records)
        if problems:
            raise ValueError("invalid registry record:\n  " + "\n  ".join(problems))
        stored = to_jsonable(rec)
        stored["id"] = record_id(rec)
        stored.setdefault("created_utc", utc_now())
        stored["seq"] = len(records)
        stored["prev_hash"] = records[-1]["record_hash"] if records else GENESIS
        stored["record_hash"] = record_hash(stored)
        line = (canonical_json(stored) + "\n").encode("utf-8")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
        try:
            n = os.write(fd, line)
            if n != len(line):
                raise OSError(f"short write to registry ({n} of {len(line)} bytes)")
            os.fsync(fd)
        finally:
            os.close(fd)
        atomic_write_json(head_p, {"seq": stored["seq"], "record_hash": stored["record_hash"],
                                   "id": stored["id"], "updated_utc": utc_now()})
    return stored


def repair_head(path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """Advance a head file that lags the chain by exactly ONE valid record (crash between the
    append and the head update). Anything else (head ahead = tail deleted, edits) is refused."""
    p, head_p, lock_p = _paths(path)
    with RegistryLock(lock_p):
        records = load_records(p)
        if not records:
            raise RegistryIntegrityError("empty registry: nothing to repair")
        head = read_head(p)
        last = records[-1]
        if head is not None and head.get("seq") == last["seq"] \
                and head.get("record_hash") == last["record_hash"]:
            return head
        lagging = (head is not None and len(records) >= 2
                   and head.get("seq") == records[-2]["seq"]
                   and head.get("record_hash") == records[-2]["record_hash"])
        if not lagging:
            raise RegistryIntegrityError("head does not lag by exactly one record; refusing to "
                                         "repair (possible truncation or tampering)")
        tmp_ok = True
        for i, r in enumerate(records):
            tmp_ok &= (r.get("seq") == i and record_hash(r) == r.get("record_hash")
                       and r.get("prev_hash") == (records[i - 1]["record_hash"] if i else GENESIS))
        if not tmp_ok:
            raise RegistryIntegrityError("chain itself is broken; refusing to repair the head")
        new = {"seq": last["seq"], "record_hash": last["record_hash"], "id": last.get("id"),
               "updated_utc": utc_now(), "repaired": True}
        atomic_write_json(head_p, new)
        return new


def withdraw(rec_id: str, reason: str, path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """Append a superseding copy of `rec_id` with status 'withdrawn' (append-only retraction)."""
    latest = latest_by_id(load_records(path))
    if rec_id not in latest:
        raise KeyError(rec_id)
    old = {k: v for k, v in latest[rec_id].items()
           if k not in RESERVED_FIELDS + ("created_utc",)}
    old.update(status="withdrawn", withdraw_reason=reason, supersedes=latest[rec_id]["record_hash"])
    return append_record(old, path)


# ----------------------------------------------------------------------------- model refs
def model_index(latest: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """{lower-cased id / run_id / model_key / alias: record id} over model records."""
    idx: Dict[str, str] = {}
    for rid, r in latest.items():
        if r.get("type") not in MODEL_TYPES:
            continue
        for key in [rid, r.get("run_id"), r.get("model_key"), *(r.get("aliases") or [])]:
            if key:
                idx.setdefault(str(key).lower(), rid)
    return idx


def resolve_model(key: str, latest: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Model record for an id, alias, model_key or run_bench name ('v2-l+r9' -> v2-l)."""
    idx = model_index(latest)
    k = str(key).lower()
    for cand in (k, k.split("+", 1)[0]):
        if cand in idx:
            return latest[idx[cand]]
    return None


def model_ref_for(model_name: str, records: Optional[List[Dict[str, Any]]] = None, *,
                  ckpt_sha256: Optional[str] = None,
                  path: Optional[os.PathLike] = None) -> Optional[str]:
    """Registry id of the model behind a run_bench model name or checkpoint hash (or None).

    'v2-l+r9' -> 'baseline:v2-l', 'student-b1_mix+w8' -> 'legacy:b1_mix',
    'yolo26n-depth' -> 'baseline:yolo26n-depth'; a V2 checkpoint resolves by checkpoint_sha256.
    """
    latest = latest_by_id(records if records is not None else load_records(path))
    if ckpt_sha256:
        for rid, r in latest.items():
            if r.get("type") in MODEL_TYPES and r.get("checkpoint_sha256") == ckpt_sha256:
                return rid
    rec = resolve_model(model_name or "", latest) if model_name else None
    return rec.get("id") if rec else None


def snapshot_result(result_path: os.PathLike, *, repo_root: os.PathLike = REPO_ROOT,
                    store_dir: str = DEFAULT_STORE, dry_run: bool = False) -> Tuple[Path, str]:
    """Copy a result file into the content-addressed store; returns (snapshot path, sha256).

    Store names are '<sha256[:16]>__<basename>', so a snapshot never changes once written."""
    rp = Path(result_path)
    sha = sha256_file(rp)
    store = Path(store_dir)
    if not store.is_absolute():
        store = Path(repo_root) / store
    snap = store / f"{sha[:16]}__{rp.name}"
    if snap.exists():
        if sha256_file(snap) != sha:
            raise RegistryIntegrityError(f"snapshot {snap} exists with different content; a "
                                         "registry-linked result was edited")
    elif not dry_run:
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rp, snap)
    return snap, sha


_OPTIONAL_RESULT_FIELDS = ("artefact", "artefact_path", "artefact_sha256", "calibration",
                           "claim_level", "image_ids_sha256", "crop_mask", "checkpoint_sha256",
                           "resolution", "camera_k")


def eval_record_from_result(result_path: os.PathLike, model_ref: Optional[str] = None, *,
                            camera_mode: Optional[str] = None,
                            records: Optional[List[Dict[str, Any]]] = None,
                            snapshot: bool = True, repo_root: os.PathLike = REPO_ROOT,
                            store_dir: str = DEFAULT_STORE, **extra: Any) -> Dict[str, Any]:
    """Build (not append) an eval record for a run_bench result JSON.

    camera_mode comes from the argument or the result's own 'camera_mode' field; a result
    without either is refused (plan 6.2: every result records its camera mode)."""
    rp = Path(result_path)
    with open(rp, encoding="utf-8") as f:
        res = json.load(f)
    cm = normalize_camera_mode(camera_mode or res.get("camera_mode"))
    if cm is None:
        raise ValueError(f"{rp}: no camera_mode in the result and none given")
    if res.get("camera_mode") and camera_mode and \
            normalize_camera_mode(res["camera_mode"]) != normalize_camera_mode(camera_mode):
        raise ValueError(f"{rp}: camera_mode {camera_mode!r} contradicts the result's "
                         f"{res['camera_mode']!r}")
    ck_sha = extra.get("checkpoint_sha256") or res.get("checkpoint_sha256") \
        or res.get("ckpt_sha256")
    if model_ref is None:
        model_ref = model_ref_for(res.get("model", ""), records, ckpt_sha256=ck_sha)
    src_sha = sha256_file(rp)
    if snapshot:
        snap, _ = snapshot_result(rp, repo_root=repo_root, store_dir=store_dir)
        result_file = repo_relpath(snap, repo_root)
    else:
        result_file = repo_relpath(rp, repo_root)
    rec: Dict[str, Any] = {
        "type": "eval",
        "model_ref": model_ref,
        "model_name": res.get("model"),
        "dataset": res.get("dataset"),
        "variant": res.get("variant", "default"),
        "benchmark": canonical_dataset(res.get("dataset")),
        "split": dataset_split(res.get("dataset"), res.get("variant")),
        "camera_mode": cm,
        "num_images": res.get("num_images"),
        "limit": res.get("limit"),
        "result_file": result_file,
        "result_sha256": src_sha,
        "source_file": repo_relpath(rp, repo_root),
        "source_sha256": src_sha,
        "result_date": res.get("created_utc") or res.get("date"),
        "status": "registered",
    }
    for k in _OPTIONAL_RESULT_FIELDS:
        if k in res and k not in extra:
            rec[k] = res[k]
    if ck_sha:
        rec["checkpoint_sha256"] = ck_sha
    rec.update(extra)
    return {k: v for k, v in rec.items() if v is not None or k == "model_ref"}


def register_result(result_path: os.PathLike, *, path: Optional[os.PathLike] = None,
                    **kw: Any) -> Dict[str, Any]:
    """eval_record_from_result + append_record (what run_bench --register calls)."""
    records = load_records(path)
    return append_record(eval_record_from_result(result_path, records=records, **kw), path)


# ----------------------------------------------------------------------------- legacy backfill
# Facts below are cited; policy_check COMPUTES the zero-shot labels from them (never by hand).
UNIDEPTHV2_TRAIN = ["a2d2", "argoverse2", "arkitscenes", "bedlam", "blendedmvs", "dl3dv",
                    "drivingstereo", "dynamicreplica", "eden", "hoi4d", "hm3d", "matterport3d",
                    "mapillary_psd", "matrixcity", "megadepth", "nianticmapfree", "pointodyssey",
                    "scannet", "scannetpp", "tartanair", "taskonomy", "waymo", "wildrgbd"]
_UDV2_EVID = ("docs/PROTOCOL_NOTES.md section 1 (arXiv 2502.20110v2 Sec. IV-A): 23 training "
              "datasets as listed; NOT in training: NYU-Depth-v2, KITTI, SUN RGB-D, IBims-1, "
              "TUM-RGBD, ETH3D. Owner brief 2026-11-19: 23 datasets excluding NYU/SUN/IBims.")
YOLO26_TRAIN = ["imagenet", "arkitscenes", "hypersim", "tartanair", "kitti", "vkitti2", "diode",
                "sun"]
_YOLO_EVID = ("outputs/v2/design_workflow_result.json research 'yolo-depth' + fact-check: "
              "training mix ~2.19M = ImageNet pseudo-labels 1,281,167, ARKitScenes 676,080, "
              "Hypersim 68,242, TartanAir 55,660, KITTI 55,198, Virtual KITTI 2 25,780, DIODE "
              "25,458, SUN RGB-D 9,245 (docs.ultralytics.com/datasets/depth); vendor declares "
              "NYU, iBims-1, ETH3D, Make3D zero-shot. SUN RGB-D pools NYU, so NYU is not clean. "
              "Teachers DA3MONO-LARGE + DA3METRIC-LARGE (docs/V2_PLAN.md section 13).")

BACKFILL_BASELINES: List[Dict[str, Any]] = (
    [{"model_key": k, "family": "UniDepthV2", "weights": hub, "licence": "CC BY-NC 4.0",
      "aliases": [f"unidepthv2-{k[-1]}", hub], "train_sources": list(UNIDEPTHV2_TRAIN),
      "lineage": {"status": "published-list",
                  "declared_excluded": ["nyu", "kitti", "sun", "ibims", "tum", "eth3d"],
                  "evidence": _UDV2_EVID}}
     for k, hub in (("v2-s", "lpiccinelli/unidepth-v2-vits14"),
                    ("v2-b", "lpiccinelli/unidepth-v2-vitb14"),
                    ("v2-l", "lpiccinelli/unidepth-v2-vitl14"))]
    + [{"model_key": k, "family": "UniK3D", "weights": hub, "licence": "unverified",
        "aliases": [hub], "train_sources": ["unik3d-train-mix"],
        "lineage": {"status": "declared-exclusions",
                    "declared_excluded": ["nyu", "sun", "ibims", "eth3d", "diode"],
                    "evidence": "docs/V2_PLAN.md 9.1 zero-shot matrix (UniDepthV2, UniK3D: "
                                "clean on IBims, ETH3D, DIODE, NYU, SUN); the UniK3D training "
                                "list itself was not re-verified in this repo."}}
       for k, hub in (("unik3d-s", "lpiccinelli/unik3d-vits"),
                      ("unik3d-b", "lpiccinelli/unik3d-vitb"))]
    + [{"model_key": k, "family": "MoGe-2", "weights": hub, "licence": "unverified",
        "aliases": [hub, k.replace("moge2", "moge-2")], "train_sources": [],
        "lineage": {"status": "unverified", "declared_excluded": [],
                    "evidence": "docs/V2_PLAN.md 9.1: MoGe-2 training lineage unverified "
                                "(benchmark/fetch_v2.py LINEAGE: paper Table A.1 read via an "
                                "automated reader; checkpoint lineage unverified)."}}
       for k, hub in (("moge2-s", "Ruicheng/moge-2-vits-normal"),
                      ("moge2-b", "Ruicheng/moge-2-vitb-normal"),
                      ("moge2-l", "Ruicheng/moge-2-vitl-normal"))]
    + [{"model_key": f"yolo26{s}-depth", "family": "YOLO26-Depth", "weights": f"yolo26{s}-depth",
        "licence": "AGPL-3.0", "aliases": [f"yolo26{s}", f"yolo26{s}-depth-768"],
        "params_M": {"n": 6.3, "s": 13.2, "m": 23.3, "l": 27.7, "x": 57.0}[s],
        "train_sources": list(YOLO26_TRAIN), "teachers": ["da3mono-large", "da3metric-large"],
        "lineage": {"status": "published-list",
                    "declared_excluded": ["nyu", "ibims", "eth3d", "make3d"],
                    "evidence": _YOLO_EVID}, "status": "lineage-only"}
       for s in "nsmlx"]
    + [{"model_key": f"depthart-{s}", "family": "DepthART", "weights": None,
        "licence": "unverified", "aliases": [], "params_M": {"s": 6.0, "b": 11.4}[s],
        "train_sources": ["nyu", "depthart-stage1-mix"],
        "lineage": {"status": "partial", "declared_excluded": ["ibims"],
                    "evidence": "research report + fact-check (arXiv 2607.17099): stage-1 "
                                "distillation on a 1.7M pool drawn from ~44M images, then NYU "
                                "fine-tuning with known intrinsics; IBims-1 reported zero-shot "
                                "by the paper."},
        "status": "lineage-only"} for s in "sb"]
    + [{"model_key": k, "family": "LightlyTrain", "weights": None, "licence": "AGPL-3.0 or "
        "commercial (LightlyTrain repo); DAv3 weight licence not stated",
        "aliases": [], "params_M": pm, "train_sources": ["lightlytrain-distillation-mix"],
        "lineage": {"status": "partial", "declared_excluded": ["nyu"],
                    "evidence": "research report + fact-check: the vendor page states NYUv2 "
                                "was not used during training (docs.lightly.ai); other sets "
                                "unverified."},
        "status": "lineage-only"}
       for k, pm in (("dav3-metric-tiny", 6.2), ("dav3-metric-tiny-plus", 7.9))]
)

BACKFILL_LEGACY_STUDENTS: List[Dict[str, Any]] = [
    {"run_id": "legacy:b1_mix", "model_key": "student-b1_mix", "aliases": ["b1_mix", "S2-mix", "b1"],
     "checkpoint_path": "outputs/checkpoints/b1_mix/best.pt", "backbone": "efficientvit_b1",
     "params_M": 5.12, "camera_input": "FiLM intrinsics (use_intrinsics=true)",
     "recipe": "multi-source (own+NYU+SUN) + FiLM + EMA, 480x640 (benchmark/registry.json S2-mix)",
     "summary_file": "outputs/checkpoints/b1_mix/train_summary.json"},
    {"run_id": "legacy:S16_prox", "model_key": "student-S16_prox", "aliases": ["S16_prox", "S16-prox", "S16"],
     "checkpoint_path": "outputs/checkpoints/S16_prox/best.pt", "backbone": "efficientvit_b2.r224_in1k",
     "params_M": 15.48, "camera_input": "ray map from intrinsics",
     "recipe": "S2-mix + s4,s5,ray,range,prox (benchmark/registry.json S16-prox)",
     "summary_file": "outputs/checkpoints/S16_prox/train_summary.json"},
]
_LEGACY_COMMON = {
    "type": "legacy-import", "legacy": True, "legacy_provenance": True,
    "train_sources": ["own", "nyu", "sun"],
    "selection_sources": ["own", "nyu", "sun"], "teachers": ["v2-l"],
    "checkpoint_kind": "best_val_absrel",
    "selection": "best.pt = epoch with the lowest validation AbsRel on the 3% own+NYU+SUN carve",
    "lineage": {"status": "complete", "declared_excluded": [],
                "evidence": "Owner brief 2026-11-19: b1/S16 trained on own+NYU+SUN; checkpoint "
                            "cfg data_gt.mix own .4 / nyu .4 / sun .2, val_frac .03 (read from "
                            "best.pt 2026-11-19); own labels are the UniDepthV2-L cache."},
    "provenance": "legacy-import: hashes computed at import time, not when the run was made",
    "status": "legacy",
}
# StudentModelRunner feeds GT intrinsics whenever the loader has K (benchmark/run_bench.py
# StudentModelRunner.predict, checked 2026-11-19), edge/onnx_runner.py does the same, while the
# result JSONs say gt_cam_prompted=false.
_LEGACY_CAMERA_NOTE = ("legacy student result: GT intrinsics were fed (StudentModelRunner / "
                       "OnnxStudent) although the JSON says gt_cam_prompted=false")
_STANDARD_TAGS = {None, "pi"}
_LEAK_EVIDENCE = {"ibims": ("outputs/v2/roles/own_vs_ibims1.csv",
                            "outputs/v2/roles/nyu_vs_ibims1.csv")}


def _legacy_artefact(suffix: str) -> str:
    if not suffix:
        return "torch-fp32"
    if re.fullmatch(r"w\d+(pt)?", suffix):
        return f"sim-{suffix}"
    return suffix


def _ckpt_meta(path: Path) -> Dict[str, Any]:
    """epoch / val_absrel / data_gt.mix from a legacy checkpoint (torch lazily; {} without it)."""
    try:
        import torch                                           # noqa: WPS433 (lazy on purpose)
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") or {}
        dg = cfg.get("data_gt") or {}
        return {"epoch": ck.get("epoch"), "val_absrel": ck.get("val_absrel"),
                "data_gt_mix": dg.get("mix"), "val_frac": dg.get("val_frac"),
                "ema_in_checkpoint": any("ema" in str(k) for k in ck)}
    except Exception as e:                                     # noqa: BLE001
        return {"meta_error": f"{type(e).__name__}: {e}"}


def _csv_rows(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def backfill_legacy(path: Optional[os.PathLike] = None, *, repo_root: os.PathLike = REPO_ROOT,
                    results_dir: str = "results_bench", store_dir: str = DEFAULT_STORE,
                    dry_run: bool = False, read_checkpoints: bool = True
                    ) -> List[Dict[str, Any]]:
    """Register baselines, legacy students and their results_bench results. Idempotent.

    Only standard-tag results (<dataset>__<model>.json and the '__pi' re-runs) of known models on
    known benchmarks are imported; probes, scale-gap files and unknown students are skipped.
    Result files are snapshotted (snapshot_result) and the snapshot is the registered file.
    """
    root = Path(repo_root).resolve()
    records = [] if dry_run else load_records(path)
    latest = latest_by_id(records)
    planned: List[Dict[str, Any]] = []
    tree_at_import = code_tree_hash(root=root)

    def add(rec):
        rid = record_id(rec)
        if rid in latest or any(record_id(r) == rid for r in planned):
            return
        planned.append(rec)

    for b in BACKFILL_BASELINES:
        add({"type": "baseline", "run_id": f"baseline:{b['model_key']}", "status": "active",
             "legacy": False, "legacy_provenance": True, **b})
    for s in BACKFILL_LEGACY_STUDENTS:
        rec = {**_LEGACY_COMMON, **s}
        ck = root / s["checkpoint_path"]
        rec["checkpoint_sha256"] = sha256_file(ck) if ck.is_file() else None
        if ck.is_file():
            rec["checkpoint_bytes"] = ck.stat().st_size
            if read_checkpoints:
                rec["checkpoint_meta"] = _ckpt_meta(ck)
        else:
            rec["import_note"] = "checkpoint missing at import time"
        sf = root / s["summary_file"]
        if sf.is_file():
            rec["train_summary"] = json.loads(sf.read_text(encoding="utf-8"))
            rec["train_summary_sha256"] = sha256_file(sf)
        leak = {}
        for bench, files in _LEAK_EVIDENCE.items():
            n = [_csv_rows(root / f) for f in files]
            if any(v is not None for v in n):
                leak[bench] = {"flagged_training_rows": sum(v or 0 for v in n),
                               "files": list(files),
                               "rule": "V2 blocker (crop pHash <= 12 or NCC >= .85); "
                                       "not inspected; legacy training did not apply it"}
        if leak:
            rec["leak_evidence"] = leak
        rec["code_tree_hash_at_import"] = tree_at_import
        add(rec)

    model_refs = {**{b["model_key"]: f"baseline:{b['model_key']}" for b in BACKFILL_BASELINES},
                  **{s["model_key"]: s["run_id"] for s in BACKFILL_LEGACY_STUDENTS}}
    rdir = root / results_dir
    for rp in sorted(rdir.glob("*.json")) if rdir.is_dir() else []:
        parts = rp.stem.split("__")
        if len(parts) not in (2, 3):
            continue
        ds_key, model_name = parts[0], parts[1]
        tag = parts[2] if len(parts) == 3 else None
        if tag not in _STANDARD_TAGS or canonical_dataset(ds_key) not in KNOWN_BENCHMARKS:
            continue
        try:
            res = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if res.get("model") != model_name or "depth" not in res:
            continue
        base, _, suffix = model_name.partition("+")
        ref = model_refs.get(base)
        if ref is None:
            continue
        is_student = ref.startswith("legacy:")
        snap, src_sha = snapshot_result(rp, repo_root=root, store_dir=store_dir, dry_run=dry_run)
        rec = {
            "type": "eval", "model_ref": ref, "model_name": model_name,
            "dataset": res.get("dataset"), "variant": res.get("variant", "default"),
            "benchmark": canonical_dataset(res.get("dataset")),
            "split": dataset_split(res.get("dataset"), res.get("variant")),
            "camera_mode": "gt-k" if (is_student or res.get("gt_cam_prompted")) else "none",
            "num_images": res.get("num_images"), "tag": tag,
            "result_file": Path(os.path.relpath(snap, root)).as_posix(),
            "result_sha256": src_sha,
            "source_file": repo_relpath(rp, root), "source_sha256": src_sha,
            "result_date": res.get("date"), "legacy": True, "legacy_provenance": True,
            "provenance": "legacy-import", "status": "registered",
        }
        if is_student:
            rec["camera_note"] = _LEGACY_CAMERA_NOTE
            art = _legacy_artefact(suffix)
            rec["artefact"] = art
            if art.startswith("onnx-"):
                s_name = base.replace("student-", "")
                onnx = root / "outputs" / "edge" / s_name / f"{art[5:]}.onnx"
                side = onnx.with_suffix(".json")
                if onnx.is_file():
                    rec["artefact_path"] = Path(os.path.relpath(onnx, root)).as_posix()
                    rec["artefact_sha256"] = sha256_file(onnx)
                if side.is_file():
                    meta = json.loads(side.read_text(encoding="utf-8"))
                    rec["artefact_meta_sha256"] = sha256_file(side)
                    q = meta.get("quant")
                    if q:
                        rec["calibration"] = {
                            "sources": q.get("calibration_sources"),
                            "n": q.get("calibration_images"), "seed": q.get("calibration_seed"),
                            "method": q.get("calibration_method"), "ids": None,
                            "note": "ids not recorded by edge/ort_int8.py; frozen-id exclusion "
                                    "unverified"}
        else:
            rec["setting"] = suffix or "default"
        rec["id"] = record_id(rec)
        add(rec)

    if dry_run:
        return planned
    return [append_record(rec, path) for rec in planned]


# ----------------------------------------------------------------------------- CLI
def _parse_anchor(s: Optional[str]) -> Optional[Tuple[int, str]]:
    if not s:
        return None
    seq, _, h = s.partition(":")
    return int(seq), h


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="V2 append-only registry")
    ap.add_argument("--registry", default=None, help="registry path (default benchmark/"
                                                     "registry_v2.jsonl or $UNIDEPTH_REGISTRY_V2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("verify")
    pv.add_argument("--anchor", default=None, help="SEQ:HASH published by an independent audit")
    pl = sub.add_parser("list")
    pl.add_argument("--type", default=None)
    sub.add_parser("head")
    pb = sub.add_parser("backfill-legacy")
    pb.add_argument("--dry-run", action="store_true")
    pb.add_argument("--no-checkpoints", action="store_true",
                    help="do not open legacy checkpoints with torch for epoch/mix metadata")
    pr = sub.add_parser("register-result")
    pr.add_argument("result")
    pr.add_argument("--model-ref", default=None)
    pr.add_argument("--camera-mode", default=None)
    pr.add_argument("--no-snapshot", action="store_true")
    sub.add_parser("repair-head")
    a = ap.parse_args(argv)
    reg = a.registry
    if a.cmd == "verify":
        ok, problems = verify_chain(reg, anchor=_parse_anchor(a.anchor))
        n = len(load_records(reg)) if ok else "?"
        print(f"registry {_paths(reg)[0]}: {'OK' if ok else 'BROKEN'} ({n} records)")
        for pr_ in problems:
            print("  -", pr_)
        return 0 if ok else 1
    if a.cmd == "list":
        for r in load_records(reg):
            if a.type and r.get("type") != a.type:
                continue
            print(f"{r.get('seq'):>4} {r.get('type'):<13} {r.get('id')}  [{r.get('status')}]")
        return 0
    if a.cmd == "head":
        print(json.dumps(read_head(reg), indent=2))
        return 0
    if a.cmd == "backfill-legacy":
        recs = backfill_legacy(reg, dry_run=a.dry_run, read_checkpoints=not a.no_checkpoints)
        for r in recs:
            print(("PLAN " if a.dry_run else "ADD  ") + f"{r.get('type'):<13} {record_id(r)}")
        print(f"{len(recs)} record(s) {'planned' if a.dry_run else 'appended'}")
        return 0
    if a.cmd == "register-result":
        r = register_result(a.result, path=reg, model_ref=a.model_ref, camera_mode=a.camera_mode,
                            snapshot=not a.no_snapshot)
        print("appended", r["id"])
        return 0
    if a.cmd == "repair-head":
        print(json.dumps(repair_head(reg), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
