"""Policy checks P1-P10 (+ amendments A4, A9, A19) and the COMPUTED zero-shot matrix for V2.

Contract (docs/V2_PLAN.md 9.1 matrix, 9.4 checks, amendments A4/A9/A19; task 7). Runs in any
env with PyYAML (depth, depth_cmp); CPU only, no torch, no network.

Zero-shot labels are never hand-written: zero_shot_label() derives them from the model's
registry record (training / development / selection sources, teachers, lineage status), from
configs/v2/roles.yaml (dev set, report-only sets, leak-check status, NYU leak rule) and from the
eval's calibration sources and camera mode.

    zero_shot_label(model_rec, benchmark, *, latest, roles=None, calibration_sources=None,
                    camera_mode=None) -> ZSLabel(label, reasons, qualifiers, camera_mode)
        label (worst wins): in-domain > dev > not-clean > unverified > vendor-claimed > clean.
          in-domain      the benchmark is a training source of the model
          dev            it is the model's development / selection set (SUN for every V2 run)
          not-clean      overlap: a training/dev/teacher/calibration source pools its images
                         (SUN RGB-D contains NYU), int8 calibration used it, a teacher trained
                         on it, or the NYU leak rule (plan 5.1) is not yet satisfied
          unverified     lineage does not let us decide (status unverified, opaque training
                         list, no sources recorded, roles.yaml missing for a V2 run)
          vendor-claimed zero-shot per vendor/paper only ('clean per vendor'; 'paper-reported')
          clean          zero-shot (qualifiers say on what evidence)
        ZSLabel.effective is 'gt-k-camera' when a zero-shot label meets camera_mode gt-k (P3:
        a row that saw GT intrinsics is never a zero-shot row); ZSLabel.zero_shot is
        effective in ZERO_SHOT_LABELS.
    zero_shot_matrix(records, roles=None, benchmarks=DEFAULT_MATRIX_BENCHMARKS)
        -> {model_id: {benchmark: ZSLabel}} over active model records.
    check_registry(records=None, *, registry_path=None, roles=None, verify=True) -> PolicyReport
    check_table(rows, *, records, roles=None, kind="zero-shot", camera_labelled=False)
        -> PolicyReport. rows: {"eval_id", "metric", "value", "label", "qualifiers",
        "camera_mode", "camera_label", "claim_level", "claim"}.
    run_policy(registry_path=None, roles_path=None) -> PolicyReport (registry + roles)
    load_roles(path=None) -> Roles (normalised view of roles.yaml; .present False if missing)
    extract_metric(result_dict, "depth.d1") -> value (dotted path)

Rules (severity FAIL blocks CI and table generation; WARN is printed as a table qualifier):
    REG          hash chain broken (registry_v2.verify_chain)
    ROLES        V2 trained runs exist but roles.yaml is missing; roles changed after a run
    P1           V2 runs must record training sources; benchmark domains (all except NYU,
                 the keep-NYU branch) never train; training sources must be frozen roles; a
                 record or table row may claim zero-shot only if the computed label is
                 zero-shot
    P2           a V2 zero-shot claim needs the roles.yaml leak check of that benchmark done;
                 legacy training sets with blocker hits get a WARN qualifier
    P3           zero-shot rows have camera_mode none or default-k
    P4           every eval's result file exists and matches result_sha256; every table
                 value equals the value in its registry-linked result file
    P5           ablation: parent exists and exactly one (resolved) factor changed; root: no
                 parent; finalist: components exist, ablation components are 'accepted'
    P6           int8 calibration ids are recorded, are train.own rows, never frozen/video ids
    CALIB-BENCH  calibration used benchmark-domain images (any benchmark, SUN/D_SUN included)
    P7           V2 checkpoints are the final-epoch EMA with selection 'none' (legacy: WARN)
    P8           lineage recorded for every model, teachers recorded and resolvable for V2
                 runs; unverified lineage => WARN + required qualifier in tables
    P9           Core ML rows carry a claim level; converted-not-executed rows carry no
                 accuracy numbers and are never called validated
    P10          competitor (baseline) rows carry the computed zero-shot label
    A4           models trained on SUN (S16, b1) are barred from D_SUN accuracy tables
    A9           a V2 checkpoint's test result on a report-only set must come after (registry
                 order AND result timestamp) a freeze record covering that run
    CAM-MIX      camera_mode differs between rows of one table and the rows are not labelled

CLI:  python -m benchmark.policy_check [--registry P] [--roles P] [--json OUT] [--matrix]
      exit 0 = no FAIL, 1 = at least one FAIL.

Tests: tests/test_policy.py (env depth or depth_cmp; CPU, offline; roles passed as dicts).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from benchmark.provenance import (DEFAULT_ROLES, REPO_ROOT, load_yaml, resolve_repo_path,
                                  sha256_file)
from benchmark.registry_v2 import (CAMERA_MODES, CORE_ML_CLAIM_LEVELS, KNOWN_BENCHMARKS,
                                   KNOWN_SOURCES, LINEAGE_STATUSES, MODEL_TYPES, TRAINED_TYPES,
                                   canonical_dataset, check_one_factor, is_active, latest_by_id,
                                   load_records, normalize_camera_mode, normalize_factors,
                                   normalize_teachers, pooled_with, resolve_model,
                                   trained_sources, verify_chain)

FAIL, WARN = "fail", "warn"
ZERO_SHOT_LABELS = ("clean", "vendor-claimed")
_LABEL_RANK = {"in-domain": 0, "dev": 1, "not-clean": 2, "unverified": 3, "vendor-claimed": 4,
               "clean": 5}
LABEL_DISPLAY = {"in-domain": "in-domain", "dev": "dev, never zero-shot",
                 "not-clean": "not clean", "unverified": "unverified",
                 "vendor-claimed": "clean per vendor", "clean": "clean",
                 "gt-k-camera": "GT intrinsics: not zero-shot"}
DEFAULT_MATRIX_BENCHMARKS = ("ibims", "eth3d", "diode", "tum", "diml", "nyu", "sun")
DEFAULT_REPORT_ONLY = ("ibims", "eth3d", "diode", "tum", "diml", "nyu")
V2_DEV_DEFAULT = ("sun",)
# Benchmark domains a V2 run may never train on. NYU is excluded: the keep-NYU branch trains on
# it and is then labelled in-domain on NYU (plan 5.1).
V2_TRAIN_ALLOWED_BENCHMARKS = frozenset({"nyu"})
INT8_ARTEFACT_HINTS = ("int8", "u8s8", "qdq", "ptq", "qat")


# ----------------------------------------------------------------------------- results
@dataclass
class Finding:
    rule: str
    severity: str
    subject: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"rule": self.rule, "severity": self.severity, "subject": self.subject,
                "message": self.message}

    def __str__(self) -> str:
        return f"[{self.severity.upper()} {self.rule}] {self.subject}: {self.message}"


@dataclass
class PolicyReport:
    findings: List[Finding] = field(default_factory=list)

    def add(self, rule: str, severity: str, subject: str, message: str) -> None:
        self.findings.append(Finding(rule, severity, str(subject), message))

    def extend(self, other: "PolicyReport") -> "PolicyReport":
        self.findings.extend(other.findings)
        return self

    @property
    def failures(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == FAIL]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def rules_failed(self) -> Set[str]:
        return {f.rule for f in self.failures}

    def summary(self, max_lines: int = 200) -> str:
        head = (f"policy: {'PASS' if self.ok else 'FAIL'} ({len(self.failures)} fail, "
                f"{len(self.warnings)} warn)")
        lines = [head] + [str(f) for f in (self.failures + self.warnings)[:max_lines]]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "n_fail": len(self.failures), "n_warn": len(self.warnings),
                "findings": [f.to_dict() for f in self.findings]}


@dataclass
class ZSLabel:
    label: str
    reasons: List[str] = field(default_factory=list)
    qualifiers: List[str] = field(default_factory=list)
    camera_mode: Optional[str] = None

    @property
    def effective(self) -> str:
        if self.label in ZERO_SHOT_LABELS and self.camera_mode == "gt-k":
            return "gt-k-camera"
        return self.label

    @property
    def zero_shot(self) -> bool:
        return self.effective in ZERO_SHOT_LABELS

    def display(self) -> str:
        return LABEL_DISPLAY.get(self.effective, self.effective)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "effective": self.effective, "zero_shot": self.zero_shot,
                "reasons": list(self.reasons), "qualifiers": list(self.qualifiers),
                "camera_mode": self.camera_mode}


# ----------------------------------------------------------------------------- roles
def _canon_set(values: Iterable[Any]) -> Set[str]:
    return {canonical_dataset(v) for v in values if v is not None and str(v).strip()}


class Roles:
    """Normalised, read-only view of configs/v2/roles.yaml (or of a dict in tests)."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None, *, path: Optional[Path] = None,
                 sha256: Optional[str] = None):
        self.raw = raw or {}
        self.present = raw is not None
        self.path, self.sha256 = path, sha256
        r = self.raw
        self.dev_benchmarks: Set[str] = set(V2_DEV_DEFAULT)
        ro = r.get("report_only")
        if isinstance(ro, dict) and ro:
            self.report_only, self.freeze_required = set(), set()
            for k, v in ro.items():
                if k == "rule":
                    continue
                c = canonical_dataset(k)
                self.freeze_required.add(c)
                never = isinstance(v, dict) and "never" in str(v.get("zero_shot", "")).lower()
                if not never:
                    self.report_only.add(c)
        else:
            self.report_only = set(DEFAULT_REPORT_ONLY)
            self.freeze_required = set(DEFAULT_REPORT_ONLY) | {"sun"}
        self.leak_done: Dict[str, bool] = {}
        lc = r.get("leak_check") or {}
        status = dict(lc.get("status") or {})
        if isinstance(ro, dict):
            for k, v in ro.items():
                if isinstance(v, dict) and v.get("leak_check") is not None:
                    status.setdefault(k, v["leak_check"])
        for k, v in status.items():
            c = canonical_dataset(k)
            self.leak_done[c] = self.leak_done.get(c, False) or str(v).startswith("done")
        nds = r.get("nyu_derived_sun") or {}
        self.nyu_leak_ready = bool(self.leak_done.get("nyu")) and \
            nds.get("matched_to_nyu_eigen_test") is not None
        tr = r.get("train") or {}
        self.train_sources: Optional[Set[str]] = None
        if self.present:
            srcs = {canonical_dataset(k) for k in ("own", "nyu") if tr.get(k)}
            pool = tr.get("pool") or {}
            if isinstance(pool, dict):
                srcs |= _canon_set((pool.get("by_pool") or {}).keys())
            self.train_sources = srcs
        self.train_own_ids: Set[str] = set(tr.get("own") or [])
        self.calibration_ids: Set[str] = set((r.get("calibration") or {}).get("own") or [])
        te = r.get("test") or {}
        self.frozen_ids: Set[str] = set(te.get("frozen_own") or []) | set(te.get("video_own") or [])
        self.hashes: Dict[str, Any] = dict((r.get("hashes") or {}).get("roles") or {})

    def benchmark_domains(self) -> Set[str]:
        return set(KNOWN_BENCHMARKS) | self.report_only | self.freeze_required | \
            self.dev_benchmarks


def load_roles(path: Optional[os.PathLike] = None) -> Roles:
    """Roles from `path` (default $UNIDEPTH_ROLES_V2 or configs/v2/roles.yaml); a missing file
    gives Roles(None) with .present False, which check_registry reports (never silent)."""
    p = Path(path) if path is not None else Path(os.environ.get("UNIDEPTH_ROLES_V2",
                                                                str(DEFAULT_ROLES)))
    if not p.is_file():
        return Roles(None, path=p)
    return Roles(load_yaml(p) or {}, path=p, sha256=sha256_file(p))


def _as_roles(roles: Any) -> Roles:
    if isinstance(roles, Roles):
        return roles
    if roles is None:
        return Roles(None)
    if isinstance(roles, dict):
        return Roles(roles)
    return load_roles(roles)


# ----------------------------------------------------------------------------- lineage
@dataclass
class Lineage:
    status: str
    train: Set[str]
    dev: Set[str]
    selection: Set[str]
    declared_excluded: Set[str]
    teachers: List[str]
    opaque: Set[str]
    trained: bool


def model_lineage(rec: Dict[str, Any]) -> Lineage:
    trained = rec.get("type") in TRAINED_TYPES
    if trained:
        status = "complete"
        dev = _canon_set(normalize_factors(rec.get("dev_sources")) or V2_DEV_DEFAULT)
        sel = _canon_set(normalize_factors(rec.get("selection_sources"))) or set(dev)
        excl: Set[str] = set()
    else:
        lin = rec.get("lineage") or {}
        status = lin.get("status") if lin.get("status") in LINEAGE_STATUSES else "missing"
        dev = _canon_set(normalize_factors(rec.get("dev_sources")))
        sel = _canon_set(normalize_factors(rec.get("selection_sources")))
        excl = _canon_set(normalize_factors(lin.get("declared_excluded")))
    train = set(trained_sources(rec))
    return Lineage(status=status, train=train, dev=dev, selection=sel, declared_excluded=excl,
                   teachers=normalize_teachers(rec.get("teachers")),
                   opaque={s for s in train if s not in KNOWN_SOURCES}, trained=trained)


def _nyu_rule_confirmed(latest: Dict[str, Dict[str, Any]]) -> bool:
    return any(r.get("type") == "freeze" and is_active(r) and r.get("nyu_leak_rule_confirmed")
               is True for r in latest.values())


def _base_label(lin: Lineage, b: str) -> Tuple[str, Optional[str], List[str]]:
    """(label, reason, qualifiers) from the lineage status alone (no contamination found)."""
    st = lin.status
    if st == "missing":
        return "unverified", "no lineage recorded", []
    if st == "unverified":
        return "unverified", "training lineage unverified", ["lineage unverified"]
    if st == "complete":
        return "clean", None, []
    if st == "published-list":
        if not lin.opaque:
            return "clean", None, ["per published training list"]
        if b in lin.declared_excluded:
            return "clean", None, ["declared exclusion"]
        return "unverified", f"training list has opaque entries {sorted(lin.opaque)}", \
            [f"lineage unverified for {b}"]
    if st == "declared-exclusions":
        if b in lin.declared_excluded:
            return "clean", None, ["declared exclusion; training list not verified here"]
        return "unverified", f"{b} is not among the declared exclusions", \
            [f"lineage unverified for {b}"]
    if st == "partial":
        if b in lin.declared_excluded:
            return "vendor-claimed", None, ["zero-shot per vendor/paper; composition unknown"]
        return "unverified", f"composition unknown and {b} not declared zero-shot", \
            [f"lineage unverified for {b}"]
    return "unverified", f"unknown lineage status {st!r}", []


def zero_shot_label(model_rec: Dict[str, Any], benchmark: str, *,
                    latest: Dict[str, Dict[str, Any]], roles: Any = None,
                    calibration_sources: Optional[Sequence[str]] = None,
                    camera_mode: Optional[str] = None) -> ZSLabel:
    """Computed zero-shot label of one model on one benchmark (see module docstring)."""
    rl = _as_roles(roles)
    b = canonical_dataset(benchmark)
    lin = model_lineage(model_rec)
    pooled = pooled_with(b)
    cands: List[Tuple[str, str]] = []
    quals: List[str] = []

    if b in lin.train:
        cands.append(("in-domain", f"{b} is a training source"))
    elif lin.train & pooled:
        cands.append(("not-clean", f"training source(s) {sorted(lin.train & pooled)} contain "
                                   f"{b} images"))
    devsel = lin.dev | lin.selection
    if b in devsel:
        cands.append(("dev", f"{b} is a development/selection set of this model"))
    elif devsel & pooled and b not in lin.train:
        if lin.trained and b == "nyu" and rl.nyu_leak_ready and _nyu_rule_confirmed(latest):
            quals.append("NYU leak rule applied (D_SUN hashed against the official NYU test "
                         "frames; ladder decisions confirmed unchanged)")
        else:
            cands.append(("not-clean", f"development set {sorted(devsel & pooled)} contains "
                                       f"{b}-derived frames"
                                       + ("; NYU leak rule (plan 5.1) not satisfied"
                                          if b == "nyu" else "")))
    calib = _canon_set(calibration_sources or [])
    if b in calib or calib & pooled:
        cands.append(("not-clean", f"int8 calibration used {sorted(calib & (pooled | {b}))} "
                                   "images"))
    for t in lin.teachers:
        trec = resolve_model(t, latest)
        if trec is None:
            quals.append(f"teacher {t} has no registry lineage record")
            continue
        tl = model_lineage(trec)
        tkey = trec.get("model_key") or trec.get("id")
        if b in tl.train:
            cands.append(("not-clean", f"teacher {tkey} trained on {b}"))
        elif tl.train & pooled:
            cands.append(("not-clean", f"teacher {tkey} trained on {sorted(tl.train & pooled)}, "
                                       f"which contains {b} images"))
        elif tl.status in ("unverified", "missing"):
            quals.append(f"teacher {tkey} lineage unverified")
        elif tl.status in ("declared-exclusions", "partial") or \
                (tl.status == "published-list" and tl.opaque):
            if b not in tl.declared_excluded:
                quals.append(f"teacher {tkey} lineage unverified for {b}")
    if lin.trained:
        if not lin.train:
            cands.append(("unverified", "no training sources recorded"))
        if not rl.present:
            cands.append(("unverified", "roles.yaml missing: dev/calibration roles unknown"))
    if not cands:
        lab, reason, q = _base_label(lin, b)
        quals = q + quals
        cands.append((lab, reason or f"{b} absent from training, development, selection and "
                                     "calibration"))
    ev = (model_rec.get("leak_evidence") or {}).get(b) or {}
    if ev.get("flagged_training_rows"):
        quals.append(f"V2 leak blocker flags {ev['flagged_training_rows']} legacy training rows "
                     f"vs {b} (uninspected)")
    worst = min(cands, key=lambda c: _LABEL_RANK.get(c[0], 3))[0]
    reasons = [r for lab, r in cands if lab == worst] + \
        [f"(also {lab}: {r})" for lab, r in cands if lab != worst]
    return ZSLabel(label=worst, reasons=reasons, qualifiers=sorted(set(quals), key=quals.index),
                   camera_mode=normalize_camera_mode(camera_mode))


def zero_shot_matrix(records: Sequence[Dict[str, Any]], roles: Any = None,
                     benchmarks: Sequence[str] = DEFAULT_MATRIX_BENCHMARKS
                     ) -> Dict[str, Dict[str, ZSLabel]]:
    latest = latest_by_id(records)
    out: Dict[str, Dict[str, ZSLabel]] = {}
    for rid, r in latest.items():
        if r.get("type") in MODEL_TYPES and is_active(r):
            out[rid] = {b: zero_shot_label(r, b, latest=latest, roles=roles) for b in benchmarks}
    return out


# ----------------------------------------------------------------------------- helpers
def extract_metric(result: Dict[str, Any], dotted: str) -> Any:
    cur: Any = result
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _parse_time(s: Any) -> Optional[_dt.datetime]:
    """ISO UTC ('...Z' or offset) -> aware; 'YYYY-MM-DD HH:MM[:SS]' -> local time -> aware."""
    if not s:
        return None
    txt = str(s).strip()
    try:
        d = _dt.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.astimezone()            # naive run_bench 'date' is local wall-clock time
    return d.astimezone(_dt.timezone.utc)


def _first_seq(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        rid = r.get("id")
        if rid is not None and rid not in out and isinstance(r.get("seq"), int):
            out[rid] = r["seq"]
    return out


def _calibration_ids(cal: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[str]]:
    """(ids or None, problem or None) from calibration.ids or ids_file (+ ids_sha256)."""
    if cal.get("ids"):
        return [str(i) for i in cal["ids"]], None
    f = cal.get("ids_file")
    if not f:
        return None, None
    p = resolve_repo_path(f)
    if not p.is_file():
        return None, f"calibration ids_file missing: {f}"
    if cal.get("ids_sha256") and sha256_file(p) != cal["ids_sha256"]:
        return None, f"calibration ids_file {f} does not match ids_sha256"
    txt = p.read_text(encoding="utf-8")
    try:
        data = json.loads(txt)
        ids = data.get("ids") if isinstance(data, dict) else data
        return [str(i) for i in ids], None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return [ln.strip() for ln in txt.splitlines() if ln.strip()], None


def _is_int8(rec: Dict[str, Any]) -> bool:
    art = str(rec.get("artefact") or "").lower()
    return not art.startswith("sim-") and any(h in art for h in INT8_ARTEFACT_HINTS)


def _model_of(ev: Dict[str, Any], latest: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ref = ev.get("model_ref")
    if ref:
        return latest.get(ref)
    sha = ev.get("checkpoint_sha256")
    if sha:
        for r in latest.values():
            if r.get("type") in MODEL_TYPES and r.get("checkpoint_sha256") == sha:
                return r
    return None


def _label_for_eval(ev: Dict[str, Any], model: Dict[str, Any], latest, roles) -> ZSLabel:
    cal = ev.get("calibration") or {}
    return zero_shot_label(model, ev.get("benchmark") or ev.get("dataset"), latest=latest,
                           roles=roles, calibration_sources=normalize_factors(cal.get("sources")),
                           camera_mode=ev.get("camera_mode"))


# ----------------------------------------------------------------------------- registry check
def check_registry(records: Optional[Sequence[Dict[str, Any]]] = None, *,
                   registry_path: Optional[os.PathLike] = None, roles: Any = None,
                   verify: bool = True) -> PolicyReport:
    """Every registry-level rule over the active records (see module docstring)."""
    rep = PolicyReport()
    if records is None:
        records = load_records(registry_path)
    if verify:
        ok, problems = verify_chain(registry_path)
        for pr in problems:
            rep.add("REG", FAIL, "registry", pr)
    rl = _as_roles(roles)
    latest = latest_by_id(records)
    first_seq = _first_seq(records)
    active = {k: v for k, v in latest.items() if is_active(v)}
    if any(r.get("type") in TRAINED_TYPES for r in active.values()) and not rl.present:
        rep.add("ROLES", FAIL, "roles.yaml", f"V2 trained runs exist but roles.yaml is missing "
                                             f"({rl.path or 'not given'}): labels cannot be computed")
    freezes = [r for r in active.values() if r.get("type") == "freeze"]
    for rid, rec in active.items():
        t = rec.get("type")
        if t in TRAINED_TYPES:
            _check_trained(rep, rec, latest, rl)
        elif t in ("baseline", "legacy-import"):
            _check_third_party(rep, rec, latest)
        elif t == "freeze":
            for c in normalize_factors(rec.get("covers")):
                cr = latest.get(c)
                if cr is None or cr.get("type") not in TRAINED_TYPES:
                    rep.add("A9", FAIL, rid, f"freeze covers '{c}', which is not a trained run")
        elif t == "eval":
            _check_eval(rep, rec, latest, rl, freezes, first_seq)
    return rep


def _check_trained(rep: PolicyReport, rec: Dict[str, Any], latest, rl: Roles) -> None:
    rid, t = rec["id"], rec["type"]
    # P5
    if t == "ablation":
        for p in check_one_factor(rec, latest):
            rep.add("P5", FAIL, rid, p)
    elif t == "root" and rec.get("parent"):
        rep.add("P5", FAIL, rid, "root record has a parent")
    elif t == "finalist":
        comps = normalize_factors(rec.get("components"))
        if not comps:
            rep.add("P5", FAIL, rid, "finalist lists no accepted components (A19)")
        for c in comps:
            cr = latest.get(c)
            if cr is None or cr.get("type") not in TRAINED_TYPES:
                rep.add("P5", FAIL, rid, f"component '{c}' is not a trained run in the registry")
            elif cr.get("type") == "ablation" and str(cr.get("status")).lower() != "accepted":
                rep.add("P5", FAIL, rid, f"component '{c}' is not accepted "
                                         f"(status {cr.get('status')!r})")
    # P7
    kind = str(rec.get("checkpoint_kind") or "").lower()
    sel = str(rec.get("selection") if rec.get("selection") is not None else "").lower()
    if kind != "final_ema" or sel != "none":
        rep.add("P7", FAIL, rid, f"checkpoint must be the final-epoch EMA with selection 'none' "
                                 f"(got checkpoint_kind={rec.get('checkpoint_kind')!r}, "
                                 f"selection={rec.get('selection')!r})")
    _check_checkpoint_hash(rep, rec, required=True)
    # P8
    if "teachers" not in rec:
        rep.add("P8", FAIL, rid, "teachers not recorded (use [] for sensor-only training)")
    for tch in normalize_teachers(rec.get("teachers")):
        tr = resolve_model(tch, latest)
        if tr is None:
            rep.add("P8", FAIL, rid, f"teacher '{tch}' has no lineage record in the registry")
        elif model_lineage(tr).status in ("unverified", "missing"):
            rep.add("P8", WARN, rid, f"teacher '{tch}' lineage unverified (qualifier printed)")
    # P1 training sources
    srcs = set(trained_sources(rec))
    if not srcs:
        rep.add("P1", FAIL, rid, "no training sources recorded (train_sources / draw_weights)")
    bad = srcs & (rl.benchmark_domains() - V2_TRAIN_ALLOWED_BENCHMARKS)
    if bad:
        rep.add("P1", FAIL, rid, f"benchmark-domain images used for training: {sorted(bad)}")
    if rl.present and rl.train_sources is not None:
        unknown = srcs - rl.train_sources - bad
        if unknown:
            rep.add("P1", FAIL, rid, f"training sources {sorted(unknown)} are not frozen train "
                                     "roles in roles.yaml")
    # ROLES fingerprint
    if rl.present:
        rs, rh = rec.get("roles_sha256"), rec.get("roles_hashes")
        if not rs and not rh:
            rep.add("ROLES", WARN, rid, "no roles fingerprint recorded on the run")
        elif rs and rl.sha256 and rs != rl.sha256:
            if rh and rl.hashes and rh == rl.hashes:
                rep.add("ROLES", WARN, rid, "roles.yaml file changed since the run but every "
                                            "id-list hash is unchanged")
            else:
                rep.add("ROLES", FAIL, rid, "roles.yaml changed after this run (frozen roles "
                                            "violated)")
        elif rh and rl.hashes and rh != rl.hashes:
            rep.add("ROLES", FAIL, rid, "role id-list hashes differ from roles.yaml")
    for k in ("code_tree_hash", "config_hash"):
        if not rec.get(k):
            rep.add("PROV", WARN, rid, f"{k} not recorded")


def _check_checkpoint_hash(rep: PolicyReport, rec: Dict[str, Any], required: bool) -> None:
    rid = rec["id"]
    sha = rec.get("checkpoint_sha256")
    if not sha:
        if required:
            rep.add("P4", FAIL, rid, "checkpoint_sha256 not recorded")
        return
    cp = rec.get("checkpoint_path")
    if cp:
        p = resolve_repo_path(cp)
        if p.is_file() and sha256_file(p) != sha:
            rep.add("P4", FAIL, rid, f"checkpoint {cp} no longer matches checkpoint_sha256")


def _check_third_party(rep: PolicyReport, rec: Dict[str, Any], latest) -> None:
    rid = rec["id"]
    lin = rec.get("lineage") or {}
    if lin.get("status") not in LINEAGE_STATUSES:
        rep.add("P8", FAIL, rid, f"lineage.status must be one of {LINEAGE_STATUSES}")
    elif lin.get("status") == "unverified":
        rep.add("P8", WARN, rid, "training lineage unverified (qualifier printed)")
    if not lin.get("evidence"):
        rep.add("P8", WARN, rid, "lineage has no evidence citation")
    if rec.get("type") == "legacy-import":
        if str(rec.get("checkpoint_kind") or "").lower() != "final_ema":
            rep.add("P7", WARN, rid, f"legacy checkpoint selected by validation "
                                     f"({rec.get('selection')!r}); not a V2 final-EMA checkpoint")
        _check_checkpoint_hash(rep, rec, required=True)


def _check_eval(rep: PolicyReport, ev: Dict[str, Any], latest, rl: Roles,
                freezes: List[Dict[str, Any]], first_seq: Dict[str, int]) -> None:
    rid = ev["id"]
    legacy = bool(ev.get("legacy_provenance") or ev.get("legacy"))
    model = _model_of(ev, latest)
    if model is None:
        rep.add("P4", FAIL, rid, f"model {ev.get('model_ref') or ev.get('checkpoint_sha256')!r} "
                                 "is not in the registry")
        return
    if not is_active(model):
        rep.add("P4", WARN, rid, f"evaluates inactive model {model['id']} "
                                 f"(status {model.get('status')!r}); excluded from tables")
    # P4 result hash
    rf = ev.get("result_file")
    if not rf:
        rep.add("P4", FAIL, rid, "no result_file")
    else:
        p = resolve_repo_path(rf)
        if not p.is_file():
            rep.add("P4", FAIL, rid, f"result file missing: {rf}")
        elif sha256_file(p) != ev.get("result_sha256"):
            rep.add("P4", FAIL, rid, f"result file {rf} does not match its registry hash")
    sf = ev.get("source_file")
    if sf and ev.get("source_sha256") and sf != rf:
        sp = resolve_repo_path(sf)
        if sp.is_file() and sha256_file(sp) != ev["source_sha256"]:
            rep.add("P4", WARN, rid, f"source {sf} changed since registration (the registered "
                                     "snapshot is still valid)")
    cm = normalize_camera_mode(ev.get("camera_mode"))
    if cm not in CAMERA_MODES:
        rep.add("P3", FAIL, rid, f"camera_mode {ev.get('camera_mode')!r} not in {CAMERA_MODES}")
    # calibration
    cal = ev.get("calibration")
    if cal or _is_int8(ev):
        cal = cal or {}
        srcs = _canon_set(normalize_factors(cal.get("sources")))
        bench = srcs & rl.benchmark_domains()
        if bench:
            rep.add("CALIB-BENCH", FAIL, rid, f"int8 calibration used benchmark-domain images "
                                              f"{sorted(bench)}")
        if not srcs:
            rep.add("P6", WARN if legacy else FAIL, rid, "calibration sources not recorded")
        elif srcs - {"own"}:
            rep.add("P6", FAIL, rid, f"calibration sources must be own training rows only, got "
                                     f"{sorted(srcs)}")
        ids, prob = _calibration_ids(cal)
        if prob:
            rep.add("P6", FAIL, rid, prob)
        elif ids is None:
            rep.add("P6", WARN if legacy else FAIL, rid,
                    "calibration ids not recorded: frozen-id exclusion cannot be verified")
        elif rl.present:
            fz = set(ids) & rl.frozen_ids
            if fz:
                rep.add("P6", FAIL, rid, f"{len(fz)} calibration ids are frozen/video test ids")
            notrain = set(ids) - rl.train_own_ids
            if notrain:
                rep.add("P6", FAIL, rid, f"{len(notrain)} calibration ids are not train.own rows")
            if rl.calibration_ids and not set(ids) <= rl.calibration_ids:
                rep.add("P6", WARN, rid, "calibration ids are not the frozen roles.calibration set")
        elif not legacy:
            rep.add("P6", FAIL, rid, "roles.yaml missing: calibration ids cannot be checked")
    # A9 freeze before report-only results of V2 checkpoints
    bench = canonical_dataset(ev.get("benchmark") or ev.get("dataset"))
    split = ev.get("split") or "test"
    if model.get("type") in TRAINED_TYPES and split == "test" and bench in rl.freeze_required:
        covering = [f for f in freezes if model["id"] in normalize_factors(f.get("covers"))
                    or (f.get("recipe_hash") and f.get("recipe_hash") == model.get("recipe_hash"))]
        eseq = first_seq.get(rid, ev.get("seq", 0))
        prior = [f for f in covering if first_seq.get(f["id"], f.get("seq", 0)) < eseq]
        if not prior:
            rep.add("A9", FAIL, rid, f"report-only result on {bench} for V2 run {model['id']} "
                                     "predates (or has no) recipe-freeze record")
        else:
            ft = min((_parse_time(f.get("created_utc")) for f in prior),
                     key=lambda d: d or _dt.datetime.max.replace(tzinfo=_dt.timezone.utc))
            rt = _parse_time(ev.get("result_date"))
            if ft and rt and rt < ft - _dt.timedelta(seconds=60):
                rep.add("A9", FAIL, rid, f"result computed at {rt.isoformat()} before the freeze "
                                         f"record ({ft.isoformat()})")
    # P9 Core ML
    if str(ev.get("artefact") or "").lower().startswith("coreml"):
        lvl = ev.get("claim_level")
        if lvl not in CORE_ML_CLAIM_LEVELS:
            rep.add("P9", FAIL, rid, f"Core ML row without a claim level in {CORE_ML_CLAIM_LEVELS}")
        if lvl == "converted-not-executed" and rf:
            p = resolve_repo_path(rf)
            try:
                res = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
            except (OSError, json.JSONDecodeError):
                res = {}
            if isinstance(res, dict) and res.get("depth"):
                rep.add("P9", FAIL, rid, "converted-not-executed Core ML row carries accuracy "
                                         "numbers (it was never executed)")
        if str(ev.get("claim") or "").lower() == "validated" and lvl != "validated-on-device":
            rep.add("P9", FAIL, rid, "Core ML row called validated without device validation")
    # P1 / P2 / P3 on labels
    lab = _label_for_eval(ev, model, latest, rl)
    if str(ev.get("claim") or "").lower() == "zero-shot":
        if cm == "gt-k":
            rep.add("P3", FAIL, rid, "zero-shot claim on a gt-k (GT intrinsics) result")
        if lab.label not in ZERO_SHOT_LABELS:
            rep.add("P1", FAIL, rid, f"claims zero-shot but the computed label is {lab.label}: "
                                     + "; ".join(lab.reasons))
    if lab.label == "clean" and split == "test":
        if model.get("type") in TRAINED_TYPES and not rl.leak_done.get(bench):
            rep.add("P2", FAIL, rid, f"leak check against {bench} not done in roles.yaml")
        elif any("leak blocker" in q for q in lab.qualifiers):
            rep.add("P2", WARN, rid, next(q for q in lab.qualifiers if "leak blocker" in q))
    if split == "dev" and bench in model_lineage(model).train:
        rep.add("A4", WARN, rid, "model trained on the development domain; barred from D_SUN "
                                 "accuracy tables")


# ----------------------------------------------------------------------------- table check
def check_table(rows: Sequence[Dict[str, Any]], *, records: Sequence[Dict[str, Any]],
                roles: Any = None, kind: str = "zero-shot",
                camera_labelled: bool = False) -> PolicyReport:
    """Rules that apply to one generated table (kind: zero-shot | ablation | comparison)."""
    rep = PolicyReport()
    rl = _as_roles(roles)
    latest = latest_by_id(records)
    modes = set()
    unlabelled = 0
    cache: Dict[str, Optional[Dict[str, Any]]] = {}
    for i, row in enumerate(rows):
        subj = f"row {i} ({row.get('eval_id')})"
        ev = latest.get(row.get("eval_id") or "")
        if ev is None or ev.get("type") != "eval" or not is_active(ev):
            rep.add("P4", FAIL, subj, "table number does not trace to an active registry eval")
            continue
        model = _model_of(ev, latest)
        if model is None:
            rep.add("P4", FAIL, subj, "eval's model is not in the registry")
            continue
        rf = ev.get("result_file")
        if rf not in cache:
            p = resolve_repo_path(rf) if rf else None
            if p is None or not p.is_file() or sha256_file(p) != ev.get("result_sha256"):
                cache[rf] = None
            else:
                try:
                    cache[rf] = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cache[rf] = None
        res = cache[rf]
        if res is None:
            rep.add("P4", FAIL, subj, "result file missing or hash mismatch")
        elif row.get("metric"):
            true = extract_metric(res, row["metric"])
            val = row.get("value")
            tol = float(row.get("tol", 1e-9))
            if (true is None) != (val is None) or (
                    true is not None and abs(float(true) - float(val)) > tol):
                rep.add("P4", FAIL, subj, f"value {val!r} != {row['metric']}={true!r} in the "
                                          "registry-linked result")
        cm = normalize_camera_mode(row.get("camera_mode") or ev.get("camera_mode"))
        if row.get("camera_mode") and cm != normalize_camera_mode(ev.get("camera_mode")):
            rep.add("P3", FAIL, subj, f"row camera_mode {row.get('camera_mode')!r} != registry "
                                      f"{ev.get('camera_mode')!r}")
        modes.add(cm)
        if not row.get("camera_label"):
            unlabelled += 1
        lab = _label_for_eval(ev, model, latest, rl)
        claimed = row.get("label")
        rule = "P10" if model.get("type") == "baseline" else "P1"
        if claimed is None:
            if kind == "zero-shot":
                rep.add(rule, FAIL, subj, "row has no zero-shot label")
        elif claimed != lab.effective:
            rep.add(rule, FAIL, subj, f"label {claimed!r} != computed {lab.effective!r} "
                                      f"({'; '.join(lab.reasons)})")
        if claimed in ZERO_SHOT_LABELS and cm == "gt-k":
            rep.add("P3", FAIL, subj, "zero-shot label on a gt-k row")
        if kind == "zero-shot":
            have = " | ".join(normalize_factors(row.get("qualifiers")))
            for q in lab.qualifiers:
                if ("unverified" in q or "lineage" in q) and q not in have:
                    rep.add("P8", FAIL, subj, f"missing lineage qualifier: {q}")
        if str(ev.get("artefact") or "").lower().startswith("coreml"):
            if row.get("claim_level") != ev.get("claim_level"):
                rep.add("P9", FAIL, subj, "Core ML row must print the registry claim level")
            if str(row.get("claim") or "").lower() == "validated" and \
                    ev.get("claim_level") != "validated-on-device":
                rep.add("P9", FAIL, subj, "converted-only Core ML row called validated")
        bench = canonical_dataset(ev.get("benchmark") or ev.get("dataset"))
        if (ev.get("split") == "dev" or kind == "ablation") and \
                bench in model_lineage(model).train:
            rep.add("A4", FAIL, subj, f"{model['id']} trained on {bench}; barred from D_SUN "
                                      "accuracy tables")
        if model.get("type") in TRAINED_TYPES and \
                str(model.get("checkpoint_kind") or "").lower() != "final_ema":
            rep.add("P7", FAIL, subj, "V2 row from a checkpoint that is not the final EMA")
    if len(modes) > 1 and not camera_labelled and unlabelled:
        rep.add("CAM-MIX", FAIL, "table", f"camera_mode differs between rows {sorted(modes)} and "
                                          f"{unlabelled} row(s) carry no camera label")
    return rep


# ----------------------------------------------------------------------------- entry points
def run_policy(registry_path: Optional[os.PathLike] = None,
               roles_path: Optional[os.PathLike] = None) -> PolicyReport:
    return check_registry(registry_path=registry_path, roles=load_roles(roles_path))


def _matrix_markdown(matrix: Dict[str, Dict[str, ZSLabel]], benchmarks: Sequence[str]) -> str:
    lines = ["| Model | " + " | ".join(benchmarks) + " |",
             "|---|" + "---|" * len(benchmarks)]
    for mid in sorted(matrix):
        lines.append(f"| {mid} | " + " | ".join(matrix[mid][b].display() for b in benchmarks)
                     + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="V2 policy checks P1-P10 and zero-shot matrix")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--roles", default=None)
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--matrix", action="store_true", help="print the computed label matrix")
    a = ap.parse_args(argv)
    roles = load_roles(a.roles)
    rep = check_registry(registry_path=a.registry, roles=roles)
    print(rep.summary())
    if not roles.present:
        print(f"note: roles file not found at {roles.path}")
    if a.matrix:
        recs = load_records(a.registry)
        print(_matrix_markdown(zero_shot_matrix(recs, roles), DEFAULT_MATRIX_BENCHMARKS))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
