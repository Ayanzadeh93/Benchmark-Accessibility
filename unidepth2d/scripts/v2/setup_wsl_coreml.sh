#!/usr/bin/env bash
# setup_wsl_coreml.sh - build the WSL (Ubuntu-24.04) Python venv used for Core ML conversion.
#
# CONTRACT
#   Run INSIDE WSL, e.g. from Windows:
#     wsl -d Ubuntu-24.04 -- bash /mnt/c/Tim/nature/unidepth2d/scripts/v2/setup_wsl_coreml.sh
#   Creates (or reuses) the venv  ${COREML_VENV:-$HOME/venvs/coreml}  with exactly:
#     torch==2.7.0+cpu, torchvision==0.22.0+cpu   (from download.pytorch.org/whl/cpu; CPU only)
#     coremltools==9.0, numpy<2.4, timm==1.0.27   (from PyPI, torch pinned by a constraints file)
#   then verifies the versions, runs `pip check`, confirms coremltools does NOT emit its
#   "torch version not tested" warning, converts a tiny traced conv net to an ML Program with
#   skip_model_load=True, and writes a JSON environment stamp to $COREML_VENV/coreml_env.json.
#   Exit code 0 only if every check passes. Idempotent: re-running reuses the venv.
#
# WHY THESE PINS (verified 2026-11-19)
#   - coremltools 9.0 (PyPI latest, 2025-11-10) ships manylinux wheels for cp37..cp313, no Windows
#     wheels, so conversion needs WSL or a Mac. Its _deps sets _TORCH_MAX_VERSION = 2.7.0 and it
#     warns for anything newer, so torch is pinned to 2.7.0 (2.7.1 would already warn).
#   - torchvision 0.22.0 is the release paired with torch 2.7.0. timm requires torchvision; without
#     this pin pip would pull the newest torchvision from PyPI and silently upgrade torch to a CUDA
#     build (multi-GB). The constraints file makes that impossible.
#   - numpy<2.4: coremltools issue 2633 (NumPy 2.4 breaks some 9.0 conversions; fixed after 9.0).
#   - Ubuntu 24.04 ships Python 3.12.3 without ensurepip; python3.12-venv is installed with apt
#     when missing (root or passwordless sudo only; otherwise the script stops with instructions).
#   Download sizes: torch 2.7.0+cpu cp312 wheel 175.8 MB, torchvision 2.0 MB, coremltools 2.3 MB,
#   timm 2.6 MB, plus small pure-Python deps (sympy, networkx, protobuf, ...).
#
# Core ML artefacts produced in this venv can be CONVERTED but never EXECUTED on Linux: Apple's
# Core ML runtime only exists on macOS/iOS. Every artefact is therefore labelled
# "converted, not executed" by edge/coreml_convert.py.
set -euo pipefail

VENV="${COREML_VENV:-$HOME/venvs/coreml}"
TORCH_SPEC="torch==2.7.0+cpu"
TV_SPEC="torchvision==0.22.0+cpu"
CT_SPEC="coremltools==9.0"
NUMPY_SPEC="numpy<2.4"
TIMM_SPEC="timm==1.0.27"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

log() { printf '[setup_wsl_coreml] %s\n' "$*"; }
die() { printf '[setup_wsl_coreml] ERROR: %s\n' "$*" >&2; exit 2; }

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  log "warning: this does not look like WSL; continuing anyway (plain Linux also works)"
fi

# ---- 1. a Python 3.10-3.13 interpreter --------------------------------------------------------
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v "$cand")"; break; fi
  done
fi
[ -n "$PY" ] || die "no python3 found"
PYVER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
case "$PYVER" in
  3.10|3.11|3.12|3.13) ;;
  *) die "python $PYVER is outside 3.10-3.13 (torch 2.7.0 + coremltools 9.0 wheels)";;
esac
log "python: $PY ($PYVER)"

# ---- 2. venv support (Ubuntu strips ensurepip into python3.X-venv) ----------------------------
if [ ! -x "$VENV/bin/python" ] && ! "$PY" -c "import ensurepip" >/dev/null 2>&1; then
  PKG="python${PYVER}-venv"
  if [ "$(id -u)" = "0" ]; then
    log "installing $PKG with apt (running as root)"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$PKG"
  elif sudo -n true >/dev/null 2>&1; then
    log "installing $PKG with sudo apt"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$PKG"
  else
    die "ensurepip is missing. Run: sudo apt-get install -y $PKG   then re-run this script."
  fi
fi

# ---- 3. venv + pinned packages ---------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv $VENV"
  mkdir -p "$(dirname "$VENV")"
  "$PY" -m venv "$VENV"
else
  log "reusing venv $VENV"
fi
VPY="$VENV/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$VPY" -m pip install --quiet --upgrade pip

CONS="$VENV/constraints.txt"
printf '%s\n%s\n' "$TORCH_SPEC" "$TV_SPEC" > "$CONS"

log "installing $TORCH_SPEC $TV_SPEC from $TORCH_INDEX"
"$VPY" -m pip install --quiet --index-url "$TORCH_INDEX" "$TORCH_SPEC" "$TV_SPEC"
log "installing $CT_SPEC '$NUMPY_SPEC' $TIMM_SPEC (torch pinned by $CONS)"
"$VPY" -m pip install --quiet -c "$CONS" "$CT_SPEC" "$NUMPY_SPEC" "$TIMM_SPEC"
log "pip check"
"$VPY" -m pip check

# ---- 4. verification + tiny conversion smoke test --------------------------------------------
log "verifying versions and converting a tiny traced conv net"
"$VPY" - "$VENV/coreml_env.json" <<'PYEOF'
import json, logging, os, platform, sys, tempfile, warnings

out_path = sys.argv[1]
records = []

class _Grab(logging.Handler):
    def emit(self, rec):
        records.append(rec.getMessage())

logging.getLogger().addHandler(_Grab())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import numpy as np
    import torch, torchvision, timm
    import coremltools as ct
msgs = [str(w.message) for w in caught] + records
untested = [m for m in msgs if "not been tested" in m or "untested" in m.lower()]

def base(v):
    return v.split("+")[0]

problems = []
if base(torch.__version__) != "2.7.0": problems.append(f"torch {torch.__version__} != 2.7.0")
if "+cpu" not in torch.__version__: problems.append(f"torch {torch.__version__} is not the CPU build")
if base(torchvision.__version__) != "0.22.0": problems.append(f"torchvision {torchvision.__version__}")
if ct.__version__ != "9.0": problems.append(f"coremltools {ct.__version__} != 9.0")
if timm.__version__ != "1.0.27": problems.append(f"timm {timm.__version__} != 1.0.27")
npv = tuple(int(x) for x in np.__version__.split(".")[:2])
if npv >= (2, 4): problems.append(f"numpy {np.__version__} is not < 2.4")
if untested: problems.append(f"coremltools warned about torch: {untested}")

# tiny conversion: conv -> relu -> bilinear x2, ML Program, Float16, skip_model_load=True
net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                          torch.nn.Upsample(scale_factor=2.0, mode="bilinear")).eval()
ex = torch.rand(1, 3, 32, 48)
traced = torch.jit.trace(net, ex)
mlm = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=ex.shape)],
                 convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                 minimum_deployment_target=ct.target.iOS17, skip_model_load=True)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "tiny.mlpackage")
    mlm.save(p)
    smoke_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)

stamp = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "coremltools": ct.__version__,
    "numpy": np.__version__,
    "timm": timm.__version__,
    "coremltools_torch_untested_warning": bool(untested),
    "smoke_conversion_mlpackage_bytes": smoke_bytes,
    "core_ml_runtime_available": False,
    "problems": problems,
}
with open(out_path, "w") as f:
    json.dump(stamp, f, indent=2)
print(json.dumps(stamp, indent=2))
sys.exit(1 if problems else 0)
PYEOF
log "OK: venv ready at $VENV (stamp: $VENV/coreml_env.json)"
