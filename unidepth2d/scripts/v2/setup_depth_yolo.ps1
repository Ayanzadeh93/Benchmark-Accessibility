<#
.SYNOPSIS
  Create the isolated `depth_yolo` conda env for the YOLO26-Depth baseline and fetch its weights.

.DESCRIPTION
  Contract (V2 plan, owner decision 2026-11-19, amendment A6):
    1. `depth_yolo` is a clone of `depth` (python 3.13, torch 2.11.0+cu128), created with
       `conda create -n depth_yolo --clone depth`. It is the ONLY env that receives ultralytics
       >= 8.4.104 (the release that added the depth task). The `depth` env is never modified:
       its ultralytics and torch versions are read before and after and must be unchanged.
    2. ultralytics is pinned to an exact version (default 8.4.202, uploaded to PyPI
       2026-11-03T23:32 UTC, i.e. more than two weeks old on 2026-11-19). pip runs with a
       constraints file that pins the clone's own torch / torchvision / numpy / opencv versions,
       so installing ultralytics can never replace torch. xformers is never installed.
    3. Weights are downloaded ONLY for the sizes requested (default n and s, 39.8 MB in total)
       into outputs/v2/weights/yolo26/ from the official GitHub release asset URL
       https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26<s>-depth.pt.
       The exact byte sizes below were taken from an HTTP HEAD on 2026-11-19; a file whose size
       differs is rejected. sha256, size, URL and ultralytics version go to weights_manifest.json.
    4. A partial clone (env folder present but no python.exe, e.g. an interrupted earlier run) is
       NOT silently reused: the script stops unless -ReplaceBroken is given, in which case that
       folder (and only a folder named depth_yolo under <CondaRoot>\envs) is removed first.
    5. Licence: Ultralytics YOLO26 code and weights are AGPL-3.0. They live only in this env and
       this weights folder; V2 code never imports ultralytics (benchmark/adapters/yolo26.py imports
       it lazily, only inside depth_yolo).

  Usage (from the repo root, PowerShell 7):
    pwsh -File scripts/v2/setup_depth_yolo.ps1                    # clone + install + n,s weights
    pwsh -File scripts/v2/setup_depth_yolo.ps1 -ReplaceBroken     # same, replacing a partial clone
    pwsh -File scripts/v2/setup_depth_yolo.ps1 -Sizes n,s,m,l,x   # all five (245 MiB, owner-approved)
    pwsh -File scripts/v2/setup_depth_yolo.ps1 -SkipClone         # env exists: install/verify only
  Then run the baseline (Z: or DATASETS_ROOT must hold IBims-1):
    $env:PYTHONUTF8=1; & "$env:LOCALAPPDATA\anaconda3\envs\depth_yolo\python.exe" -m benchmark.run_bench --dataset ibims --model yolo26n-depth --limit 5
#>
param(
  [string]$Version = "8.4.202",
  [string[]]$Sizes = @("n", "s"),
  [string]$CondaRoot = "$env:LOCALAPPDATA\anaconda3",
  [string]$WeightsDir = "",
  [switch]$SkipClone,
  [switch]$SkipWeights,
  [switch]$ReplaceBroken
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $WeightsDir) { $WeightsDir = Join-Path $RepoRoot "outputs\v2\weights\yolo26" }
$Conda = Join-Path $CondaRoot "condabin\conda.bat"
$EnvsDir = Join-Path $CondaRoot "envs"
$PyDepth = Join-Path $EnvsDir "depth\python.exe"
$YoloDir = Join-Path $EnvsDir "depth_yolo"
$PyYolo = Join-Path $YoloDir "python.exe"
# exact Content-Length of the v8.4.0 release assets (HTTP HEAD, 2026-11-19)
$WeightBytes = @{ n = 12998295; s = 26764887; m = 46926343; l = 55899735; x = 114629975 }

function Assert-Ok([string]$what) {
  if ($LASTEXITCODE -ne 0) { throw "setup_depth_yolo: $what failed (exit $LASTEXITCODE)" }
}

function Get-DepthStamp {
  $s = & $PyDepth -c "import importlib.metadata as m; print(m.version('ultralytics'), m.version('torch'))" 2>$null
  return ($s | Out-String).Trim()
}

foreach ($s in $Sizes) {
  if (-not $WeightBytes.ContainsKey($s)) { throw "unknown YOLO26 size '$s' (choose from n,s,m,l,x)" }
}
if (-not (Test-Path $PyDepth)) { throw "source env 'depth' not found at $PyDepth" }
$depthBefore = Get-DepthStamp
Write-Host "[depth] ultralytics/torch before: $depthBefore (must stay unchanged)"

# 1. clone -------------------------------------------------------------------------------
if (Test-Path $PyYolo) {
  Write-Host "[depth_yolo] env already exists at $PyYolo"
} elseif ($SkipClone) {
  throw "-SkipClone given but $PyYolo does not exist"
} else {
  if (Test-Path $YoloDir) {
    if (-not $ReplaceBroken) {
      throw ("$YoloDir exists but has no python.exe (partial clone from an interrupted run). " +
             "Re-run with -ReplaceBroken to delete that folder and clone again.")
    }
    $resolved = (Resolve-Path $YoloDir).Path
    if ((Split-Path $resolved -Leaf) -ne "depth_yolo" -or
        (Split-Path $resolved -Parent) -ne (Resolve-Path $EnvsDir).Path) {
      throw "refusing to delete unexpected path $resolved"
    }
    Write-Host "[depth_yolo] removing partial clone $resolved"
    & $Conda env remove -n depth_yolo -y 2>$null | Out-Null
    if (Test-Path $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
  }
  Write-Host "[depth_yolo] cloning depth -> depth_yolo (several minutes)"
  & $Conda create -n depth_yolo --clone depth -y
  Assert-Ok "conda create --clone"
  if (-not (Test-Path $PyYolo)) { throw "clone finished but $PyYolo is missing" }
}

# 2. install ultralytics (clone only), torch & co pinned by constraints -------------------
$constraints = Join-Path $env:TEMP "depth_yolo_constraints.txt"
& $PyYolo -c @"
import importlib.metadata as m
pins = []
for p in ('torch', 'torchvision', 'numpy', 'opencv-python', 'opencv-python-headless', 'pillow'):
    try:
        pins.append(f'{p}=={m.version(p)}')
    except m.PackageNotFoundError:
        pass
open(r'$constraints', 'w').write('\n'.join(pins) + '\n')
print('[depth_yolo] constraints:', ', '.join(pins))
"@
Assert-Ok "write constraints"
Write-Host "[depth_yolo] pip install ultralytics==$Version"
& $PyYolo -m pip install --disable-pip-version-check "ultralytics==$Version" -c $constraints
Assert-Ok "pip install ultralytics"

# 3. verify ------------------------------------------------------------------------------
& $PyYolo -c @"
import importlib.util, sys
import torch, ultralytics
v = tuple(int(x) for x in ultralytics.__version__.split('.')[:3])
assert v >= (8, 4, 104), f'ultralytics {ultralytics.__version__} predates the depth task (8.4.104)'
assert importlib.util.find_spec('ultralytics.models.yolo.depth') is not None, 'no ultralytics.models.yolo.depth'
assert importlib.util.find_spec('xformers') is None, 'xformers must not be installed'
print('[depth_yolo] python', sys.version.split()[0], 'torch', torch.__version__,
      'cuda', torch.cuda.is_available(), 'ultralytics', ultralytics.__version__)
"@
Assert-Ok "env verification"

$depthAfter = Get-DepthStamp
if ($depthAfter -ne $depthBefore) {
  throw "the depth env changed ($depthBefore -> $depthAfter); investigate"
}
Write-Host "[depth] ultralytics/torch after: $depthAfter (unchanged)"

# 4. weights -----------------------------------------------------------------------------
if (-not $SkipWeights) {
  New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null
  $spec = ($Sizes | ForEach-Object { "$($_):$($WeightBytes[$_])" }) -join ","
  & $PyYolo -c @"
import datetime, hashlib, json, os, urllib.request
from pathlib import Path
import ultralytics
wdir = Path(r'$WeightsDir')
base = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/'
man_path = wdir / 'weights_manifest.json'
man = json.loads(man_path.read_text()) if man_path.exists() else {}
for item in '$spec'.split(','):
    s, want = item.split(':')
    want = int(want)
    name = f'yolo26{s}-depth.pt'
    dest, part = wdir / name, wdir / (name + '.part')
    if dest.exists() and dest.stat().st_size != want:
        raise SystemExit(f'{dest} has {dest.stat().st_size} bytes, expected {want}; delete it and rerun')
    if not dest.exists():
        req = urllib.request.Request(base + name, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r, open(part, 'wb') as f:
            n = int(r.headers.get('Content-Length') or -1)
            if n != want:
                raise SystemExit(f'{name}: server says {n} bytes, expected {want}; refusing')
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        if part.stat().st_size != want:
            raise SystemExit(f'{name}: got {part.stat().st_size} bytes, expected {want}')
        os.replace(part, dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    man[name] = {'path': str(dest), 'url': base + name, 'bytes': dest.stat().st_size, 'sha256': h,
                 'ultralytics': ultralytics.__version__, 'licence': 'AGPL-3.0',
                 'fetched': datetime.date.today().isoformat()}
    print(f'[weights] {name} {dest.stat().st_size} bytes sha256={h}')
man_path.write_text(json.dumps(man, indent=2))
print('[weights] manifest ->', man_path)
"@
  Assert-Ok "weights download"
}
Write-Host "[depth_yolo] ready: $PyYolo"
