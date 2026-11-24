# b10 / b20 queue (single GPU, strictly sequential). b10 already trained (outputs/checkpoints/b10).
# 1) train b20 (full CanonScale-Rep recipe)  2) train b20core (same minus rep4, attribution control)
# 3) isolate best/last checkpoints  4) dense NYU/SUN/IBims + OCMR-SUN + q:8 + latency  5) paired stats.
# Spec: docs/B20_SPEC.md. Log: outputs/b20_queue.log

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\b20_queue.log"
Add-Content $log ("`n===== B20 QUEUE START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$C = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints"
$R = "s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,selgt"
$train = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone","efficientvit_b1.r224_in1k")

function Run-Step($name, $envname, $argv) {
    $t0 = Get-Date
    Add-Content $log ("[{0}] START {1}" -f $t0.ToString("HH:mm:ss"), $name)
    if ($envname -eq "base") { & python -u @argv *>> $log }
    else { & conda run -n $envname --no-capture-output python -u @argv *>> $log }
    $code = $LASTEXITCODE
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content $log ("[{0}] END   {1}  exit={2}  ({3}s)" -f (Get-Date -Format "HH:mm:ss"), $name, $code, $dt)
    return $code
}

# ---- training (base env, same as b10) ----
Run-Step "T1_train_b20" "base" ($train + @("--recipe","$R,rep4","--ckpt-subdir","b20","--name","b20"))
Run-Step "T2_train_b20core" "base" ($train + @("--recipe",$R,"--ckpt-subdir","b20core","--name","b20core"))

# ---- isolate checkpoints (run_bench names outputs after the parent dir) ----
foreach ($m in "b10","b20","b20core") { foreach ($k in "best","last") {
    $src = "outputs/checkpoints/$m/$k.pt"
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force "outputs/checkpoints/${m}_$k" | Out-Null
        Copy-Item $src "outputs/checkpoints/${m}_$k/best.pt" -Force } } }

$five = @("--model","$C/b10_last/best.pt","--model","$C/b10_best/best.pt","--model","$C/b20_last/best.pt",
          "--model","$C/b20_best/best.pt","--model","$C/b20core_last/best.pt")
$three = @("--model","$C/b10_last/best.pt","--model","$C/b20_last/best.pt","--model","$C/b20core_last/best.pt")

# ---- benchmark datasets live on the Z: drive; wait for it (up to 12 h) ----
$waited = 0
while (-not (Test-Path "Z:\Datasets\IBims-1") -and $waited -lt 144) {
    if ($waited -eq 0) { Add-Content $log ("[{0}] WAIT  Z:\Datasets not mounted - polling every 5 min" -f (Get-Date -Format "HH:mm:ss")) }
    Start-Sleep -Seconds 300; $waited++ }
if (-not (Test-Path "Z:\Datasets\IBims-1")) {
    Add-Content $log "ABORT eval: Z:\Datasets still missing after 12 h. Rerun from E1 when the drive is attached."
    exit 2 }

# ---- accuracy (depth env, official v2 harness) ----
Run-Step "E1_ibims" "depth" (@("-m","benchmark.run_bench","--dataset","ibims","--boundary","--save-per-image") + $five)
Run-Step "E2_nyu"   "depth" (@("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--save-per-image") + $five)
Run-Step "E3_sun"   "depth" (@("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--save-per-image") + $three)
foreach ($m in "b10_last","b20_last","b20core_last") {
    Run-Step "E4_ocmr_$m" "depth" @("-m","benchmark.ocmr_eval_ann","--model","$C/$m/best.pt") }
Run-Step "E5_ibims_q8" "depth" @("-m","benchmark.run_bench","--dataset","ibims","--boundary","--tag","q8",
    "--model","q:8:C:/Tim/nature/unidepth2d/outputs/checkpoints/b10_last/best.pt",
    "--model","q:8:C:/Tim/nature/unidepth2d/outputs/checkpoints/b20_last/best.pt")

# ---- latency (GPU idle; same rig as RESULTS_DIGEST) ----
if (Test-Path outputs/checkpoints/fps_bench.json) { Copy-Item outputs/checkpoints/fps_bench.json outputs/checkpoints/fps_bench_prev.json -Force }
Run-Step "L1_fps" "depth" @("-m","benchmark.fps_bench","--config","config.yaml",
    "--model","$C/b10_last/best.pt","--model","$C/b20_last/best.pt","--model","$C/b20core_last/best.pt",
    "--model","$C/b1_mix/best.pt","--model","$C/S16_prox/best.pt")

Add-Content $log ("===== B20 QUEUE DONE {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Output "b20 queue finished; log: $log"
