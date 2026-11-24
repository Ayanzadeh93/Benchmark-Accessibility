# b20dws = b20core recipe + dws4 (plain depthwise-separable stride-4 blocks; replaces the unstable
# multi-branch rep4). Waits for the MAE re-run to free the GPU, trains 40 ep, evaluates everything.
$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\b20_queue.log"
while (-not (Select-String -Path $log -Pattern "MAE RERUN DONE" -SimpleMatch -Quiet)) { Start-Sleep -Seconds 60 }
Add-Content $log ("`n===== B20DWS START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

function Run-Step($name, $envname, $argv) {
    $t0 = Get-Date
    Add-Content $log ("[{0}] START {1}" -f $t0.ToString("HH:mm:ss"), $name)
    if ($envname -eq "base") { & python -u @argv *>> $log }
    else { & conda run -n $envname --no-capture-output python -u @argv *>> $log }
    Add-Content $log ("[{0}] END   {1}  exit={2}  ({3}s)" -f (Get-Date -Format "HH:mm:ss"), $name, $LASTEXITCODE, [int]((Get-Date) - $t0).TotalSeconds)
}
$C = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints"
Run-Step "D1_train_b20dws" "base" @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics",
    "--backbone","efficientvit_b1.r224_in1k","--recipe","s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt",
    "--ckpt-subdir","b20dws","--name","b20dws")
foreach ($k in "best","last") {
    New-Item -ItemType Directory -Force "outputs/checkpoints/b20dws_$k" | Out-Null
    Copy-Item "outputs/checkpoints/b20dws/$k.pt" "outputs/checkpoints/b20dws_$k/best.pt" -Force }
$m2 = @("--model","$C/b20dws_last/best.pt","--model","$C/b20dws_best/best.pt")
Run-Step "D2_ibims" "depth" (@("-m","benchmark.run_bench","--dataset","ibims","--boundary","--save-per-image") + $m2)
Run-Step "D3_nyu"   "depth" (@("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--save-per-image") + $m2)
Run-Step "D4_sun"   "depth" @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--save-per-image","--model","$C/b20dws_last/best.pt")
Run-Step "D5_ocmr"  "depth" @("-m","benchmark.ocmr_eval_ann","--model","$C/b20dws_last/best.pt")
Run-Step "D6_ocmr_b20last" "depth" @("-m","benchmark.ocmr_eval_ann","--model","$C/b20_last/best.pt")
Run-Step "D7_fps" "depth" @("-m","benchmark.fps_bench","--config","config.yaml",
    "--model","$C/b10_last/best.pt","--model","$C/b20core_last/best.pt","--model","$C/b20dws_last/best.pt",
    "--model","$C/b1_mix/best.pt","--model","$C/S16_prox/best.pt")
Add-Content $log ("===== B20DWS DONE {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
