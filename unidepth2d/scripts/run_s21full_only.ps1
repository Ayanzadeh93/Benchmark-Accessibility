# S21-scale2 FULL 40-epoch from-scratch training + evals (resume of the interrupted
# Phase T from run_ssdi_and_s21full.ps1 — Phase H already completed and saved 2026-07-20;
# this relaunches ONLY the training that died at epoch 2/40 when the prior session ended).
# Tests whether a full from-scratch run beats the 15-ep warm-start S21 probe that SSDI
# currently uses (NYU 90.12 / SUN 89.75 / IBims 82.75 official-protocol, all beating V2-S).

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\s21full_only.log"
Add-Content $log ("`n===== S21FULL-ONLY START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$S21F  = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S21_scale2_full/best.pt"
$HYBF  = "hybrid:C:/Tim/nature/unidepth2d/outputs/checkpoints/S21_scale2_full/best.pt+v2-l+r0"

$steps = @(
    @{ n = "T1_train_S21_full";  e = "IJCAI"; a = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone","efficientvit_b2.r224_in1k","--recipe","s4,s5,ray,range,prox,scale2","--ckpt-subdir","S21_scale2_full","--name","S21-scale2") },
    @{ n = "T2_ibims_S21full";   e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$S21F,"--boundary") },
    @{ n = "T3_nyu_S21full";     e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$S21F) },
    @{ n = "T4_sun_S21full";     e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$S21F) },
    @{ n = "T5_ocmr_S21full";    e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$S21F) },
    @{ n = "T6_scalegap";        e = "depth"; a = @("-m","benchmark.scale_gap","--probe","--dataset","ibims","--model",$S21F) },
    @{ n = "T7_ibims_hybridF";   e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$HYBF,"--boundary") },
    @{ n = "T8_nyu_hybridF";     e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$HYBF) },
    @{ n = "T9_sun_hybridF";     e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$HYBF) },
    @{ n = "T10_ocmr_hybridF";   e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$HYBF) },
    @{ n = "T11_tables";         e = "depth"; a = @("-m","benchmark.paper_tables") }
)

$fail = 0
foreach ($s in $steps) {
    $t0 = Get-Date
    Add-Content $log ("[{0}] START {1}" -f $t0.ToString("HH:mm:ss"), $s.n)
    & conda run -n $s.e --no-capture-output python @($s.a) *>> $log
    $code = $LASTEXITCODE
    if ($code -ne 0) { $fail++ }
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content $log ("[{0}] END   {1}  exit={2}  ({3}s)" -f (Get-Date -Format "HH:mm:ss"), $s.n, $code, $dt)
}
Add-Content $log ("===== S21FULL-ONLY DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "s21full-only finished; failed steps: $fail; log: $log"
