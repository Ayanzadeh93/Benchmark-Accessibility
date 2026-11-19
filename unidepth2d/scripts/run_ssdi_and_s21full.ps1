# SSDI paper-grade rows + S21 from-scratch full training (2026-07-20 overnight queue).
# Phase H: official-protocol hybrid rows (ssdi-S21+v2-l+r0) on ibims/nyu/sun + OCMR
#   object rows for hybrid + S21-probe + refreshed tables.
# Phase T: S21-scale2 FULL 40-epoch from-scratch training (subdir S21_scale2_full so the
#   15-ep probe ckpt stays intact), then its dense + OCMR evals + tables.
# Recipe evidence: docs/SCALE_GAP.md + memory 2026-07-20 (scale latent is per-scene,
# view-stable, K-uncorrelated; donor at res-level 0 reads it; median log-ratio in the
# 0.2-8m working range, no GT anywhere).

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\ssdi_s21full.log"
Add-Content $log ("`n===== SSDI+S21FULL START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$HYB  = "hybrid:C:/Tim/nature/unidepth2d/outputs/checkpoints/S21_scale2/best.pt+v2-l+r0"
$S21P = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S21_scale2/best.pt"
$S21F = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S21_scale2_full/best.pt"
$S16CK = "C:/Tim/nature/unidepth2d/outputs/checkpoints/S16_prox/best.pt"

$steps = @(
    # ── Phase H: hybrid + S21-probe paper rows ───────────────────────────────
    @{ n = "H1_ibims_hybrid";  e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$HYB,"--boundary") },
    @{ n = "H2_nyu_hybrid";    e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$HYB) },
    @{ n = "H3_sun_hybrid";    e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$HYB) },
    @{ n = "H4_ocmr_hybrid";   e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$HYB) },
    @{ n = "H5_ocmr_S21probe"; e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$S21P) },
    @{ n = "H6_tables";        e = "depth"; a = @("-m","benchmark.paper_tables") },
    # ── Phase T: S21 full from-scratch (40 ep, config lr) ────────────────────
    @{ n = "T1_train_S21_full"; e = "IJCAI"; a = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone","efficientvit_b2.r224_in1k","--recipe","s4,s5,ray,range,prox,scale2","--ckpt-subdir","S21_scale2_full","--name","S21-scale2") },
    @{ n = "T2_ibims_S21full"; e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$S21F,"--boundary") },
    @{ n = "T3_nyu_S21full";   e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$S21F) },
    @{ n = "T4_sun_S21full";   e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$S21F) },
    @{ n = "T5_ocmr_S21full";  e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$S21F) },
    @{ n = "T6_scalegap";      e = "depth"; a = @("-m","benchmark.scale_gap","--probe","--dataset","ibims","--model",$S21F) },
    @{ n = "T7_tables";        e = "depth"; a = @("-m","benchmark.paper_tables") }
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
Add-Content $log ("===== SSDI+S21FULL DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "ssdi+s21full finished; failed steps: $fail; log: $log"
