# Retry queue for the 2026-07-18 run's failed steps (see outputs/eval_queue.log).
# Fixes applied: v1-l DROPPED (needs removed xformers.components API — PROTOCOL_NOTES §2c),
# args passed as token arrays (the original Invoke-Expression split "3,6,9" into 3 args),
# fps_bench hardened per-model. Appends to the same eval_queue.log (monitor stays valid).

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\eval_queue.log"
Add-Content $log ("`n===== RETRY START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$S10 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S10_ultra_L/best.pt"
$S15 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S15_anch_icd/best.pt"
$B1  = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/b1_mix/best.pt"

$steps = @(
    @{ n = "R1_nyu_v2b";        a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model","v2-b","--res-level","9") },
    @{ n = "R2_ibims_v2b";      a = @("-m","benchmark.run_bench","--dataset","ibims","--model","v2-b","--res-level","9","--boundary") },
    @{ n = "R3_resmarg_ibims";  a = @("-m","prototypes.res_marginalize","--dataset","ibims","--model","v2-l","--levels","3,6,9","--boundary") },
    @{ n = "R4_fps_bench";      a = @("-m","benchmark.fps_bench","--config","config.yaml","--model","v2-s","--model","v2-b","--model","v2-l","--model",$S10,"--model",$S15,"--model",$B1,"--res-level","9") },
    @{ n = "R5_sun_v2b";        a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model","v2-b","--res-level","9") },
    @{ n = "R6_gate_final";     a = @("-m","benchmark.gate_check") },
    @{ n = "R7_paper_tables";   a = @("-m","benchmark.paper_tables") },
    @{ n = "R8_ablation_table"; a = @("-m","benchmark.ablation_table","--config","config.yaml") }
)

$fail = 0
foreach ($s in $steps) {
    $t0 = Get-Date
    Add-Content $log ("[{0}] START {1}" -f $t0.ToString("HH:mm:ss"), $s.n)
    & conda run -n depth --no-capture-output python @($s.a) *>> $log
    $code = $LASTEXITCODE
    if ($code -ne 0) { $fail++ }
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content $log ("[{0}] END   {1}  exit={2}  ({3}s)" -f (Get-Date -Format "HH:mm:ss"), $s.n, $code, $dt)
}
Add-Content $log ("===== RETRY DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "retry finished; failed steps: $fail; log: $log"
