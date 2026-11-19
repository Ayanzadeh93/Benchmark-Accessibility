# Phase-4 training queue (IJCAI env, single GPU — launch ONLY after eval retries finish).
# S18-zsclean first (paper-critical honesty row: no SUN in training -> honest SUN/IBims
# zero-shot student), then S16-prox (proximity-weighted SILog safety-band loss).
# S17-anchloss DEFERRED: own corpus is 99.5% class-less (RESULTS_DIGEST #3).
# After each training: dense NYU/SUN/IBims + OCMR-SUN eval of the new student (depth env),
# then refreshed tables.

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\eval_queue.log"
Add-Content $log ("`n===== PHASE4 START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$S18 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S18_zsclean/best.pt"
$S16 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S16_prox/best.pt"

$steps = @(
    @{ n = "P1_train_S18_zsclean"; e = "IJCAI"; a = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone","efficientvit_b2.r224_in1k","--recipe","s4,s5,ray,range","--mix","own:0.5,nyu:0.5","--ckpt-subdir","S18_zsclean","--name","S18-zsclean") },
    @{ n = "P2_train_S16_prox";    e = "IJCAI"; a = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone","efficientvit_b2.r224_in1k","--recipe","s4,s5,ray,range,prox","--ckpt-subdir","S16_prox","--name","S16-prox") },
    @{ n = "P3_nyu_new";           e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$S18,"--model",$S16) },
    @{ n = "P4_ibims_new";         e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$S18,"--model",$S16,"--boundary") },
    @{ n = "P5_sun_new";           e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$S18,"--model",$S16) },
    @{ n = "P6_ocmrsun_S18";       e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$S18) },
    @{ n = "P7_ocmrsun_S16";       e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$S16) },
    @{ n = "P8_paper_tables";      e = "depth"; a = @("-m","benchmark.paper_tables") },
    @{ n = "P9_ablation_table";    e = "depth"; a = @("-m","benchmark.ablation_table","--config","config.yaml") }
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
Add-Content $log ("===== PHASE4 DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "phase4 finished; failed steps: $fail; log: $log"
