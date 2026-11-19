# S21-scale2 warm-start PROBE (2026-07-20): S16 recipe + 'scale2' — per-image scale-bias
# toward the row target on ALL rows (own rows = TEACHER's per-image scale, nyu/sun = GT).
# Evidence chain: (1) scale error is a view-stable per-scene commitment (view_var probe:
# only 11% of variance is view-dependent -> consistency/TTA cannot fix it; S19-sced
# probe FAILED IBims .689 vs S16 .707); (2) teacher scale is 2x more stable OOD
# (b-std .10 vs .18, corr .59); (3) transplanting V2-L per-image scale onto S16
# structure scores IBims d1 .8425 > official V2-S .8253 (the training ceiling).
# Gate for promotion: NYU d1 >= 88.0 held AND IBims d1 > .75.

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\s21_probe.log"
Add-Content $log ("`n===== S21 PROBE START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$S16CKPT = "C:/Tim/nature/unidepth2d/outputs/checkpoints/S16_prox/best.pt"
$S21 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S21_scale2/best.pt"

$steps = @(
    @{ n = "T1_train_S21_probe"; e = "IJCAI"; a = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone","efficientvit_b2.r224_in1k","--recipe","s4,s5,ray,range,prox,scale2","--init-from",$S16CKPT,"--epochs","15","--lr","1e-4","--ckpt-subdir","S21_scale2","--name","S21-scale2-probe") },
    @{ n = "E1_ibims_S21";       e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$S21,"--boundary") },
    @{ n = "E2_nyu_S21";         e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$S21) },
    @{ n = "E3_sun_S21";         e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$S21) },
    @{ n = "E4_scalegap_ibims";  e = "depth"; a = @("-m","benchmark.scale_gap","--probe","--dataset","ibims","--model",$S21) },
    @{ n = "E5_tables";          e = "depth"; a = @("-m","benchmark.paper_tables") }
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
Add-Content $log ("===== S21 PROBE DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "s21 probe finished; failed steps: $fail; log: $log"
