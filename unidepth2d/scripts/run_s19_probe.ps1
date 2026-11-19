# S19-sced warm-start PROBE (2026-07-20): S16 recipe + 'scale' (per-image scale-bias on
# real-GT rows) + 'sced' (camera-equivariant crop consistency). Motivated by
# benchmark/scale_gap.py findings: student OOD failure is a per-image log-scale latent
# (IBims d1 .71 raw -> .90 with one scalar/image; K-uncorrelated; std ~0.18, ~65% of
# variance independent of the teacher). 15-epoch fine-tune from S16 at lr 1e-4 = fast
# signal before committing to a 40-epoch from-scratch run (S19 full / S20 sced-only /
# S21 scale-only ablations).
# Gate for promotion: NYU d1 >= 88.0 held AND IBims d1 > .74 (beats +global-shift .745).

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\s19_probe.log"
Add-Content $log ("`n===== S19 PROBE START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$S16CKPT = "C:/Tim/nature/unidepth2d/outputs/checkpoints/S16_prox/best.pt"
$S19 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S19_sced/best.pt"

$steps = @(
    @{ n = "T1_train_S19_probe"; e = "IJCAI"; a = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone","efficientvit_b2.r224_in1k","--recipe","s4,s5,ray,range,prox,scale,sced","--init-from",$S16CKPT,"--epochs","15","--lr","1e-4","--ckpt-subdir","S19_sced","--name","S19-sced-probe") },
    @{ n = "E1_ibims_S19";       e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$S19,"--boundary") },
    @{ n = "E2_nyu_S19";         e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$S19) },
    @{ n = "E3_sun_S19";         e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$S19) },
    @{ n = "E4_scalegap_ibims";  e = "depth"; a = @("-m","benchmark.scale_gap","--probe","--dataset","ibims","--model",$S19) },
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
Add-Content $log ("===== S19 PROBE DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "s19 probe finished; failed steps: $fail; log: $log"
