# P-series: FAIR zero-shot-clean distillation ladder (2026-07-22).
# Fairness contract: training mix = own:1.0 ONLY (7,534 teacher-pseudo-labeled images;
# no NYU/SUN/KITTI/IBims exposure, matching UniDepthV2's own zero-shot discipline);
# checkpoint selection on own-val (select_metric: own in config.yaml — never eval-domain);
# registry now records train_mix + zero_shot_clean per run. W&B project: UnidepthV2+.
#
# Ladder (each rung adds exactly one ingredient; identical chassis s4,s5,ray,range on
# efficientvit_b2 15.5M, 30 epochs; P5 deployment model gets 40):
#   P0-naive  output distillation, metric SILog anchor          (fair reference)
#   P1-ssi    SSI-only (Distill-Any-Depth-style)                 (prior-art failure mode)
#   P2-sdd    + explicit per-image scale loss (scale2)           (M1)
#   P3-csd    + consensus-scale-refined pseudo-labels            (M2 — core novelty)
#   P4-rel    + teacher scale-reliability weighting              (M3)
#   P5-full   + proximity-weighted SILog (prox)                  (deployment, 40 ep)
# Prereq step C0 builds the consensus-scale targets (offline teacher, ~1.2 h).

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\p_series.log"
Add-Content $log ("`n===== P-SERIES START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$BB = "efficientvit_b2.r224_in1k"
$COMMON = @("-m","distill.train","--config","config.yaml","--data","gt","--use-intrinsics","--backbone",$BB,"--mix","own:1.0")

$steps = @(
    @{ n = "C0_consensus_scale"; e = "depth"; a = @("-m","scripts.build_consensus_scale","--source","own") },
    @{ n = "T0_P0_naive"; e = "IJCAI"; a = $COMMON + @("--recipe","s4,s5,ray,range","--epochs","30","--ckpt-subdir","P0_naive","--name","P0-naive") },
    @{ n = "T1_P1_ssi";   e = "IJCAI"; a = $COMMON + @("--recipe","s4,s5,ray,range,ssionly","--epochs","30","--ckpt-subdir","P1_ssi","--name","P1-ssi") },
    @{ n = "T2_P2_sdd";   e = "IJCAI"; a = $COMMON + @("--recipe","s4,s5,ray,range,scale2","--epochs","30","--ckpt-subdir","P2_sdd","--name","P2-sdd") },
    @{ n = "T3_P3_csd";   e = "IJCAI"; a = $COMMON + @("--recipe","s4,s5,ray,range,scale2,csd","--epochs","30","--ckpt-subdir","P3_csd","--name","P3-csd") },
    @{ n = "T4_P4_rel";   e = "IJCAI"; a = $COMMON + @("--recipe","s4,s5,ray,range,scale2,csd,rel","--epochs","30","--ckpt-subdir","P4_rel","--name","P4-rel") },
    @{ n = "T5_P5_full";  e = "IJCAI"; a = $COMMON + @("--recipe","s4,s5,ray,range,scale2,csd,rel,prox","--epochs","40","--ckpt-subdir","P5_full","--name","P5-full") }
)

# Dense zero-shot evals (UniDepthV2's own indoor eval sets + NYU, all now genuinely
# zero-shot for the P-series) + OCMR object track for the deployment model.
foreach ($p in @("P0_naive","P1_ssi","P2_sdd","P3_csd","P4_rel","P5_full")) {
    $ck = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/$p/best.pt"
    $steps += @{ n = "E_${p}_ibims"; e = "depth"; a = @("-m","benchmark.run_bench","--dataset","ibims","--model",$ck,"--boundary") }
    $steps += @{ n = "E_${p}_nyu";   e = "depth"; a = @("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--model",$ck) }
    $steps += @{ n = "E_${p}_sun";   e = "depth"; a = @("-m","benchmark.run_bench","--dataset","sun","--variant","raw","--model",$ck) }
}
$P5 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/P5_full/best.pt"
$steps += @{ n = "E_P5_ocmr";   e = "depth"; a = @("-m","benchmark.ocmr_eval_ann","--model",$P5) }
$steps += @{ n = "E_scalegap";  e = "depth"; a = @("-m","benchmark.scale_gap","--probe","--dataset","ibims","--model",$P5) }
$steps += @{ n = "E_tables";    e = "depth"; a = @("-m","benchmark.paper_tables") }

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
Add-Content $log ("===== P-SERIES DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "p-series finished; failed steps: $fail; log: $log"
