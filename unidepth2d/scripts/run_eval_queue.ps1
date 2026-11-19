# Paper evaluation queue — sequential (single 12GB GPU), logged, continue-on-error.
# Launch:  powershell -ExecutionPolicy Bypass -File scripts\run_eval_queue.ps1
# Monitor: Get-Content outputs\eval_queue.log -Tail 30
# Phases: A fast/foundational -> B anchoring (C2) -> C OCMR breadth -> D long dense tail -> E assembly.

$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\eval_queue.log"
Add-Content $log ("`n===== QUEUE START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$S10 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S10_ultra_L/best.pt"
$S15 = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/S15_anch_icd/best.pt"
$B1  = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints/b1_mix/best.pt"

$steps = @(
    # ── Phase A: fast / foundational ─────────────────────────────────────────
    @{ n = "A1_gate_initial";    a = "-m benchmark.gate_check" },
    @{ n = "A2_nyu_v2l_v2s";     a = "-m benchmark.run_bench --dataset nyu --variant filled --model v2-l --model v2-s --res-level 9" },
    @{ n = "A3_nyu_v2b_v1l";     a = "-m benchmark.run_bench --dataset nyu --variant filled --model v2-b --model v1-l --res-level 9" },
    @{ n = "A4_ibims_v2l_v2s";   a = "-m benchmark.run_bench --dataset ibims --model v2-l --model v2-s --res-level 9 --boundary" },
    @{ n = "A5_ibims_v2b_v1l";   a = "-m benchmark.run_bench --dataset ibims --model v2-b --model v1-l --res-level 9 --boundary" },
    @{ n = "A6_nyu_students";    a = "-m benchmark.run_bench --dataset nyu --variant filled --model $S10 --model $S15 --model $B1" },
    @{ n = "A7_ibims_students";  a = "-m benchmark.run_bench --dataset ibims --model $S10 --model $S15 --model $B1 --boundary" },
    @{ n = "A8_resmarg_ibims";   a = "-m prototypes.res_marginalize --dataset ibims --model v2-l --levels 3,6,9 --boundary" },
    @{ n = "A9_fps_bench";       a = "-m benchmark.fps_bench --config config.yaml --model v2-s --model v2-b --model v2-l --model v1-l --model $S10 --model $S15 --model $B1 --res-level 9" },
    # ── Phase B: training-free anchoring (headline C2) ───────────────────────
    @{ n = "B1_anchor_v2l";      a = "-m prototypes.anchor_lock --model v2-l --res-level 9" },
    @{ n = "B2_anchor_S10";      a = "-m prototypes.anchor_lock --model $S10" },
    # ── Phase C: OCMR breadth (KITTI outdoor + SUN model coverage) ───────────
    @{ n = "C1_kitti_build";     a = "-m benchmark.ocmr_kitti build" },
    @{ n = "C2_kitti_v2l";       a = "-m benchmark.ocmr_kitti eval --model v2-l --res-level 9" },
    @{ n = "C3_kitti_v2s";       a = "-m benchmark.ocmr_kitti eval --model v2-s --res-level 9" },
    @{ n = "C4_kitti_S10";       a = "-m benchmark.ocmr_kitti eval --model $S10" },
    @{ n = "C5_ocmrsun_v2b";     a = "-m benchmark.ocmr_eval_ann --model v2-b --res-level 9" },
    @{ n = "C6_ocmrsun_v1l";     a = "-m benchmark.ocmr_eval_ann --model v1-l --res-level 9" },
    @{ n = "C7_ocmrsun_S15";     a = "-m benchmark.ocmr_eval_ann --model $S15" },
    # ── Phase D: long dense tail (full SUN) ──────────────────────────────────
    @{ n = "D1_sun_v2b_v1l";     a = "-m benchmark.run_bench --dataset sun --variant raw --model v2-b --model v1-l --res-level 9" },
    @{ n = "D2_sun_students";    a = "-m benchmark.run_bench --dataset sun --variant raw --model $S10 --model $S15 --model $B1" },
    # ── Phase E: assembly ────────────────────────────────────────────────────
    @{ n = "E1_gate_final";      a = "-m benchmark.gate_check" },
    @{ n = "E2_paper_tables";    a = "-m benchmark.paper_tables" },
    @{ n = "E3_ablation_table";  a = "-m benchmark.ablation_table --config config.yaml" }
)

$fail = 0
foreach ($s in $steps) {
    $t0 = Get-Date
    Add-Content $log ("[{0}] START {1}" -f $t0.ToString("HH:mm:ss"), $s.n)
    Invoke-Expression ("conda run -n depth --no-capture-output python {0} *>> `"{1}`"" -f $s.a, $log)
    $code = $LASTEXITCODE
    if ($code -ne 0) { $fail++ }
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content $log ("[{0}] END   {1}  exit={2}  ({3}s)" -f (Get-Date -Format "HH:mm:ss"), $s.n, $code, $dt)
}
Add-Content $log ("===== QUEUE DONE {0}  failed_steps={1} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $fail)
Write-Output "queue finished; failed steps: $fail; log: $log"
