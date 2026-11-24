# Re-run IBims + NYU for every student and baseline so all rows carry MAE (protocol.py now emits "mae").
# Waits for the b20 eval queue to finish (GPU must be free). Log: outputs/b20_queue.log
$ErrorActionPreference = "Continue"
Set-Location C:\Tim\nature\unidepth2d
$env:PYTHONPATH = "C:\Tim\nature\unidepth2d"
$log = "C:\Tim\nature\unidepth2d\outputs\b20_queue.log"
# first queue wrote one "B20 QUEUE DONE"; the eval rerun writes the second
while (@(Select-String -Path $log -Pattern "B20 QUEUE DONE" -SimpleMatch).Count -lt 2) { Start-Sleep -Seconds 60 }
Add-Content $log ("`n===== MAE RERUN START {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

function Run-Step($name, $envname, $argv) {
    $t0 = Get-Date
    Add-Content $log ("[{0}] START {1}" -f $t0.ToString("HH:mm:ss"), $name)
    & conda run -n $envname --no-capture-output python -u @argv *>> $log
    Add-Content $log ("[{0}] END   {1}  exit={2}  ({3}s)" -f (Get-Date -Format "HH:mm:ss"), $name, $LASTEXITCODE, [int]((Get-Date) - $t0).TotalSeconds)
}
$C = "student:C:/Tim/nature/unidepth2d/outputs/checkpoints"
$stud = @()
foreach ($m in "b10_last","b10_best","b20_last","b20_best","b20core_last","b1_mix","S10_ultra_L","S16_prox","S21_scale2") { $stud += @("--model","$C/$m/best.pt") }
$hyb = @("--model","hybrid:C:/Tim/nature/unidepth2d/outputs/checkpoints/S21_scale2/best.pt+v2-l+r0")
$off = @("--model","v2-s","--model","v2-b","--model","v2-l","--res-level","9")

Run-Step "M1_ibims_students" "depth" (@("-m","benchmark.run_bench","--dataset","ibims","--boundary","--save-per-image") + $stud + $hyb)
Run-Step "M2_ibims_v2"       "depth" (@("-m","benchmark.run_bench","--dataset","ibims","--boundary") + $off)
Run-Step "M3_ibims_2025"     "depth_cmp" @("-m","benchmark.run_bench","--dataset","ibims","--boundary","--res-level","9",
    "--model","moge2-s","--model","moge2-b","--model","moge2-l","--model","unik3d-s","--model","unik3d-b")
Run-Step "M4_nyu_students"   "depth" (@("-m","benchmark.run_bench","--dataset","nyu","--variant","filled","--save-per-image") + $stud + $hyb)
Run-Step "M5_nyu_v2"         "depth" (@("-m","benchmark.run_bench","--dataset","nyu","--variant","filled") + $off)
Add-Content $log ("===== MAE RERUN DONE {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
