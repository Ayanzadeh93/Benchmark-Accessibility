#!/bin/bash
# YOLO26n-depth (6.3M, AGPL-3.0) in OUR harness on all five indoor sets, plain and flip-TTA.
# Needs the isolated env from scripts/v2/setup_depth_yolo.ps1 (weights in Z:/Models/yolo26).
# Accuracy only (no timing), so sharing the GPU with the other eval queue is harmless.
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d PYTHONUTF8=1 YOLO26_WEIGHTS=Z:/Models/yolo26
PY=/c/Users/aydina1/AppData/Local/anaconda3/envs/depth_yolo/python.exe
L=outputs/yolo26n.log
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
[ -x "$PY" ] || { log "ABORT: $PY missing (run setup_depth_yolo.ps1 first)"; exit 1; }
# small model (6.36M, <1 GiB): runs alongside the DIODE/ETH3D queue; finishes long before the latency step
until [ -d Z:/Datasets/ETH3D/office/ground_truth_depth ]; do sleep 60; done
M="--model yolo26n-depth --model yolo26n-depth@flip"
log "START ibims"; $PY -u -m benchmark.run_bench --dataset ibims --boundary --save-per-image $M >> $L 2>&1; log "END rc=$?"
log "START nyu";   $PY -u -m benchmark.run_bench --dataset nyu --variant filled --save-per-image $M >> $L 2>&1; log "END rc=$?"
log "START sun";   $PY -u -m benchmark.run_bench --dataset sun --variant raw --save-per-image $M >> $L 2>&1; log "END rc=$?"
log "START diode"; $PY -u -m benchmark.run_bench --dataset diode --boundary --save-per-image $M >> $L 2>&1; log "END rc=$?"
log "START eth3d"; $PY -u -m benchmark.run_bench --dataset eth3d --boundary --save-per-image $M >> $L 2>&1; log "END rc=$?"
log "YOLO_DONE"
