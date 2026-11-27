#!/bin/bash
# UniK3D (S,B) and MoGe-2 (S,B,L) on NYU-654 (filled) and SUN-raw (4396), same protocol as the IBims rows (+r9).
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d
L=outputs/baselines_nyu_sun.log
M="--model unik3d-s --model unik3d-b --model moge2-s --model moge2-b --model moge2-l"
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
log "START nyu"; conda run --no-capture-output -n depth_cmp python -u -m benchmark.run_bench --dataset nyu --variant filled --save-per-image $M >> $L 2>&1; log "END nyu rc=$?"
log "START sun"; conda run --no-capture-output -n depth_cmp python -u -m benchmark.run_bench --dataset sun --variant raw --save-per-image $M >> $L 2>&1; log "END sun rc=$?"
log "BASEDONE"
