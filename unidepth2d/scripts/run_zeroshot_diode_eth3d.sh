#!/bin/bash
# Two new zero-shot sets (DIODE val/indoors n=325, ETH3D indoor n=219) for every model in the paper
# tables, same protocol as IBims (students get the dataset K; foundation models self-calibrate, +r9).
#   students        -> depth env (base env aborts with a duplicate-OpenMP error on >1024x768 inputs;
#                      depth env reproduces the base-env IBims student number to 1e-4)
#   UniDepthV2/SSDI -> depth env (--res-level 9; SSDI donor r0 as before)
#   UniK3D/MoGe-2   -> depth_cmp env
# ETH3D waits for scripts.fetch_eth3d_parallel ([ETH3D_DL_DONE]) and extracts first.
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d HF_HUB_OFFLINE=1
L=outputs/zeroshot_diode_eth3d.log
C=C:/Tim/nature/unidepth2d/outputs/checkpoints
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
STU="--model student:$C/b20dws_last/best.pt --model student:$C/b20dws_last/best.pt@flip \
 --model student:$C/b20dws_ft2_all_last/best.pt --model student:$C/b20dws_ft2_all_last/best.pt@flip \
 --model student:$C/b20dws_ft2_ctrl_last/best.pt --model student:$C/b20core_last/best.pt \
 --model student:$C/b10_last/best.pt --model student:$C/b1_mix/best.pt \
 --model student:$C/S16_prox/best.pt --model student:$C/S21_scale2/best.pt"
V2="--model v2-s --model v2-b --model v2-l"
EXT="--model unik3d-s --model unik3d-b --model moge2-s --model moge2-b --model moge2-l"
run_set(){ # $1 = dataset
  log "START $1 students"; conda run --no-capture-output -n depth python -u -m benchmark.run_bench --dataset $1 --boundary --save-per-image $STU >> $L 2>&1; log "END $1 students rc=$?"
  log "START $1 v2"; conda run --no-capture-output -n depth python -u -m benchmark.run_bench --dataset $1 --res-level 9 --boundary --save-per-image $V2 >> $L 2>&1; log "END $1 v2 rc=$?"
  log "START $1 ssdi"; conda run --no-capture-output -n depth python -u -m benchmark.run_bench --dataset $1 --boundary --save-per-image --model hybrid:$C/S21_scale2/best.pt+v2-l+r0 >> $L 2>&1; log "END $1 ssdi rc=$?"
  log "START $1 ext"; conda run --no-capture-output -n depth_cmp python -u -m benchmark.run_bench --dataset $1 --boundary --save-per-image $EXT >> $L 2>&1; log "END $1 ext rc=$?"
}
run_set diode
log "DIODE_DONE"
until grep -q "ETH3D_DL_" outputs/fetch_eth3d.log 2>/dev/null; do sleep 60; done
if ! grep -q "\[ETH3D_DL_DONE\]" outputs/fetch_eth3d.log; then log "ABORT eth3d download incomplete"; exit 1; fi
log "START eth3d extract"; python -u -m benchmark.fetch_v2 --only eth3d_indoor --extract >> $L 2>&1; log "END eth3d extract rc=$?"
N=$(python -c "from benchmark.datasets_tier1 import list_eth3d; print(len(list_eth3d()))" 2>/dev/null | tail -1)
log "eth3d listed n=$N"
if [ "${N:-0}" -lt 200 ]; then log "ABORT eth3d layout (n=$N)"; exit 1; fi
run_set eth3d
log "ZS_ALLDONE"
