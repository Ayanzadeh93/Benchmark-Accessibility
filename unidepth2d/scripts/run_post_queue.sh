#!/bin/bash
# Runs AFTER scripts/run_zeroshot_diode_eth3d.sh (waits for ZS_ALLDONE / ABORT), one GPU job at a time:
#  1. IBims boundary-F1 v2 for every paper model lacking it  -> results_bf1v2/ (depth metrics unchanged)
#  2. restored-share fine-tune (own .30 / nyu .35 / sun .15 / xarkit .15 / xhyper .05), same lr/epochs/init
#     as ft2_all, evaluated on IBims/NYU/SUN (base env, like the Original/Control rows) + DIODE/ETH3D (depth env)
#  3. same-rig latency: GPU (fps_bench) for v2 / MoGe-2 / UniK3D / students, CPU ONNX for exported students
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d HF_HUB_OFFLINE=1
L=outputs/post_queue.log
C=C:/Tim/nature/unidepth2d/outputs/checkpoints
B=C:/Tim/nature/unidepth2d/results_bf1v2
O=C:/Tim/nature/unidepth2d/results_finetune
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
log "waiting for zero-shot run"
until grep -q "ZS_ALLDONE\|ABORT" outputs/zeroshot_diode_eth3d.log 2>/dev/null; do sleep 60; done
log "zero-shot run finished: $(grep -o 'ZS_ALLDONE\|ABORT.*' outputs/zeroshot_diode_eth3d.log | tail -1)"

# 1. boundary F1 v2 on IBims
STU="--model student:$C/b20dws_last/best.pt --model student:$C/b20core_last/best.pt --model student:$C/b10_last/best.pt \
 --model student:$C/b1_mix/best.pt --model student:$C/S16_prox/best.pt --model student:$C/S21_scale2/best.pt"
log "START bf1v2 students"; conda run --no-capture-output -n depth python -u -m benchmark.run_bench --out-dir $B --dataset ibims --boundary --save-per-image $STU >> $L 2>&1; log "END rc=$?"
log "START bf1v2 v2"; conda run --no-capture-output -n depth python -u -m benchmark.run_bench --out-dir $B --dataset ibims --res-level 9 --boundary --save-per-image --model v2-s --model v2-b --model v2-l >> $L 2>&1; log "END rc=$?"
log "START bf1v2 ssdi"; conda run --no-capture-output -n depth python -u -m benchmark.run_bench --out-dir $B --dataset ibims --boundary --save-per-image --model hybrid:$C/S21_scale2/best.pt+v2-l+r0 >> $L 2>&1; log "END rc=$?"
log "START bf1v2 ext"; conda run --no-capture-output -n depth_cmp python -u -m benchmark.run_bench --out-dir $B --dataset ibims --boundary --save-per-image --model unik3d-s --model unik3d-b --model moge2-s --model moge2-b --model moge2-l >> $L 2>&1; log "END rc=$?"
log "BF1V2_DONE"

# 2. restored-share fine-tune
REC="s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt"
COMMON="--data gt --use-intrinsics --backbone efficientvit_b1.r224_in1k --recipe $REC --init-from $C/b20dws/last.pt --epochs 10 --lr 2e-5 --no-wandb"
log "START rest train"
python -u -m distill.train --config configs/finetune_extra.yaml $COMMON --mix own:0.30,nyu:0.35,sun:0.15,xarkit:0.15,xhyper:0.05 --ckpt-subdir b20dws_ft2_rest --name b20dws-ft2-rest >> $L 2>&1; rc=$?
log "END rest train rc=$rc"
if [ $rc -eq 0 ]; then
  mkdir -p $C/b20dws_ft2_rest_last; cp $C/b20dws_ft2_rest/last.pt $C/b20dws_ft2_rest_last/best.pt
  M="--model student:$C/b20dws_ft2_rest_last/best.pt"
  python -u -m benchmark.run_bench --out-dir $O --dataset ibims --boundary --save-per-image --tag ft2rest $M >> $L 2>&1
  python -u -m benchmark.run_bench --out-dir $O --dataset nyu --variant filled --save-per-image --tag ft2rest $M >> $L 2>&1
  python -u -m benchmark.run_bench --out-dir $O --dataset sun --variant raw --save-per-image --tag ft2rest $M >> $L 2>&1
  conda run --no-capture-output -n depth python -u -m benchmark.run_bench --dataset diode --boundary --save-per-image $M >> $L 2>&1
  conda run --no-capture-output -n depth python -u -m benchmark.run_bench --dataset eth3d --boundary --save-per-image $M >> $L 2>&1
  log "REST_EVAL_DONE"
fi

# 3. latency (GPU idle now)
log "START fps"
conda run --no-capture-output -n depth python -u -m benchmark.fps_bench --res-level 9 --model v2-s --model v2-b --model v2-l --out outputs/fps_v2_r9.json >> $L 2>&1
conda run --no-capture-output -n depth_cmp python -u -m benchmark.fps_bench --model unik3d-s --model unik3d-b --model moge2-s --model moge2-b --model moge2-l --out outputs/fps_ext.json >> $L 2>&1
conda run --no-capture-output -n depth python -u -m benchmark.fps_bench --model student:$C/b20dws_last/best.pt --model student:$C/b20core_last/best.pt --model student:$C/b10_last/best.pt --model student:$C/b1_mix/best.pt --model student:$C/S16_prox/best.pt --out outputs/fps_students.json >> $L 2>&1
conda run --no-capture-output -n depth_cmp python -u -m edge.ort_latency --onnx outputs/edge/b20dws_last/fp32.onnx --onnx outputs/edge/b20core_last/fp32.onnx --onnx outputs/edge/b10_last/fp32.onnx --onnx outputs/edge/S16_prox/fp32.onnx --threads 1 --threads 4 --threads 8 >> $L 2>&1
log "FPS_DONE"
log "POST_ALLDONE"
