#!/bin/bash
# Fair fine-tune test from the ORIGINAL b20dws checkpoint: low LR (2e-5, built-in 500-step warmup + cosine), 10 epochs.
#   all  = original mix scaled down + ARKitScenes + Hypersim
#   ctrl = same settings, original data only
# Both are scored on IBims / NYU / SUN and compared with the original b20dws.
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d
L=outputs/finetune_all.log
C=C:/Tim/nature/unidepth2d/outputs/checkpoints
O=C:/Tim/nature/unidepth2d/results_finetune
REC="s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt"
COMMON="--data gt --use-intrinsics --backbone efficientvit_b1.r224_in1k --recipe $REC --init-from $C/b20dws/last.pt --epochs 10 --lr 2e-5 --no-wandb"
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
ev(){ # $1 = ckpt subdir, $2 = tag
  mkdir -p $C/$1_last; cp $C/$1/last.pt $C/$1_last/best.pt
  M="--model student:$C/$1_last/best.pt"
  python -u -m benchmark.run_bench --out-dir $O --dataset ibims --boundary --save-per-image --tag $2 $M >> $L 2>&1
  python -u -m benchmark.run_bench --out-dir $O --dataset nyu --variant filled --save-per-image --tag $2 $M >> $L 2>&1
  python -u -m benchmark.run_bench --out-dir $O --dataset sun --variant raw --save-per-image --tag $2 $M >> $L 2>&1
}
python -u -m scripts.build_extra_gt --which all >> $L 2>&1; rc=$?; log "build rc=$rc"; [ $rc -ne 0 ] && { log "ABORT build"; exit 1; }
log "START all"; python -u -m distill.train --config configs/finetune_extra.yaml $COMMON --mix own:0.26,nyu:0.26,sun:0.13,xarkit:0.25,xhyper:0.10 --ckpt-subdir b20dws_ft2_all --name b20dws-ft2-all >> $L 2>&1; rc=$?; log "END all rc=$rc"
[ $rc -eq 0 ] && { log "eval all"; ev b20dws_ft2_all ft2all; log "EVAL_ALL_DONE"; }
log "START ctrl"; python -u -m distill.train --config config.yaml $COMMON --ckpt-subdir b20dws_ft2_ctrl --name b20dws-ft2-ctrl >> $L 2>&1; rc=$?; log "END ctrl rc=$rc"
[ $rc -eq 0 ] && { log "eval ctrl"; ev b20dws_ft2_ctrl ft2ctrl; }
log "ALLDONE"
