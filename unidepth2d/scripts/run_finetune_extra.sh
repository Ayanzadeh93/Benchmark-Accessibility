#!/bin/bash
# b20dws fine-tune: control (original data) vs +ARKitScenes/+Hypersim, same init, same epochs/LR.
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d
L=outputs/finetune_extra.log
C=C:/Tim/nature/unidepth2d/outputs/checkpoints
O=C:/Tim/nature/unidepth2d/results_finetune
REC="s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt"
COMMON="--data gt --use-intrinsics --backbone efficientvit_b1.r224_in1k --recipe $REC --init-from $C/b20dws/last.pt --epochs 15 --lr 1e-4 --no-wandb"
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
until grep -q ALLDONE results_rescore/logs/queue2.log 2>/dev/null; do sleep 60; done
log "rescore queue done"
log "START ctrl"; python -u -m distill.train --config config.yaml $COMMON --ckpt-subdir b20dws_ft_ctrl --name b20dws-ft-ctrl >> $L 2>&1; log "END ctrl rc=$?"
until grep -q "^DONE" /z/Datasets/ARKitScenes/download.log && grep -q "^DONE" /z/Datasets/Hypersim/download.log; do sleep 120; done
log "downloads done"
python -u -m scripts.build_extra_gt --which all >> $L 2>&1; log "build_extra_gt rc=$?"
log "START extra"; python -u -m distill.train --config configs/finetune_extra.yaml $COMMON --mix own:0.3,nyu:0.3,sun:0.1,xarkit:0.2,xhyper:0.1 --ckpt-subdir b20dws_ft_extra --name b20dws-ft-extra >> $L 2>&1; log "END extra rc=$?"
for r in ctrl extra; do mkdir -p $C/b20dws_ft_${r}_last; cp $C/b20dws_ft_$r/last.pt $C/b20dws_ft_${r}_last/best.pt; done
M="--model student:$C/b20dws_ft_ctrl_last/best.pt --model student:$C/b20dws_ft_extra_last/best.pt"
log "eval ibims"; python -u -m benchmark.run_bench --out-dir $O --dataset ibims --boundary --save-per-image --tag ft $M >> $L 2>&1
log "eval nyu";   python -u -m benchmark.run_bench --out-dir $O --dataset nyu --variant filled --save-per-image --tag ft $M >> $L 2>&1
log "eval sun";   python -u -m benchmark.run_bench --out-dir $O --dataset sun --variant raw --save-per-image --tag ft $M >> $L 2>&1
log "ALLDONE"
