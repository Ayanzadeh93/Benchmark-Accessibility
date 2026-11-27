#!/bin/bash
# Evaluate the control fine-tune now, then fine-tune b20dws with +Hypersim only (ARKit later), and evaluate.
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d
L=outputs/finetune_hyper.log
C=C:/Tim/nature/unidepth2d/outputs/checkpoints
O=C:/Tim/nature/unidepth2d/results_finetune
REC="s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt"
COMMON="--data gt --use-intrinsics --backbone efficientvit_b1.r224_in1k --recipe $REC --init-from $C/b20dws/last.pt --epochs 15 --lr 1e-4 --no-wandb"
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
ev(){ # $1 = model args, $2 = tag
  python -u -m benchmark.run_bench --out-dir $O --dataset ibims --boundary --save-per-image --tag $2 $1 >> $L 2>&1
  python -u -m benchmark.run_bench --out-dir $O --dataset nyu --variant filled --save-per-image --tag $2 $1 >> $L 2>&1
  python -u -m benchmark.run_bench --out-dir $O --dataset sun --variant raw --save-per-image --tag $2 $1 >> $L 2>&1
}
if [ "$1" != "--skip-ctrl" ]; then
mkdir -p $C/b20dws_ft_ctrl_last; cp $C/b20dws_ft_ctrl/last.pt $C/b20dws_ft_ctrl_last/best.pt
log "eval ctrl"; ev "--model student:$C/b20dws_ft_ctrl_last/best.pt" ftctrl; log "EVAL_CTRL_DONE"
fi
python -u -m scripts.build_extra_gt --which hyper >> $L 2>&1; rc=$?; log "build hyper rc=$rc"; [ $rc -ne 0 ] && { log "ABORT"; exit 1; }
log "START hyper"; python -u -m distill.train --config configs/finetune_extra.yaml $COMMON --mix own:0.35,nyu:0.35,sun:0.15,xhyper:0.15 --ckpt-subdir b20dws_ft_hyper --name b20dws-ft-hyper >> $L 2>&1; log "END hyper rc=$?"
mkdir -p $C/b20dws_ft_hyper_last; cp $C/b20dws_ft_hyper/last.pt $C/b20dws_ft_hyper_last/best.pt
log "eval hyper"; ev "--model student:$C/b20dws_ft_hyper_last/best.pt" fthyper
log "ALLDONE"
