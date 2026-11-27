#!/bin/bash
# Leave-one-out ablation of the b20dws recipe (reviewer-requested attribution table).
# Every run trains FROM SCRATCH with the exact b20dws command (scripts/run_b20dws.ps1: config.yaml, --data gt,
# --use-intrinsics, efficientvit_b1.r224_in1k, 40 epochs) minus ONE component. Nothing else changes.
#   rep1     unchanged recipe again. train.py sets no seed, so |b20dws - rep1| is the run-to-run noise floor;
#            an ablation gap smaller than that is not evidence.
#   no_dsh   - decoupled global scale head
#   no_fov   - zoom-out uncrop (cstm kept)
#   no_cstm  - canonical-focal depth space (fov must go too: train.py asserts fov requires cstm)
#   no_ownar - aspect-true own rows
#   no_trim  - top-10% residual trim on own rows
#   no_own   - the ~7k teacher-pseudo-labelled own images (mix nyu:0.67,sun:0.33; ownar/trim become no-ops).
#              NOTE: 50 leaked NYU-test ids came via own rows, so compare NYU on the CLEAN 602 split only.
#   rep2     second replicate (second noise estimate)
# Already available, no training needed: b20core (= - dws4), b10 (= all five b20 changes removed, no dws4).
# Eval (last.pt, no selection): IBims/NYU/SUN in base env (as the original b20dws rows), DIODE/ETH3D in env depth.
# Results -> results_ablation/. Log -> outputs/ablation_b20.log. One GPU job at a time.
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d HF_HUB_OFFLINE=1 PYTHONUTF8=1
L=outputs/ablation_b20.log
C=C:/Tim/nature/unidepth2d/outputs/checkpoints
O=C:/Tim/nature/unidepth2d/results_ablation
FULL="s4,s5,ray,range,prox,ownar,fov,cstm,dsh,trim,dws4,selgt"
COMMON="--config config.yaml --data gt --use-intrinsics --backbone efficientvit_b1.r224_in1k --no-wandb"
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
drop(){ # $1 = comma list of tags to remove from FULL
  local r=",$FULL,"; for t in ${1//,/ }; do r="${r//,$t,/,}"; done; r="${r#,}"; echo "${r%,}"; }

ev(){ # $1 = run name
  local d=abl_$1_last; mkdir -p $C/$d; cp $C/abl_$1/last.pt $C/$d/best.pt
  local M="--model student:$C/$d/best.pt"
  python -u -m benchmark.run_bench --out-dir $O --dataset ibims --boundary --save-per-image $M >> $L 2>&1; log "  ibims rc=$?"
  python -u -m benchmark.run_bench --out-dir $O --dataset nyu --variant filled --save-per-image $M >> $L 2>&1; log "  nyu rc=$?"
  conda run --no-capture-output -n depth python -u -m benchmark.run_bench --out-dir $O --dataset diode --save-per-image $M >> $L 2>&1; log "  diode rc=$?"
  conda run --no-capture-output -n depth python -u -m benchmark.run_bench --out-dir $O --dataset eth3d --save-per-image $M >> $L 2>&1; log "  eth3d rc=$?"
  python -u -m benchmark.run_bench --out-dir $O --dataset sun --variant raw --save-per-image $M >> $L 2>&1; log "  sun rc=$?"
}

run(){ # $1 = name, $2 = recipe, $3 = extra args
  if [ -f $C/abl_$1/last.pt ] && grep -q "EVAL_DONE $1\$" $L 2>/dev/null; then log "SKIP $1 (done)"; return; fi
  log "START $1 recipe=$2 $3"
  python -u -m distill.train $COMMON --recipe $2 $3 --ckpt-subdir abl_$1 --name abl-$1 >> $L 2>&1; local rc=$?
  log "END train $1 rc=$rc"
  if [ $rc -eq 0 ] && [ -f $C/abl_$1/last.pt ]; then ev $1; log "EVAL_DONE $1"; else log "FAILED $1"; fi
}

[ "$1" = "--print" ] && { for x in dsh cstm,fov fov ownar trim; do echo "$x -> $(drop $x)"; done; exit 0; }
log "===== ABLATION START ====="
run rep1     "$FULL"
run no_dsh   "$(drop dsh)"
run no_cstm  "$(drop cstm,fov)"
run no_fov   "$(drop fov)"
run no_ownar "$(drop ownar)"
run no_trim  "$(drop trim)"
run no_own   "$FULL" "--mix nyu:0.67,sun:0.33"
run rep2     "$FULL"
log "ABL_ALLDONE"
