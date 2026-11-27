#!/bin/bash
# Post-hoc inference options (no retraining) on our proposed checkpoints: base / larger input / flip / multi-scale / both.
cd /c/Tim/nature/unidepth2d
export PYTHONPATH=C:/Tim/nature/unidepth2d
L=outputs/tta_eval.log
C=outputs/checkpoints
M=""
for ck in b20dws_last b20dws_ft2_all_last; do
  for o in "" "@size=576x768" "@flip" "@ms" "@flip,ms"; do M="$M --model student:$C/$ck/best.pt$o"; done
done
echo "[$(date '+%F %T')] START" >> $L
python -u -m benchmark.run_bench --out-dir results_tta --dataset ibims --boundary --save-per-image $M >> $L 2>&1; echo "[$(date '+%F %T')] END ibims rc=$?" >> $L
python -u -m benchmark.run_bench --out-dir results_tta --dataset nyu --variant filled --save-per-image $M >> $L 2>&1; echo "[$(date '+%F %T')] END nyu rc=$?" >> $L
echo "[$(date '+%F %T')] TTADONE" >> $L
