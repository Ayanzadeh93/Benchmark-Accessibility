#!/bin/bash
cd /c/Tim/nature/unidepth2d
O=C:/Tim/nature/unidepth2d/results_rescore
C=C:/Tim/nature/unidepth2d/outputs/checkpoints
L=results_rescore/logs/queue2.log
run(){ echo "[$(date +%T)] START $1" >> $L; python -m benchmark.run_bench --out-dir $O "${@:2}" > results_rescore/logs/q2_$1.log 2>&1; echo "[$(date +%T)] END $1 rc=$?" >> $L; }
for m in b1_mix b10_best b20_last b20_best S10_ultra_L S15_anch_icd S16_prox S18_zsclean S19_sced S21_scale2; do
  run a04_$m --dataset ibims --boundary --save-per-image --tag a04 --model student:$C/$m/best.pt
done
for ds in "nyu --variant filled" "sun --variant raw"; do
  n=${ds%% *}
  for p in legacy default_k; do
    run nok_${n}_$p --dataset $ds --no-k --k-none-policy $p --tag a05 --model student:$C/b10_last/best.pt --model student:$C/b20core_last/best.pt --model student:$C/b20dws_last/best.pt
  done
done
echo "[$(date +%T)] ALLDONE" >> $L
