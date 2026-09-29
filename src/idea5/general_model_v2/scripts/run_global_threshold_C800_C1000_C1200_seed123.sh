#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model_v2/results_global_threshold_C800_C1000_C1200_seed123"
LOGDIR="src/idea5/general_model_v2/logs/batch"
STATUS_CSV="$LOGDIR/global_threshold_C800_C1000_C1200_seed123_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "model,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  CVAL="$1"
  SEED=123
  NAME="global_threshold_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "======================================================================"

  python src/idea5/general_model_v2/scripts/run_threshold_benchmark.py \
    --train_regime MIX12_EQ \
    --C "$CVAL" \
    --F 3 \
    --T 1000 \
    --episodes 1000 \
    --eval_episodes 200 \
    --seed "$SEED" \
    --money_p 1.0 \
    --money_tau 100.0 \
    --drop_penalty 0.0 \
    --flush_levels 17 \
    --threshold_mode global \
    --save_mode full \
    --result_root "$ROOT" \
    2>&1 | tee "$LOG_PATH"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "global_threshold,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

for CVAL in 800 1000 1200; do
  run_one "$CVAL"
done

python src/idea5/general_model_v2/scripts/evaluate_v2_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/global_threshold_C800_C1000_C1200_seed123_aggregate.log"

echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
