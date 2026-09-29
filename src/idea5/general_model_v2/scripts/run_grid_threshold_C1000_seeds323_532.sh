#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model_v2/results_grid_threshold_C1000_seeds323_532"
LOGDIR="src/idea5/general_model_v2/logs/batch"
STATUS_CSV="$LOGDIR/grid_threshold_C1000_seeds323_532_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  SEED="$1"
  NAME="grid_threshold_C1000_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "SEED: $SEED"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model_v2/scripts/run_threshold_benchmark.py \
    --train_regime MIX12_EQ \
    --C 1000 \
    --F 3 \
    --T 1000 \
    --episodes 1000 \
    --eval_episodes 200 \
    --seed "$SEED" \
    --money_p 1.0 \
    --money_tau 100.0 \
    --drop_penalty 0.0 \
    --flush_levels 17 \
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

  echo "GridThreshold,1000,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo
  echo "FINISHED: $NAME"
  echo "STATUS: $STATUS"
}

for SEED in 323 532; do
  run_one "$SEED"
done

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating this result root..."
echo "======================================================================"

python src/idea5/general_model_v2/scripts/evaluate_v2_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/grid_threshold_C1000_seeds323_532_aggregate.log"

AGG_EXIT_CODE=${PIPESTATUS[0]}
AGG_END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

if [ "$AGG_EXIT_CODE" -eq 0 ]; then
  echo "aggregate,ALL,ALL,SUCCESS,NA,$AGG_END_TIME,$LOGDIR/grid_threshold_C1000_seeds323_532_aggregate.log" >> "$STATUS_CSV"
else
  echo "aggregate,ALL,ALL,FAILED_EXIT_${AGG_EXIT_CODE},NA,$AGG_END_TIME,$LOGDIR/grid_threshold_C1000_seeds323_532_aggregate.log" >> "$STATUS_CSV"
fi

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
