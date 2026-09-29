#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model_v2/results_pressure_maskSafe_C1000_ep1500_seeds323_532"
LOGDIR="src/idea5/general_model_v2/logs/batch"
STATUS_CSV="$LOGDIR/pressure_maskSafe_C1000_ep1500_seeds323_532_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "model,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  SEED="$1"
  NAME="conditional_pressure_maskSafe_C1000_ep1500_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "SEED: $SEED"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model_v2/scripts/run_conditional_factorized_ac.py \
    --train_regime MIX12_EQ \
    --C 1000 \
    --F 3 \
    --T 1000 \
    --episodes 1500 \
    --eval_episodes 200 \
    --seed "$SEED" \
    --money_p 1.0 \
    --money_tau 100.0 \
    --drop_penalty 0.0 \
    --device cpu \
    --hidden_size 128 \
    --settle_embed_dim 32 \
    --conditional_hidden_size 256 \
    --flush_levels 17 \
    --learning_rate 0.0001 \
    --clip_eps 0.1 \
    --reward_scale 100 \
    --state_feature_mode pressure \
    --mask_mode safe \
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

  echo "conditional_pressure_maskSafe,1000,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

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
  2>&1 | tee "$LOGDIR/pressure_maskSafe_C1000_ep1500_seeds323_532_aggregate.log"

AGG_EXIT_CODE=${PIPESTATUS[0]}
AGG_END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

if [ "$AGG_EXIT_CODE" -eq 0 ]; then
  echo "aggregate,ALL,ALL,SUCCESS,NA,$AGG_END_TIME,$LOGDIR/pressure_maskSafe_C1000_ep1500_seeds323_532_aggregate.log" >> "$STATUS_CSV"
else
  echo "aggregate,ALL,ALL,FAILED_EXIT_${AGG_EXIT_CODE},NA,$AGG_END_TIME,$LOGDIR/pressure_maskSafe_C1000_ep1500_seeds323_532_aggregate.log" >> "$STATUS_CSV"
fi

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
