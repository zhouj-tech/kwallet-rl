#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model/results_formal_main_grid_stable_to_10seeds"
LOGDIR="src/idea5/general_model/logs/batch"
STATUS_CSV="$LOGDIR/formal_main_grid_stable_to_10seeds_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

CS=(1000 1200 1600)

# Grid Threshold currently has seed=123 only.
GRID_MISSING_SEEDS=(323 532 777 999 2027 3407 4501 6101 8888)

# Stable Conditional AC currently has seeds=123,323,532.
STABLE_MISSING_SEEDS=(777 999 2027 3407 4501 6101 8888)

run_grid_threshold () {
  CVAL="$1"
  SEED="$2"
  NAME="grid_threshold_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "C: $CVAL"
  echo "SEED: $SEED"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model/scripts/run_threshold_benchmark.py \
    --mode grid \
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
    --device cpu \
    --flush_levels 17 \
    --flush_grid uniform \
    --state_feature_mode base \
    --mask_mode none \
    --threshold_advantage_mode none \
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

  echo "Grid Threshold,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

run_stable_conditional () {
  CVAL="$1"
  SEED="$2"
  NAME="stable_conditional_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "C: $CVAL"
  echo "SEED: $SEED"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model/scripts/run_conditional_factorized_ac.py \
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
    --device cpu \
    --hidden_size 128 \
    --settle_embed_dim 32 \
    --conditional_hidden_size 256 \
    --flush_levels 17 \
    --flush_grid uniform \
    --state_feature_mode base \
    --mask_mode none \
    --imitation_mode none \
    --threshold_advantage_mode none \
    --learning_rate 0.0001 \
    --clip_eps 0.1 \
    --reward_scale 100 \
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

  echo "Stable Conditional AC,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

echo
echo "======================================================================"
echo "Part 1: Grid Threshold missing seeds"
echo "======================================================================"

for CVAL in "${CS[@]}"; do
  for SEED in "${GRID_MISSING_SEEDS[@]}"; do
    run_grid_threshold "$CVAL" "$SEED"
  done
done

echo
echo "======================================================================"
echo "Part 2: Stable Conditional AC missing seeds"
echo "======================================================================"

for CVAL in "${CS[@]}"; do
  for SEED in "${STABLE_MISSING_SEEDS[@]}"; do
    run_stable_conditional "$CVAL" "$SEED"
  done
done

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating this result root..."
echo "======================================================================"

python src/idea5/general_model/scripts/evaluate_general_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/formal_main_grid_stable_to_10seeds_aggregate.log"

AGG_EXIT_CODE=${PIPESTATUS[0]}
AGG_END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

if [ "$AGG_EXIT_CODE" -eq 0 ]; then
  echo "aggregate,ALL,ALL,SUCCESS,NA,$AGG_END_TIME,$LOGDIR/formal_main_grid_stable_to_10seeds_aggregate.log" >> "$STATUS_CSV"
else
  echo "aggregate,ALL,ALL,FAILED_EXIT_${AGG_EXIT_CODE},NA,$AGG_END_TIME,$LOGDIR/formal_main_grid_stable_to_10seeds_aggregate.log" >> "$STATUS_CSV"
fi

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
