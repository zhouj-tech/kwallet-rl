#!/usr/bin/env bash

set +e

SCRIPT="src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py"

LOG_DIR="src/idea4/ac/logs/conditional_factorized_sanity"
STATUS_CSV="src/idea4/ac/results/conditional_factorized_sanity_status/conditional_factorized_sanity_6runs_status.csv"

TRAIN_REGIME="MIX12_EQ"
MODEL_MODE="conditional_factorized_ac"

K=24
F=3
T=1000
EPISODES=1000
EVAL_EPISODES=200
DEVICE="cpu"
SAVE_MODE="full"

SEEDS=(123 323 532)
CS=(800 1200)

echo "model,C,k,F,T,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  local C="$1"
  local SEED="$2"

  local RUN_ID="${MODEL_MODE}_C${C}_k${K}_F${F}_T${T}_seed${SEED}"
  local LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

  local START_TIME
  local END_TIME
  START_TIME=$(date "+%Y-%m-%d %H:%M:%S")

  echo ""
  echo "============================================================"
  echo "RUN: $RUN_ID"
  echo "START: $START_TIME"
  echo "LOG: $LOG_PATH"
  echo "============================================================"

  python -u "$SCRIPT" \
    --model_mode "$MODEL_MODE" \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --eval_episodes "$EVAL_EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    --save_mode "$SAVE_MODE" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_PATH"

  local EXIT_CODE=${PIPESTATUS[0]}
  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "$MODEL_MODE,$C,$K,$F,$T,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo "END: $END_TIME"
  echo "STATUS: $STATUS"
}

echo "============================================================"
echo "Conditional Factorized AC Sanity Batch"
echo "============================================================"
echo "Model: $MODEL_MODE"
echo "Settings: C=800/1200, k=24, F=3, T=1000"
echo "Seeds: ${SEEDS[*]}"
echo "Status CSV: $STATUS_CSV"
echo "============================================================"

for C in "${CS[@]}"
do
  for SEED in "${SEEDS[@]}"
  do
    run_one "$C" "$SEED"
  done
done

echo ""
echo "============================================================"
echo "CONDITIONAL FACTORIZED SANITY BATCH FINISHED"
echo "Status file:"
echo "$STATUS_CSV"
echo "============================================================"
