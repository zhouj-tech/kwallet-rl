#!/usr/bin/env bash

set +e

BASIC_SCRIPT="src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py"
FACT_SCRIPT="src/idea4/ac/code/run_factorized_ac_benchmark.py"

LOG_DIR="src/idea4/ac/logs/k24_10seed_final"
STATUS_CSV="src/idea4/ac/results/k24_10seed_final_status/k24_basic_vs_factorized_10seeds_status.csv"

TRAIN_REGIME="MIX12_EQ"
F=3
T=1000
K=24
EPISODES=1000
EVAL_EPISODES=200
DEVICE="cpu"
SAVE_MODE="full"

SEEDS=(123 323 532 777 999 2027 3407 4501 6101 8888)
CS=(800 1200)

echo "model,C,k,F,T,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

has_existing_result () {
  local MODEL="$1"
  local C="$2"
  local SEED="$3"

  if [ "$MODEL" = "basic_ppo" ]; then
    find src/idea4/ac/results/basic_ppo/runs \
      -path "*basic_ppo_train${TRAIN_REGIME}_C${C}_k${K}_T${T}_F${F}_seed${SEED}*/cross_regime_results.json" \
      -print -quit 2>/dev/null | grep -q .
  elif [ "$MODEL" = "factorized_ac" ]; then
    find src/idea4/ac/results/factorized_ac/runs \
      -path "*factorized_ac_train${TRAIN_REGIME}_C${C}_k${K}_T${T}_F${F}_seed${SEED}*/cross_regime_results.json" \
      -print -quit 2>/dev/null | grep -q .
  else
    return 1
  fi
}

run_one () {
  local MODEL="$1"
  local C="$2"
  local SEED="$3"

  local SCRIPT=""
  local MODEL_MODE=""

  if [ "$MODEL" = "basic_ppo" ]; then
    SCRIPT="$BASIC_SCRIPT"
    MODEL_MODE="basic_ppo"
  elif [ "$MODEL" = "factorized_ac" ]; then
    SCRIPT="$FACT_SCRIPT"
    MODEL_MODE="factorized_ac"
  else
    echo "Unknown model: $MODEL"
    return 1
  fi

  local RUN_ID="${MODEL}_C${C}_k${K}_F${F}_T${T}_seed${SEED}"
  local LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

  if has_existing_result "$MODEL" "$C" "$SEED"; then
    local NOW
    NOW=$(date "+%Y-%m-%d %H:%M:%S")
    echo "SKIP existing result: $RUN_ID"
    echo "$MODEL,$C,$K,$F,$T,$SEED,SKIPPED_EXISTING,$NOW,$NOW,$LOG_PATH" >> "$STATUS_CSV"
    return 0
  fi

  local START_TIME
  local END_TIME
  START_TIME=$(date "+%Y-%m-%d %H:%M:%S")

  echo ""
  echo "============================================================"
  echo "RUN: $RUN_ID"
  echo "SCRIPT: $SCRIPT"
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

  echo "$MODEL,$C,$K,$F,$T,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo "END: $END_TIME"
  echo "STATUS: $STATUS"
}

echo "============================================================"
echo "K=24 Basic PPO vs Factorized AC 10-Seed Final Batch"
echo "============================================================"
echo "Device: $DEVICE"
echo "Seeds: ${SEEDS[*]}"
echo "C values: ${CS[*]}"
echo "Status CSV: $STATUS_CSV"
echo "============================================================"

for C in "${CS[@]}"
do
  for SEED in "${SEEDS[@]}"
  do
    run_one basic_ppo "$C" "$SEED"
    run_one factorized_ac "$C" "$SEED"
  done
done

echo ""
echo "============================================================"
echo "K=24 10-SEED BATCH FINISHED"
echo "Status file:"
echo "$STATUS_CSV"
echo "============================================================"
