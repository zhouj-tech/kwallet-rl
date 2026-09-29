#!/usr/bin/env bash

set +e

SCRIPT="src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py"

LOG_DIR="src/idea4/ac/logs/conditional_E32_H256_k24_10seed"
STATUS_CSV="src/idea4/ac/results/conditional_E32_H256_k24_10seed_status/conditional_E32_H256_k24_extend_status.csv"

TRAIN_REGIME="MIX12_EQ"
MODEL_MODE="conditional_factorized_ac"

F=3
T=1000
K=24

EPISODES=1000
EVAL_EPISODES=200
DEVICE="cpu"
SAVE_MODE="full"

EMBED=32
COND_H=256

# 已有 123,323,532,777,999；这里只补新 seeds
SEEDS=(2027 3407 4501 6101 8888)

CS=(800 1200)

echo "model,C,k,F,T,seed,embed_dim,conditional_hidden_size,status,start_time,end_time,log_path" > "$STATUS_CSV"

has_existing_result () {
  local C="$1"
  local SEED="$2"

  find src/idea4/ac/results/conditional_factorized_ac/runs \
    -path "*conditional_factorized_ac_train${TRAIN_REGIME}_C${C}_k${K}_T${T}_F${F}_seed${SEED}*condE${EMBED}_condH${COND_H}*/cross_regime_results.json" \
    -print -quit 2>/dev/null | grep -q .
}

run_one () {
  local C="$1"
  local SEED="$2"

  local RUN_ID="${MODEL_MODE}_E${EMBED}_H${COND_H}_C${C}_k${K}_F${F}_T${T}_seed${SEED}"
  local LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

  if has_existing_result "$C" "$SEED"; then
    local NOW
    NOW=$(date "+%Y-%m-%d %H:%M:%S")
    echo "SKIP existing result: $RUN_ID"
    echo "$MODEL_MODE,$C,$K,$F,$T,$SEED,$EMBED,$COND_H,SKIPPED_EXISTING,$NOW,$NOW,$LOG_PATH" >> "$STATUS_CSV"
    return 0
  fi

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
    --conditional_embed_dim "$EMBED" \
    --conditional_hidden_size "$COND_H" \
    --reward_mode original \
    --val_metric value_accept_ratio \
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

  echo "$MODEL_MODE,$C,$K,$F,$T,$SEED,$EMBED,$COND_H,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo "END: $END_TIME"
  echo "STATUS: $STATUS"
}

echo "============================================================"
echo "Conditional Factorized AC E32,H256 k=24 10-seed Extension"
echo "============================================================"
echo "Settings: C=800/1200, k=24, F=3, T=1000"
echo "New seeds: ${SEEDS[*]}"
echo "Params: E=$EMBED, H=$COND_H"
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
echo "CONDITIONAL E32,H256 K24 10-SEED EXTENSION FINISHED"
echo "Status file:"
echo "$STATUS_CSV"
echo "============================================================"
