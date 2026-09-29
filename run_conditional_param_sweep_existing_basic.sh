#!/usr/bin/env bash

set +e

SCRIPT="src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py"

LOG_DIR="src/idea4/ac/logs/conditional_param_sweep"
STATUS_CSV="src/idea4/ac/results/conditional_param_sweep_status/conditional_param_sweep_existing_basic_status.csv"

TRAIN_REGIME="MIX12_EQ"
MODEL_MODE="conditional_factorized_ac"

F=3
T=1000
SEED=123
EPISODES=1000
EVAL_EPISODES=200
DEVICE="cpu"
SAVE_MODE="full"

# Existing Basic PPO comparison settings
SETTINGS=(
  "800 12"
  "800 24"
  "1200 12"
  "1200 24"
)

# variant_name embed_dim conditional_hidden_size
VARIANTS=(
  "compact 8 64"
  "base 16 128"
  "embed_wide 32 128"
  "flush_wide 32 256"
)

echo "model,variant,C,k,F,T,seed,embed_dim,conditional_hidden_size,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  local VARIANT="$1"
  local EMBED="$2"
  local COND_H="$3"
  local C="$4"
  local K="$5"

  local RUN_ID="${MODEL_MODE}_${VARIANT}_C${C}_k${K}_F${F}_T${T}_seed${SEED}_E${EMBED}_H${COND_H}"
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

  echo "$MODEL_MODE,$VARIANT,$C,$K,$F,$T,$SEED,$EMBED,$COND_H,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo "END: $END_TIME"
  echo "STATUS: $STATUS"
}

echo "============================================================"
echo "Conditional Factorized AC Parameter Sweep"
echo "============================================================"
echo "Seed: $SEED"
echo "Settings: ${SETTINGS[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "Status CSV: $STATUS_CSV"
echo "============================================================"

for SETTING in "${SETTINGS[@]}"
do
  read -r C K <<< "$SETTING"

  for VARIANT_ROW in "${VARIANTS[@]}"
  do
    read -r VARIANT EMBED COND_H <<< "$VARIANT_ROW"
    run_one "$VARIANT" "$EMBED" "$COND_H" "$C" "$K"
  done
done

echo ""
echo "============================================================"
echo "CONDITIONAL PARAM SWEEP FINISHED"
echo "Status file:"
echo "$STATUS_CSV"
echo "============================================================"
