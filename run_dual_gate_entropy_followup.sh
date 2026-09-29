#!/usr/bin/env bash

set +e

SCRIPT="src/idea4/ac/code/run_dual_branch_ac_dual_critic_gate_entropy.py"
LOG_DIR="src/idea4/ac/logs/dual_gate_entropy_followup"
STATUS_CSV="src/idea4/ac/results/overnight_status/dual_gate_entropy_followup_status.csv"

MODEL_MODE="dual_branch_factorized_ac_gate_balanced_entropy"
TRAIN_REGIME="MIX12_EQ"

C=1200
K=12
F=3
T=1000
EPISODES=1000
EVAL_EPISODES=200
DEVICE="cpu"

echo "run_id,seed,C,k,F,T,gate_min,gate_max,gate_entropy_coef,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  local SEED="$1"
  local GMIN="$2"
  local GMAX="$3"
  local ECOEF="$4"

  local RUN_ID="C${C}_k${K}_F${F}_seed${SEED}_g${GMIN}_${GMAX}_ent${ECOEF}"
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
    --gate_min "$GMIN" \
    --gate_max "$GMAX" \
    --gate_entropy_coef "$ECOEF" \
    --save_mode full \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_PATH"

  local EXIT_CODE=${PIPESTATUS[0]}
  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "$RUN_ID,$SEED,$C,$K,$F,$T,$GMIN,$GMAX,$ECOEF,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo "END: $END_TIME"
  echo "STATUS: $STATUS"
}

echo "============================================================"
echo "Dual Gate Entropy Follow-up Batch"
echo "============================================================"

# 1. Best candidate: complete 3 seeds
run_one 323 0.15 0.85 0.001
run_one 532 0.15 0.85 0.001

# 2. Range-only ablation: is improvement from gate range or entropy?
run_one 123 0.15 0.85 0.000
run_one 323 0.15 0.85 0.000
run_one 532 0.15 0.85 0.000

# 3. Second candidate: wider range + stronger entropy
run_one 323 0.10 0.90 0.010
run_one 532 0.10 0.90 0.010

echo ""
echo "============================================================"
echo "FOLLOW-UP BATCH FINISHED"
echo "Status file:"
echo "$STATUS_CSV"
echo "============================================================"
