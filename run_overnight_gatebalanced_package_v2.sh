#!/usr/bin/env bash

# Do not stop the full script when one command fails.
set +e

cd /Users/qiubi/kwallet-rl
source .venv/bin/activate

LOG_DIR="src/idea4/ac/logs/overnight_gatebalanced_package_v2"
STATUS_DIR="$LOG_DIR/status"
STATUS_FILE="$STATUS_DIR/overnight_status.txt"
RESULT_FILE="$STATUS_DIR/overnight_task_results.csv"

mkdir -p "$LOG_DIR"
mkdir -p "$STATUS_DIR"

# Per-task timeout.
# 7200s = 2 hours per task.
# If one task hangs, it will be killed and the script will continue.
TASK_TIMEOUT_SECONDS=7200

echo "task_name,status,exit_code,start_time,end_time,log_file" > "$RESULT_FILE"

echo "============================================================" | tee "$STATUS_FILE"
echo "Overnight K-Wallet experiment package V2 started: $(date)" | tee -a "$STATUS_FILE"
echo "Goal: fair 4-seed comparison for C=1200,k=12,F=3 + optional k=24 Gate-balanced signal" | tee -a "$STATUS_FILE"
echo "Task timeout: ${TASK_TIMEOUT_SECONDS}s per task" | tee -a "$STATUS_FILE"
echo "Logs saved under: $LOG_DIR" | tee -a "$STATUS_FILE"
echo "============================================================" | tee -a "$STATUS_FILE"

run_and_log () {
  NAME="$1"
  shift

  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
  LOG_FILE="$LOG_DIR/${NAME}.log"

  echo "" | tee -a "$STATUS_FILE"
  echo "============================================================" | tee -a "$STATUS_FILE"
  echo "START: $NAME | $START_TIME" | tee -a "$STATUS_FILE"
  echo "COMMAND: $*" | tee -a "$STATUS_FILE"
  echo "LOG: $LOG_FILE" | tee -a "$STATUS_FILE"
  echo "============================================================" | tee -a "$STATUS_FILE"

  timeout "$TASK_TIMEOUT_SECONDS" "$@" 2>&1 | tee "$LOG_FILE"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
    echo "END: $NAME | SUCCESS | exit_code=$EXIT_CODE | $END_TIME" | tee -a "$STATUS_FILE"
  elif [ "$EXIT_CODE" -eq 124 ]; then
    STATUS="TIMEOUT"
    echo "END: $NAME | TIMEOUT | exit_code=$EXIT_CODE | $END_TIME" | tee -a "$STATUS_FILE"
    echo "WARNING: $NAME timed out after ${TASK_TIMEOUT_SECONDS}s. Continue to next task." | tee -a "$STATUS_FILE"
  else
    STATUS="FAILED"
    echo "END: $NAME | FAILED | exit_code=$EXIT_CODE | $END_TIME" | tee -a "$STATUS_FILE"
    echo "WARNING: $NAME failed. Continue to next task." | tee -a "$STATUS_FILE"
  fi

  echo "$NAME,$STATUS,$EXIT_CODE,\"$START_TIME\",\"$END_TIME\",\"$LOG_FILE\"" >> "$RESULT_FILE"
}

echo "" | tee -a "$STATUS_FILE"
echo "P0. Fair comparison completion: C=1200, k=12, F=3, T=1000" | tee -a "$STATUS_FILE"
echo "Models: Factorized AC + Original Dual AC | New seeds: 532, 999" | tee -a "$STATUS_FILE"

for SEED in 532 999
do
  run_and_log "P0_factorized_ac_C1200_k12_F3_seed${SEED}" \
    python -u src/idea4/ac/code/run_factorized_ac_benchmark.py \
      --model_mode factorized_ac \
      --train_regime MIX12_EQ \
      --seed "$SEED" \
      --episodes 1000 \
      --eval_episodes 200 \
      --C 1200 \
      --k 12 \
      --F 3 \
      --T 1000 \
      --device cpu

  run_and_log "P0_dual_branch_ac_C1200_k12_F3_seed${SEED}" \
    python -u src/idea4/ac/code/run_dual_branch_ac_benchmark.py \
      --model_mode dual_branch_factorized_ac \
      --train_regime MIX12_EQ \
      --seed "$SEED" \
      --episodes 1000 \
      --eval_episodes 200 \
      --C 1200 \
      --k 12 \
      --F 3 \
      --T 1000 \
      --device cpu
done

echo "" | tee -a "$STATUS_FILE"
echo "P2. Optional scalability signal: Gate-balanced Dual AC on C=1200, k=24, F=3, seed=123" | tee -a "$STATUS_FILE"

run_and_log "P2_gate_balanced_dual_ac_C1200_k24_F3_seed123" \
  python -u src/idea4/ac/code/run_dual_branch_ac_benchmark.py \
    --model_mode dual_branch_factorized_ac_gate_balanced \
    --train_regime MIX12_EQ \
    --seed 123 \
    --episodes 1000 \
    --eval_episodes 200 \
    --C 1200 \
    --k 24 \
    --F 3 \
    --T 1000 \
    --device cpu

echo "" | tee -a "$STATUS_FILE"
echo "R. Rebuilding experiment map and final tables" | tee -a "$STATUS_FILE"

run_and_log "R1_build_experiment_map" \
  python -u src/idea4/analysis/experiment_map/build_experiment_map.py

run_and_log "R2_build_final_tables_v1" \
  python -u src/idea4/analysis/final_tables_v1/build_final_tables_v1.py

echo "" | tee -a "$STATUS_FILE"
echo "============================================================" | tee -a "$STATUS_FILE"
echo "Overnight package finished: $(date)" | tee -a "$STATUS_FILE"
echo "============================================================" | tee -a "$STATUS_FILE"

echo "" | tee -a "$STATUS_FILE"
echo "Task result summary:" | tee -a "$STATUS_FILE"
cat "$RESULT_FILE" | tee -a "$STATUS_FILE"

echo "" | tee -a "$STATUS_FILE"
echo "Counts:" | tee -a "$STATUS_FILE"
echo "SUCCESS: $(grep -c ',SUCCESS,' "$RESULT_FILE")" | tee -a "$STATUS_FILE"
echo "FAILED:  $(grep -c ',FAILED,' "$RESULT_FILE")" | tee -a "$STATUS_FILE"
echo "TIMEOUT: $(grep -c ',TIMEOUT,' "$RESULT_FILE")" | tee -a "$STATUS_FILE"

echo "" | tee -a "$STATUS_FILE"
echo "Files to check tomorrow:" | tee -a "$STATUS_FILE"
echo "- $STATUS_FILE" | tee -a "$STATUS_FILE"
echo "- $RESULT_FILE" | tee -a "$STATUS_FILE"
echo "- src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv" | tee -a "$STATUS_FILE"
echo "- src/idea4/analysis/final_tables_v1/ablation_summary.csv" | tee -a "$STATUS_FILE"
echo "- src/idea4/analysis/final_tables_v1/seed_level_runs_used.csv" | tee -a "$STATUS_FILE"

