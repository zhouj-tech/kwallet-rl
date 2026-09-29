#!/usr/bin/env bash
set -u

cd /Users/qiubi/kwallet-rl
source .venv/bin/activate

mkdir -p src/idea4/ac/logs/overnight_stress_15runs

MASTER_LOG="src/idea4/ac/logs/overnight_stress_15runs/master_overnight_stress_15runs.log"

echo "============================================================" | tee -a "$MASTER_LOG"
echo "Overnight Stress Batch: 15 runs" | tee -a "$MASTER_LOG"
echo "Start time: $(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "Waiting for current benchmark python process to finish..." | tee -a "$MASTER_LOG"

while pgrep -af "python.*src/idea4/ac/code/.*benchmark.py" >/dev/null
do
  echo "$(date '+%Y-%m-%d %H:%M:%S') benchmark still running... wait 60s" | tee -a "$MASTER_LOG"
  pgrep -af "python.*src/idea4/ac/code/.*benchmark.py" | tee -a "$MASTER_LOG"
  sleep 60
done

echo "" | tee -a "$MASTER_LOG"
echo "No active benchmark process detected. Start overnight batch." | tee -a "$MASTER_LOG"
echo "Batch start time: $(date)" | tee -a "$MASTER_LOG"

EPISODES=1000
EVAL_EPISODES=200
SEED=123
T=1000
TRAIN_REGIME=MIX12_EQ
REWARD_MODE=original

run_basic_ppo () {
  local C=$1
  local K=$2
  local F=$3
  local TAG="basic_ppo_C${C}_k${K}_F${F}_seed${SEED}"
  local LOG="src/idea4/ac/logs/overnight_stress_15runs/${TAG}.log"

  echo "" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "RUN: $TAG" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"

  python -u src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py \
    --model_mode basic_ppo \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --eval_episodes "$EVAL_EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    --reward_mode "$REWARD_MODE" \
    --save_mode full \
    2>&1 | tee "$LOG"

  local STATUS=${PIPESTATUS[0]}
  echo "STATUS $TAG = $STATUS" | tee -a "$MASTER_LOG"
}

run_factorized_ac () {
  local C=$1
  local K=$2
  local F=$3
  local TAG="factorized_ac_C${C}_k${K}_F${F}_seed${SEED}"
  local LOG="src/idea4/ac/logs/overnight_stress_15runs/${TAG}.log"

  echo "" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "RUN: $TAG" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"

  python -u src/idea4/ac/code/run_factorized_ac_benchmark.py \
    --model_mode factorized_ac \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --eval_episodes "$EVAL_EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    --reward_mode "$REWARD_MODE" \
    --save_mode full \
    2>&1 | tee "$LOG"

  local STATUS=${PIPESTATUS[0]}
  echo "STATUS $TAG = $STATUS" | tee -a "$MASTER_LOG"
}

run_dual_model () {
  local MODEL_MODE=$1
  local C=$2
  local K=$3
  local F=$4
  local TAG="${MODEL_MODE}_C${C}_k${K}_F${F}_seed${SEED}"
  local LOG="src/idea4/ac/logs/overnight_stress_15runs/${TAG}.log"

  echo "" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "RUN: $TAG" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"

  if [ "$MODEL_MODE" = "dual_branch_residual_risk" ]; then
    python -u src/idea4/ac/code/run_dual_branch_ac_benchmark.py \
      --model_mode "$MODEL_MODE" \
      --train_regime "$TRAIN_REGIME" \
      --seed "$SEED" \
      --episodes "$EPISODES" \
      --eval_episodes "$EVAL_EPISODES" \
      --C "$C" \
      --k "$K" \
      --F "$F" \
      --T "$T" \
      --reward_mode "$REWARD_MODE" \
      --risk_residual_scale 0.2 \
      --value_residual_scale 0.1 \
      --save_mode full \
      2>&1 | tee "$LOG"
  else
    python -u src/idea4/ac/code/run_dual_branch_ac_benchmark.py \
      --model_mode "$MODEL_MODE" \
      --train_regime "$TRAIN_REGIME" \
      --seed "$SEED" \
      --episodes "$EPISODES" \
      --eval_episodes "$EVAL_EPISODES" \
      --C "$C" \
      --k "$K" \
      --F "$F" \
      --T "$T" \
      --reward_mode "$REWARD_MODE" \
      --save_mode full \
      2>&1 | tee "$LOG"
  fi

  local STATUS=${PIPESTATUS[0]}
  echo "STATUS $TAG = $STATUS" | tee -a "$MASTER_LOG"
}

run_five_models_for_setting () {
  local C=$1
  local K=$2
  local F=$3

  echo "" | tee -a "$MASTER_LOG"
  echo "############################################################" | tee -a "$MASTER_LOG"
  echo "SETTING: C=$C k=$K F=$F T=$T seed=$SEED" | tee -a "$MASTER_LOG"
  echo "############################################################" | tee -a "$MASTER_LOG"

  run_basic_ppo "$C" "$K" "$F"
  run_factorized_ac "$C" "$K" "$F"
  run_dual_model "dual_branch_factorized_ac" "$C" "$K" "$F"
  run_dual_model "dual_branch_capacity_only" "$C" "$K" "$F"
  run_dual_model "dual_branch_residual_risk" "$C" "$K" "$F"
}

# ============================================================
# Overnight batch order
# ============================================================

run_five_models_for_setting 800 24 3
run_five_models_for_setting 1200 12 6
run_five_models_for_setting 800 12 6

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "OVERNIGHT STRESS BATCH FINISHED" | tee -a "$MASTER_LOG"
echo "End time: $(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
