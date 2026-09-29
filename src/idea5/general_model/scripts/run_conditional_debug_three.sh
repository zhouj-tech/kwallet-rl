#!/usr/bin/env bash
set -u

cd /Users/qiubi/kwallet-rl

mkdir -p src/idea5/general_model/logs/batch
mkdir -p src/idea5/general_model/logs/batch_status

STATUS_FILE="src/idea5/general_model/logs/batch_status/conditional_debug_three_status.csv"
echo "timestamp,experiment,C,flush_levels,lr,clip_eps,reward_scale,seed,status,log_file" > "${STATUS_FILE}"

SEED=123
F=3
T=1000
EPISODES=1000
EVAL_EPISODES=200
MONEY_P=1.0
MONEY_TAU=100.0
DROP_PENALTY=0.0
DEVICE=cpu
TRAIN_REGIME=MIX12_EQ

RESULT_ROOT="src/idea5/general_model/results_conditional_debug_three"
LOG_ROOT="src/idea5/general_model/logs_conditional_debug_three"

run_and_record() {
  local exp_name="$1"
  local C="$2"
  local flush_levels="$3"
  local lr="$4"
  local clip_eps="$5"
  local reward_scale="$6"
  local log_file="$7"

  echo
  echo "======================================================================"
  echo "Experiment: ${exp_name}"
  echo "C=${C}, flush_levels=${flush_levels}, lr=${lr}, clip_eps=${clip_eps}, reward_scale=${reward_scale}"
  echo "Log: ${log_file}"
  echo "======================================================================"

  python src/idea5/general_model/scripts/run_conditional_factorized_ac.py \
    --train_regime ${TRAIN_REGIME} \
    --C ${C} \
    --F ${F} \
    --T ${T} \
    --episodes ${EPISODES} \
    --eval_episodes ${EVAL_EPISODES} \
    --seed ${SEED} \
    --money_p ${MONEY_P} \
    --money_tau ${MONEY_TAU} \
    --drop_penalty ${DROP_PENALTY} \
    --device ${DEVICE} \
    --hidden_size 128 \
    --settle_embed_dim 32 \
    --conditional_hidden_size 256 \
    --flush_levels ${flush_levels} \
    --learning_rate ${lr} \
    --clip_eps ${clip_eps} \
    --reward_scale ${reward_scale} \
    --save_mode full \
    --result_root ${RESULT_ROOT} \
    --log_root ${LOG_ROOT} \
    2>&1 | tee "${log_file}"

  local exit_code=${PIPESTATUS[0]}
  local ts
  ts=$(date "+%Y-%m-%d %H:%M:%S")

  if [ "${exit_code}" -eq 0 ]; then
    echo "${ts},${exp_name},${C},${flush_levels},${lr},${clip_eps},${reward_scale},${SEED},SUCCESS,${log_file}" >> "${STATUS_FILE}"
  else
    echo "${ts},${exp_name},${C},${flush_levels},${lr},${clip_eps},${reward_scale},${SEED},FAILED,${log_file}" >> "${STATUS_FILE}"
  fi
}

run_and_record \
  "check_old_setting_C800_flush5_default" \
  800 \
  5 \
  0.0003 \
  0.2 \
  1.0 \
  "src/idea5/general_model/logs/batch/conditional_debug_C800_flush5_default_seed123.log"

run_and_record \
  "stable_fix_C1000_flush17" \
  1000 \
  17 \
  0.0001 \
  0.1 \
  100 \
  "src/idea5/general_model/logs/batch/conditional_debug_C1000_flush17_stable_seed123.log"

run_and_record \
  "stable_fix_C1200_flush17" \
  1200 \
  17 \
  0.0001 \
  0.1 \
  100 \
  "src/idea5/general_model/logs/batch/conditional_debug_C1200_flush17_stable_seed123.log"

echo
echo "======================================================================"
echo "Finished conditional debug three experiments."
echo "Status file:"
echo "${STATUS_FILE}"
echo "======================================================================"
