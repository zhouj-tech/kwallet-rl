#!/usr/bin/env bash
set -u

cd /Users/qiubi/kwallet-rl

mkdir -p src/idea5/general_model/logs/batch
mkdir -p src/idea5/general_model/logs/batch_status

STATUS_FILE="src/idea5/general_model/logs/batch_status/threshold_conditional_C_flush_sweep_status.csv"
echo "timestamp,flush_levels,C,model,seed,status,log_file" > "${STATUS_FILE}"

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

CS=(1000 1200 1600)
FLUSH_LEVELS_LIST=(5 9 17)

RESULT_ROOT_BASE="src/idea5/general_model/results_threshold_conditional_sweep"
LOG_ROOT_BASE="src/idea5/general_model/logs_threshold_conditional_sweep"

run_and_record() {
  local flush_levels="$1"
  local C="$2"
  local model="$3"
  local cmd="$4"
  local log_file="$5"

  echo
  echo "======================================================================"
  echo "Run: flush_levels=${flush_levels}, C=${C}, model=${model}, seed=${SEED}"
  echo "Log: ${log_file}"
  echo "======================================================================"

  bash -lc "${cmd}" 2>&1 | tee "${log_file}"
  local exit_code=${PIPESTATUS[0]}

  local ts
  ts=$(date "+%Y-%m-%d %H:%M:%S")

  if [ "${exit_code}" -eq 0 ]; then
    echo "${ts},${flush_levels},${C},${model},${SEED},SUCCESS,${log_file}" >> "${STATUS_FILE}"
  else
    echo "${ts},${flush_levels},${C},${model},${SEED},FAILED,${log_file}" >> "${STATUS_FILE}"
  fi
}

for FLUSH_LEVELS in "${FLUSH_LEVELS_LIST[@]}"; do

  echo
  echo "######################################################################"
  echo "Starting flush_levels=${FLUSH_LEVELS}"
  echo "######################################################################"

  RESULT_ROOT="${RESULT_ROOT_BASE}/flush${FLUSH_LEVELS}"
  LOG_ROOT="${LOG_ROOT_BASE}/flush${FLUSH_LEVELS}"

  mkdir -p "${RESULT_ROOT}"
  mkdir -p "${LOG_ROOT}"

  for C in "${CS[@]}"; do

    # ------------------------------------------------------------------
    # 1. Grid Threshold Baseline
    # ------------------------------------------------------------------
    MODEL="grid_threshold"
    LOG_FILE="src/idea5/general_model/logs/batch/${MODEL}_flush${FLUSH_LEVELS}_C${C}_seed${SEED}.log"

    CMD="python src/idea5/general_model/scripts/run_threshold_benchmark.py \
      --mode grid \
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
      --flush_levels ${FLUSH_LEVELS} \
      --save_mode full \
      --result_root ${RESULT_ROOT} \
      --log_root ${LOG_ROOT}"

    run_and_record "${FLUSH_LEVELS}" "${C}" "${MODEL}" "${CMD}" "${LOG_FILE}"


    # ------------------------------------------------------------------
    # 2. Conditional Factorized AC
    # ------------------------------------------------------------------
    MODEL="conditional_factorized_ac"
    LOG_FILE="src/idea5/general_model/logs/batch/${MODEL}_flush${FLUSH_LEVELS}_C${C}_seed${SEED}.log"

    CMD="python src/idea5/general_model/scripts/run_conditional_factorized_ac.py \
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
      --flush_levels ${FLUSH_LEVELS} \
      --save_mode full \
      --result_root ${RESULT_ROOT} \
      --log_root ${LOG_ROOT}"

    run_and_record "${FLUSH_LEVELS}" "${C}" "${MODEL}" "${CMD}" "${LOG_FILE}"

  done
done

echo
echo "======================================================================"
echo "Batch finished."
echo "Status file:"
echo "${STATUS_FILE}"
echo "======================================================================"
