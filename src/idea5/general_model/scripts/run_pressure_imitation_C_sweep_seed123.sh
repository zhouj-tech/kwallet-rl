#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model/results_pressure_imitation_C_sweep_seed123"
LOGDIR="src/idea5/general_model/logs/batch"
STATUS_CSV="$LOGDIR/pressure_imitation_C_sweep_seed123_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,C,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  NAME="$1"
  CVAL="$2"
  EXTRA="$3"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "C: $CVAL"
  echo "EXTRA: $EXTRA"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model/scripts/run_conditional_factorized_ac.py \
    --train_regime MIX12_EQ \
    --C "$CVAL" \
    --F 3 \
    --T 1000 \
    --episodes 1000 \
    --eval_episodes 200 \
    --seed 123 \
    --money_p 1.0 \
    --money_tau 100.0 \
    --drop_penalty 0.0 \
    --device cpu \
    --hidden_size 128 \
    --settle_embed_dim 32 \
    --conditional_hidden_size 256 \
    --flush_levels 17 \
    --learning_rate 0.0001 \
    --clip_eps 0.1 \
    --reward_scale 100 \
    --state_feature_mode pressure \
    --imitation_mode threshold_pretrain \
    --imitation_batch_size 512 \
    --imitation_lr 0.0003 \
    --save_mode full \
    --result_root "$ROOT" \
    $EXTRA \
    2>&1 | tee "$LOG_PATH"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "$NAME,$CVAL,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo
  echo "FINISHED: $NAME"
  echo "STATUS: $STATUS"
  echo
}

# ------------------------------------------------------------
# A. Current best: Pressure + Imit50e1
# Already have C=1000 from previous sweep.
# Run C=1200 and C=1600.
# ------------------------------------------------------------
run_one "pressure_imit50e1_C1200_seed123" "1200" \
  "--imitation_episodes 50 --imitation_epochs 1"

run_one "pressure_imit50e1_C1600_seed123" "1600" \
  "--imitation_episodes 50 --imitation_epochs 1"

# ------------------------------------------------------------
# B. New candidate: Pressure + Imit100e1
# Run C=1000, 1200, 1600.
# ------------------------------------------------------------
run_one "pressure_imit100e1_C1000_seed123" "1000" \
  "--imitation_episodes 100 --imitation_epochs 1"

run_one "pressure_imit100e1_C1200_seed123" "1200" \
  "--imitation_episodes 100 --imitation_epochs 1"

run_one "pressure_imit100e1_C1600_seed123" "1600" \
  "--imitation_episodes 100 --imitation_epochs 1"

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating..."
echo "======================================================================"

python src/idea5/general_model/scripts/evaluate_general_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/pressure_imitation_C_sweep_seed123_aggregate.log"

AGG_EXIT_CODE=${PIPESTATUS[0]}
AGG_END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

if [ "$AGG_EXIT_CODE" -eq 0 ]; then
  echo "aggregate,ALL,SUCCESS,NA,$AGG_END_TIME,$LOGDIR/pressure_imitation_C_sweep_seed123_aggregate.log" >> "$STATUS_CSV"
else
  echo "aggregate,ALL,FAILED_EXIT_${AGG_EXIT_CODE},NA,$AGG_END_TIME,$LOGDIR/pressure_imitation_C_sweep_seed123_aggregate.log" >> "$STATUS_CSV"
fi

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
