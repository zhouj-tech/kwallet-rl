#!/usr/bin/env bash

# Do not stop the whole batch if one run fails.
set +e

ROOT="src/idea5/general_model/results_conditional_variant_sweep_C1000_seed123"
LOGDIR="src/idea5/general_model/logs/batch"
STATUS_CSV="$LOGDIR/conditional_variant_sweep_C1000_seed123_status.csv"
MASTER_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,status,start_time,end_time,log_path" > "$STATUS_CSV"

COMMON_ARGS="\
  --train_regime MIX12_EQ \
  --C 1000 \
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
  --save_mode full \
  --result_root $ROOT"

run_one () {
  NAME="$1"
  EXTRA="$2"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "START: $START_TIME"
  echo "EXTRA: $EXTRA"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model/scripts/run_conditional_factorized_ac.py $COMMON_ARGS $EXTRA \
    2>&1 | tee "$LOG_PATH"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "$NAME,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo
  echo "FINISHED: $NAME"
  echo "STATUS: $STATUS"
  echo "END: $END_TIME"
  echo
}

echo "Master batch started at: $MASTER_START_TIME"
echo "Result root: $ROOT"
echo "Status CSV: $STATUS_CSV"

# 1. Stable baseline rerun
run_one "01_stable_C1000_flush17_seed123" \
""

# 2. Pressure
run_one "02_pressure_C1000_flush17_seed123" \
"--state_feature_mode pressure"

# 3. Pressure + Nonuniform
run_one "03_pressure_nonuniform_C1000_flush17_seed123" \
"--state_feature_mode pressure --flush_grid nonuniform_v1"

# 4. Pressure + Nonuniform + Mask
run_one "04_pressure_nonuniform_mask_C1000_flush17_seed123" \
"--state_feature_mode pressure --flush_grid nonuniform_v1 --mask_mode safe"

# 5. Pressure + Light Imitation
run_one "05_pressure_imit50e1_C1000_flush17_seed123" \
"--state_feature_mode pressure --imitation_mode threshold_pretrain --imitation_episodes 50 --imitation_epochs 1 --imitation_batch_size 512 --imitation_lr 0.0003"

# 6. Pressure + Medium Imitation
run_one "06_pressure_imit200e2_C1000_flush17_seed123" \
"--state_feature_mode pressure --imitation_mode threshold_pretrain --imitation_episodes 200 --imitation_epochs 2 --imitation_batch_size 512 --imitation_lr 0.0003"

# 7. Nonuniform only
run_one "07_nonuniform_C1000_flush17_seed123" \
"--flush_grid nonuniform_v1"

# 8. Mask only
run_one "08_mask_C1000_flush17_seed123" \
"--mask_mode safe"

# 9. Pressure + Mask
run_one "09_pressure_mask_C1000_flush17_seed123" \
"--state_feature_mode pressure --mask_mode safe"

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating successful saved runs..."
echo "======================================================================"

python src/idea5/general_model/scripts/evaluate_general_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/conditional_variant_sweep_C1000_seed123_aggregate.log"

AGG_EXIT_CODE=${PIPESTATUS[0]}
AGG_END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

if [ "$AGG_EXIT_CODE" -eq 0 ]; then
  echo "aggregate,SUCCESS,$MASTER_START_TIME,$AGG_END_TIME,$LOGDIR/conditional_variant_sweep_C1000_seed123_aggregate.log" >> "$STATUS_CSV"
else
  echo "aggregate,FAILED_EXIT_${AGG_EXIT_CODE},$MASTER_START_TIME,$AGG_END_TIME,$LOGDIR/conditional_variant_sweep_C1000_seed123_aggregate.log" >> "$STATUS_CSV"
fi

echo
echo "Batch finished."
echo "Status CSV:"
echo "$STATUS_CSV"
echo
echo "Results root:"
echo "$ROOT"
