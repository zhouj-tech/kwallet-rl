#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model/results_stable_vs_pressure_imit50_C1800_C2000_seed123"
LOGDIR="src/idea5/general_model/logs/batch"
STATUS_CSV="$LOGDIR/stable_vs_pressure_imit50_C1800_C2000_seed123_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  VARIANT="$1"
  CVAL="$2"
  EXTRA="$3"

  SEED=123
  NAME="${VARIANT}_C${CVAL}_seed${SEED}"
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
    --seed "$SEED" \
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

  echo "$VARIANT,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

for CVAL in 1800 2000; do
  run_one "stable" "$CVAL" \
    "--state_feature_mode base --mask_mode none --imitation_mode none"

  run_one "pressure_imit50e1" "$CVAL" \
    "--state_feature_mode pressure --imitation_mode threshold_pretrain --imitation_episodes 50 --imitation_epochs 1 --imitation_batch_size 512 --imitation_lr 0.0003"
done

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating..."
echo "======================================================================"

python src/idea5/general_model/scripts/evaluate_general_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/stable_vs_pressure_imit50_C1800_C2000_seed123_aggregate.log"

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
