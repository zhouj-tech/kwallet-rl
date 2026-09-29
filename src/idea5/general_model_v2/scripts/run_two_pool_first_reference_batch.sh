#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model_v2/results_first_reference_batch"
LOGDIR="src/idea5/general_model_v2/logs/batch"
STATUS_CSV="$LOGDIR/two_pool_first_reference_batch_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "model,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

COMMON_ARGS_BASE="\
  --train_regime MIX12_EQ \
  --F 3 \
  --T 1000 \
  --episodes 1000 \
  --eval_episodes 200 \
  --money_p 1.0 \
  --money_tau 100.0 \
  --drop_penalty 0.0 \
  --flush_levels 17 \
  --save_mode full \
  --result_root $ROOT"

COMMON_RL_ARGS="\
  --device cpu \
  --hidden_size 128 \
  --learning_rate 0.0001 \
  --clip_eps 0.1 \
  --reward_scale 100"

run_one () {
  MODEL="$1"
  CVAL="$2"
  SEED="$3"
  EXTRA="$4"

  NAME="${MODEL}_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "MODEL: $MODEL"
  echo "C: $CVAL"
  echo "SEED: $SEED"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  if [ "$MODEL" = "grid_threshold" ]; then
    python src/idea5/general_model_v2/scripts/run_threshold_benchmark.py \
      $COMMON_ARGS_BASE \
      --C "$CVAL" \
      --seed "$SEED" \
      $EXTRA \
      2>&1 | tee "$LOG_PATH"

  elif [ "$MODEL" = "flat_ppo" ]; then
    python src/idea5/general_model_v2/scripts/run_flat_ppo.py \
      $COMMON_ARGS_BASE \
      $COMMON_RL_ARGS \
      --C "$CVAL" \
      --seed "$SEED" \
      $EXTRA \
      2>&1 | tee "$LOG_PATH"

  elif [ "$MODEL" = "factorized_ac" ]; then
    python src/idea5/general_model_v2/scripts/run_factorized_ac.py \
      $COMMON_ARGS_BASE \
      $COMMON_RL_ARGS \
      --C "$CVAL" \
      --seed "$SEED" \
      $EXTRA \
      2>&1 | tee "$LOG_PATH"

  elif [ "$MODEL" = "conditional_factorized_ac" ]; then
    python src/idea5/general_model_v2/scripts/run_conditional_factorized_ac.py \
      $COMMON_ARGS_BASE \
      $COMMON_RL_ARGS \
      --C "$CVAL" \
      --seed "$SEED" \
      --settle_embed_dim 32 \
      --conditional_hidden_size 256 \
      $EXTRA \
      2>&1 | tee "$LOG_PATH"

  else
    echo "Unknown model: $MODEL"
    false
  fi

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "$MODEL,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo
  echo "FINISHED: $NAME"
  echo "STATUS: $STATUS"
}

echo
echo "======================================================================"
echo "Block A: Core model comparison, seed=123, C=800/1000/1200"
echo "======================================================================"

for CVAL in 800 1000 1200; do
  run_one "grid_threshold" "$CVAL" 123 ""
  run_one "flat_ppo" "$CVAL" 123 ""
  run_one "factorized_ac" "$CVAL" 123 ""
  run_one "conditional_factorized_ac" "$CVAL" 123 ""
done

echo
echo "======================================================================"
echo "Block B: Small seed check for Grid Threshold and Conditional AC at C=1000"
echo "======================================================================"

for SEED in 323 532; do
  run_one "grid_threshold" 1000 "$SEED" ""
  run_one "conditional_factorized_ac" 1000 "$SEED" ""
done

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating..."
echo "======================================================================"

python src/idea5/general_model_v2/scripts/evaluate_v2_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/two_pool_first_reference_batch_aggregate.log"

AGG_EXIT_CODE=${PIPESTATUS[0]}
AGG_END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

if [ "$AGG_EXIT_CODE" -eq 0 ]; then
  echo "aggregate,ALL,ALL,SUCCESS,NA,$AGG_END_TIME,$LOGDIR/two_pool_first_reference_batch_aggregate.log" >> "$STATUS_CSV"
else
  echo "aggregate,ALL,ALL,FAILED_EXIT_${AGG_EXIT_CODE},NA,$AGG_END_TIME,$LOGDIR/two_pool_first_reference_batch_aggregate.log" >> "$STATUS_CSV"
fi

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
