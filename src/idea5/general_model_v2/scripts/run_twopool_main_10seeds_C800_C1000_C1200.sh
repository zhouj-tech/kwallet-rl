#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model_v2/results_twopool_main_10seeds_C800_C1000_C1200"
LOGDIR="src/idea5/general_model_v2/logs/batch"
STATUS_CSV="$LOGDIR/twopool_main_10seeds_C800_C1000_C1200_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "method,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

SEEDS=(123 323 532 777 999 2027 3407 4501 6101 8888)
CS=(800 1000 1200)

run_threshold () {
  MODE="$1"
  CVAL="$2"
  SEED="$3"

  NAME="twopool_${MODE}_threshold_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "======================================================================"

  python src/idea5/general_model_v2/scripts/run_threshold_benchmark.py \
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
    --flush_levels 17 \
    --threshold_mode "$MODE" \
    --save_mode full \
    --result_root "$ROOT" \
    2>&1 | tee "$LOG_PATH"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "${MODE}_threshold,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

run_conditional_imit100 () {
  CVAL="$1"
  SEED="$2"

  NAME="twopool_conditional_pressure_maskSafe_imit100_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "======================================================================"

  python src/idea5/general_model_v2/scripts/run_conditional_factorized_ac.py \
    --train_regime MIX12_EQ \
    --C "$CVAL" \
    --F 3 \
    --T 1000 \
    --episodes 1500 \
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
    --state_feature_mode pressure \
    --mask_mode safe \
    --imitation_mode threshold_pretrain \
    --imitation_episodes 100 \
    --imitation_epochs 1 \
    --imitation_batch_size 512 \
    --imitation_lr 0.0003 \
    --save_mode full \
    --result_root "$ROOT" \
    2>&1 | tee "$LOG_PATH"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "conditional_ac_imit100,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

echo
echo "======================================================================"
echo "1/3 Running Global Threshold"
echo "======================================================================"

for CVAL in "${CS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_threshold global "$CVAL" "$SEED"
  done
done

echo
echo "======================================================================"
echo "2/3 Running Pool-Specific Threshold"
echo "======================================================================"

for CVAL in "${CS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_threshold pool_specific "$CVAL" "$SEED"
  done
done

echo
echo "======================================================================"
echo "3/3 Running Conditional AC + Imit100"
echo "======================================================================"

for CVAL in "${CS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_conditional_imit100 "$CVAL" "$SEED"
  done
done

echo
echo "======================================================================"
echo "All requested two-pool runs attempted. Aggregating..."
echo "======================================================================"

python src/idea5/general_model_v2/scripts/evaluate_v2_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/twopool_main_10seeds_C800_C1000_C1200_aggregate.log"

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
