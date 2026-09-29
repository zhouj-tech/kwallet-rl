#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model/results_onepool_imit100_C900_C1000_C1100_to10seeds"
LOGDIR="src/idea5/general_model/logs/batch"
STATUS_CSV="$LOGDIR/onepool_imit100_C900_C1000_C1100_to10seeds_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,C,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  CVAL="$1"
  SEED="$2"

  NAME="onepool_conditional_imit100_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "C: $CVAL"
  echo "SEED: $SEED"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model/scripts/run_conditional_factorized_ac.py \
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

  echo "OnePool+Conditional+Imit100,$CVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo
  echo "FINISHED: $NAME"
  echo "STATUS: $STATUS"
}

# C=1000: only remaining seeds. Existing completed seeds: 123, 323, 532.
for SEED in 777 999 2027 3407 4501 6101 8888; do
  run_one 1000 "$SEED"
done

# C=900: full 10 seeds.
for SEED in 123 323 532 777 999 2027 3407 4501 6101 8888; do
  run_one 900 "$SEED"
done

# C=1100: full 10 seeds.
for SEED in 123 323 532 777 999 2027 3407 4501 6101 8888; do
  run_one 1100 "$SEED"
done

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating this result root..."
echo "======================================================================"

python src/idea5/general_model/scripts/evaluate_general_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/onepool_imit100_C900_C1000_C1100_to10seeds_aggregate.log"

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
