#!/usr/bin/env bash

set +e

DQN_SCRIPT="src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py"
ROOT="src/idea3/results/dqn_baseline_k24_C800_C900_C1000_C1200_10seeds"
LOGDIR="logs/kwallet_dqn_batch"
STATUS_CSV="$LOGDIR/kwallet_dqn_baseline_k24_C800_C900_C1000_C1200_10seeds_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "model,C,k,seed,status,start_time,end_time,log_path" > "$STATUS_CSV"

SEEDS=(123 323 532 777 999 2027 3407 4501 6101 8888)
CS=(800 900 1000 1200)

run_one () {
  CVAL="$1"
  SEED="$2"
  KVAL=24

  NAME="kwallet_dqn_baseline_C${CVAL}_k${KVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "SCRIPT: $DQN_SCRIPT"
  echo "======================================================================"

  python "$DQN_SCRIPT" \
    --model_mode baseline \
    --train_regime MIX12_EQ \
    --C "$CVAL" \
    --k "$KVAL" \
    --F 3 \
    --T 1000 \
    --episodes 1000 \
    --eval_episodes 200 \
    --val_every 100 \
    --val_episodes 200 \
    --seed "$SEED" \
    --device cpu \
    --save_mode full \
    --output_dir "$ROOT" \
    2>&1 | tee "$LOG_PATH"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED_EXIT_${EXIT_CODE}"
  fi

  echo "DQN_baseline,$CVAL,$KVAL,$SEED,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

for CVAL in "${CS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "$CVAL" "$SEED"
  done
done

echo
echo "======================================================================"
echo "All DQN baseline runs attempted."
echo "Trying aggregate-only mode..."
echo "======================================================================"

python "$DQN_SCRIPT" \
  --aggregate_only \
  --output_dir "$ROOT" \
  2>&1 | tee "$LOGDIR/kwallet_dqn_baseline_k24_C800_C900_C1000_C1200_10seeds_aggregate.log"

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
