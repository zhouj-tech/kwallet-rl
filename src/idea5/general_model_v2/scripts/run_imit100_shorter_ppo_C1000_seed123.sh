#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model_v2/results_imit100_shorter_ppo_C1000_seed123"
LOGDIR="src/idea5/general_model_v2/logs/batch"
STATUS_CSV="$LOGDIR/imit100_shorter_ppo_C1000_seed123_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,C,seed,episodes,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  EPISODES="$1"
  SEED=123
  NAME="imit100_shorterPPO_C1000_ep${EPISODES}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "EPISODES: $EPISODES"
  echo "LOG: $LOG_PATH"
  echo "======================================================================"

  python src/idea5/general_model_v2/scripts/run_conditional_factorized_ac.py \
    --train_regime MIX12_EQ \
    --C 1000 \
    --F 3 \
    --T 1000 \
    --episodes "$EPISODES" \
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

  echo "Imit100_shorterPPO,1000,$SEED,$EPISODES,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"
}

for EP in 800 1000 1200; do
  run_one "$EP"
done

python src/idea5/general_model_v2/scripts/evaluate_v2_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/imit100_shorter_ppo_C1000_seed123_aggregate.log"

echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
