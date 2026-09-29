#!/usr/bin/env bash

set +e

ROOT="src/idea5/general_model_v2/results_imit100_regcoef_C1000_seed123"
LOGDIR="src/idea5/general_model_v2/logs/batch"
STATUS_CSV="$LOGDIR/imit100_regcoef_C1000_seed123_status.csv"

mkdir -p "$LOGDIR"
mkdir -p "$ROOT"

echo "variant,C,seed,imitation_reg_coef,status,start_time,end_time,log_path" > "$STATUS_CSV"

run_one () {
  REG="$1"
  SEED=123
  CVAL=1000

  REG_TAG=$(python - <<PY
reg = float("$REG")
print(f"{int(round(reg * 1000)):03d}")
PY
)

  NAME="conditional_pressure_maskSafe_imit100_reg${REG_TAG}_C${CVAL}_seed${SEED}"
  LOG_PATH="$LOGDIR/${NAME}.log"
  START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "REG: $REG"
  echo "LOG: $LOG_PATH"
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
    --imitation_reg_coef "$REG" \
    --imitation_reg_batch_size 512 \
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

  echo "Pressure+MaskSafe+Imit100+Reg${REG_TAG},$CVAL,$SEED,$REG,$STATUS,$START_TIME,$END_TIME,$LOG_PATH" >> "$STATUS_CSV"

  echo
  echo "FINISHED: $NAME"
  echo "STATUS: $STATUS"
}

run_one 0.01
run_one 0.03
run_one 0.05

echo
echo "======================================================================"
echo "All requested runs attempted."
echo "Now aggregating..."
echo "======================================================================"

python src/idea5/general_model_v2/scripts/evaluate_v2_results.py \
  --result_root "$ROOT" \
  2>&1 | tee "$LOGDIR/imit100_regcoef_C1000_seed123_aggregate.log"

echo
echo "DONE"
echo "Status CSV: $STATUS_CSV"
echo "Results root: $ROOT"
