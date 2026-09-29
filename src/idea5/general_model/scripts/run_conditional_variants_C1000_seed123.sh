#!/usr/bin/env bash
set -euo pipefail

ROOT="src/idea5/general_model/results_conditional_variants_C1000_seed123"
LOGDIR="src/idea5/general_model/logs/batch"
mkdir -p "$LOGDIR"

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
  echo
  echo "======================================================================"
  echo "RUN: $NAME"
  echo "======================================================================"
  python src/idea5/general_model/scripts/run_conditional_factorized_ac.py $COMMON_ARGS $EXTRA \
    2>&1 | tee "$LOGDIR/${NAME}.log"
}

run_one "stable_C1000_flush17_seed123" ""
run_one "pressure_C1000_flush17_seed123" "--state_feature_mode pressure"
run_one "nonuniform_C1000_flush17_seed123" "--flush_grid nonuniform_v1"
run_one "pressure_nonuniform_C1000_flush17_seed123" "--state_feature_mode pressure --flush_grid nonuniform_v1"
run_one "pressure_nonuniform_mask_C1000_flush17_seed123" "--state_feature_mode pressure --flush_grid nonuniform_v1 --mask_mode safe"

python src/idea5/general_model/scripts/evaluate_general_results.py \
  --result_root "$ROOT"

echo
echo "DONE. Results root: $ROOT"
