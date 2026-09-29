#!/usr/bin/env bash
set -euo pipefail

cd /Users/qiubi/kwallet-rl

SCRIPT="src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py"
OUT_DIR="src/idea4/ac/results/basic_ppo"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "$LOG_DIR"

KS=(3 6 12)
SEEDS=(123 323)

C=1200
F=3
T=1000
EPISODES=1000
EVAL_EPISODES=200
DEVICE="cpu"

echo "============================================================"
echo "Run Basic PPO full benchmark"
echo "model_mode=basic_ppo"
echo "k=${KS[*]}"
echo "seeds=${SEEDS[*]}"
echo "C=$C F=$F T=$T episodes=$EPISODES eval_episodes=$EVAL_EPISODES"
echo "============================================================"

for K in "${KS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "START basic_ppo | k=$K | seed=$SEED"
    echo "============================================================"

    python -u "$SCRIPT" \
      --seed "$SEED" \
      --C "$C" \
      --k "$K" \
      --F "$F" \
      --T "$T" \
      --episodes "$EPISODES" \
      --eval_episodes "$EVAL_EPISODES" \
      --device "$DEVICE" \
      --output_dir "$OUT_DIR" \
      2>&1 | tee "${LOG_DIR}/basic_ppo_C${C}_k${K}_T${T}_F${F}_seed${SEED}.log"

    echo ""
    echo "FINISHED basic_ppo | k=$K | seed=$SEED"
  done
done

echo ""
echo "============================================================"
echo "Rebuild Basic PPO aggregate CSV"
echo "============================================================"

python -u "$SCRIPT" \
  --aggregate_only \
  --output_dir "$OUT_DIR"

echo ""
echo "============================================================"
echo "Basic PPO full benchmark finished"
echo "Results root: $OUT_DIR"
echo "Aggregate: ${OUT_DIR}/aggregates/basic_ppo_fair_benchmark_aggregated.csv"
echo "Logs: $LOG_DIR"
echo "============================================================"
