#!/bin/bash
set -e

cd /Users/qiubi/kwallet-rl
source .venv/bin/activate

echo "============================================================"
echo "K=12 FORMAL SEQUENTIAL RUN"
echo "Order: DQN baseline -> factorized AC -> dual-branch AC"
echo "============================================================"

C=1200
K=12
F=3
T=1000
EPISODES=1000
TRAIN_REGIME="MIX12_EQ"
SEEDS="123 323"

DQN_SCRIPT="src/ideaextra/kwallet_ideaextra_dqn_k12_formal.py"
FACTOR_SCRIPT="src/idea4/ac/code/run_factorized_ac_benchmark.py"
DUAL_SCRIPT="src/idea4/ac/code/run_dual_branch_ac_benchmark.py"

echo ""
echo "Checking script paths..."
for SCRIPT in "$DQN_SCRIPT" "$FACTOR_SCRIPT" "$DUAL_SCRIPT"
do
  if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT"
    exit 1
  fi
  echo "Found: $SCRIPT"
done

echo ""
echo "============================================================"
echo "1. FORMAL RUN: DQN baseline k=12"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Running DQN baseline | seed=$SEED | C=$C | k=$K | F=$F | T=$T | episodes=$EPISODES"

  python -u "$DQN_SCRIPT" \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    2>&1 | tee "src/idea4/ac/logs/k12_sequential_formal/dqn_baseline_MIX12_EQ_C${C}_k${K}_T${T}_F${F}_seed${SEED}_e${EPISODES}_local.log"
done

echo ""
echo "============================================================"
echo "2. FORMAL RUN: Factorized AC k=12"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Running factorized_ac | seed=$SEED | C=$C | k=$K | F=$F | T=$T | episodes=$EPISODES"

  python -u "$FACTOR_SCRIPT" \
    --model_mode factorized_ac \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    2>&1 | tee "src/idea4/ac/logs/k12_sequential_formal/factorized_ac_MIX12_EQ_C${C}_k${K}_T${T}_F${F}_seed${SEED}_e${EPISODES}_local.log"
done

echo ""
echo "============================================================"
echo "3. FORMAL RUN: Dual-branch AC k=12"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Running dual_branch_factorized_ac | seed=$SEED | C=$C | k=$K | F=$F | T=$T | episodes=$EPISODES"

  python -u "$DUAL_SCRIPT" \
    --model_mode dual_branch_factorized_ac \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    2>&1 | tee "src/idea4/ac/logs/k12_sequential_formal/dual_branch_ac_MIX12_EQ_C${C}_k${K}_T${T}_F${F}_seed${SEED}_e${EPISODES}_local.log"
done

echo ""
echo "============================================================"
echo "K=12 FORMAL SEQUENTIAL RUN FINISHED"
echo "============================================================"
