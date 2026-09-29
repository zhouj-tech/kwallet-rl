#!/bin/bash
set -e

cd /Users/qiubi/kwallet-rl
source .venv/bin/activate

echo "============================================================"
echo "K=12 SEQUENTIAL MOCK TEST"
echo "This checks DQN, factorized AC, and dual AC."
echo "============================================================"

C=1200
K=12
F=3
T=1000
EPISODES=3
TRAIN_REGIME="MIX12_EQ"
SEEDS="123 323"

DQN_SCRIPT="src/ideaextra/kwallet_ideaextra_dqn.py"
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
echo "1. MOCK RUN: DQN baseline"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Mock DQN baseline | seed=$SEED | k=$K"

  python -u "$DQN_SCRIPT" \
    --model_mode baseline \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    2>&1 | tee "src/idea4/ac/logs/k12_sequential_mock/mock_dqn_baseline_k${K}_seed${SEED}.log"
done

echo ""
echo "============================================================"
echo "2. MOCK RUN: Factorized AC"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Mock factorized_ac | seed=$SEED | k=$K"

  python -u "$FACTOR_SCRIPT" \
    --model_mode factorized_ac \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    2>&1 | tee "src/idea4/ac/logs/k12_sequential_mock/mock_factorized_ac_k${K}_seed${SEED}.log"
done

echo ""
echo "============================================================"
echo "3. MOCK RUN: Dual-branch AC"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Mock dual_branch_factorized_ac | seed=$SEED | k=$K"

  python -u "$DUAL_SCRIPT" \
    --model_mode dual_branch_factorized_ac \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    2>&1 | tee "src/idea4/ac/logs/k12_sequential_mock/mock_dual_branch_ac_k${K}_seed${SEED}.log"
done

echo ""
echo "============================================================"
echo "MOCK TEST FINISHED"
echo "If all three models printed env C=1200, k=12, F=3, T=1000, formal run is safe."
echo "============================================================"
