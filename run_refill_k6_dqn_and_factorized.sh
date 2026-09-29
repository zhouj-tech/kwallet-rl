#!/bin/bash
set -e

cd /Users/qiubi/kwallet-rl
source .venv/bin/activate

C=1200
K=6
F=3
T=1000
EPISODES=1000
TRAIN_REGIME="MIX12_EQ"
SEEDS="123 323"

DQN_SCRIPT="src/ideaextra/kwallet_ideaextra_dqn_k6_formal.py"
FACTOR_SCRIPT="src/idea4/ac/code/run_factorized_ac_benchmark.py"

echo "============================================================"
echo "REFILL K=6 FORMAL RUN"
echo "Models: DQN baseline + Plain Factorized AC"
echo "Seeds: 123, 323"
echo "Env: C=${C}, k=${K}, F=${F}, T=${T}"
echo "============================================================"

echo ""
echo "Checking scripts..."
for SCRIPT in "$DQN_SCRIPT" "$FACTOR_SCRIPT"
do
  if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT"
    exit 1
  fi
  echo "Found: $SCRIPT"
done

echo ""
echo "============================================================"
echo "1. DQN BASELINE k=6"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Running DQN baseline | seed=${SEED}"

  python -u "$DQN_SCRIPT" \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    2>&1 | tee "src/idea4/ac/logs/refill_k6_all/dqn_baseline_MIX12_EQ_C${C}_k${K}_T${T}_F${F}_seed${SEED}_e${EPISODES}.log"
done

echo ""
echo "============================================================"
echo "2. PLAIN FACTORIZED AC k=6"
echo "============================================================"

for SEED in $SEEDS
do
  echo ""
  echo "Running factorized_ac | seed=${SEED}"

  python -u "$FACTOR_SCRIPT" \
    --model_mode factorized_ac \
    --train_regime "$TRAIN_REGIME" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --C "$C" \
    --k "$K" \
    --F "$F" \
    --T "$T" \
    2>&1 | tee "src/idea4/ac/logs/refill_k6_all/factorized_ac_MIX12_EQ_C${C}_k${K}_T${T}_F${F}_seed${SEED}_e${EPISODES}.log"
done

echo ""
echo "============================================================"
echo "REFILL K=6 FORMAL RUN FINISHED"
echo "============================================================"
