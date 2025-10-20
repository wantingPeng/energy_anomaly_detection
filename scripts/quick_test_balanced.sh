#!/bin/bash
# Quick test: Single strategy for balanced Random Forest

echo "========================================"
echo "Quick Test: BalancedRandomForest + SMOTE"
echo "========================================"

python src/training/random_forest/train_random_forest_balanced.py \
    --config configs/random_forest_balanced_config.yaml \
    --resample smote \
    --use_balanced_rf \
    --output_suffix "pcb"

echo ""
echo "========================================"
echo "Test completed!"
echo "Check experiments/random_forest_balanced/ for results"
echo "========================================"

