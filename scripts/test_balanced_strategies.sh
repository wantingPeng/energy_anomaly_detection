#!/bin/bash
# Test different strategies for handling extreme class imbalance

echo "========================================"
echo "Testing Balanced Random Forest Strategies"
echo "========================================"

CONFIG="configs/random_forest_balanced_config.yaml"

# Strategy 1: Standard RF with SMOTE
echo ""
echo "Strategy 1: Standard RF + SMOTE"
echo "----------------------------------------"
python src/training/random_forest/train_random_forest_balanced.py \
    --config $CONFIG \
    --resample smote \
    --output_suffix "pcb"

# Strategy 2: Standard RF with ADASYN
echo ""
echo "Strategy 2: Standard RF + ADASYN"
echo "----------------------------------------"
python src/training/random_forest/train_random_forest_balanced.py \
    --config $CONFIG \
    --resample adasyn \
    --output_suffix "pcb"

# Strategy 3: BalancedRandomForest without resampling
echo ""
echo "Strategy 3: BalancedRandomForest (no resampling)"
echo "----------------------------------------"
python src/training/random_forest/train_random_forest_balanced.py \
    --config $CONFIG \
    --resample none \
    --use_balanced_rf \
    --output_suffix "pcb"

# Strategy 4: BalancedRandomForest with SMOTE
echo ""
echo "Strategy 4: BalancedRandomForest + SMOTE"
echo "----------------------------------------"
python src/training/random_forest/train_random_forest_balanced.py \
    --config $CONFIG \
    --resample smote \
    --use_balanced_rf \
    --output_suffix "pcb"

echo ""
echo "========================================"
echo "All strategies tested!"
echo "Check experiments/random_forest_balanced/ for results"
echo "========================================"

