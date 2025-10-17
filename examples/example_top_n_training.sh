#!/bin/bash
# Example: Training XGBoost with different top-N feature configurations

# 激活虚拟环境
source venv/bin/activate

echo "=========================================="
echo "Training XGBoost with Top-N Features"
echo "=========================================="

# 示例 1: 使用前 10 个最重要特征
echo -e "\n[Example 1] Training with top 10 features..."
# 需要先手动修改配置文件中的 top_n 为 10，或者创建专门的配置文件
python src/training/xgboost/train_xgboost.py \
    --config configs/xgboost_timeseries_config.yaml \
    --experiment_name "xgboost_top10_features"

# 示例 2: 使用前 20 个最重要特征
echo -e "\n[Example 2] Training with top 20 features..."
# top_n: 20
python src/training/xgboost/train_xgboost.py \
    --config configs/xgboost_timeseries_config.yaml \
    --experiment_name "xgboost_top20_features"

# 示例 3: 使用前 40 个最重要特征（当前配置文件默认值）
echo -e "\n[Example 3] Training with top 40 features..."
python src/training/xgboost/train_xgboost.py \
    --config configs/xgboost_timeseries_config.yaml \
    --experiment_name "xgboost_top40_features"

# 示例 4: 使用所有特征（将 top_n 设为 null）
echo -e "\n[Example 4] Training with all features..."
# top_n: null
python src/training/xgboost/train_xgboost.py \
    --config configs/xgboost_timeseries_config.yaml \
    --experiment_name "xgboost_all_features"

echo -e "\n=========================================="
echo "Training completed!"
echo "Check results in experiments/xgboost_timeseries/experiments/"
echo "=========================================="



