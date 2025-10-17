# Examples - Top-N Feature Selection

本目录包含使用 Top-N 特征选择功能的示例脚本。

## 文件说明

### `example_top_n_usage.py`
Python 示例脚本，演示四种不同的使用方式：
1. 直接使用 XGBoostDataLoader
2. 通过配置文件使用
3. 对比不同 top_n 值的效果
4. 不使用 top_n（向后兼容）

**运行方法：**
```bash
# 激活虚拟环境
source venv/bin/activate

# 运行示例
python examples/example_top_n_usage.py
```

### `example_top_n_training.sh`
Shell 脚本示例，演示如何使用不同的 top_n 配置训练多个模型。

**运行方法：**
```bash
chmod +x examples/example_top_n_training.sh
./examples/example_top_n_training.sh
```

**注意：** 需要在运行不同实验前修改配置文件中的 `top_n` 值。

## 快速开始

### 1. 测试功能是否正常

```bash
python test_top_n_feature_loading.py
```

### 2. 运行使用示例

```bash
python examples/example_top_n_usage.py
```

### 3. 训练模型

```bash
# 使用配置文件中的 top_n 设置
python src/training/xgboost/train_xgboost.py \
    --config configs/xgboost_timeseries_config.yaml
```

## 配置示例

在 `configs/xgboost_timeseries_config.yaml` 中：

```yaml
data:
  data_path: "experiments/statistic_40_window_features_ring/window_features.parquet"
  
  # 使用前 40 个最重要特征
  top_n: 40
  feature_importance_path: "experiments/statistic_40_window_features_ring/feature_importance_summary.csv"
  
  target_column: "anomaly_label"
  exclude_columns: [TimeStamp]
```

## 实验建议

### 对比实验

测试不同数量特征对模型性能的影响：

```bash
# 实验 1: top 10 features
# 修改配置: top_n: 10
python src/training/xgboost/train_xgboost.py --experiment_name "top10"

# 实验 2: top 20 features
# 修改配置: top_n: 20
python src/training/xgboost/train_xgboost.py --experiment_name "top20"

# 实验 3: top 40 features
# 修改配置: top_n: 40
python src/training/xgboost/train_xgboost.py --experiment_name "top40"

# 实验 4: all features
# 修改配置: top_n: null
python src/training/xgboost/train_xgboost.py --experiment_name "all_features"
```

然后对比 `experiments/xgboost_timeseries/experiments/` 下各个实验的结果。

## 更多信息

详细文档请参考：
- [Top-N 特征选择使用指南](../docs/top_n_feature_selection_usage.md)
- [更新日志](../CHANGELOG_top_n_feature.md)



