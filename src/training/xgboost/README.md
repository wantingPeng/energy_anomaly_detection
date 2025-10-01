# XGBoost 时序能源异常检测

本目录包含基于 XGBoost 的时间序列能源异常检测模型实现。

## 文件说明

- `xgboost_model.py`: XGBoost 模型封装类
- `dataloader.py`: 数据加载和特征工程
- `train_xgboost.py`: 完整的训练脚本
- `README.md`: 使用说明（本文件）

## 特性

### 1. 简化的数据加载
- **预计算特征**: 直接加载已计算好的窗口统计特征
- **无需标准化**: XGBoost 是树模型，不需要特征标准化
- **时序分割**: 按时间顺序分割训练/验证/测试集
- **自动处理**: 自动识别特征列和标签列

### 2. 类别不平衡处理
- **scale_pos_weight**: 自动计算正负样本权重
- **阈值优化**: 基于验证集优化决策阈值
- **点调整评估**: 针对时序异常的评估策略

### 3. 模型评估
- 标准指标：Accuracy, Precision, Recall, F1
- AUC 指标：AUROC, AUPRC
- 混淆矩阵
- 点调整后的指标（更符合时序异常检测实际场景）

### 4. 可视化
- 训练历史曲线
- 特征重要性分析
- 混淆矩阵热图

## 使用方法

### 1. 激活虚拟环境

```bash
cd /home/wanting/energy_anomaly_detection
source .venv/bin/activate
```

### 2. 训练模型

#### 使用默认配置

```bash
python src/training/xgboost/train_xgboost.py
```

#### 使用自定义配置

```bash
python src/training/xgboost/train_xgboost.py \
    --config configs/xgboost_timeseries_config.yaml \
    --experiment_name xgboost_exp1
```

#### 使用预设模型变体

**快速模式** (适合快速测试):
```bash
python src/training/xgboost/train_xgboost.py \
    --variant fast \
    --experiment_name xgboost_fast
```

**标准模式** (推荐):
```bash
python src/training/xgboost/train_xgboost.py \
    --variant standard \
    --experiment_name xgboost_standard
```

**深度模式** (最佳性能):
```bash
python src/training/xgboost/train_xgboost.py \
    --variant deep \
    --experiment_name xgboost_deep
```

## 配置文件说明

配置文件位于 `configs/xgboost_timeseries_config.yaml`

### 数据配置 (data)

```yaml
data:
  # 预计算的窗口统计特征（30分钟窗口）
  data_path: "experiments/statistic_30_window_features_contact/filtered_window_features.parquet"
  train_ratio: 0.7        # 训练集比例
  val_ratio: 0.15         # 验证集比例
  test_ratio: 0.15        # 测试集比例
  target_column: "anomaly_label"
  exclude_columns: []     # 会自动排除 target_column
  
  # 注意：特征已经预先计算好了，不需要额外的特征工程
  # parquet 文件包含从30分钟窗口计算的统计特征
```

### 模型配置 (model)

```yaml
model:
  objective: "binary:logistic"
  eval_metric: ["logloss", "auc", "aucpr"]
  tree_method: "hist"
  max_depth: 6              # 树的最大深度
  learning_rate: 0.1        # 学习率
  n_estimators: 300         # 树的数量
  subsample: 0.8           # 样本采样比例
  colsample_bytree: 0.8    # 特征采样比例
  scale_pos_weight: 10     # 正样本权重（自动计算设为 "auto"）
```

### 训练配置 (training)

```yaml
training:
  early_stopping_rounds: 30
  verbose_eval: 10
  use_best_iteration: true
  optimize_threshold: true
  threshold_metric: "f1"    # 用于优化阈值的指标
```

### 评估配置 (evaluation)

```yaml
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auprc", "auroc"]
  use_point_adjustment: true    # 使用点调整评估
  save_predictions: true
  save_probabilities: true
```

## 模型变体

配置文件中定义了三种预设模型变体：

### 1. Fast (快速)
- 快速训练和测试
- 适合快速迭代和调试
- 参数：max_depth=4, n_estimators=100
- 窗口：30 分钟

### 2. Standard (标准，推荐)
- 平衡性能和速度
- 适合大多数场景
- 参数：max_depth=6, n_estimators=300
- 窗口：60 分钟

### 3. Deep (深度)
- 最佳性能
- 需要更多计算时间
- 参数：max_depth=9, n_estimators=500
- 窗口：120 分钟

## 输出结果

训练完成后，会在 `experiments/xgboost_timeseries/experiments/<experiment_name>/` 生成：

### 模型文件
- `model/xgboost_model.json`: XGBoost 模型
- `model/model_metadata.pkl`: 模型元数据
- `model/config.json`: 训练配置
- `model/scaler.pkl`: 特征标准化器

### 结果文件
- `results/metrics.json`: 评估指标（JSON 格式）
- `results/summary.txt`: 结果摘要
- `results/feature_importance.csv`: 特征重要性
- `results/test_predictions.csv`: 测试集预测结果

### 可视化
- `plots/training_history.png`: 训练历史曲线
- `plots/confusion_matrix_test.png`: 混淆矩阵
- `plots/feature_importance.png`: 特征重要性图

## 特征工程详解

### 原始特征
- 从数据文件中直接读取的能源数据特征

### 滞后特征 (Lag Features)
- 例如：`feature_lag_1`, `feature_lag_5`, `feature_lag_10`, `feature_lag_30`
- 捕获时间依赖性

### 滚动统计特征 (Rolling Features)
对每个滚动窗口计算：
- **均值**: `feature_roll_mean_5`, `feature_roll_mean_10`, ...
- **标准差**: `feature_roll_std_5`, `feature_roll_std_10`, ...
- **最小值**: `feature_roll_min_5`, ...
- **最大值**: `feature_roll_max_5`, ...

### 统计特征 (Statistical Features)
- **变化率**: `feature_roc` (percentage change)
- **差分**: `feature_diff` (difference)

## 点调整评估 (Point Adjustment)

针对时序异常检测的特殊评估策略：
- 如果检测到异常段中的任意一点，则认为整个异常段被检测到
- 更符合实际应用场景
- 提供调整前后的指标对比

## 性能优化建议

### 提高 Precision（减少误报）
1. 增大 `scale_pos_weight` 值
2. 调整 `threshold` 向上
3. 增大 `gamma` 参数（更保守的分裂）
4. 减小 `max_depth`（防止过拟合）

### 提高 Recall（减少漏报）
1. 减小 `scale_pos_weight` 值
2. 调整 `threshold` 向下
3. 增大 `max_depth`（捕获更复杂模式）
4. 增加 `n_estimators`

### 加速训练
1. 使用 `fast` 变体
2. 减小 `n_estimators`
3. 减小特征工程的滚动窗口数量
4. 使用 `tree_method='hist'`（已默认）

## 依赖包

主要依赖：
- `xgboost`: XGBoost 模型
- `scikit-learn`: 数据处理和评估
- `pandas`: 数据处理
- `numpy`: 数值计算
- `matplotlib`, `seaborn`: 可视化
- `pyyaml`: 配置文件解析

安装方法：
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn pyyaml
```

## 常见问题

### 1. 内存不足
- 使用 `fast` 变体
- 减小 `window_size`
- 减少滚动统计窗口数量
- 减少特征工程选项

### 2. 训练太慢
- 减小 `n_estimators`
- 使用 `fast` 变体
- 减小 `max_depth`
- 调整 `n_jobs` 参数

### 3. 过拟合
- 增大 `gamma`
- 增大 `lambda` (L2 正则化)
- 增大 `alpha` (L1 正则化)
- 减小 `max_depth`
- 增大 `min_child_weight`
- 减小 `subsample` 和 `colsample_bytree`

### 4. 欠拟合
- 增大 `max_depth`
- 增大 `n_estimators`
- 减小正则化参数
- 增加更多特征

## 示例完整训练流程

```bash
# 1. 激活环境
cd /home/wanting/energy_anomaly_detection
source .venv/bin/activate

# 2. 使用标准配置训练
python src/training/xgboost/train_xgboost.py \
    --config configs/xgboost_timeseries_config.yaml \
    --experiment_name contact_xgboost_standard \
    --variant standard

# 3. 查看结果
ls experiments/xgboost_timeseries/experiments/contact_xgboost_standard/

# 4. 查看日志
tail -f experiments/logs/train_xgboost_*.log
```

## 与其他模型对比

| 特性 | XGBoost | LSTM-CNN | Transformer |
|------|---------|----------|-------------|
| 训练速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 内存占用 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 可解释性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 长序列建模 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 特征工程 | 需要 | 自动 | 自动 |

## 参考资料

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Time Series Feature Engineering](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

