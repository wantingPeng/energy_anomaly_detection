# LSTM Late Fusion 实现方案

## 概述

本文档描述了如何将计算完的窗口统计特征作为单独分支走FC（全连接层）进行Late Fusion，作为LSTM滑动窗口的补充信息，送入LSTM模型进行训练。

## 实现步骤

### 1. 修改统计特征计算和保存方式

在 `src/preprocessing/energy/lsmt_statisticalFeatures/sliding_window_features.py` 中，我们修改了以下内容：

- 在 `process_batch` 函数中，除了保存Parquet格式的统计特征外，还保存了NPZ格式的数据，包含以下内容：
  - `stat_features`: 统计特征数组
  - `window_starts`: 窗口开始时间戳
  - `window_ends`: 窗口结束时间戳
  - `segment_ids`: 分段ID

这样可以确保统计特征与LSTM滑动窗口数据能够正确对齐。

### 2. 创建Late Fusion模型

在 `src/training/lsmt/lstm_late_fusion_model.py` 中，我们创建了一个新的模型类 `LSTMLateFusionModel`，它包含以下组件：

- LSTM分支：处理时序数据
- 统计特征分支：处理统计特征
- 融合层：将两个分支的特征连接起来，进行最终的分类

模型架构如下：

```
输入：
- 时序数据 (batch_size, sequence_length, input_size)
- 统计特征 (batch_size, stat_features_size)

LSTM分支：
- LSTM层
- 全连接层

统计特征分支：
- 全连接层

融合：
- 连接LSTM分支和统计特征分支的输出
- 全连接层
- 输出层
```

### 3. 创建数据加载器

在 `src/training/lsmt/lstm_late_fusion_dataset.py` 中，我们创建了一个数据集类 `LSTMLateFusionDataset`，它可以：

- 加载LSTM滑动窗口数据
- 加载统计特征数据
- 根据时间戳和分段ID将它们对齐
- 返回对齐后的数据对

### 4. 创建训练脚本

在 `src/training/lsmt/train_late_fusion.py` 中，我们创建了一个训练脚本，它可以：

- 加载配置文件
- 创建数据加载器
- 创建模型
- 训练模型
- 评估模型
- 保存模型和训练结果

### 5. 创建配置文件

在 `configs/lstm_late_fusion.yaml` 中，我们创建了一个配置文件，它包含以下内容：

- 模型参数
- 训练参数
- 数据参数
- 路径配置

## 使用方法

### 1. 生成统计特征

```bash
python -m src.preprocessing.energy.lsmt_statisticalFeatures.sliding_window_features
```

这将生成统计特征数据，并保存在 `Data/processed/lsmt_statisticalFeatures/statistic_features_filtered/` 目录下。

### 2. 训练Late Fusion模型

```bash
python -m src.training.lsmt.train_late_fusion
```

这将使用配置文件 `configs/lstm_late_fusion.yaml` 训练模型，并保存结果在 `experiments/lstm_late_fusion/` 目录下。

## 数据流程

1. 原始数据 → 滑动窗口数据 → LSTM模型输入
2. 原始数据 → 统计特征计算 → 统计特征数据 → FC分支输入
3. LSTM输出 + FC分支输出 → Late Fusion → 最终分类结果

## 优势

1. 利用了时序数据和统计特征的互补性
2. 通过Late Fusion可以更好地融合不同类型的特征
3. 模块化设计，便于后续扩展和修改 