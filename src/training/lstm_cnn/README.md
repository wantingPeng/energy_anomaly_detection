# LSTM-CNN 模型用于能源异常检测

本目录包含基于 LSTM 和 CNN 混合架构的时间序列能源异常检测模型。

## 文件说明

- `lstm_cnn_model.py`: LSTM-CNN 混合模型架构实现
- `train_lstm_cnn.py`: 完整的训练脚本
- `README.md`: 使用说明（本文件）

## 模型架构

### LSTMCNN（标准模型）

混合架构结合了两个分支：

1. **CNN 分支**: 多尺度 1D 卷积层，用于局部特征提取
   - 多层 Conv1D + BatchNorm + ReLU + Dropout
   - 捕获局部时间模式和特征

2. **LSTM 分支**: 双向 LSTM，用于时序建模
   - 多层双向 LSTM
   - 捕获长期时间依赖关系

3. **特征融合**: 连接 CNN 和 LSTM 特征
4. **分类头**: 全连接层，逐时间步二分类

### SimpleLSTMCNN（轻量级模型）

简化版本，参数更少，训练更快：
- 单层 CNN
- 单层双向 LSTM
- 适合小数据集或快速实验

## 使用方法

### 1. 激活虚拟环境

```bash
cd /home/wanting/energy_anomaly_detection
source venv/bin/activate
```

### 2. 训练模型

使用默认配置：

```bash
python src/training/lstm_cnn/train_lstm_cnn.py
```

使用自定义配置文件：

```bash
python src/training/lstm_cnn/train_lstm_cnn.py --config configs/lstm_cnn_config.yaml
```

指定实验名称：

```bash
python src/training/lstm_cnn/train_lstm_cnn.py \
    --config configs/lstm_cnn_config.yaml \
    --experiment_name lstm_cnn_exp1
```

从检查点继续训练：

```bash
python src/training/lstm_cnn/train_lstm_cnn.py \
    --config configs/lstm_cnn_config.yaml \
    --load_model experiments/lstm_cnn/model_save/lstm_cnn_20250101_120000/best_f1/best_model.pt
```

### 3. 监控训练

使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir experiments/lstm_cnn/tensorboard
```

然后在浏览器中打开 `http://localhost:6006`

## 配置文件说明

配置文件位于 `configs/lstm_cnn_config.yaml`，主要配置项：

### 数据配置 (data)

```yaml
data:
  train_ratio: 0.7              # 训练集比例
  val_ratio: 0.15               # 验证集比例
  test_ratio: 0.15              # 测试集比例
  window_size: 60               # 滑动窗口大小（时间步数）
  step_size: 10                 # 滑动窗口步长
  target_anomaly_ratio: 0.25    # 训练集目标异常比例（可选）
```

### 模型配置 (model)

```yaml
model:
  variant: "standard"           # 模型变体：'standard' 或 'simple'
  cnn_channels: [64, 128, 128]  # CNN 层通道数
  cnn_kernel_sizes: [3, 3, 3]   # CNN 卷积核大小
  lstm_hidden_dim: 128          # LSTM 隐藏层维度
  lstm_num_layers: 2            # LSTM 层数
  use_bidirectional: true       # 是否使用双向 LSTM
  dropout: 0.3                  # Dropout 概率
```

### 训练配置 (training)

```yaml
training:
  batch_size: 64                # 批次大小
  num_epochs: 100               # 最大训练轮数
  learning_rate: 0.001          # 初始学习率
  optimizer: "adam"             # 优化器：'adam' 或 'adamw'
  lr_scheduler: "reduce_on_plateau"  # 学习率调度器
  use_focal_loss: true          # 使用 Focal Loss（适合不平衡数据）
  focal_loss_alpha: 0.75        # Focal Loss alpha 参数
  focal_loss_gamma: 2.0         # Focal Loss gamma 参数
  early_stopping_patience: 20   # 早停耐心值
```

## 模型变体

配置文件中定义了三种预设模型变体：

### 1. Lightweight（轻量级）
- 快速训练和推理
- 适合小数据集或快速实验
- 参数较少

### 2. Standard（标准）
- 平衡性能和速度
- 推荐用于大多数场景
- 默认配置

### 3. Large（大型）
- 最佳性能
- 需要更多计算资源
- 适合大数据集

## 输出结果

训练完成后，会生成以下输出：

### 模型检查点

保存在 `experiments/lstm_cnn/model_save/<experiment_name>/` 下：

- `best_loss/`: 最佳验证损失模型
- `best_f1/`: 最佳 F1 分数模型
- `best_auprc/`: 最佳 AUPRC 模型
- `best_adj_f1/`: 最佳调整后 F1 分数模型

每个目录包含：
- `best_model.pt`: 模型检查点
- `config/config.yaml`: 训练配置

### TensorBoard 日志

保存在 `experiments/lstm_cnn/tensorboard/<experiment_name>/`

包含的指标：
- 训练/验证损失
- F1 分数、精确率、召回率
- AUPRC（平均精确率-召回率曲线下面积）
- 调整后的评估指标（Point Adjustment）
- 学习率变化

### 日志文件

保存在 `experiments/logs/train_lstm_cnn_<timestamp>.log`

## 评估指标

模型使用多种评估指标：

1. **AUPRC**: 平均精确率-召回率曲线下面积
2. **Optimal F1**: 基于最优阈值的 F1 分数
3. **Adjusted F1**: 使用 Point Adjustment 策略的 F1 分数
   - Point Adjustment: 如果异常段中任何点被检测到，整个段被视为已检测
   - 更符合实际应用场景

## 数据加载

模型使用 `src/preprocessing/row_energyData_subsample_Transform/dataloader.py` 中的数据加载器。

数据格式要求：
- Parquet 格式
- 必须包含 `TimeStamp` 和 `anomaly_label` 列
- 其他列作为特征

默认数据路径：
```
Data/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet
```

## 依赖项

主要依赖：
- PyTorch
- pandas
- numpy
- scikit-learn
- tensorboard
- tqdm
- pyyaml

如果缺少依赖，请运行：
```bash
pip install torch pandas numpy scikit-learn tensorboard tqdm pyyaml
```

并更新 `requirements.txt`。

## 测试模型

测试模型架构（无需数据）：

```bash
cd /home/wanting/energy_anomaly_detection
python -c "from src.training.lstm_cnn.lstm_cnn_model import LSTMCNN, SimpleLSTMCNN; import torch; x = torch.randn(32, 60, 10); model = LSTMCNN(10); print(model(x).shape); print(model.get_num_params())"
```

## 常见问题

### 1. 内存不足
- 减小 `batch_size`
- 使用 `simple` 模型变体
- 减小 `window_size`

### 2. 训练速度慢
- 增大 `batch_size`（如果内存允许）
- 减小 `num_workers`
- 使用 `simple` 模型变体

### 3. 过拟合
- 增大 `dropout`
- 增大 `weight_decay`
- 减小模型容量（使用更小的 `cnn_channels` 和 `lstm_hidden_dim`）
- 启用数据增强

### 4. 欠拟合
- 增大模型容量
- 增加训练轮数
- 减小 `dropout`
- 调整学习率

## 下一步

训练完成后，你可以：

1. 使用最佳模型进行推理
2. 分析模型在测试集上的性能
3. 可视化异常检测结果
4. 调整超参数进一步优化性能
5. 与其他模型（如 Transformer、纯 LSTM）进行对比

## 联系与支持

如有问题，请查看：
- 日志文件：`experiments/logs/`
- TensorBoard 可视化
- 模型架构代码：`lstm_cnn_model.py`


