# Models 模块说明

本目录包含用于能源异常检测的深度学习模型实现。

## FEDformer 实现

目前有两个 FEDformer 实现版本：

### 1. 官方完整版本 ⭐ **推荐使用**

**文件**: `fedformer_anomaly_detection.py`
**配置**: `src/config/fedformer_config.py`
**训练脚本**: `examples/train_official_fedformer_anomaly.py`

**特点**:
- ✅ 基于 ICML 2022 官方论文完整实现
- ✅ 包含完整的 Fourier Enhanced 层
- ✅ 支持分钟级数据的正确周期性配置 (1440分钟=24小时)
- ✅ 内存优化和混合精度训练
- ✅ 多尺度损失函数 (seasonal + trend + combined)
- ✅ 完整的时间特征工程

**适用场景**:
- 生产环境部署
- 需要最佳性能的场景
- 分钟级能源数据异常检测

### 2. 简化版本 (Legacy)

**文件**: `fedformer.py`
**训练脚本**: `src/training/fedformer/`

**特点**:
- ⚠️ 简化实现，缺少关键组件
- ⚠️ 原配置为 seq_len=168，不适合分钟级数据
- ⚠️ 功能不完整

**适用场景**:
- 教学和学习目的
- 快速原型验证
- 资源受限环境测试

## 推荐使用方式

### 新项目
使用官方完整版本：
```python
from src.models.fedformer_anomaly_detection import FEDformerAnomalyDetector
from src.config import FEDformerConfig

# 标准配置 (24小时窗口)
config = FEDformerConfig()

# 内存优化配置 (12小时窗口)  
config = FEDformerConfig().get_memory_config()

# 周模式配置 (36小时窗口)
config = FEDformerConfig().get_weekly_config()

model = FEDformerAnomalyDetector(config)
```

### 训练
```bash
python examples/train_official_fedformer_anomaly.py
```

## 配置说明

配置文件位于 `src/config/fedformer_config.py`，支持：

- **YAML 文件加载**: `FEDformerConfig.from_yaml()`
- **预设配置**: 标准/内存优化/周模式
- **自动时间特征**: 分钟级数据的周期性特征
- **动态参数**: 根据数据自动调整

## 注意事项

1. **数据分辨率**: 确认使用正确的序列长度
   - 分钟级数据: seq_len=1440 (24小时)
   - 小时级数据: seq_len=24 (24小时)

2. **内存要求**: 
   - 标准配置需要 ~8GB GPU 内存
   - 内存优化配置需要 ~4GB GPU 内存

3. **迁移指南**: 如果从简化版本迁移，请：
   - 更新导入路径
   - 使用新的配置系统
   - 重新配置序列长度 