"""
训练示例：使用官方FEDformer架构进行能源异常检测
基于ICML 2022论文，适配二分类异常检测任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.models.fedformer_anomaly_detection import FEDformerAnomalyDetector, create_fedformer_anomaly_model
from src.config import FEDformerConfig
from src.utils.logger import logger

class MinuteLevelEnergyDataset(torch.utils.data.Dataset):
    """
    分钟级能源数据集，适配官方FEDformer格式
    """
    
    def __init__(self, data_path, seq_len=1440, target_col='rTotalActivePower'):
        self.seq_len = seq_len
        self.target_col = target_col
        
        # 加载数据
        logger.info(f"Loading data from {data_path}")
        self.df = pd.read_parquet(data_path)
        
        # 数据预处理
        self._preprocess_data()
        self._create_time_features()
        self._create_sequences()
        
    def _preprocess_data(self):
        """数据预处理"""
        # 确保时间戳格式正确
        if 'TimeStamp' in self.df.columns:
            self.df['TimeStamp'] = pd.to_datetime(self.df['TimeStamp'])
            self.df = self.df.sort_values('TimeStamp')
        
        # 选择数值特征
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['ID', 'segment_id'] if 'segment_id' in self.df.columns else ['ID'] if 'ID' in self.df.columns else []
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.feature_data = self.df[feature_cols].values
        self.feature_names = feature_cols
        
        logger.info(f"Features shape: {self.feature_data.shape}")
        logger.info(f"Feature columns: {feature_cols}")
        
        # 标准化
        self.mean = np.mean(self.feature_data, axis=0)
        self.std = np.std(self.feature_data, axis=0)
        self.feature_data = (self.feature_data - self.mean) / (self.std + 1e-8)
        
    def _create_time_features(self):
        """创建时间特征（适配官方FEDformer格式）"""
        if 'TimeStamp' not in self.df.columns:
            # 如果没有时间戳，创建虚拟时间特征
            self.time_features = np.zeros((len(self.df), 5))
            return
        
        df_time = self.df.copy()
        
        # 分钟级时间特征
        df_time['minute'] = df_time['TimeStamp'].dt.minute / 59.0  # 标准化到[0,1]
        df_time['hour'] = df_time['TimeStamp'].dt.hour / 23.0
        df_time['day_of_week'] = df_time['TimeStamp'].dt.dayofweek / 6.0
        df_time['day_of_month'] = df_time['TimeStamp'].dt.day / 31.0
        df_time['month'] = df_time['TimeStamp'].dt.month / 12.0
        
        self.time_features = df_time[['month', 'day_of_month', 'day_of_week', 'hour', 'minute']].values
        
        logger.info(f"Time features shape: {self.time_features.shape}")
        
    def _create_sequences(self):
        """创建序列数据"""
        self.sequences = []
        
        logger.info(f"Creating sequences with seq_len={self.seq_len}")
        
        # 简单异常标签生成（基于统计阈值）
        # 在实际应用中，您应该使用真实的异常标签
        target_values = self.feature_data[:, 0]  # 使用第一个特征作为目标
        threshold = np.percentile(target_values, 95)  # 95%分位数作为异常阈值
        anomaly_labels = (target_values > threshold).astype(int)
        
        for i in range(len(self.feature_data) - self.seq_len + 1):
            # 特征序列
            seq_x = self.feature_data[i:i+self.seq_len]
            # 时间特征序列
            seq_mark = self.time_features[i:i+self.seq_len]
            # 序列级别标签（如果序列中有异常点，则整个序列标记为异常）
            seq_label = int(np.any(anomaly_labels[i:i+self.seq_len]))
            # 点级别标签
            point_labels = anomaly_labels[i:i+self.seq_len]
            
            self.sequences.append((seq_x, seq_mark, seq_label, point_labels))
        
        logger.info(f"Created {len(self.sequences)} sequences")
        
        # 统计标签分布
        seq_labels = [seq[2] for seq in self.sequences]
        normal_count = sum([1 for label in seq_labels if label == 0])
        anomaly_count = sum([1 for label in seq_labels if label == 1])
        logger.info(f"Label distribution - Normal: {normal_count}, Anomaly: {anomaly_count}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_x, seq_mark, seq_label, point_labels = self.sequences[idx]
        return (
            torch.FloatTensor(seq_x),           # [seq_len, features]
            torch.FloatTensor(seq_mark),        # [seq_len, time_features]
            torch.LongTensor([seq_label]),      # [1]
            torch.LongTensor(point_labels)      # [seq_len]
        )

def train_model():
    """训练官方FEDformer异常检测模型"""
    logger.info("=== 官方FEDformer异常检测训练 ===")
    
    # 1. 数据配置
    data_path = "experiments/data/combined_energy_data.parquet"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run the periodicity analysis first to generate the combined data")
        return
    
    # 2. 创建数据集
    dataset = MinuteLevelEnergyDataset(
        data_path=data_path,
        seq_len=1440  # 24小时
    )
    
    # 3. 创建模型配置
    config = FEDformerConfig(
        # 数据参数（基于分钟级数据）
        seq_len=1440,           # 24小时输入
        enc_in=dataset.feature_data.shape[1],  # 实际特征数量
        
        # 模型架构（针对异常检测优化）
        d_model=256,            # 降低维度以处理长序列
        n_heads=8,
        e_layers=3,             # 编码器层数
        d_ff=1024,              # 前馈网络维度
        
        # FEDformer特定参数
        modes=64,               # 增加Fourier模式数以捕获周期性
        mode_select='random',   # 随机模式选择
        version='Fourier',      # 使用Fourier版本
        moving_avg=25,          # 移动平均窗口
        
        # 训练参数
        dropout=0.1,
        activation='gelu',
        
        # 时间嵌入（分钟级）
        embed='timeF',
        freq='t',               # 分钟级频率
        
        # 异常检测
        num_classes=2
    )
    
    logger.info(f"Model config: seq_len={config.seq_len}, enc_in={config.enc_in}, d_model={config.d_model}")
    
    # 4. 创建模型
    model = FEDformerAnomalyDetector(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Using device: {device}")
    
    # 5. 数据加载器
    batch_size = 8  # 小批次应对长序列
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 6. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 7. 训练循环（简化演示版本）
    model.train()
    logger.info("Starting training demonstration...")
    
    # 只演示几个批次
    max_batches = 5
    for batch_idx, (seq_x, seq_mark, seq_labels, point_labels) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
            
        # 移到设备
        seq_x = seq_x.to(device)          # [B, L, F]
        seq_mark = seq_mark.to(device)    # [B, L, T]
        seq_labels = seq_labels.to(device).squeeze(-1)  # [B]
        point_labels = point_labels.to(device)  # [B, L]
        
        logger.info(f"Batch {batch_idx}: seq_x shape {seq_x.shape}, seq_mark shape {seq_mark.shape}")
        
        try:
            # 前向传播
            results = model(seq_x, seq_mark)
            
            # 多任务损失
            # 1. 序列级分类损失
            seq_loss = criterion(results['combined_logits'], seq_labels)
            
            # 2. 点级分类损失（可选）
            point_logits = results['pointwise_logits'].view(-1, 2)
            point_labels_flat = point_labels.view(-1)
            point_loss = criterion(point_logits, point_labels_flat)
            
            # 组合损失
            total_loss = seq_loss + 0.5 * point_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 计算准确率
            seq_pred = torch.argmax(results['combined_logits'], dim=1)
            seq_acc = (seq_pred == seq_labels).float().mean()
            
            logger.info(f"Batch {batch_idx}: Total Loss = {total_loss.item():.6f}, "
                      f"Seq Loss = {seq_loss.item():.6f}, Point Loss = {point_loss.item():.6f}, "
                      f"Seq Acc = {seq_acc.item():.4f}")
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            break
    
    logger.info("Training demonstration completed!")
    
    # 8. 简单验证
    model.eval()
    logger.info("Running validation...")
    
    val_predictions = []
    val_targets = []
    val_scores = []
    
    with torch.no_grad():
        for batch_idx, (seq_x, seq_mark, seq_labels, point_labels) in enumerate(val_loader):
            if batch_idx >= 3:  # 只验证几个批次
                break
                
            seq_x = seq_x.to(device)
            seq_mark = seq_mark.to(device)
            seq_labels = seq_labels.to(device).squeeze(-1)
            
            results = model(seq_x, seq_mark)
            
            # 预测和分数
            seq_pred = torch.argmax(results['combined_logits'], dim=1)
            seq_probs = torch.softmax(results['combined_logits'], dim=1)[:, 1]  # 异常概率
            
            val_predictions.extend(seq_pred.cpu().numpy())
            val_targets.extend(seq_labels.cpu().numpy())
            val_scores.extend(seq_probs.cpu().numpy())
    
    # 计算指标
    if val_predictions:
        accuracy = accuracy_score(val_targets, val_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_predictions, average='binary')
        auc = roc_auc_score(val_targets, val_scores)
        
        logger.info(f"Validation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")

def demonstrate_model_structure():
    """演示模型结构和特性"""
    logger.info("=== FEDformer异常检测模型结构演示 ===")
    
    # 创建配置
    config = FEDformerConfig(
        seq_len=1440,           # 24小时分钟级数据
        enc_in=33,              # 能源特征数量
        d_model=256,
        modes=64,
        version='Fourier'
    )
    
    model = FEDformerAnomalyDetector(config)
    
    logger.info("模型架构特点：")
    logger.info("1. 基于官方FEDformer ICML 2022实现")
    logger.info("2. 使用AutoCorrelation机制捕获周期性")
    logger.info("3. Fourier增强的频域处理")
    logger.info("4. 季节性-趋势分解")
    logger.info("5. 多尺度异常检测（序列级+点级）")
    
    logger.info(f"\n核心组件：")
    logger.info(f"  - 输入序列长度: {config.seq_len} (24小时)")
    logger.info(f"  - 特征维度: {config.enc_in}")
    logger.info(f"  - 模型维度: {config.d_model}")
    logger.info(f"  - Fourier模式数: {config.modes}")
    logger.info(f"  - 编码器层数: {config.e_layers}")
    
    # 测试前向传播
    batch_size = 2
    seq_x = torch.randn(batch_size, config.seq_len, config.enc_in)
    seq_mark = torch.randn(batch_size, config.seq_len, 5)
    
    results = model(seq_x, seq_mark)
    
    logger.info(f"\n输出结果：")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n参数统计：")
    logger.info(f"  总参数数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")

if __name__ == "__main__":
    logger.info("=== 官方FEDformer异常检测演示 ===")
    
    # 演示模型结构
    demonstrate_model_structure()
    
    print("\n" + "="*50)
    
    # 询问是否继续训练
    response = input("是否继续训练演示？(y/n): ")
    if response.lower() == 'y':
        train_model()
    else:
        logger.info("训练演示已跳过")
        
    logger.info("官方FEDformer异常检测演示完成！") 