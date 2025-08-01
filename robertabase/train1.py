import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import random
from transformers import AutoModel, AutoConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.nn.utils.rnn import pad_sequence
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        positive_probs = predictions[:, 1]
    else:
        positive_probs = 1 / (1 + np.exp(-predictions))

    # 计算PR曲线和PR-AUC
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, positive_probs)
    prauc = auc(recall_curve, precision_curve)
    
    # 异常检测任务推荐的阈值策略
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # 调试信息：打印预测概率分布
    print(f"[DEBUG] 预测概率范围: [{positive_probs.min():.6f}, {positive_probs.max():.6f}]")
    print(f"[DEBUG] 预测概率均值: {positive_probs.mean():.6f}, 中位数: {np.median(positive_probs):.6f}")
    print(f"[DEBUG] 正样本比例: {np.mean(labels):.4f}")
    
    # 策略1: 固定阈值0.5 (标准做法)
    threshold_05 = 0.5
    pred_05 = (positive_probs >= threshold_05).astype(int)
    
    # 策略2: 基于Youden Index (Sensitivity + Specificity - 1 最大化)
    # 对于不平衡数据更稳定
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_thresholds = roc_curve(labels, positive_probs)
    youden_scores = tpr - fpr
    youden_idx = np.argmax(youden_scores)
    
    # 添加边界条件处理
    if youden_idx < len(roc_thresholds):
        threshold_youden = roc_thresholds[youden_idx]
        # 防止极端阈值
        if np.isinf(threshold_youden) or threshold_youden > 0.99:
            threshold_youden = 0.5
        elif threshold_youden < 0.01:
            threshold_youden = 0.01
    else:
        threshold_youden = 0.5
    
    pred_youden = (positive_probs >= threshold_youden).astype(int)
    
    # 调试信息：Youden Index策略结果
    print(f"[DEBUG] Youden阈值: {threshold_youden:.6f}")
    print(f"[DEBUG] Youden策略预测正样本比例: {np.mean(pred_youden):.4f}")
    
    # 策略3: 基于分位数的异常检测阈值 (保守策略)
    # 使用95分位数作为阈值，假设5%的数据是异常
    threshold_percentile = np.percentile(positive_probs, 95)
    pred_percentile = (positive_probs >= threshold_percentile).astype(int)
    
    # 策略4: 在合理的recall范围内找最佳precision (如recall >= 0.7) 
    valid_indices = recall_curve >= 0.7
    if np.any(valid_indices):
        best_idx = np.argmax(precision_curve[valid_indices])
        actual_idx = np.where(valid_indices)[0][best_idx]
        threshold_balanced = thresholds[actual_idx] if actual_idx < len(thresholds) else 0.5
    else:
        threshold_balanced = 0.5
    pred_balanced = (positive_probs >= threshold_balanced).astype(int)
    
    # 调试信息：分位数策略结果
    print(f"[DEBUG] 95分位数阈值: {threshold_percentile:.6f}")
    print(f"[DEBUG] 分位数策略预测正样本比例: {np.mean(pred_percentile):.4f}")
    
    # 计算各策略的指标
    metrics_05 = {
        "accuracy_05": accuracy_score(labels, pred_05),
        "precision_05": precision_score(labels, pred_05, zero_division=0),
        "recall_05": recall_score(labels, pred_05, zero_division=0),
        "f1_05": f1_score(labels, pred_05, zero_division=0),
    }
    
    metrics_youden = {
        "accuracy_youden": accuracy_score(labels, pred_youden),
        "precision_youden": precision_score(labels, pred_youden, zero_division=0),
        "recall_youden": recall_score(labels, pred_youden, zero_division=0),
        "f1_youden": f1_score(labels, pred_youden, zero_division=0),
    }
    
    metrics_percentile = {
        "accuracy_percentile": accuracy_score(labels, pred_percentile),
        "precision_percentile": precision_score(labels, pred_percentile, zero_division=0),
        "recall_percentile": recall_score(labels, pred_percentile, zero_division=0),
        "f1_percentile": f1_score(labels, pred_percentile, zero_division=0),
    }
    
    metrics_balanced = {
        "accuracy_balanced": accuracy_score(labels, pred_balanced),
        "precision_balanced": precision_score(labels, pred_balanced, zero_division=0),
        "recall_balanced": recall_score(labels, pred_balanced, zero_division=0),
        "f1_balanced": f1_score(labels, pred_balanced, zero_division=0),
    }
    
    # 主要指标使用分位数策略 (对异常检测任务更稳定)
    main_metrics = {
        "accuracy": metrics_percentile["accuracy_percentile"],
        "precision": metrics_percentile["precision_percentile"], 
        "recall": metrics_percentile["recall_percentile"],
        "f1": metrics_percentile["f1_percentile"],
        "prauc": prauc,
        "threshold": threshold_percentile,
        "pos_ratio": np.mean(labels),  # 添加数据分布信息
        "pred_ratio": np.mean(pred_percentile),  # 预测正样本比例
    }
    
    # 合并所有指标
    return {**main_metrics, **metrics_05, **metrics_youden, **metrics_balanced}
    

class NumericDataset(Dataset):
    def __init__(self, data, mean=None, std=None, normalize=True):
        super().__init__()

        # 1. 把所有数据提前堆成 tensor，方便后面一次性做归一化
        self.inputs = torch.tensor([d["input"] for d in data], dtype=torch.float32)
        self.labels = torch.tensor([d["label"] for d in data], dtype=torch.long)

        self.normalize = normalize
        if normalize:
            if mean is None or std is None:       
                self.mean = self.inputs.mean(dim=0)
                self.std  = self.inputs.std (dim=0)
            else:                              
                self.mean = mean
                self.std  = std
            self.std[self.std == 0] = 1.0
            self.inputs = (self.inputs - self.mean) / self.std
        else:
            self.mean, self.std = None, None

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],   # 已经是 float32 tensor
            "labels": self.labels[idx],   # long tensor
        }

    # 方便外部拿到训练集的统计量
    def get_stats(self):
        return self.mean, self.std

# def collate_numeric(features):
#     x_seqs  = [f["inputs"] for f in features]
#     labels  = torch.stack([f["labels"] for f in features])
#     seq_lens   = [len(x) for x in x_seqs]
#     padded_x   = pad_sequence(x_seqs, batch_first=True, padding_value=0.0)
#     attn_mask  = torch.zeros_like(padded_x, dtype=torch.long)
#     for i,l in enumerate(seq_lens):
#         attn_mask[i, :l] = 1

#     return {
#         "inputs": padded_x,  
#         "mask": attn_mask, 
#         "labels": labels     
#     }
def collate_numeric(batch):
    # 每条样本都是 {"inputs": [27], "labels": int}
    inputs = torch.stack([b["inputs"] for b in batch])   # [B, 27]
    labels = torch.stack([b["labels"] for b in batch])   # [B]
    attn_mask = torch.ones_like(inputs, dtype=torch.long)
    return {"inputs": inputs, "mask": attn_mask, "labels": labels}

class NumericProjector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)   # 输入 1-dim → 输出 hidden

    def forward(self, x):  # x: [B, L] float
        return self.proj(x.unsqueeze(-1))       # [B, L, hidden]

# class NumericClassifier(nn.Module):
#     def __init__(self, backbone_name, num_labels):
#         super().__init__()
#         cfg = AutoConfig.from_pretrained(backbone_name)
#         self.backbone = AutoModel.from_pretrained(backbone_name)
#         self.num_hidden = cfg.hidden_size
#         self.numeric_proj = NumericProjector(cfg.hidden_size)

#         self.classifier = nn.Linear(cfg.hidden_size, num_labels)

#     def forward(self, inputs, mask, labels=None):
#         num_emb = self.numeric_proj(inputs)  
#         attention_mask = mask.long()     
#         outputs = self.backbone(
#             inputs_embeds=num_emb,
#             attention_mask=attention_mask,
#         )
#         cls_rep = outputs.last_hidden_state[:, 0]   # 用第 0 位置作 [CLS]
#         logits = self.classifier(cls_rep)

#         if labels is not None:
#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(logits, labels)
#             return {"loss": loss, "logits": logits}
#         else:
#             return {"logits": logits}

class NumericProjector(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(1, hidden)

    def forward(self, x):          # x: [B, 27]
        return self.proj(x.unsqueeze(-1))   # -> [B, 27, hidden]


class NumericClassifier(nn.Module):
    def __init__(self, backbone_name: str = "microsoft/deberta-v3-small", num_labels: int = 2):
        super().__init__()
        cfg = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=cfg)
        self.proj = NumericProjector(cfg.hidden_size)
        self.cls = nn.Linear(cfg.hidden_size, num_labels)

    def forward(self, inputs, mask, labels=None):
        emb = self.proj(inputs)          # [B, 27, hidden]
        outputs = self.backbone(inputs_embeds=emb, attention_mask=mask)
        cls_repr = outputs.last_hidden_state[:, 0]   # 取第 0 位置
        logits = self.cls(cls_repr)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class MLPClassifier(nn.Module):
    """
    一个 27 → 128 → 64 → 2 的全连接网络，可以自行调 hidden_dim / dropout
    """
    def __init__(self,
                 input_dim: int = 27,
                 hidden_dims=(128, 64),
                 dropout: float = 0.3,
                 num_labels: int = 2):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_labels)

    def forward(self, inputs, labels=None, mask=None, **kwargs):
        # mask 在 MLP 中用不上，直接忽略
        x = self.feature_extractor(inputs)
        logits = self.classifier(x)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


def process_rawdata(df, model, neg_rate):
    if model == 'train':
        count_map = {
            0:0,
            1:0
        }
        data_list = []
        data_0_list = []
        dict_list = df.to_dict(orient="records")
        for line in tqdm(dict_list):
            count_map[line['anomaly_label']] += 1
            cur_input = []
            for l in line:
                if l not in ('TimeStamp', 'segment_id', 'anomaly_label'):
                    cur_input.append(line[l])
            if line['anomaly_label'] == 1:
                data_list.append({
                    'input':cur_input,
                    'label':line['anomaly_label']
                })
            else:
                data_0_list.append({
                    'input':cur_input,
                    'label':line['anomaly_label']
                })
        data_list += random.sample(data_0_list, count_map[1]*neg_rate)
        random.shuffle(data_list)
        return data_list
    else:
        data_list = []
        dict_list = df.to_dict(orient="records")
        for line in tqdm(dict_list):
            cur_input = []
            for l in line:
                if l not in ('TimeStamp', 'segment_id', 'anomaly_label'):
                    cur_input.append(line[l])
            data_list.append({
                'input':cur_input,
                'label':line['anomaly_label']
            })
        return data_list


def main_train(neg_rate, learning_rate, batch_size, raw_train_data, raw_val_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[INFO] Using device: {device}')
    print('[INFO] processing data...')
    samples = process_rawdata(raw_train_data, 'train', neg_rate)
    evalsamples = process_rawdata(raw_val_data, 'val', neg_rate)
    
    # 打印数据分布信息
    train_pos = sum(1 for s in samples if s['label'] == 1)
    val_pos = sum(1 for s in evalsamples if s['label'] == 1)
    print(f'[INFO] 训练集: {len(samples)} 样本, 正样本: {train_pos} ({train_pos/len(samples)*100:.2f}%)')
    print(f'[INFO] 验证集: {len(evalsamples)} 样本, 正样本: {val_pos} ({val_pos/len(evalsamples)*100:.2f}%)')
    print('[INFO] data processed...')
    
    dataset = NumericDataset(samples)
    mean, std = dataset.get_stats()
    evaldataset = NumericDataset(evalsamples, mean, std)

    # 加载预训练模型
    model = NumericClassifier(backbone_name="Microsoft/deberta-v3-base", num_labels=2)
    # model = MLPClassifier()
    model.to(device)
    outputpath = 'trained_models_1/'+str(neg_rate)+'_'+str(learning_rate)+'_'+str(batch_size)
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=outputpath,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1024,
        num_train_epochs=50,
        weight_decay=0.01,
        save_total_limit=10,
        disable_tqdm=False,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        load_best_model_at_end=True,
        metric_for_best_model="prauc",  # PR-AUC对异常检测任务最稳定
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=1,
        report_to="tensorboard",
        learning_rate=learning_rate,
        gradient_accumulation_steps=1
    )

    # 配置 Early Stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=10,      # 连续5个epoch没有改善就停止
        early_stopping_threshold=0.001  # 改善的最小阈值 (0.1%)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=evaldataset,
        data_collator=collate_numeric,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]      # 添加 early stopping callback
    )
    trainer.train()
    trainer.save_model(f"{outputpath}/bestmodel")
    return trainer
    
if __name__ == '__main__':
    # 预加载数据，避免重复读取
    print("🔄 Pre-loading training data...")
    raw_train_data = pd.read_parquet('Data/row_energyData_subsample_Transform/labeled/train/contact/part.0.parquet')
    raw_val_data = pd.read_parquet('Data/row_energyData_subsample_Transform/labeled/val/contact/part.0.parquet')
    print("✅ Data pre-loaded successfully!")
    
    # 🎯 优化建议：先用较少参数组合快速筛选，再精细化搜索
    #negrate_list = [1, 2, 3, 4, 5]
    negrate_list = [3]
    # 建议：先用这个较小的范围快速找到合理区间
    learning_rate_list = [0.0001]  # 快速搜索
    #learning_rate_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-6, 5e-6, 7e-6]  # 快速搜索
    # learning_rate_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-6, 5e-6, 7e-6]  # 完整搜索
    
    # 建议：batch_size 通常 128-512 就够了，太大可能效果不好
    # batch_size_list = [128, 256, 512, 1024]
    #batch_size_list = [128, 256, 512]
    batch_size_list = [256]
    total_ = len(negrate_list) * len(learning_rate_list) * len(batch_size_list)
    count_ = 0
    
    # 跟踪最佳模型
    best_prauc = -1
    best_config = None
    best_model_path = None
    results_log = []
    
    # 时间追踪
    import time
    start_time = time.time()
    completed_count = 0
    
    for neg_rate in negrate_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                count_ += 1
                print(f'------------------------------ [{count_}/{total_}] ------------------------------')
                
                config = f"neg_rate={neg_rate}, lr={learning_rate}, batch_size={batch_size}"
                model_path = f'trained_models/{neg_rate}_{learning_rate}_{batch_size}/bestmodel'
                
                try:
                    # 训练模型并获取最终结果
                    trainer = main_train(neg_rate, learning_rate, batch_size, raw_train_data, raw_val_data)
                    
                    # 获取最终评估结果
                    eval_results = trainer.evaluate()
                    current_prauc = eval_results.get('eval_prauc', -1)
                    
                    # 记录结果 - 包含多种阈值策略的指标
                    results_log.append({
                        'config': config,
                        'path': model_path,
                        'prauc': current_prauc,
                        # 主要指标 (分位数策略)
                        'f1': eval_results.get('eval_f1', -1),
                        'accuracy': eval_results.get('eval_accuracy', -1),
                        'precision': eval_results.get('eval_precision', -1),
                        'recall': eval_results.get('eval_recall', -1),
                        'threshold': eval_results.get('eval_threshold', -1),
                        # 固定阈值0.5的指标
                        'f1_05': eval_results.get('eval_f1_05', -1),
                        'accuracy_05': eval_results.get('eval_accuracy_05', -1),
                        'precision_05': eval_results.get('eval_precision_05', -1),
                        'recall_05': eval_results.get('eval_recall_05', -1),
                        # Youden Index策略
                        'f1_youden': eval_results.get('eval_f1_youden', -1),
                        'accuracy_youden': eval_results.get('eval_accuracy_youden', -1),
                        'precision_youden': eval_results.get('eval_precision_youden', -1),
                        'recall_youden': eval_results.get('eval_recall_youden', -1),
                        # 平衡策略
                        'f1_balanced': eval_results.get('eval_f1_balanced', -1),
                        'precision_balanced': eval_results.get('eval_precision_balanced', -1),
                        'recall_balanced': eval_results.get('eval_recall_balanced', -1),
                        # 数据分布信息
                        'pos_ratio': eval_results.get('eval_pos_ratio', -1),
                        'pred_ratio': eval_results.get('eval_pred_ratio', -1),
                        'status': 'completed'
                    })
                    
                    # 更新最佳模型
                    if current_prauc > best_prauc:
                        best_prauc = current_prauc
                        best_config = config
                        best_model_path = model_path
                    
                    completed_count += 1
                    
                    # 时间估算
                    elapsed_time = time.time() - start_time
                    avg_time_per_model = elapsed_time / completed_count
                    remaining_models = total_ - completed_count
                    estimated_remaining_time = avg_time_per_model * remaining_models
                    
                    print(f"✅ Current PR-AUC: {current_prauc:.4f}, Best PR-AUC so far: {best_prauc:.4f}")
                    print(f"⏱️  Avg time per model: {avg_time_per_model/60:.1f}min, ETA: {estimated_remaining_time/3600:.1f}h")
                    
                except Exception as e:
                    print(f"❌ Training failed for {config}: {str(e)}")
                    results_log.append({
                        'config': config,
                        'path': model_path,
                        'f1': -1,
                        'accuracy': -1,
                        'precision': -1,
                        'recall': -1,
                        'prauc': -1,
                        'status': 'failed',
                        'error': str(e)
                    })
                    
                    # 清理GPU内存（如果使用GPU）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    continue
    
    # 输出最终结果
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best PR-AUC Score: {best_prauc:.4f}")
    print(f"Best Configuration: {best_config}")
    print(f"Best Model Path: {best_model_path}")
    print("="*80)
    
    # 保存所有结果到文件
    import json
    with open('training_results.json', 'w') as f:
        json.dump({
            'best_prauc': best_prauc,
            'best_config': best_config,
            'best_model_path': best_model_path,
            'all_results': results_log
        }, f, indent=2)
    
    print("All results saved to 'training_results.json'")
