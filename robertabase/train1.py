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
from sklearn.metrics import precision_recall_curve, auc

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        positive_probs = predictions[:, 1]
    else:
        positive_probs = 1 / (1 + np.exp(-predictions))

    # 计算PR曲线
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, positive_probs)
    
    # 计算每个阈值下的F1分数
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    
    # 找到最佳F1阈值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_f1_idx]
    best_precision = precision_curve[best_f1_idx]
    best_recall = recall_curve[best_f1_idx]
    
    # 使用最佳阈值生成预测标签
    predictions_labels = (positive_probs >= best_threshold).astype(int)
    
    # 计算准确率（使用最佳阈值）
    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(predictions=predictions_labels, references=labels)
    
    # 计算PR-AUC
    prauc = auc(recall_curve, precision_curve)
    
    # 合并结果并返回
    return {
        "accuracy": accuracy["accuracy"],
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "prauc": prauc,
        "best_threshold": best_threshold
    }
    

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


def process_rawdata(input_path, model, neg_rate):
    import pandas as pd
    if model == 'train':
        count_map = {
            0:0,
            1:0
        }
        data_list = []
        data_0_list = []
        df = pd.read_parquet(input_path)
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
        df = pd.read_parquet(input_path)
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


def main_train(neg_rate, learning_rate, batch_size):
    device = torch.device("cpu")
    print('[INFO] loading data...')
    samples = process_rawdata('Data/row_energyData_subsample_Transform/labeled/train/contact/part.0.parquet', 'train', neg_rate)
    evalsamples = process_rawdata('Data/row_energyData_subsample_Transform/labeled/val/contact/part.0.parquet', 'val', neg_rate)
    print('[INFO] data loaded...')
    dataset = NumericDataset(samples)
    mean, std = dataset.get_stats()
    evaldataset = NumericDataset(evalsamples, mean, std)

    # 加载预训练模型
    model = NumericClassifier(backbone_name="Microsoft/deberta-v3-base", num_labels=2)
    # model = MLPClassifier()
    model.to(device)
    outputpath = 'trained_models/'+str(neg_rate)+'_'+str(learning_rate)+'_'+str(batch_size)
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
        metric_for_best_model="prauc",
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=1,
        report_to="tensorboard",
        learning_rate=learning_rate,
        gradient_accumulation_steps=1
    )

    # 配置 Early Stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,      # 连续5个epoch没有改善就停止
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
    negrate_list = [1, 2, 3, 4, 5]
    learning_rate_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-6, 5e-6, 7e-6]
    batch_size_list = [128, 256, 512, 1024]
    total_ = len(negrate_list) * len(learning_rate_list) * len(batch_size_list)
    count_ = 0
    
    # 跟踪最佳模型
    best_prauc = -1
    best_config = None
    best_model_path = None
    results_log = []
    
    for neg_rate in negrate_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                count_ += 1
                print(f'------------------------------ [{count_}/{total_}] ------------------------------')
                
                # 训练模型并获取最终结果
                trainer = main_train(neg_rate, learning_rate, batch_size)
                
                # 获取最终评估结果
                eval_results = trainer.evaluate()
                current_prauc = eval_results.get('eval_prauc', -1)
                
                # 记录结果
                config = f"neg_rate={neg_rate}, lr={learning_rate}, batch_size={batch_size}"
                model_path = f'trained_models/{neg_rate}_{learning_rate}_{batch_size}/bestmodel'
                
                results_log.append({
                    'config': config,
                    'path': model_path,
                    'f1': eval_results.get('eval_f1', -1),
                    'accuracy': eval_results.get('eval_accuracy', -1),
                    'precision': eval_results.get('eval_precision', -1),
                    'recall': eval_results.get('eval_recall', -1),
                    'prauc': current_prauc
                })
                
                # 更新最佳模型
                if current_prauc > best_prauc:
                    best_prauc = current_prauc
                    best_config = config
                    best_model_path = model_path
                
                print(f"Current PR-AUC: {current_prauc:.4f}, Best PR-AUC so far: {best_prauc:.4f}")
    
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
