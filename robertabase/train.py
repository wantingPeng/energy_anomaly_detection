import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import AutoModel, AutoConfig, Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NumericDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "inputs": torch.tensor(sample["input"], dtype=torch.float32),
            "labels": torch.tensor(sample["label"], dtype=torch.long)
        }

def collate_numeric(features):
    x_seqs  = [f["inputs"] for f in features]
    labels  = torch.stack([f["labels"] for f in features])
    seq_lens   = [len(x) for x in x_seqs]
    padded_x   = pad_sequence(x_seqs, batch_first=True, padding_value=0.0)
    attn_mask  = torch.zeros_like(padded_x, dtype=torch.long)
    for i,l in enumerate(seq_lens):
        attn_mask[i, :l] = 1

    return {
        "inputs": padded_x,  
        "mask": attn_mask, 
        "labels": labels     
    }

class NumericProjector(nn.Module):
    """
    把单个实数映射到 hidden_size 维向量
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)   # 输入 1-dim → 输出 hidden

    def forward(self, x):  # x: [B, L] float
        return self.proj(x.unsqueeze(-1))       # [B, L, hidden]

class NumericClassifier(nn.Module):
    def __init__(self, backbone_name, num_labels):
        super().__init__()
        cfg = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.num_hidden = cfg.hidden_size
        self.numeric_proj = NumericProjector(cfg.hidden_size)

        self.classifier = nn.Linear(cfg.hidden_size, num_labels)

    def forward(self, inputs, mask, labels=None):
        num_emb = self.numeric_proj(inputs)  
        attention_mask = mask.long()     
        outputs = self.backbone(
            inputs_embeds=num_emb,
            attention_mask=attention_mask,
        )
        cls_rep = outputs.last_hidden_state[:, 0]   # 用第 0 位置作 [CLS]
        logits = self.classifier(cls_rep)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

def process_rawdata(input_path, model):
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
        data_list += random.sample(data_0_list, count_map[1]*9)
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

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main_train():
    device = torch.device("cuda")
    print('[INFO] loading data...')
    samples = process_rawdata('Data/row_energyData_subsample_Transform/labeled/train/contact/part.0.parquet','train')
    val_samples = process_rawdata('Data/row_energyData_subsample_Transform/labeled/val/contact/part.0.parquet','val')
    print('[INFO] data loaded...')
    train_dataset = NumericDataset(samples)
    val_dataset = NumericDataset(val_samples)

    # 加载预训练模型
    model = NumericClassifier(backbone_name="arampacha/roberta-tiny", num_labels=2)
    model.to(device)
    outputpath = 'trained_models'
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=outputpath,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=1024,  
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=30,
        disable_tqdm=False,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # 对于loss，值越小越好
        bf16=True,
        dataloader_num_workers=16,
        report_to="tensorboard",
        learning_rate=7e-6,
        gradient_accumulation_steps=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_numeric,
        compute_metrics=compute_metrics  # 添加这一行
    )
    trainer.train()
    trainer.save_model(f"{outputpath}/bestmodel")
    
if __name__ == '__main__':
    main_train()