# -*- coding: utf-8 -*-
"""
2层中文 BERT (MLM) Proxy-Student
功能：
1. 对 JSONL QA 数据计算 MLM loss（不训练！）
2. 保存所有样本的 loss 到 CSV
3. 筛选 loss 最高的 TOP_RATIO 样本作为困难池（jsonl 比例不变）
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
import json
from tqdm import tqdm
import pandas as pd
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------
# 配置
# ---------------------------ed
DATA_PATH = "./train_data/main_data.jsonl"
BATCH_SIZE = 32
SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_PROB = 0.15
TOP_RATIO = 0.6  # 困难池比例

# ---------------------------
# 数据集
# ---------------------------
class MedQADataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_len=SEQ_LEN):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        text = f"Q: {row['input']} A: {row['output']}"
        tokenized = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "idx": idx
        }

# ---------------------------
# 模型
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased32")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased")

# 剪枝到前 2 层（Proxy-Student）
model.bert.encoder.layer = torch.nn.ModuleList(
    model.bert.encoder.layer[:6]
)

model.to(DEVICE)
model.eval()

# ---------------------------
# DataLoader
# ---------------------------
dataset = MedQADataset(DATA_PATH, tokenizer, SEQ_LEN)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True if DEVICE == "cuda" else False
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MASK_PROB
)

sample_losses = {}  # idx -> loss

# ---------------------------
# 推理（无梯度）
# ---------------------------
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Computing MLM Loss"):
        original_idxs = batch["idx"]

        inputs_for_collator = [
            {
                "input_ids": batch["input_ids"][i],
                "attention_mask": batch["attention_mask"][i]
            }
            for i in range(batch["input_ids"].size(0))
        ]
        collated = collator(inputs_for_collator)

        input_ids = collated["input_ids"].to(DEVICE)
        attention_mask = collated["attention_mask"].to(DEVICE)
        labels = collated["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        ).view(labels.size())

        for i, idx in enumerate(original_idxs):
            idx = int(idx)
            mask = labels[i] != -100
            if mask.sum() > 0:
                sample_loss = token_loss[i][mask].mean().item()
            else:
                sample_loss = 0.0
            sample_losses[idx] = sample_loss

# ---------------------------
# ① 保存【所有样本】loss
# ---------------------------
all_loss_df = pd.DataFrame(
    sorted(sample_losses.items()),
    columns=["idx", "loss"]
)
all_loss_df.to_csv("all_sample_losses.csv", index=False)

# ---------------------------
# ② 筛选困难池
# ---------------------------
loss_df = all_loss_df.sort_values(
    by="loss", ascending=False
).reset_index(drop=True)

top_n = math.ceil(len(loss_df) * TOP_RATIO)
hard_pool = loss_df.iloc[:top_n]

hard_pool.to_csv("hard_pool.csv", index=False)

# ---------------------------
# ③ 输出困难池 jsonl（比例不变）
# ---------------------------
hard_indices = set(hard_pool["idx"].values)

with open("hard_pool_samples.jsonl", "w", encoding="utf-8") as f_out:
    with open(DATA_PATH, "r", encoding="utf-8") as f_in:
        for i, line in enumerate(f_in):
            if i in hard_indices:
                f_out.write(line)

print(f"总样本数: {len(all_loss_df)}")
print(f"困难池样本数: {len(hard_pool)} ({TOP_RATIO * 100:.0f}%)")
print("✔ 全量 loss 已保存: all_sample_losses.csv")
print("✔ 困难池 CSV: hard_pool.csv")
print("✔ 困难池 JSONL: hard_pool_samples.jsonl")
