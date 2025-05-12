import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, config):
        self.texts = texts
        self.tokenizer = tokenizer
        self.padding = config.preprocess.padding
        self.max_length = config.preprocess.max_length
        self.truncation = config.preprocess.truncation
        self.return_tensors = config.preprocess.return_tensors

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        for k in encoded:
            encoded[k] = encoded[k].squeeze(0)
        return encoded, idx

def extract_bert_features(config):
    print("Loading data...")

    texts, labels = list(), list()
    for file in os.listdir(config.data.path):
        if file.endswith(".tsv"):
            data = pd.read_csv(os.path.join(config.data.path, file), sep="\t")
            texts.extend(data["text"].tolist())
            labels.extend(data["label"].tolist())
        elif file.endswith(".csv"):
            data = pd.read_csv(os.path.join(config.data.path, file))
            texts.extend(data["text"].tolist())
            labels.extend(data["label"].tolist())
        else:
            print(f"[WARNING] Unsupported file format: {file}. Only .tsv and .csv are supported.")

    print(f"Total samples: {len(texts)}")

    if config.data.train_ratio < 1.0:
        combined = list(zip(texts, labels))
        random_state = np.random.RandomState(config.seed)
        random_state.shuffle(combined)
        num_samples = int(len(combined) * config.data.train_ratio)
        combined = combined[:num_samples]
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)
        print(f"Using {len(texts)} samples for training and validation.")

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=config.data.val_size, random_state=config.seed)

    print("Loading tokenizer and bert model...")

    # 加载BERT tokenizer和模型
    assert (os.path.exists("pretrain/chinese-roberta-wwm-ext-large/pytorch_model.bin"),
            "Please download the pre-trained model first.")

    tokenizer = BertTokenizer.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")
    bert_model = BertModel.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")
    bert_model.eval().cuda()

    # 创建数据集和加载器
    train_ds = TextDataset(X_train, tokenizer, config)
    val_ds = TextDataset(X_val, tokenizer, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.preprocess.batch_size,
        shuffle=False,
        num_workers=config.preprocess.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.preprocess.batch_size,
        shuffle=False,
        num_workers=config.preprocess.num_workers,
        pin_memory=True
    )

    # 预处理并保存特征
    output_dir = os.path.join(config.exp_dir, config.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    train_features = []
    with torch.no_grad():
        for batch, _ in tqdm(train_loader, desc="Extracting train features"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            train_features.append(outputs.cpu())

    train_features = torch.cat(train_features, dim=0)
    torch.save({
        "features": train_features,
        "labels": torch.tensor(y_train, dtype=torch.long)
    }, os.path.join(output_dir, "train_features.pt"))

    print(f"Saving train features to: {os.path.join(output_dir, 'train_features.pt')}")

    val_features = []
    with torch.no_grad():
        for batch, _ in tqdm(val_loader, desc="Extracting val features"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            val_features.append(outputs.cpu())
    
    val_features = torch.cat(val_features, dim=0)
    torch.save({
        "features": val_features,
        "labels": torch.tensor(y_val, dtype=torch.long)
    }, os.path.join(output_dir, "val_features.pt"))

    print(f"Saving val features to: {os.path.join(output_dir, 'val_features.pt')}")

    info = {
        "hidden_size": train_features.size(1),
        "train_samples": len(train_features),
        "val_samples": len(val_features)
    }
    print(f"Feature info: {info}")
    torch.save(info, os.path.join(output_dir, "features_info.pt"))
    print(f"Saving feature info to: {os.path.join(output_dir, 'features_info.pt')}")