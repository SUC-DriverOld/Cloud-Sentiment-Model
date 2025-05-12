import os
import torch
import pandas as pd
import numpy as np
import argparse
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import CloudSentimentModel, FocalLoss


class TextDataset(Dataset):
    def __init__(self, config, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.padding = config.data.padding
        self.truncation = config.data.truncation
        self.max_length = config.data.max_length
        self.return_tensors = config.data.return_tensors

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        for k in encoded:
            encoded[k] = encoded[k].squeeze(0)
        return encoded, torch.tensor(self.labels[idx], dtype=torch.long)


class CloudSentimentModelLightning(LightningModule):
    def __init__(self, model: CloudSentimentModel, lr=1e-4, decay=0.01, gamma=2):
        super(CloudSentimentModelLightning, self).__init__()
        self.strict_loading=False
        self.model=model
        self.lr=lr
        self.decay=decay
        self.loss_fn=FocalLoss(gamma=gamma)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        encoded, labels = batch
        logits, _, _, _ = self(encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device))
        loss = self.loss_fn(logits, labels.to(self.device))
        return loss

    def validation_step(self, batch, batch_idx):
        encoded, labels = batch
        logits, _, _, _ = self(encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device))
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log("val_acc", acc, prog_bar=True, logger=True, rank_zero_only=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        non_bert_state_dict = {k: v for k, v in pl_module.state_dict().items() if "bert" not in k}
        checkpoint["state_dict"] = non_bert_state_dict
        checkpoint["config"] = config
        super().on_save_checkpoint(trainer, pl_module, checkpoint)


class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def train(config, model_path=None):
    seed_everything(config.seed, workers=True)

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
            print(f"[WARNING] Unsupported file format: {file}")

    if config.data.train_ratio < 1.0:
        combined = list(zip(texts, labels))
        random_state = np.random.RandomState(config.seed)
        random_state.shuffle(combined)
        num_samples = int(len(combined) * config.data.train_ratio)
        combined = combined[:num_samples]
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)

    tokenizer = BertTokenizer.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")
    bert_model = BertModel.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=config.data.val_size, random_state=config.seed)

    train_ds = TextDataset(config, X_train, y_train, tokenizer)
    val_ds = TextDataset(config, X_val, y_val, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers
    )

    model = CloudSentimentModel(
        bert_model,
        cloud_drop_num=config.model.cloud_drop_num,
        features=config.model.features,
        dropout=config.model.dropout
    )

    lightning_model = CloudSentimentModelLightning(
        model,
        lr=config.train.lr,
        decay=config.train.weight_decay,
        gamma=config.train.gamma
    )

    checkpoint_callback = CustomModelCheckpoint(
        config=config,
        monitor=config.train.monitor,
        dirpath=os.path.join(config.exp_dir, config.exp_name),
        filename=config.train.save_name,
        save_top_k=config.train.save_top_k,
        mode=config.train.mode,
        save_last=config.train.save_last,
        verbose=config.train.verbose,
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.train.devices,
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo' if os.name == "nt" else 'nccl'),
        max_epochs=config.train.max_epochs,
        logger=TensorBoardLogger(name="logs", save_dir=os.path.join(config.exp_dir, config.exp_name)),
        log_every_n_steps=config.train.log_every_n_steps,
        callbacks=[checkpoint_callback, LitProgressBar()],
        default_root_dir=os.path.join(config.exp_dir, config.exp_name),
    )

    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=model_path)
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml", help="Path to the config file")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the checkpoint model for resuming training")
    args = parser.parse_args()

    os.environ["USE_LIBUV"] = "0"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    config = OmegaConf.load(args.config)
    os.makedirs(os.path.join(config.exp_dir, config.exp_name), exist_ok=True)
    OmegaConf.save(config, os.path.join(config.exp_dir, config.exp_name, "config.yaml"))
    print(config)

    train(config, args.model)
