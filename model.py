import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class CloudSentimentModel(nn.Module):
    def __init__(
        self,
        bert_model: BertModel,
        cloud_drop_num: int = 512,
        features: list[int] = [128, 64],
        dropout: float = 0.3,
    ):
        super(CloudSentimentModel, self).__init__()
        self.bert = bert_model
        self.bert.requires_grad_(False)  # 冻结BERT层
        self.cloud_drop_num = cloud_drop_num  # 可配置的云模型采样数量
        self.dropout = dropout
        self.features = features

        hidden_size = self.bert.config.hidden_size

        # 云模型映射器
        self.fc_ex = nn.Linear(hidden_size, 1)
        self.fc_en = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())
        self.fc_he = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(cloud_drop_num, features[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features[0], features[1]),
            nn.ReLU(),
            nn.Linear(features[1], 2)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, H]
        ex = self.fc_ex(cls).squeeze(1)  # [B]
        en = self.fc_en(cls).squeeze(1)  # [B]
        he = self.fc_he(cls).squeeze(1)  # [B]

        device = next(self.parameters()).device
        B = ex.size(0)
        D = self.cloud_drop_num  # 云模型采样点数

        noise1 = torch.randn(B, D).to(device)
        noise2 = torch.randn(B, D).to(device)
        drops = ex.unsqueeze(1) + (en.unsqueeze(1) + he.unsqueeze(1) * noise1) * noise2  # [B, D]

        mu = torch.exp(-((drops - ex.unsqueeze(1)) ** 2) / (2 * (en.unsqueeze(1) ** 2 + 1e-6)))  # [B, D]

        return self.classifier(mu), ex, en, he


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=1)
        p = torch.exp(logp)
        loss = self.ce(input, target)
        modulating_factor = (1 - p[range(len(target)), target]) ** self.gamma
        return (modulating_factor * loss).mean()

