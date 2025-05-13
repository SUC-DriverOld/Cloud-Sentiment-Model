import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudSentimentModel(nn.Module):
    def __init__(
        self,
        input_size: int,  # BERT特征维度
        cloud_drop_num: int = 512,
        features: list[int] = [128, 64],
        dropout: float = 0.3,
    ):
        super(CloudSentimentModel, self).__init__()
        self.cloud_drop_num = cloud_drop_num  # 可配置的云模型采样数量
        self.dropout = dropout
        self.features = features

        # 云模型映射器
        self.fc_ex = nn.Linear(input_size, 1)
        self.fc_en = nn.Sequential(nn.Linear(input_size, 1), nn.Softplus())
        self.fc_he = nn.Sequential(nn.Linear(input_size, 1), nn.Softplus())

        # 分类器
        self.classifier = nn.Sequential()
        input_dim = cloud_drop_num

        for i, feature_dim in enumerate(features[:-1]):
            self.classifier.add_module(f'linear_{i}', nn.Linear(input_dim, feature_dim))
            self.classifier.add_module(f'relu_{i}', nn.ReLU())
            if dropout > 0:
                self.classifier.add_module(f'dropout_{i}', nn.Dropout(dropout))
            input_dim = feature_dim

        self.classifier.add_module('output', nn.Linear(input_dim, 2))

    def forward(self, features):
        ex = self.fc_ex(features).squeeze(1)  # [B]
        en = self.fc_en(features).squeeze(1)  # [B]
        he = self.fc_he(features).squeeze(1)  # [B]

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