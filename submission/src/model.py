from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGCNN(nn.Module):
    """
    Lightweight 1D CNN for multichannel sEMG gesture classification.

    Input shape: (batch_size, 8, T)
    Output: logits of shape (batch_size, num_classes)
    """

    def __init__(
        self,
        num_channels: int = 8,
        num_classes: int = 5,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (7, 5, 3),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        c1, c2, c3 = conv_channels
        k1, k2, k3 = kernel_sizes

        self.conv1 = nn.Conv1d(num_channels, c1, kernel_size=k1, padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(c1)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2)
        self.bn2 = nn.BatchNorm1d(c2)

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2)
        self.bn3 = nn.BatchNorm1d(c3)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c3, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc_out(x)
        return logits


def build_model_from_config(config) -> EMGCNN:
    """
    Helper builder that reads basic hyperparameters from config dict.
    """
    model_cfg = config.get("model", {})
    num_channels = int(model_cfg.get("num_channels", 8))
    num_classes = int(model_cfg.get("num_classes", 5))
    conv_channels = tuple(model_cfg.get("conv_channels", [32, 64, 128]))
    kernel_sizes = tuple(model_cfg.get("kernel_sizes", [7, 5, 3]))
    dropout = float(model_cfg.get("dropout", 0.3))
    return EMGCNN(
        num_channels=num_channels,
        num_classes=num_classes,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    )


