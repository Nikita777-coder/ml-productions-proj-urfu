from __future__ import annotations

import torch
from torch import nn


class FashionCNN(nn.Module):
    """Two-block CNN for 28x28 grayscale FashionMNIST images."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer1(images)
        features = self.layer2(features)
        features = torch.flatten(features, start_dim=1)
        features = self.dropout(features)
        return self.fc(features)
