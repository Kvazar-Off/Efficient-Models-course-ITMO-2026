"""MNIST models used in the paper: logistic regression and LeNet (two convolutions, two fully connected layers)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(x, 1))


class LeNet(nn.Module):
    """Caffe-style LeNet (conv-conv-fc-fc) used in Table 2 of the paper."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        for m in (self.conv1, self.conv2, self.fc1, self.fc2):
            fan_in = m.weight[0].numel()
            nn.init.uniform_(m.weight, -(3 / fan_in)**.5, (3 / fan_in)**.5)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
