"""Small CNN for MNIST (~21k parameters).

Architecture:

    Conv2d(1, 16, 5)  -> ReLU -> MaxPool2d(2)
    Conv2d(16, 32, 5) -> ReLU -> MaxPool2d(2)
    Flatten -> Linear(32*4*4, 64) -> ReLU -> Linear(64, 10)

Actual parameter count: 46,706 (the 21,706 figure in
docs/specifications.md is incorrect — see design-decisions.md D9).
The architecture is the contract, not the count.
"""

from __future__ import annotations

import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_mnist_cnn() -> nn.Module:
    """Return a small CNN suitable for MNIST classification."""
    return MnistCNN()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
