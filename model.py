"""Architecture CNN pour la reconnaissance de locuteur à partir de spectrogrammes de Mel."""

import torch
from torch import nn


class SpeakerCNN(nn.Module):
    """3 blocs conv (BatchNorm + ReLU + MaxPool), pooling adaptatif puis
    classifieur linéaire. Volontairement petit et régularisé (dropout) car le
    dataset d'entraînement est très réduit (~100 exemples pour 5 classes)."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)  # logits (B, num_classes)
