"""3-class CNN classifier for Mel-spectrogram sound type recognition."""

import torch
import torch.nn as nn


class SoundClassifier(nn.Module):
    """
    Lightweight CNN for 3-class sound type classification.

    Input:  (B, 1, 128, 65) — batch of Mel-spectrograms
    Output: (B, 3)          — raw logits (hum / whistle / clap)

    Architecture: 3 conv blocks → Global Average Pooling → linear head.
    ~25,700 parameters — right-sized for ~150 training samples.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: (B, 1, 128, 65) → (B, 16, 64, 32)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: (B, 16, 64, 32) → (B, 32, 32, 16)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: (B, 32, 32, 16) → (B, 64, 16, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global Average Pooling: (B, 64, H, W) → (B, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.gap(x)
        x = x.flatten(1)        # (B, 64, 1, 1) → (B, 64)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
