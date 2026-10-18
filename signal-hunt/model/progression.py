"""
Phase 4b — CNN-RNN hybrid for chord progression classification.

Architecture:
    (B, 1, 128, T)
        ↓  Phase 4 chord-name CNN backbone (conv_blocks)
    (B, 64, 16, T')          T' ≈ T/8 after 3× MaxPool2d(2)
        ↓  mean over freq axis
    (B, T', 64)              time-step feature sequence
        ↓  bidirectional GRU (hidden=64, 1 layer)
    (B, hidden*2)            last hidden state, both directions concatenated
        ↓  linear head
    (B, num_progressions)    logits

Loaded from Phase 4 chord-name checkpoint (backbone weights reused).
The GRU and head are trained fresh.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from model.cnn import SoundClassifier


@dataclass
class ProgressionConfig:
    checkpoint_path: Path = field(
        default_factory=lambda: Path("output/chords/name/best_model.pt")
    )
    processed_dir: Path = field(
        default_factory=lambda: Path("data/processed/progressions")
    )
    output_dir: Path = field(default_factory=lambda: Path("output/progressions"))
    num_progressions: int = 4
    gru_hidden: int = 64
    gru_layers: int = 1
    dropout: float = 0.3
    batch_size: int = 8         # small — only 16 source clips
    lr: float = 5e-4
    epochs: int = 100
    seed: int = 42
    patience: int = 15
    lr_scheduler_patience: int = 6
    lr_scheduler_factor: float = 0.5


class ProgressionClassifier(nn.Module):
    """
    CNN feature extractor (Phase 4 backbone) + bidirectional GRU + linear head.

    Input:  (B, 1, 128, T)  — full-length progression Mel-spectrogram
    Output: (B, num_progressions)  — logits
    """

    def __init__(self, backbone: SoundClassifier, num_progressions: int, gru_hidden: int,
                 gru_layers: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv_blocks = backbone.conv_blocks  # (B,1,128,T) → (B,64,16,T')

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden * 2, num_progressions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN: (B, 1, 128, T) → (B, 64, H, T')
        features = self.conv_blocks(x)

        # Mean over freq axis: (B, 64, H, T') → (B, 64, T')
        features = features.mean(dim=2)

        # Transpose for GRU: (B, 64, T') → (B, T', 64)
        features = features.permute(0, 2, 1)

        # GRU: (B, T', 64) → (B, T', hidden*2)
        out, _ = self.gru(features)

        # Use last time step
        last = out[:, -1, :]            # (B, hidden*2)
        last = self.dropout(last)
        return self.head(last)          # (B, num_progressions)


def load_progression_model(config: ProgressionConfig) -> ProgressionClassifier:
    """Load Phase 4 chord-name backbone, attach GRU + head."""
    ckpt = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
    original_num_classes = len(ckpt["label_map"])

    from model.chord import load_chord_name_model, ChordConfig
    chord_cfg = ChordConfig(dropout=config.dropout)
    backbone = load_chord_name_model(chord_cfg)

    return ProgressionClassifier(
        backbone=backbone,
        num_progressions=config.num_progressions,
        gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers,
        dropout=config.dropout,
    )
