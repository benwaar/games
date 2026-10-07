"""
Phase 3 transfer learning: load Phase 2 SoundClassifier, swap head for 12 note classes.

Transfer strategy
-----------------
The Phase 2 conv blocks learned frequency-band patterns (texture, onset, harmonic shape).
Those features are directly useful for pitch — the fundamental frequency and overtone ladder
both appear as frequency-band patterns on the Mel spectrogram.

We keep the backbone (conv_blocks + gap) and replace just the classifier head:
    Phase 2 head: Linear(64→32) → ReLU → Dropout → Linear(32→3)
    Phase 3 head: Linear(64→32) → ReLU → Dropout → Linear(32→12)

Two training modes (controlled by TransferConfig.freeze_backbone):
    freeze_backbone=True  — conv_blocks frozen, only the head trains.
                            Fast convergence. Good first experiment.
    freeze_backbone=False — everything trains end-to-end.
                            Slower, but allows the backbone to adapt to piano timbre.
"""

from pathlib import Path

import torch
import torch.nn as nn

from model.cnn import SoundClassifier
from model.config import TransferConfig


def load_transfer_model(config: TransferConfig) -> SoundClassifier:
    """
    Load Phase 2 checkpoint, replace head for num_classes, apply freeze setting.

    Returns a SoundClassifier ready for Phase 3 training.
    """
    # weights_only=False: checkpoint dict contains non-tensor objects (label_map, config)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    # Build the Phase 2 architecture and load saved weights.
    original_num_classes = len(checkpoint["label_map"])
    source_model = SoundClassifier(
        num_classes=original_num_classes,
        dropout=config.dropout,
    )
    source_model.load_state_dict(checkpoint["model_state"])

    # Swap the final linear layer (32→original_classes) for 32→num_classes.
    # The rest of the head (Linear(64→32), ReLU, Dropout) is kept as-is.
    in_features = source_model.classifier[-1].in_features
    source_model.classifier[-1] = nn.Linear(in_features, config.num_classes)

    if config.freeze_backbone:
        _freeze(source_model.conv_blocks)
        _freeze(source_model.gap)
        # Keep the head (classifier) trainable.

    return source_model


def _freeze(module: nn.Module) -> None:
    """Disable gradient updates for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def frozen_param_count(model: SoundClassifier) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def trainable_param_count(model: SoundClassifier) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_label_map(checkpoint_path: Path) -> dict[str, int]:
    """Read the label_map from a saved checkpoint without loading the full model."""
    # weights_only=False: checkpoint dict contains non-tensor objects (label_map, config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["label_map"]
