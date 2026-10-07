"""Hyperparameter configuration for Phase 2 and Phase 3 training."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50
    dropout: float = 0.3
    seed: int = 42
    patience: int = 5
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    num_classes: int = 3


@dataclass
class TransferConfig:
    """Config for Phase 3 transfer learning — Phase 2 backbone → 12-class note head."""

    checkpoint_path: Path = field(default_factory=lambda: Path("output/best_model.pt"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed/notes"))
    output_dir: Path = field(default_factory=lambda: Path("output/transfer"))
    num_classes: int = 12
    # When True: freeze conv_blocks, train head only (fast convergence check).
    # When False: fine-tune everything end-to-end (usually better final accuracy).
    freeze_backbone: bool = True
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 60
    dropout: float = 0.3
    seed: int = 42
    patience: int = 8
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
