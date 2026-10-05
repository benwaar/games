"""Hyperparameter configuration for Phase 2 training."""

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
