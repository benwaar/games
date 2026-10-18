"""
Phase 4 chord detection: two head variants on the Phase 3 backbone.

Option B — ChordNameClassifier (6-class, CrossEntropyLoss):
    Simpler. Predicts one of 6 chord names. Same setup as Phase 3.
    Less informative — can't tell you which note was wrong.

Option A — NoteSetClassifier (12-label, BCEWithLogitsLoss):
    Predicts which of the 12 chromatic notes are present simultaneously.
    Richer: errors are interpretable per note.
    Harder to train — 12 independent binary tasks.

Both load the Phase 3 fine-tuned checkpoint and swap the final linear layer.
The backbone (conv_blocks + gap) is kept and fine-tuned end-to-end.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from model.cnn import SoundClassifier


@dataclass
class ChordConfig:
    """Config shared by both chord head variants."""

    checkpoint_path: Path = field(
        default_factory=lambda: Path("output/transfer/finetune/best_model.pt")
    )
    processed_dir: Path = field(default_factory=lambda: Path("data/processed/chords"))
    output_dir: Path = field(default_factory=lambda: Path("output/chords"))
    batch_size: int = 32
    lr: float = 5e-4
    epochs: int = 80
    dropout: float = 0.3
    seed: int = 42
    patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5


NUM_CHORD_CLASSES = 6
NUM_NOTE_CLASSES = 12

# Chord name → which of the 12 chromatic notes (C4..B4) are present.
# Order matches sorted note names used by make_label_map / load_splits.
_SORTED_NOTES = ["A4", "Ab4", "B4", "Bb4", "C4", "D4", "Db4", "E4", "Eb4", "F4", "G4", "Gb4"]
_NOTE_INDEX: dict[str, int] = {note: i for i, note in enumerate(_SORTED_NOTES)}

_CHORD_NOTES: dict[str, list[str]] = {
    "Cmaj": ["C4", "E4", "G4"],
    "Dmin": ["D4", "F4", "A4"],
    "Emin": ["E4", "G4", "B4"],
    "Fmaj": ["F4", "A4", "C4"],
    "Gmaj": ["G4", "B4", "D4"],
    "Amin": ["A4", "C4", "E4"],
}


def chord_to_note_vector(chord_name: str) -> torch.Tensor:
    """Return a 12-dim binary tensor indicating which notes are in this chord."""
    vec = torch.zeros(NUM_NOTE_CLASSES)
    for note in _CHORD_NOTES[chord_name]:
        vec[_NOTE_INDEX[note]] = 1.0
    return vec


def load_chord_name_model(config: ChordConfig) -> SoundClassifier:
    """Load Phase 3 checkpoint, swap head for 6-class chord-name prediction."""
    return _load_and_swap(config, num_classes=NUM_CHORD_CLASSES)


def load_note_set_model(config: ChordConfig) -> SoundClassifier:
    """Load Phase 3 checkpoint, swap head for 12-label note-set prediction."""
    return _load_and_swap(config, num_classes=NUM_NOTE_CLASSES)


def _load_and_swap(config: ChordConfig, num_classes: int) -> SoundClassifier:
    # weights_only=False: checkpoint dict contains non-tensor objects (label_map, config)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    original_num_classes = len(checkpoint["label_map"])
    model = SoundClassifier(num_classes=original_num_classes, dropout=config.dropout)
    model.load_state_dict(checkpoint["model_state"])

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def note_set_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss for multi-label note prediction."""
    return nn.BCEWithLogitsLoss()(logits, targets.float())


def chord_name_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropyLoss for single-label chord-name prediction."""
    return nn.CrossEntropyLoss()(logits, targets)


def note_set_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert note-set logits → binary predictions. Shape (B, 12)."""
    return (torch.sigmoid(logits) > threshold).long()


def exact_match_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of samples where all predicted notes match targets exactly."""
    correct = (preds == targets).all(dim=1).sum().item()
    return correct / targets.shape[0]
