"""Tests for model/chord.py — Phase 4 chord detection heads."""

import torch
import pytest
from pathlib import Path

from model.chord import (
    ChordConfig,
    NUM_CHORD_CLASSES,
    NUM_NOTE_CLASSES,
    chord_name_loss,
    chord_to_note_vector,
    exact_match_accuracy,
    load_chord_name_model,
    load_note_set_model,
    note_set_loss,
    note_set_predictions,
    _CHORD_NOTES,
    _NOTE_INDEX,
)

CHECKPOINT = Path("output/transfer/finetune/best_model.pt")


# --- chord_to_note_vector ---

def test_chord_to_note_vector_shape():
    vec = chord_to_note_vector("Cmaj")
    assert vec.shape == (NUM_NOTE_CLASSES,)


def test_chord_to_note_vector_cmaj_correct_notes():
    vec = chord_to_note_vector("Cmaj")
    # Cmaj = C4, E4, G4
    assert vec[_NOTE_INDEX["C4"]] == 1.0
    assert vec[_NOTE_INDEX["E4"]] == 1.0
    assert vec[_NOTE_INDEX["G4"]] == 1.0
    assert vec.sum() == 3.0


def test_chord_to_note_vector_all_chords_have_three_notes():
    for chord in _CHORD_NOTES:
        vec = chord_to_note_vector(chord)
        assert vec.sum() == 3.0, f"{chord} should have exactly 3 notes"


def test_chord_to_note_vector_amin():
    vec = chord_to_note_vector("Amin")
    # Amin = A4, C4, E4
    assert vec[_NOTE_INDEX["A4"]] == 1.0
    assert vec[_NOTE_INDEX["C4"]] == 1.0
    assert vec[_NOTE_INDEX["E4"]] == 1.0


# --- model loading (requires checkpoint) ---

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Phase 3 checkpoint not found")
def test_chord_name_model_output_shape():
    config = ChordConfig()
    model = load_chord_name_model(config)
    x = torch.randn(4, 1, 128, 65)
    out = model(x)
    assert out.shape == (4, NUM_CHORD_CLASSES)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Phase 3 checkpoint not found")
def test_note_set_model_output_shape():
    config = ChordConfig()
    model = load_note_set_model(config)
    x = torch.randn(4, 1, 128, 65)
    out = model(x)
    assert out.shape == (4, NUM_NOTE_CLASSES)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Phase 3 checkpoint not found")
def test_chord_name_model_head_size():
    config = ChordConfig()
    model = load_chord_name_model(config)
    assert model.classifier[-1].out_features == NUM_CHORD_CLASSES


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Phase 3 checkpoint not found")
def test_note_set_model_head_size():
    config = ChordConfig()
    model = load_note_set_model(config)
    assert model.classifier[-1].out_features == NUM_NOTE_CLASSES


# --- loss functions ---

def test_chord_name_loss_no_nan():
    logits = torch.randn(8, NUM_CHORD_CLASSES)
    targets = torch.randint(0, NUM_CHORD_CLASSES, (8,))
    loss = chord_name_loss(logits, targets)
    assert not loss.isnan()


def test_note_set_loss_no_nan():
    logits = torch.randn(8, NUM_NOTE_CLASSES)
    targets = torch.zeros(8, NUM_NOTE_CLASSES)
    targets[:, [4, 7, 10]] = 1.0  # Cmaj-like
    loss = note_set_loss(logits, targets)
    assert not loss.isnan()


def test_note_set_loss_shape_scalar():
    logits = torch.randn(4, NUM_NOTE_CLASSES)
    targets = torch.zeros(4, NUM_NOTE_CLASSES)
    loss = note_set_loss(logits, targets)
    assert loss.shape == torch.Size([])


# --- predictions and metrics ---

def test_note_set_predictions_shape():
    logits = torch.randn(8, NUM_NOTE_CLASSES)
    preds = note_set_predictions(logits)
    assert preds.shape == (8, NUM_NOTE_CLASSES)


def test_note_set_predictions_binary():
    logits = torch.randn(8, NUM_NOTE_CLASSES)
    preds = note_set_predictions(logits)
    assert set(preds.unique().tolist()).issubset({0, 1})


def test_note_set_predictions_threshold():
    # Large positive logit → always predicted
    logits = torch.full((2, NUM_NOTE_CLASSES), 10.0)
    preds = note_set_predictions(logits, threshold=0.5)
    assert preds.all()


def test_exact_match_accuracy_perfect():
    preds = torch.tensor([[1, 0, 1], [0, 1, 0]])
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    assert exact_match_accuracy(preds, targets) == 1.0


def test_exact_match_accuracy_none():
    preds = torch.tensor([[1, 0, 1], [0, 1, 0]])
    targets = torch.tensor([[0, 1, 0], [1, 0, 1]])
    assert exact_match_accuracy(preds, targets) == 0.0


def test_exact_match_accuracy_partial():
    preds = torch.tensor([[1, 0, 1], [0, 1, 0]])
    targets = torch.tensor([[1, 0, 1], [1, 0, 1]])  # second row wrong
    assert exact_match_accuracy(preds, targets) == 0.5
