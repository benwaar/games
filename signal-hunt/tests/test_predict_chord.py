"""Tests for model/predict_chord.py — two-pass chord inference output."""

import torch
import pytest
from pathlib import Path
from io import StringIO
import sys

from model.predict_chord import _print_result
from model.chord import _CHORD_NOTES


# --- _print_result (no checkpoint needed) ---

def _capture(result, expected=None, verbose=False) -> str:
    buf = StringIO()
    sys.stdout = buf
    try:
        _print_result(result, expected, verbose)
    finally:
        sys.stdout = sys.__stdout__
    return buf.getvalue()


def _fake_result(chord="Cmaj", notes_on=None) -> dict:
    if notes_on is None:
        notes_on = list(_CHORD_NOTES[chord])
    return {
        "chord": chord,
        "confidence": 0.9,
        "notes_on": notes_on,
        "scores": {chord: 0.9, "Amin": 0.05, "Dmin": 0.05},
    }


class TestPrintResult:
    def test_correct_chord_shows_tick(self):
        out = _capture(_fake_result("Cmaj"), expected="Cmaj")
        assert "✓" in out
        assert "✗" not in out

    def test_wrong_chord_shows_cross(self):
        out = _capture(_fake_result("Amin"), expected="Cmaj")
        assert "✗" in out
        assert "expected Cmaj" in out

    def test_missing_notes_shown(self):
        # Amin has A4, C4, E4; Cmaj needs C4, E4, G4 — G4 is missing, A4 is extra
        out = _capture(_fake_result("Amin", notes_on=["A4", "C4", "E4"]), expected="Cmaj")
        assert "G4" in out
        assert "Missing" in out

    def test_no_expected_no_verdict(self):
        out = _capture(_fake_result("Cmaj"))
        assert "✗" not in out
        assert "expected" not in out

    def test_verbose_shows_all_scores(self):
        out = _capture(_fake_result("Cmaj"), verbose=True)
        assert "Amin" in out
        assert "Dmin" in out

    def test_notes_listed(self):
        out = _capture(_fake_result("Cmaj", notes_on=["C4", "E4", "G4"]))
        assert "C4" in out
        assert "E4" in out
        assert "G4" in out

    def test_no_notes_detected(self):
        result = _fake_result("Cmaj", notes_on=[])
        out = _capture(result)
        assert "none detected" in out.lower() or out.count("✓") == 0


# --- predict_chord with fake checkpoints ---

@pytest.fixture
def fake_chord_checkpoints(tmp_path):
    """Minimal name + notes checkpoints for predict_chord smoke test."""
    import torch.nn as nn
    from model.cnn import SoundClassifier
    from model.chord import NUM_CHORD_CLASSES, NUM_NOTE_CLASSES

    label_map = {"Amin": 0, "Cmaj": 1, "Dmin": 2, "Emin": 3, "Fmaj": 4, "Gmaj": 5}

    # name checkpoint (6-class head)
    name_model = SoundClassifier(num_classes=NUM_CHORD_CLASSES, dropout=0.0)
    name_path = tmp_path / "name.pt"
    torch.save({"model_state": name_model.state_dict(), "label_map": label_map}, name_path)

    # notes checkpoint (12-class head)
    notes_model = SoundClassifier(num_classes=12, dropout=0.0)
    notes_model.classifier[-1] = nn.Linear(32, 12)
    notes_path = tmp_path / "notes.pt"
    torch.save({"model_state": notes_model.state_dict(), "label_map": label_map}, notes_path)

    return name_path, notes_path


def test_predict_chord_returns_expected_keys(fake_chord_checkpoints, tmp_path):
    import soundfile as sf
    import numpy as np
    from model.predict_chord import predict_chord

    name_path, notes_path = fake_chord_checkpoints

    # Write a minimal WAV
    wav = tmp_path / "test.wav"
    sf.write(str(wav), np.zeros(22050, dtype=np.float32), 22050)

    result = predict_chord(wav, name_checkpoint=name_path, notes_checkpoint=notes_path)
    assert "chord" in result
    assert "confidence" in result
    assert "notes_on" in result
    assert "scores" in result
    assert isinstance(result["notes_on"], list)
    assert 0.0 <= result["confidence"] <= 1.0
