"""Tests for synthesis scripts — mix_notes, load_chord_clip, dry-run counts."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from scripts.synthesise_chords import mix_notes, CHORDS, DYN_COMBOS, synthesise as synthesise_chords
from scripts.synthesise_progressions import (
    CHORD_SAMPLES,
    GAP_SAMPLES,
    PROGRESSIONS,
    load_chord_clip,
    synthesise as synthesise_progressions,
)


# --- mix_notes ---

class TestMixNotes:
    def test_output_length_matches_longest_input(self):
        a = np.ones(100, dtype=np.float32)
        b = np.ones(80, dtype=np.float32)
        c = np.ones(60, dtype=np.float32)
        result = mix_notes([a, b, c])
        assert len(result) == 100

    def test_output_does_not_clip(self):
        # Three near-peak signals — should not exceed 1.0 after normalisation
        clips = [np.ones(100, dtype=np.float32) * 0.95 for _ in range(3)]
        result = mix_notes(clips)
        assert np.abs(result).max() <= 1.0

    def test_silent_clip_handled(self):
        clips = [np.zeros(100, dtype=np.float32), np.ones(100, dtype=np.float32)]
        result = mix_notes(clips)
        assert not np.isnan(result).any()

    def test_peak_normalised_to_09(self):
        clips = [np.ones(100, dtype=np.float32)]
        result = mix_notes(clips)
        assert abs(np.abs(result).max() - 0.9) < 1e-5

    def test_single_clip_passthrough(self):
        clip = np.linspace(0, 1, 100, dtype=np.float32)
        result = mix_notes([clip])
        # shape preserved
        assert result.shape == clip.shape


# --- chord synthesis dry-run ---

def test_synthesise_chords_dry_run_count(tmp_path):
    # With a non-existent notes dir, dry-run should count 0 (all missing)
    count = synthesise_chords(
        notes_dir=tmp_path / "notes",
        output_dir=tmp_path / "chords",
        dry_run=True,
    )
    assert count == 0


def test_synthesise_chords_dry_run_with_real_notes():
    notes_dir = Path("data/raw/notes")
    if not notes_dir.exists():
        pytest.skip("Iowa piano notes not present")
    from scripts.synthesise_chords import synthesise as syn
    count = syn(notes_dir=notes_dir, output_dir=Path("/tmp/chords_dryrun"), dry_run=True)
    expected = len(CHORDS) * len(DYN_COMBOS)
    assert count == expected


# --- load_chord_clip ---

def test_load_chord_clip_truncates_to_chord_samples(tmp_path):
    chord_dir = tmp_path / "Cmaj"
    chord_dir.mkdir()
    # Write a WAV longer than CHORD_SAMPLES
    long_audio = np.zeros(CHORD_SAMPLES * 3, dtype=np.float32)
    sf.write(str(chord_dir / "chord_mf-mf-mf_Cmaj_0.wav"), long_audio, 22050)

    clip = load_chord_clip(tmp_path, "Cmaj", "mf")
    assert len(clip) == CHORD_SAMPLES


def test_load_chord_clip_pads_short_audio(tmp_path):
    chord_dir = tmp_path / "Cmaj"
    chord_dir.mkdir()
    short_audio = np.zeros(100, dtype=np.float32)
    sf.write(str(chord_dir / "chord_mf-mf-mf_Cmaj_0.wav"), short_audio, 22050)

    clip = load_chord_clip(tmp_path, "Cmaj", "mf")
    assert len(clip) == CHORD_SAMPLES


def test_load_chord_clip_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chord_clip(tmp_path, "Cmaj", "mf")


# --- progression synthesis dry-run ---

def test_synthesise_progressions_dry_run_count(tmp_path):
    # With empty chords dir all loads fail → count = 0
    count = synthesise_progressions(
        chords_dir=tmp_path / "chords",
        output_dir=tmp_path / "progressions",
        dry_run=True,
    )
    assert count == 0
