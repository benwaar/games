"""
Two-pass chord inference — verdict (chord name) + correction (missing notes).

Usage
-----
    python -m model.predict_chord path/to/chord.wav
    python -m model.predict_chord path/to/chord.wav --expected Cmaj
    python -m model.predict_chord path/to/chord.wav --verbose
    bash demo_chord.sh

Output
------
    Chord:   Amin  (87.3% confidence)
    Notes:   A4 ✓  C4 ✓  E4 ✓

With --expected Cmaj:
    Chord:   Amin        ✗  (expected Cmaj)
    Notes:   A4 ✓  C4 ✓  E4 ✓
    Missing: E4 → G4  (Cmaj needs C4 + E4 + G4)
"""

import argparse
from pathlib import Path

import torch

from model.chord import (
    ChordConfig,
    _CHORD_NOTES,
    _NOTE_INDEX,
    _SORTED_NOTES,
    load_chord_name_model,
    load_note_set_model,
    note_set_predictions,
)
from pipeline.features import extract_features
from pipeline.ingest import DEFAULT_DURATION, DEFAULT_SR, ingest


def predict_chord(
    wav_path: Path,
    name_checkpoint: Path = Path("output/chords/name/best_model.pt"),
    notes_checkpoint: Path = Path("output/chords/notes/best_model.pt"),
    threshold: float = 0.5,
) -> dict:
    """
    Two-pass chord inference on a .wav file.

    Returns:
        {
            "chord":      "Amin",
            "confidence": 0.873,
            "notes_on":   ["A4", "C4", "E4"],
            "scores":     {"Amin": 0.873, "Cmaj": 0.041, ...}
        }
    """
    signal, sr = ingest(wav_path, target_sr=DEFAULT_SR, duration=DEFAULT_DURATION)
    tensor = extract_features(signal, sr).unsqueeze(0)   # (1, 1, 128, 65)

    # Pass 1 — chord name (verdict)
    name_ckpt = torch.load(name_checkpoint, map_location="cpu", weights_only=False)
    name_cfg = ChordConfig(checkpoint_path=name_checkpoint, dropout=0.0)
    name_model = load_chord_name_model(name_cfg)
    name_model.load_state_dict(name_ckpt["model_state"])
    name_model.eval()

    label_map = name_ckpt["label_map"]
    inv_map = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        logits = name_model(tensor)[0]
        probs = torch.softmax(logits, dim=0)
    scores = {inv_map[i]: round(probs[i].item(), 4) for i in range(len(label_map))}
    chord = max(scores, key=scores.__getitem__)

    # Pass 2 — note set (correction)
    notes_ckpt = torch.load(notes_checkpoint, map_location="cpu", weights_only=False)
    from model.cnn import SoundClassifier
    import torch.nn as nn
    # Rebuild from saved label_map size (12 note outputs)
    notes_model = SoundClassifier(num_classes=12, dropout=0.0)
    # Swap head to match checkpoint
    notes_model.classifier[-1] = nn.Linear(32, 12)
    notes_model.load_state_dict(notes_ckpt["model_state"])
    notes_model.eval()

    with torch.no_grad():
        note_logits = notes_model(tensor)[0]
    active = note_set_predictions(note_logits.unsqueeze(0), threshold=threshold)[0]
    notes_on = [_SORTED_NOTES[i] for i in range(len(_SORTED_NOTES)) if active[i]]

    return {
        "chord": chord,
        "confidence": scores[chord],
        "notes_on": notes_on,
        "scores": scores,
    }


def _print_result(result: dict, expected: str | None, verbose: bool) -> None:
    chord = result["chord"]
    conf = result["confidence"]
    notes_on = result["notes_on"]

    if expected:
        verdict = "✓" if chord == expected else f"✗  (expected {expected})"
        print(f"Chord:   {chord:<8} {verdict}  ({conf:.1%} confidence)")
    else:
        print(f"Chord:   {chord}  ({conf:.1%} confidence)")

    note_line = "  ".join(f"{n} ✓" for n in notes_on) or "(none detected)"
    print(f"Notes:   {note_line}")

    if expected and chord != expected:
        expected_notes = set(_CHORD_NOTES.get(expected, []))
        detected = set(notes_on)
        missing = sorted(expected_notes - detected)
        extra = sorted(detected - expected_notes)
        if missing:
            print(f"Missing: {', '.join(missing)}  ({expected} needs {' + '.join(sorted(expected_notes))})")
        if extra:
            print(f"Extra:   {', '.join(extra)}")

    if verbose:
        print("\nAll chord scores:")
        for c, s in sorted(result["scores"].items(), key=lambda x: -x[1]):
            print(f"  {c:>10}: {s:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-pass chord inference")
    parser.add_argument("wav", type=Path, help="Path to a .wav file")
    parser.add_argument("--expected", type=str, default=None,
                        help="Expected chord (e.g. Cmaj) — shows what's missing")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true", help="Show all chord scores")
    args = parser.parse_args()

    result = predict_chord(args.wav, threshold=args.threshold)
    _print_result(result, args.expected, args.verbose)


if __name__ == "__main__":
    main()
