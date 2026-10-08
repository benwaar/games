#!/usr/bin/env python3
"""
Synthesise diatonic triad chord clips from individual Iowa piano note WAVs.

Mixes three single-note WAVs in the time domain to produce chord clips.
No new recordings needed — built entirely from data/raw/notes/.

Chords (C major diatonic triads):
    Cmaj = C4 + E4 + G4
    Dmin = D4 + F4 + A4
    Emin = E4 + G4 + B4
    Fmaj = F4 + A4 + C4
    Gmaj = G4 + B4 + D4
    Amin = A4 + C4 + E4

Each note has three dynamics (pp, mf, ff), producing multiple mix combinations
per chord — 4 mixes × 3 note combinations = up to 12 clips per chord before
augmentation. pipeline.batch then generates 7 augmented variants each.

Output: data/raw/chords/{chord_name}/chord_{dynamics}_{chord_name}_{idx}.wav

Usage:
    python scripts/synthesise_chords.py
    python scripts/synthesise_chords.py --notes-dir data/raw/notes --output data/raw/chords
    python scripts/synthesise_chords.py --dry-run
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import soundfile as sf

CHORDS: dict[str, list[str]] = {
    "Cmaj": ["C4", "E4", "G4"],
    "Dmin": ["D4", "F4", "A4"],
    "Emin": ["E4", "G4", "B4"],
    "Fmaj": ["F4", "A4", "C4"],
    "Gmaj": ["G4", "B4", "D4"],
    "Amin": ["A4", "C4", "E4"],
}

DYNAMICS = ["pp", "mf", "ff"]

# Dynamic combinations: uniform (pp+pp+pp etc.) + one mixed (pp+mf+ff)
DYN_COMBOS: list[tuple[str, str, str]] = [
    ("pp", "pp", "pp"),
    ("mf", "mf", "mf"),
    ("ff", "ff", "ff"),
    ("pp", "mf", "ff"),
]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def mix_notes(clips: list[np.ndarray]) -> np.ndarray:
    """Normalise each clip to peak 0.5, sum, renormalise."""
    normalised = []
    for clip in clips:
        peak = np.abs(clip).max()
        normalised.append(clip / peak * 0.5 if peak > 0 else clip)

    # Pad all to the same length
    max_len = max(c.shape[0] for c in normalised)
    padded = [np.pad(c, (0, max_len - c.shape[0])) for c in normalised]

    mixed = np.sum(padded, axis=0)

    # Renormalise the result
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak * 0.9

    return mixed


def synthesise(notes_dir: Path, output_dir: Path, dry_run: bool = False) -> int:
    total = 0
    for chord_name, notes in CHORDS.items():
        chord_out = output_dir / chord_name
        if not dry_run:
            chord_out.mkdir(parents=True, exist_ok=True)

        for idx, dyn_combo in enumerate(DYN_COMBOS):
            wav_paths = []
            missing = False
            for note, dyn in zip(notes, dyn_combo):
                wav = notes_dir / note / f"iowa_{dyn}_{note}.wav"
                if not wav.exists():
                    print(f"  MISSING: {wav}")
                    missing = True
                    break
                wav_paths.append(wav)

            if missing:
                continue

            out_path = chord_out / f"chord_{'-'.join(dyn_combo)}_{chord_name}_{idx}.wav"

            if dry_run:
                print(f"  [dry] {out_path.relative_to(output_dir.parent)}")
                total += 1
                continue

            clips = [load_wav(p) for p in wav_paths]
            sr = clips[0][1]
            mixed = mix_notes([c[0] for c in clips])

            sf.write(out_path, mixed, sr)
            total += 1

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesise chord WAVs from Iowa piano notes")
    parser.add_argument("--notes-dir", default="data/raw/notes", help="Source note WAVs")
    parser.add_argument("--output", default="data/raw/chords", help="Output chord directory")
    parser.add_argument("--dry-run", action="store_true", help="List files without writing")
    args = parser.parse_args()

    notes_dir = Path(args.notes_dir)
    output_dir = Path(args.output)

    if not notes_dir.exists():
        print(f"Error: notes directory not found: {notes_dir}")
        raise SystemExit(1)

    print(f"Notes dir : {notes_dir}")
    print(f"Output    : {output_dir}")
    print(f"Chords    : {', '.join(CHORDS)}")
    print(f"Mixes/chord: {len(DYN_COMBOS)}")
    print()

    total = synthesise(notes_dir, output_dir, dry_run=args.dry_run)

    label = "Would write" if args.dry_run else "Written"
    print(f"\n{label}: {total} chord clips across {len(CHORDS)} chords")


if __name__ == "__main__":
    main()
