#!/usr/bin/env python3
"""
Synthesise chord progression clips from individual chord WAVs.

Concatenates 4 chord clips with short silence gaps to produce a single
progression clip. Uses the first 1.5s of each chord WAV (same as the
pipeline's ingest step) so each chord's attack and sustain is captured.

Progressions (C major diatonic):
    I-IV-V-I   = Cmaj → Fmaj → Gmaj → Cmaj
    vi-IV-I-V  = Amin → Fmaj → Cmaj → Gmaj
    I-V-vi-IV  = Cmaj → Gmaj → Amin → Fmaj
    ii-V-I     = Dmin → Gmaj → Cmaj → Cmaj   (repeat tonic for 4-chord length)

Dynamic combinations: pp, mf, ff, mixed (one per source chord clip).
Target: 4 clips per progression before augmentation → 16+ tensors each.

Output: data/raw/progressions/{label}/prog_{dynamics}_{label}_{idx}.wav

Usage:
    python scripts/synthesise_progressions.py
    python scripts/synthesise_progressions.py --dry-run
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
CHORD_DURATION = 1.5          # seconds per chord
GAP_DURATION = 0.25           # silence between chords
CHORD_SAMPLES = int(SR * CHORD_DURATION)
GAP_SAMPLES = int(SR * GAP_DURATION)

PROGRESSIONS: dict[str, list[str]] = {
    "I-IV-V-I":  ["Cmaj", "Fmaj", "Gmaj", "Cmaj"],
    "vi-IV-I-V": ["Amin", "Fmaj", "Cmaj", "Gmaj"],
    "I-V-vi-IV": ["Cmaj", "Gmaj", "Amin", "Fmaj"],
    "ii-V-I-I":  ["Dmin", "Gmaj", "Cmaj", "Cmaj"],
}

DYNAMICS = ["pp", "mf", "ff"]

DYN_COMBOS: list[tuple[str, str, str, str]] = [
    ("pp", "pp", "pp", "pp"),
    ("mf", "mf", "mf", "mf"),
    ("ff", "ff", "ff", "ff"),
    ("pp", "mf", "ff", "mf"),
]


def load_chord_clip(chords_dir: Path, chord: str, dyn: str) -> np.ndarray:
    """Load first CHORD_SAMPLES of a chord WAV."""
    # Try uniform dynamics file first (e.g. chord_mf-mf-mf_Cmaj_1.wav)
    for f in sorted((chords_dir / chord).glob("*.wav")):
        parts = f.stem.split("_")
        if len(parts) >= 2 and dyn in parts[1].split("-")[0]:
            audio, _ = sf.read(f, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            return audio[:CHORD_SAMPLES] if len(audio) >= CHORD_SAMPLES else np.pad(
                audio, (0, CHORD_SAMPLES - len(audio))
            )
    raise FileNotFoundError(f"No {dyn} clip for {chord} in {chords_dir}")


def synthesise(chords_dir: Path, output_dir: Path, dry_run: bool = False) -> int:
    gap = np.zeros(GAP_SAMPLES, dtype=np.float32)
    total = 0

    for label, chords in PROGRESSIONS.items():
        label_dir = output_dir / label
        if not dry_run:
            label_dir.mkdir(parents=True, exist_ok=True)

        for idx, dyn_combo in enumerate(DYN_COMBOS):
            clips = []
            missing = False
            for chord, dyn in zip(chords, dyn_combo):
                try:
                    clips.append(load_chord_clip(chords_dir, chord, dyn))
                except FileNotFoundError as e:
                    print(f"  MISSING: {e}")
                    missing = True
                    break
            if missing:
                continue

            # Interleave chord clips with silence gaps
            parts = []
            for i, clip in enumerate(clips):
                parts.append(clip)
                if i < len(clips) - 1:
                    parts.append(gap)
            progression = np.concatenate(parts)

            # Normalise
            peak = np.abs(progression).max()
            if peak > 0:
                progression = progression / peak * 0.9

            out_path = label_dir / f"prog_{'|'.join(dyn_combo)}_{label}_{idx}.wav"

            if dry_run:
                dur = len(progression) / SR
                print(f"  [dry] {out_path.relative_to(output_dir.parent)}  ({dur:.2f}s)")
            else:
                sf.write(out_path, progression, SR)
            total += 1

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesise chord progression WAVs")
    parser.add_argument("--chords-dir", default="data/raw/chords")
    parser.add_argument("--output", default="data/raw/progressions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chords_dir = Path(args.chords_dir)
    output_dir = Path(args.output)

    if not chords_dir.exists():
        print(f"Error: chords directory not found: {chords_dir}")
        raise SystemExit(1)

    total_dur = (CHORD_DURATION * 4 + GAP_DURATION * 3)
    print(f"Chords dir  : {chords_dir}")
    print(f"Output      : {output_dir}")
    print(f"Progressions: {', '.join(PROGRESSIONS)}")
    print(f"Clip length : {total_dur:.2f}s per progression ({CHORD_DURATION}s × 4 + {GAP_DURATION}s gaps)")
    print()

    total = synthesise(chords_dir, output_dir, dry_run=args.dry_run)
    label = "Would write" if args.dry_run else "Written"
    print(f"\n{label}: {total} progression clips across {len(PROGRESSIONS)} labels")


if __name__ == "__main__":
    main()
