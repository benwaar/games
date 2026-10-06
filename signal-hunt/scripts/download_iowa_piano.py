#!/usr/bin/env python3
"""
Download piano note samples from the University of Iowa Electronic Music Studios.
Covers C4–B4 (one chromatic octave, 12 notes) at three velocity levels.

Source: https://theremin.music.uiowa.edu/MISPiano.html
License: Free for educational and research use.

Downloads AIFF from Iowa and converts to WAV so the existing pipeline.batch
can process them without any changes (batch globs for *.wav).

Usage:
    python scripts/download_iowa_piano.py
    python scripts/download_iowa_piano.py --output data/raw/notes  # default
    python scripts/download_iowa_piano.py --dry-run                 # list files without downloading

Output folder structure:
    data/raw/notes/
        C4/    iowa_mf_C4.wav, iowa_pp_C4.wav, iowa_ff_C4.wav
        Db4/   iowa_mf_Db4.wav, ...
        ...
        B4/

The pipeline (pipeline.batch) reads these via data/raw/notes/ and uses
the folder name as the class label (e.g. C4/, Db4/).
"""

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_URL = "https://theremin.music.uiowa.edu/sound%20files/MIS/Piano_Other/piano"

NOTES = ["C4", "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4"]
VELOCITIES = ["pp", "mf", "ff"]
TARGET_SR = 22050  # match pipeline default


def download_and_convert(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Download AIFF from Iowa, resample to 22050 Hz mono, save as WAV."""
    if dest.exists():
        print(f"  skip  {dest.name} (already exists)")
        return True
    if dry_run:
        print(f"  would download → {dest.name}")
        return True
    try:
        print(f"  downloading {dest.name} ...", end=" ", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        urllib.request.urlretrieve(url, tmp_path)

        # Load AIFF, convert to mono float32
        data, sr = sf.read(tmp_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo → mono

        # Resample to TARGET_SR if needed
        if sr != TARGET_SR:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

        sf.write(dest, data, TARGET_SR, subtype="PCM_16")
        tmp_path.unlink()
        print(f"done ({dest.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Iowa piano samples (C4–B4)")
    parser.add_argument("--output", type=Path, default=Path("data/raw/notes"))
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    args = parser.parse_args()

    output_dir = args.output
    total, failed = 0, 0

    print(f"{'DRY RUN — ' if args.dry_run else ''}Iowa piano samples (AIFF → WAV) → {output_dir}/\n")

    for note in NOTES:
        note_dir = output_dir / note
        if not args.dry_run:
            note_dir.mkdir(parents=True, exist_ok=True)
        print(f"{note}/")
        for velocity in VELOCITIES:
            aiff_name = f"Piano.{velocity}.{note}.aiff"
            url = f"{BASE_URL}/{aiff_name}"
            dest = note_dir / f"iowa_{velocity}_{note}.wav"
            ok = download_and_convert(url, dest, dry_run=args.dry_run)
            total += 1
            if not ok:
                failed += 1

    print(f"\n{'Would download' if args.dry_run else 'Downloaded and converted'} {total - failed}/{total} files")
    if failed:
        print(f"WARNING: {failed} files failed — re-run to retry")
        sys.exit(1)


if __name__ == "__main__":
    main()
