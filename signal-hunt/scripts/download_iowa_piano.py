#!/usr/bin/env python3
"""
Download piano note samples from the University of Iowa Electronic Music Studios.
Covers C4–B4 (one chromatic octave, 12 notes) at three velocity levels.

Source: https://theremin.music.uiowa.edu/MISPiano.html
License: Free for educational and research use.

Usage:
    python scripts/download_iowa_piano.py
    python scripts/download_iowa_piano.py --output data/raw/notes  # default
    python scripts/download_iowa_piano.py --dry-run                 # list files without downloading

Output folder structure:
    data/raw/notes/
        C4/    iowa_mf_C4.aiff, iowa_pp_C4.aiff, iowa_ff_C4.aiff
        Db4/   iowa_mf_Db4.aiff, ...
        ...
        B4/

The pipeline (pipeline.batch) reads these via data/raw/notes/ and uses
the folder name as the class label (e.g. C4/, Db4/).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://theremin.music.uiowa.edu/sound%20files/MIS/Piano_Other/piano"

NOTES = ["C4", "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4"]
VELOCITIES = ["pp", "mf", "ff"]


def download_file(url: str, dest: Path, dry_run: bool = False) -> bool:
    if dest.exists():
        print(f"  skip  {dest.name} (already exists)")
        return True
    if dry_run:
        print(f"  would download → {dest}")
        return True
    try:
        print(f"  downloading {dest.name} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
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

    print(f"{'DRY RUN — ' if args.dry_run else ''}Downloading Iowa piano samples → {output_dir}/\n")

    for note in NOTES:
        note_dir = output_dir / note
        if not args.dry_run:
            note_dir.mkdir(parents=True, exist_ok=True)
        print(f"{note}/")
        for velocity in VELOCITIES:
            filename = f"Piano.{velocity}.{note}.aiff"
            url = f"{BASE_URL}/{filename}"
            dest = note_dir / f"iowa_{velocity}_{note}.aiff"
            ok = download_file(url, dest, dry_run=args.dry_run)
            total += 1
            if not ok:
                failed += 1

    print(f"\n{'Would download' if args.dry_run else 'Downloaded'} {total - failed}/{total} files")
    if failed:
        print(f"WARNING: {failed} files failed — re-run to retry")
        sys.exit(1)


if __name__ == "__main__":
    main()
