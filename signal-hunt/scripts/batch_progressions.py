#!/usr/bin/env python3
"""
Batch pipeline for chord progression clips.

Unlike pipeline.batch (which truncates to 1.5s), this processes full-length
progression clips (~6.75s) and saves tensors with their natural time dimension.

Output tensors: (1, 128, T) where T ≈ 291 for 6.75s clips.

Usage:
    python scripts/batch_progressions.py
    python scripts/batch_progressions.py --input data/raw/progressions --output data/processed/progressions
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch


def process_wav(path: Path) -> torch.Tensor:
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S.astype(np.float32), ref=1.0)
    std = S_db.std()
    S_norm = (S_db - S_db.mean()) / (std + 1e-8) if std > 0 else np.zeros_like(S_db)
    return torch.tensor(S_norm).unsqueeze(0)   # (1, 128, T)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pipeline for progression clips")
    parser.add_argument("--input", default="data/raw/progressions")
    parser.add_argument("--output", default="data/processed/progressions")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    count = 0

    for label_dir in sorted(input_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        (output_dir / label).mkdir(exist_ok=True)

        for wav in sorted(label_dir.glob("*.wav")):
            tensor = process_wav(wav)
            out_name = wav.stem + ".pt"
            out_path = output_dir / label / out_name
            torch.save(tensor, out_path)
            manifest.append({
                "file": str(out_path.relative_to(output_dir)),
                "label": label,
                "source": wav.name,
                "shape": list(tensor.shape),
            })
            count += 1

    json.dump(manifest, open(output_dir / "manifest.json", "w"), indent=2)
    print(f"Processed {count} tensors → {output_dir}")
    if manifest:
        print(f"  shape: {manifest[0]['shape']}  labels: {sorted({m['label'] for m in manifest})}")


if __name__ == "__main__":
    main()
