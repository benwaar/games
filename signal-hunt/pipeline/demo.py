"""End-to-end demo — load a sample file, run the full pipeline, print stats."""

import sys
from pathlib import Path

import numpy as np

from pipeline.ingest import ingest
from pipeline.augment import add_noise, pitch_shift, time_stretch
from pipeline.features import extract_features


def demo(path: str | Path) -> None:
    path = Path(path)
    print(f"=== Signal Hunt — Pipeline Demo ===\n")
    print(f"Source: {path.name}")

    y, sr = ingest(path)
    print(f"Ingested: {len(y)} samples, {sr} Hz, {len(y)/sr:.2f}s")

    tensor = extract_features(y, sr, target_frames=65)
    print(f"Features: shape={tuple(tensor.shape)}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")

    print(f"\nAugmentation variants:")
    rng = np.random.default_rng(42)
    variants = [
        ("clean", y),
        ("noise (SNR 20)", add_noise(y, snr_db=20, rng=rng)),
        ("noise (SNR 10)", add_noise(y, snr_db=10, rng=rng)),
        ("pitch +2", pitch_shift(y, sr, n_steps=2.0)),
        ("pitch -2", pitch_shift(y, sr, n_steps=-2.0)),
        ("slow 0.85×", time_stretch(y, rate=0.85)[:len(y)]),
        ("fast 1.15×", time_stretch(y, rate=1.15)[:len(y)]),
    ]

    for name, signal in variants:
        padded = np.pad(signal, (0, max(0, len(y) - len(signal))))[:len(y)]
        t = extract_features(padded, sr, target_frames=65)
        print(f"  {name:16s}  shape={tuple(t.shape)}  mean={t.mean():.4f}  std={t.std():.4f}")

    print(f"\nPipeline: ingest → augment → Mel-spectrogram → log-dB → normalise → tensor")
    print(f"All variants: (1, 128, 65) float32 — ready for CNN")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data/raw/hum_440hz.wav"
    demo(path)


if __name__ == "__main__":
    main()
