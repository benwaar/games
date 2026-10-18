"""Batch processing — ingest, augment, extract features, save tensors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.ingest import ingest, pad_or_truncate, DEFAULT_SR, DEFAULT_DURATION
from pipeline.augment import add_noise, pitch_shift, time_stretch
from pipeline.features import extract_features, DEFAULT_N_MELS, DEFAULT_HOP_LENGTH

DEFAULT_TARGET_FRAMES = 65


def default_augmentations() -> list[dict]:
    return [
        {"name": "clean"},
        {"name": "noise_low", "fn": "add_noise", "kwargs": {"snr_db": 20.0}},
        {"name": "noise_high", "fn": "add_noise", "kwargs": {"snr_db": 10.0}},
        {"name": "pitch_up", "fn": "pitch_shift", "kwargs": {"n_steps": 2.0}},
        {"name": "pitch_down", "fn": "pitch_shift", "kwargs": {"n_steps": -2.0}},
        {"name": "slow", "fn": "time_stretch", "kwargs": {"rate": 0.85}},
        {"name": "fast", "fn": "time_stretch", "kwargs": {"rate": 1.15}},
    ]


AUGMENT_FNS = {
    "add_noise": add_noise,
    "pitch_shift": pitch_shift,
    "time_stretch": time_stretch,
}


def apply_augmentation(
    signal: np.ndarray, sr: int, aug: dict, rng: np.random.Generator,
) -> np.ndarray:
    if aug["name"] == "clean":
        return signal.copy()
    fn = AUGMENT_FNS[aug["fn"]]
    kwargs = dict(aug["kwargs"])
    if aug["fn"] == "add_noise":
        kwargs["rng"] = rng
    if aug["fn"] == "pitch_shift":
        kwargs["sr"] = sr
    result = fn(signal, **kwargs)
    return pad_or_truncate(result, len(signal))


def process_file(
    path: Path,
    output_dir: Path,
    augmentations: list[dict] | None = None,
    target_sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    seed: int = 42,
    label: str | None = None,
) -> list[dict]:
    augmentations = augmentations or default_augmentations()
    rng = np.random.default_rng(seed)
    signal, sr = ingest(path, target_sr=target_sr, duration=duration)
    label = label if label is not None else path.stem
    records = []

    for aug in augmentations:
        augmented = apply_augmentation(signal, sr, aug, rng)
        tensor = extract_features(augmented, sr, target_frames=target_frames)
        variant_name = f"{path.stem}_{aug['name']}"
        out_path = output_dir / f"{variant_name}.pt"
        torch.save(tensor, out_path)
        records.append({
            "file": out_path.name,
            "source": path.name,
            "label": label,
            "augmentation": aug["name"],
            "shape": list(tensor.shape),
        })

    return records


def process_folder(
    input_dir: Path,
    output_dir: Path,
    augmentations: list[dict] | None = None,
    target_sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    seed: int = 42,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build (path, label) pairs — subfolders take priority over flat layout
    subdirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]
    if subdirs:
        pairs = [
            (wav, subdir.name)
            for subdir in subdirs
            for wav in sorted(subdir.glob("*.wav"))
        ]
    else:
        flat = sorted(input_dir.glob("*.wav"))
        pairs = [(wav, wav.stem) for wav in flat]

    if not pairs:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    manifest = []
    for path, label in pairs:
        records = process_file(
            path, output_dir,
            augmentations=augmentations,
            target_sr=target_sr,
            duration=duration,
            target_frames=target_frames,
            seed=seed,
            label=label,
        )
        manifest.extend(records)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Batch audio → tensor pipeline")
    parser.add_argument("input_dir", type=Path, help="Folder of .wav files")
    parser.add_argument("output_dir", type=Path, help="Folder for .pt output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = process_folder(args.input_dir, args.output_dir, seed=args.seed)
    print(f"Processed {len(manifest)} tensors → {args.output_dir}")
    for record in manifest:
        print(f"  {record['file']}  shape={record['shape']}  aug={record['augmentation']}")


if __name__ == "__main__":
    main()
