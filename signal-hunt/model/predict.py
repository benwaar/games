"""Inference — raw .wav file → class prediction + confidence."""

import argparse
from pathlib import Path

import torch

from model.evaluate import load_checkpoint
from pipeline.features import extract_features
from pipeline.ingest import DEFAULT_DURATION, DEFAULT_SR, ingest


def predict(
    wav_path: Path,
    checkpoint_path: Path = Path("output/best_model.pt"),
) -> dict:
    """
    Run the full pipeline on a .wav file and return a prediction.

    Returns:
        {
            "class": "hum",
            "confidence": 0.923,
            "scores": {"clap": 0.031, "hum": 0.923, "whistle": 0.046}
        }
    """
    wav_path = Path(wav_path)
    model, label_map, cfg = load_checkpoint(Path(checkpoint_path))

    signal, sr = ingest(wav_path, target_sr=DEFAULT_SR, duration=DEFAULT_DURATION)
    tensor = extract_features(signal, sr)       # (1, 128, 65)
    batch = tensor.unsqueeze(0)                 # (1, 1, 128, 65)

    with torch.no_grad():
        logits = model(batch)                   # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0] # (num_classes,)

    inv_map = {v: k for k, v in label_map.items()}
    scores = {inv_map[i]: round(probs[i].item(), 4) for i in range(len(label_map))}
    top_class = max(scores, key=scores.__getitem__)

    return {
        "class": top_class,
        "confidence": scores[top_class],
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sound type from a .wav file")
    parser.add_argument("wav", type=Path, help="Path to .wav file")
    parser.add_argument("--checkpoint", type=Path, default=Path("output/best_model.pt"))
    parser.add_argument("--verbose", action="store_true", help="Show all class scores")
    args = parser.parse_args()

    result = predict(args.wav, args.checkpoint)
    print(f"{result['class']} ({result['confidence']:.1%} confidence)")
    if args.verbose:
        for cls, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            print(f"  {cls:>10}: {score:.1%}")


if __name__ == "__main__":
    main()
