"""Evaluation — load best checkpoint, run test set, generate report and plots."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model.cnn import SoundClassifier
from model.dataset import load_splits
from model.train import evaluate


def load_checkpoint(checkpoint_path: Path) -> tuple[SoundClassifier, dict, dict]:
    """Load model weights, label_map and config from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    cfg = ckpt["config"]
    model = SoundClassifier(num_classes=cfg["num_classes"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["label_map"], cfg


def collect_predictions(
    model: SoundClassifier,
    loader: DataLoader,
) -> tuple[list[int], list[int]]:
    """Run model over loader, return (predictions, true_labels)."""
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).argmax(dim=1).tolist())
            labels.extend(y.tolist())
    return preds, labels


def print_metrics(
    preds: list[int],
    labels: list[int],
    label_map: dict[str, int],
    output_dir: Path,
) -> str:
    """Print and save classification report."""
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    report = classification_report(labels, preds, target_names=class_names)
    print(report)
    report_path = output_dir / "eval_report.txt"
    report_path.write_text(report)
    print(f"Report saved → {report_path}")
    return report


def plot_confusion_matrix(
    preds: list[int],
    labels: list[int],
    label_map: dict[str, int],
    output_dir: Path,
) -> None:
    """Save confusion matrix heatmap to explainers/images/."""
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix — Test Set",
    )
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix → {out_path}")


def plot_loss_curves(history_path: Path, output_dir: Path) -> None:
    """Save train/val loss curves to explainers/images/."""
    history = json.loads(history_path.read_text())
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train loss", linewidth=1.5)
    ax.plot(epochs, val_loss, label="val loss", linewidth=1.5)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Train vs Val Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Loss curves → {out_path}")


def run_evaluation(
    checkpoint_path: Path,
    processed_dir: Path,
    images_dir: Path,
    output_dir: Path,
) -> dict:
    """Full evaluation pipeline. Returns summary dict."""
    model, label_map, cfg = load_checkpoint(checkpoint_path)

    _, _, test_ds = load_splits(processed_dir, seed=cfg["seed"])
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest set ({len(test_ds)} samples): loss={test_loss:.4f} | acc={test_acc:.3f}\n")

    preds, labels = collect_predictions(model, test_loader)
    print_metrics(preds, labels, label_map, output_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(preds, labels, label_map, images_dir)

    history_path = output_dir / "history.json"
    if history_path.exists():
        plot_loss_curves(history_path, images_dir)

    return {"test_loss": test_loss, "test_acc": test_acc, "num_samples": len(test_ds)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on test set")
    parser.add_argument("--checkpoint", type=Path, default=Path("output/best_model.pt"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--images-dir", type=Path, default=Path("explainers/images"))
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.processed_dir, args.images_dir, args.output_dir)


if __name__ == "__main__":
    main()
