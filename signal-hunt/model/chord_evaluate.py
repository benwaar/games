"""
Phase 4 chord evaluation — confusion matrices, per-chord F1, threshold sweep.

Usage
-----
    python -m model.chord_evaluate --mode name
    python -m model.chord_evaluate --mode notes
    python -m model.chord_evaluate --mode notes --thresholds 0.3 0.5 0.7
    python -m model.chord_evaluate --mode name --mode notes
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from model.chord import (
    ChordConfig,
    _CHORD_NOTES,
    _NOTE_INDEX,
    _SORTED_NOTES,
    chord_to_note_vector,
    load_chord_name_model,
    load_note_set_model,
    note_set_predictions,
    exact_match_accuracy,
)
from model.dataset import load_splits


def _collect_name(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())
    return all_preds, all_labels


def _collect_notes(model, loader, device, label_map, threshold=0.5):
    idx_to_chord = {v: k for k, v in label_map.items()}
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds = note_set_predictions(logits, threshold=threshold)
            targets = torch.stack([chord_to_note_vector(idx_to_chord[i.item()]) for i in y])
            all_preds.append(preds.cpu())
            all_targets.append(targets.long().cpu())
    return torch.cat(all_preds), torch.cat(all_targets)


def evaluate_name(config: ChordConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ds = load_splits(config.processed_dir, seed=config.seed)
    label_map = test_ds.label_map
    idx_to_label = {v: k for k, v in label_map.items()}
    chord_names = [idx_to_label[i] for i in range(len(label_map))]
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    ckpt_path = config.output_dir / "name" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_chord_name_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, labels = _collect_name(model, loader, device)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    print(f"\n=== Chord-name classifier ===")
    print(f"Test accuracy: {acc:.1%}  ({sum(p==l for p,l in zip(preds,labels))}/{len(labels)})\n")
    print(classification_report(labels, preds, target_names=chord_names))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(chord_names)))
    ax.set_yticks(range(len(chord_names)))
    ax.set_xticklabels(chord_names, rotation=45, ha="right")
    ax.set_yticklabels(chord_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Chord-name classifier — Confusion matrix (test set)")
    plt.colorbar(im)
    for i in range(len(chord_names)):
        for j in range(len(chord_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    fig.tight_layout()
    out = output_dir / "chord_confusion_name.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Confusion matrix saved to {out}")

    _check_musical_confusions(cm, chord_names)


def _check_musical_confusions(cm: np.ndarray, chord_names: list[str]) -> None:
    """Print which confusions are musically sensible (shared notes → more confusion)."""
    print("\nConfusion analysis (off-diagonal cells):")
    n = len(chord_names)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                a, b = chord_names[i], chord_names[j]
                shared = set(_CHORD_NOTES.get(a, [])) & set(_CHORD_NOTES.get(b, []))
                pairs.append((cm[i, j], a, b, shared))
    for count, a, b, shared in sorted(pairs, reverse=True):
        shared_str = f"share {', '.join(sorted(shared))}" if shared else "no shared notes"
        print(f"  {a} → {b}: {count} times  ({shared_str})")


def evaluate_notes(config: ChordConfig, output_dir: Path, thresholds: list[float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ds = load_splits(config.processed_dir, seed=config.seed)
    label_map = test_ds.label_map
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    ckpt_path = config.output_dir / "notes" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_note_set_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])

    print(f"\n=== Note-set classifier ===")

    # Threshold sweep
    print("\nThreshold sweep:")
    print(f"  {'Threshold':>10}  {'Exact-match':>12}  {'Per-note F1':>12}")
    best_threshold, best_f1 = 0.5, 0.0
    for thresh in thresholds:
        preds, targets = _collect_notes(model, loader, device, label_map, threshold=thresh)
        em = exact_match_accuracy(preds, targets)
        per_note_f1 = f1_score(targets.numpy(), preds.numpy(), average="macro", zero_division=0)
        print(f"  {thresh:>10.1f}  {em:>12.1%}  {per_note_f1:>12.3f}")
        if per_note_f1 > best_f1:
            best_f1, best_threshold = per_note_f1, thresh

    print(f"\nBest threshold by per-note F1: {best_threshold}")

    # Detailed per-note report at best threshold
    preds, targets = _collect_notes(model, loader, device, label_map, threshold=best_threshold)
    print(f"\nPer-note classification report (threshold={best_threshold}):")
    # Only report notes that appear in the 6 chords
    active_notes = sorted({n for notes in _CHORD_NOTES.values() for n in notes},
                          key=lambda n: _NOTE_INDEX[n])
    active_indices = [_NOTE_INDEX[n] for n in active_notes]
    t_sub = targets[:, active_indices].numpy()
    p_sub = preds[:, active_indices].numpy()
    print(classification_report(t_sub, p_sub, target_names=active_notes, zero_division=0))

    # Per-note F1 bar chart
    f1_per_note = f1_score(t_sub, p_sub, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(active_notes, f1_per_note, color="#6c8ebf", edgecolor="white")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Note-set classifier — Per-note F1 (threshold={best_threshold}, test set)")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="#aaa", linewidth=0.8, linestyle=":")
    for bar, val in zip(bars, f1_per_note):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = output_dir / "chord_per_note_f1.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"\nPer-note F1 chart saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 chord evaluation")
    parser.add_argument("--mode", choices=["name", "notes"], action="append", dest="modes")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--checkpoint", default="output/transfer/finetune/best_model.pt")
    parser.add_argument("--processed-dir", default="data/processed/chords")
    parser.add_argument("--output-dir", default="output/chords")
    parser.add_argument("--images-dir", default="explainers/images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modes = args.modes or ["name", "notes"]
    config = ChordConfig(
        checkpoint_path=Path(args.checkpoint),
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
    images_dir = Path(args.images_dir)

    if "name" in modes:
        evaluate_name(config, images_dir)
    if "notes" in modes:
        evaluate_notes(config, images_dir, args.thresholds)


if __name__ == "__main__":
    main()
