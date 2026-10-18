"""
Phase 4 chord classifier training.

Two modes:
    --mode name   Option B: 6-class chord-name classifier (CrossEntropyLoss)
    --mode notes  Option A: 12-label note-set classifier (BCEWithLogitsLoss)

Usage
-----
    python -m model.chord_train --mode name --epochs 80
    python -m model.chord_train --mode notes --epochs 80
    python -m model.chord_train --mode name --mode notes  # run both

Output saved to output/chords/{name,notes}/:
    best_model.pt   — best checkpoint (lowest val_loss)
    history.json    — per-epoch metrics
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.chord import (
    ChordConfig,
    NUM_NOTE_CLASSES,
    _CHORD_NOTES,
    _NOTE_INDEX,
    chord_name_loss,
    chord_to_note_vector,
    exact_match_accuracy,
    load_chord_name_model,
    load_note_set_model,
    note_set_loss,
    note_set_predictions,
)
from model.dataset import load_splits


def chord_to_note_vector(chord_name: str) -> torch.Tensor:
    """Return a 12-dim binary tensor indicating which notes are in this chord."""
    vec = torch.zeros(NUM_NOTE_CLASSES)
    for note in _CHORD_NOTES[chord_name]:
        vec[_NOTE_INDEX[note]] = 1.0
    return vec


# --- Name mode (Option B) ---

def _train_epoch_name(model, loader, optimiser, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(tensors)
        loss = chord_name_loss(logits, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * tensors.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += tensors.size(0)
    return total_loss / total, correct / total


def _eval_epoch_name(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for tensors, labels in loader:
            tensors, labels = tensors.to(device), labels.to(device)
            logits = model(tensors)
            total_loss += chord_name_loss(logits, labels).item() * tensors.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += tensors.size(0)
    return total_loss / total, correct / total


# --- Notes mode (Option A) ---

def _build_note_targets(label_map: dict[str, int], labels: torch.Tensor) -> torch.Tensor:
    """Convert integer chord labels → (B, 12) binary note vectors."""
    idx_to_chord = {v: k for k, v in label_map.items()}
    vecs = torch.stack([chord_to_note_vector(idx_to_chord[i.item()]) for i in labels])
    return vecs


def _train_epoch_notes(model, loader, optimiser, device, label_map):
    model.train()
    total_loss, exact, total = 0.0, 0, 0
    for tensors, labels in loader:
        tensors = tensors.to(device)
        targets = _build_note_targets(label_map, labels).to(device)
        optimiser.zero_grad()
        logits = model(tensors)
        loss = note_set_loss(logits, targets)
        loss.backward()
        optimiser.step()
        preds = note_set_predictions(logits)
        total_loss += loss.item() * tensors.size(0)
        exact += int(exact_match_accuracy(preds, targets.long()) * tensors.size(0))
        total += tensors.size(0)
    return total_loss / total, exact / total


def _eval_epoch_notes(model, loader, device, label_map):
    model.eval()
    total_loss, exact, total = 0.0, 0, 0
    with torch.no_grad():
        for tensors, labels in loader:
            tensors = tensors.to(device)
            targets = _build_note_targets(label_map, labels).to(device)
            logits = model(tensors)
            total_loss += note_set_loss(logits, targets).item() * tensors.size(0)
            preds = note_set_predictions(logits)
            exact += int(exact_match_accuracy(preds, targets.long()) * tensors.size(0))
            total += tensors.size(0)
    return total_loss / total, exact / total


# --- Shared training loop ---

def run_chord_training(mode: str, config: ChordConfig) -> dict:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.output_dir / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_splits(config.processed_dir, seed=config.seed)
    label_map = train_ds.label_map
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    if mode == "name":
        model = load_chord_name_model(config).to(device)
    else:
        model = load_note_set_model(config).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    acc_label = "exact_match" if mode == "notes" else "val_acc"
    print(f"\n[chord/{mode}] {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"trainable params | {len(train_ds)} train / {len(val_ds)} val")

    history = []
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, config.epochs + 1):
        if mode == "name":
            train_loss, train_acc = _train_epoch_name(model, train_loader, optimiser, device)
            val_loss, val_acc = _eval_epoch_name(model, val_loader, device)
        else:
            train_loss, train_acc = _train_epoch_notes(model, train_loader, optimiser, device, label_map)
            val_loss, val_acc = _eval_epoch_notes(model, val_loader, device, label_map)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, acc_label: val_acc})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {acc_label}={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_map": label_map,
                "mode": mode,
                "epoch": epoch,
                "val_loss": val_loss,
                acc_label: val_acc,
            }, output_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"  early stop at epoch {epoch}")
                break

    json.dump(history, open(output_dir / "history.json", "w"), indent=2)
    best_acc = max(e[acc_label] for e in history)
    print(f"\n[chord/{mode}] best val_loss={best_val_loss:.4f}  best {acc_label}={best_acc:.3f}")
    return {"mode": mode, "best_val_loss": best_val_loss, f"best_{acc_label}": best_acc, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 chord classifier training")
    parser.add_argument("--mode", choices=["name", "notes"], action="append", dest="modes",
                        help="Training mode(s). Pass twice to run both.")
    parser.add_argument("--checkpoint", default="output/transfer/finetune/best_model.pt")
    parser.add_argument("--processed-dir", default="data/processed/chords")
    parser.add_argument("--output-dir", default="output/chords")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modes = args.modes or ["name"]
    config = ChordConfig(
        checkpoint_path=Path(args.checkpoint),
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )

    results = [run_chord_training(mode, config) for mode in modes]

    if len(results) > 1:
        print("\n--- Comparison ---")
        for r in results:
            key = next(k for k in r if k.startswith("best_") and k != "best_val_loss")
            print(f"  {r['mode']:6s}  val_loss={r['best_val_loss']:.4f}  {key}={r[key]:.3f}")


if __name__ == "__main__":
    main()
