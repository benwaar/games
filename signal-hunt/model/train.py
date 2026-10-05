"""Training loop for Phase 2 sound type classification."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import TrainConfig
from model.cnn import SoundClassifier
from model.dataset import load_splits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
) -> tuple[float, float]:
    """One full pass over the training set. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        optimiser.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Evaluate on a DataLoader. Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


def train(config: TrainConfig) -> dict:
    """Full training run. Returns history dict."""
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_splits(config.processed_dir, seed=config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = SoundClassifier(num_classes=config.num_classes, dropout=config.dropout)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    history = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    checkpoint_path = config.output_dir / "best_model.pt"

    print(f"Training {model.num_parameters():,} params | "
          f"{len(train_ds)} train / {len(val_ds)} val samples")
    print(f"{'Epoch':>5} | {'train_loss':>10} | {'val_loss':>8} | {'val_acc':>7} | {'lr':>8}")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimiser)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        lr = optimiser.param_groups[0]["lr"]

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_acc": val_acc, "lr": lr, "elapsed": time.time() - t0}
        history.append(row)
        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>7.3f} | {lr:>8.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_map": train_ds.label_map,
                "config": config.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stop at epoch {epoch} (no improvement for {config.patience} epochs)")
                break

    history_path = config.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"\nBest val_loss={best_val_loss:.4f} → {checkpoint_path}")
    return {"history": history, "best_val_loss": best_val_loss, "checkpoint": str(checkpoint_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SoundClassifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    train(config)


if __name__ == "__main__":
    main()
