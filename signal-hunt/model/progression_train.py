"""
Phase 4b progression training.

Usage
-----
    python -m model.progression_train --epochs 100
    python -m model.progression_train --epochs 100 --lr 1e-4

Output: output/progressions/best_model.pt, history.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model.progression import ProgressionConfig, load_progression_model


class ProgressionDataset(Dataset):
    """Load (tensor, label_int) pairs from a progression manifest."""

    def __init__(self, processed_dir: Path) -> None:
        manifest = json.load(open(processed_dir / "manifest.json"))
        labels = sorted({m["label"] for m in manifest})
        self.label_map = {l: i for i, l in enumerate(labels)}
        self.items = [
            (processed_dir / m["file"], self.label_map[m["label"]])
            for m in manifest
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        return torch.load(path, weights_only=True), label


def run_progression_training(config: ProgressionConfig) -> dict:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ds = ProgressionDataset(config.processed_dir)
    label_map = ds.label_map
    n = len(ds)
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(config.seed))

    # Progression tensors have variable T — pad to same width in batch
    def collate(batch):
        tensors, labels = zip(*batch)
        max_t = max(t.shape[2] for t in tensors)
        padded = [torch.nn.functional.pad(t, (0, max_t - t.shape[2])) for t in tensors]
        return torch.stack(padded), torch.tensor(labels)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate)

    model = load_progression_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[progression] {total_params:,} trainable params | "
          f"{len(train_ds)} train / {len(val_ds)} val | "
          f"{config.num_progressions} classes: {list(label_map)}")

    history = []
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(y)

        tl = train_loss / train_total
        ta = train_correct / train_total
        vl = val_loss / val_total
        va = val_correct / val_total
        scheduler.step(vl)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_acc": va})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={va:.3f}")

        if vl < best_val_loss:
            best_val_loss = vl
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_map": label_map,
                "epoch": epoch,
                "val_loss": vl,
                "val_acc": va,
            }, config.output_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"  early stop at epoch {epoch}")
                break

    json.dump(history, open(config.output_dir / "history.json", "w"), indent=2)
    best_acc = max(e["val_acc"] for e in history)
    print(f"\n[progression] best val_loss={best_val_loss:.4f}  best val_acc={best_acc:.3f}")
    return {"best_val_loss": best_val_loss, "best_val_acc": best_acc, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4b progression training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", default="data/processed/progressions")
    parser.add_argument("--output-dir", default="output/progressions")
    args = parser.parse_args()

    config = ProgressionConfig(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        lr=args.lr,
        gru_hidden=args.gru_hidden,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        epochs=args.epochs,
    )
    run_progression_training(config)


if __name__ == "__main__":
    main()
