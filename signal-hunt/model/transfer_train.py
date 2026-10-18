"""
Phase 3 transfer learning training loop.

Usage
-----
    # Frozen backbone (head only):
    python -m model.transfer_train

    # Full fine-tune:
    python -m model.transfer_train --no-freeze

    # Compare both and save results side-by-side:
    python -m model.transfer_train --compare

Output saved to output/transfer/{frozen,finetune}/:
    best_model.pt    — best checkpoint (lowest val_loss)
    history.json     — per-epoch metrics
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import TransferConfig
from model.dataset import load_splits
from model.transfer import frozen_param_count, load_transfer_model, trainable_param_count
from model.train import run_training_loop


def run_transfer(config: TransferConfig, label: str) -> dict:
    """Single transfer learning run. Returns history dict."""
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_splits(config.processed_dir, seed=config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = load_transfer_model(config)
    criterion = nn.CrossEntropyLoss()

    # Only pass trainable parameters to the optimiser — frozen layers must not appear.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    frozen = frozen_param_count(model)
    trainable_n = trainable_param_count(model)
    print(f"\n[{label}] {trainable_n:,} trainable / {frozen:,} frozen params | "
          f"{len(train_ds)} train / {len(val_ds)} val | {config.num_classes} classes")

    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}
    result = run_training_loop(
        model, train_loader, val_loader, criterion, optimiser, scheduler,
        label_map=train_ds.label_map,
        checkpoint_path=config.output_dir / "best_model.pt",
        epochs=config.epochs,
        patience=config.patience,
        config_dict=config_dict,
    )
    result["label"] = label
    return result


def compare(checkpoint_path: Path = Path("output/best_model.pt")) -> None:
    """Run frozen then fine-tune, print a side-by-side summary."""
    frozen_cfg = TransferConfig(
        checkpoint_path=checkpoint_path,
        output_dir=Path("output/transfer/frozen"),
        freeze_backbone=True,
    )
    finetune_cfg = TransferConfig(
        checkpoint_path=checkpoint_path,
        output_dir=Path("output/transfer/finetune"),
        freeze_backbone=False,
    )

    frozen_result = run_transfer(frozen_cfg, label="frozen")
    finetune_result = run_transfer(finetune_cfg, label="finetune")

    print("\n--- Comparison ---")
    for r in [frozen_result, finetune_result]:
        best_acc = max(e["val_acc"] for e in r["history"])
        print(f"  {r['label']:10s} val_loss={r['best_val_loss']:.4f}  best_val_acc={best_acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 transfer learning")
    parser.add_argument("--checkpoint", default="output/best_model.pt",
                        help="Path to Phase 2 checkpoint")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Fine-tune entire model (default: freeze backbone)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both frozen and fine-tune, print comparison")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.compare:
        compare(Path(args.checkpoint))
        return

    freeze = not args.no_freeze
    output_dir = Path("output/transfer/frozen" if freeze else "output/transfer/finetune")
    config = TransferConfig(
        checkpoint_path=Path(args.checkpoint),
        output_dir=output_dir,
        freeze_backbone=freeze,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    run_transfer(config, label="frozen" if freeze else "finetune")


if __name__ == "__main__":
    main()
