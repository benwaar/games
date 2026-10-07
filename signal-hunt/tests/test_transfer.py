"""Tests for Phase 3 transfer learning — model/transfer.py."""

import torch
import pytest
from pathlib import Path

from model.cnn import SoundClassifier
from model.config import TransferConfig
from model.transfer import (
    load_transfer_model,
    frozen_param_count,
    trainable_param_count,
    checkpoint_label_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PHASE2_LABEL_MAP = {"clap": 0, "hum": 1, "whistle": 2}  # 3 classes, alphabetical


def _make_fake_checkpoint(tmp_path: Path, num_classes: int = 3) -> Path:
    """Write a minimal Phase 2 checkpoint to disk and return its path."""
    model = SoundClassifier(num_classes=num_classes, dropout=0.3)
    ckpt = {
        "model_state": model.state_dict(),
        "label_map": PHASE2_LABEL_MAP,
        "config": {"num_classes": num_classes},
        "epoch": 41,
        "val_loss": 0.302,
        "val_acc": 0.971,
    }
    path = tmp_path / "best_model.pt"
    torch.save(ckpt, path)
    return path


# ---------------------------------------------------------------------------
# load_transfer_model
# ---------------------------------------------------------------------------

class TestLoadTransferModel:
    def test_output_shape_12_classes(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(
            checkpoint_path=ckpt,
            num_classes=12,
            freeze_backbone=False,
        )
        model = load_transfer_model(config)
        x = torch.randn(4, 1, 128, 65)
        out = model(x)
        assert out.shape == (4, 12)

    def test_head_replaced_not_appended(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12)
        model = load_transfer_model(config)
        # Last layer in classifier must be Linear(32, 12)
        last = model.classifier[-1]
        assert isinstance(last, torch.nn.Linear)
        assert last.out_features == 12
        assert last.in_features == 32

    def test_freeze_backbone_freezes_conv(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=True)
        model = load_transfer_model(config)
        for param in model.conv_blocks.parameters():
            assert not param.requires_grad

    def test_freeze_backbone_leaves_head_trainable(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=True)
        model = load_transfer_model(config)
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_no_freeze_all_trainable(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=False)
        model = load_transfer_model(config)
        for param in model.parameters():
            assert param.requires_grad

    def test_backbone_weights_loaded(self, tmp_path):
        """Conv block weights from Phase 2 should survive the head swap."""
        ckpt_path = _make_fake_checkpoint(tmp_path)
        # Record the original conv weights.
        original = SoundClassifier(num_classes=3)
        original.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
        )
        orig_weight = original.conv_blocks[0].weight.clone()

        config = TransferConfig(checkpoint_path=ckpt_path, num_classes=12)
        transferred = load_transfer_model(config)
        transferred_weight = transferred.conv_blocks[0].weight

        assert torch.allclose(orig_weight, transferred_weight)

    def test_arbitrary_num_classes(self, tmp_path):
        for n in [2, 7, 24, 88]:
            ckpt = _make_fake_checkpoint(tmp_path)
            config = TransferConfig(checkpoint_path=ckpt, num_classes=n)
            model = load_transfer_model(config)
            assert model.classifier[-1].out_features == n


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

class TestParamCounting:
    def test_frozen_count_nonzero_when_frozen(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=True)
        model = load_transfer_model(config)
        assert frozen_param_count(model) > 0

    def test_frozen_count_zero_when_not_frozen(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=False)
        model = load_transfer_model(config)
        assert frozen_param_count(model) == 0

    def test_trainable_plus_frozen_equals_total(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=True)
        model = load_transfer_model(config)
        total = sum(p.numel() for p in model.parameters())
        assert trainable_param_count(model) + frozen_param_count(model) == total


# ---------------------------------------------------------------------------
# checkpoint_label_map
# ---------------------------------------------------------------------------

class TestCheckpointLabelMap:
    def test_returns_phase2_label_map(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        label_map = checkpoint_label_map(ckpt)
        assert label_map == PHASE2_LABEL_MAP

    def test_returns_dict(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        assert isinstance(checkpoint_label_map(ckpt), dict)


# ---------------------------------------------------------------------------
# Gradient flow sanity check
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_frozen_backbone_grads_zero(self, tmp_path):
        """A backward pass must not produce gradients for frozen conv layers."""
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=True)
        model = load_transfer_model(config)

        x = torch.randn(2, 1, 128, 65)
        y = torch.randint(0, 12, (2,))
        loss = torch.nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        for param in model.conv_blocks.parameters():
            assert param.grad is None

    def test_head_receives_grads(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path)
        config = TransferConfig(checkpoint_path=ckpt, num_classes=12, freeze_backbone=True)
        model = load_transfer_model(config)

        x = torch.randn(2, 1, 128, 65)
        y = torch.randint(0, 12, (2,))
        loss = torch.nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        for param in model.classifier.parameters():
            assert param.grad is not None
