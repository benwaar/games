#!/usr/bin/env python3
"""Export Phase 5 models to ONNX for Flutter."""

import sys
import torch
import onnx
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")

from utala.deep_learning.dqn_agent import DQNAgent

sys.path.insert(0, "scripts/train/variant_a")
from distill_dqn import TinyImitationNN

OUTPUT_DIR = Path("models/onnx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def export_full_dqn():
    """Export Full DQN (80 -> 128 -> 128 -> 95) to ONNX."""
    print("=== Full DQN ===")

    checkpoint_path = "results/dqn_v2d/dqn_v2_best.pth"
    if not Path(checkpoint_path).exists():
        print(f"ERROR: {checkpoint_path} not found")
        return False

    agent = DQNAgent.load(checkpoint_path)
    agent.set_training(False)
    model = agent.q_network
    model.eval()

    print(f"  state_dim={model.state_dim}, action_dim={model.action_dim}, hidden={model.hidden_dim}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randn(1, 80)

    output_path = str(OUTPUT_DIR / "dqn_full.onnx")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["state_features"],
        output_names=["q_values"],
        dynamic_axes={
            "state_features": {0: "batch_size"},
            "q_values": {0: "batch_size"},
        },
    )
    print(f"  Exported to {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX validation passed")

    verify_match(model, output_path, state_dim=80)
    return True


def export_tiny_nn():
    """Export Distilled NN (80 -> 32 -> 95) to ONNX."""
    print("\n=== Distilled NN (TinyNN) ===")

    checkpoint_path = "results/distill_v1/tiny_nn.pth"
    if not Path(checkpoint_path).exists():
        alt = "models/imitation_nn_32h.pth"
        if Path(alt).exists():
            checkpoint_path = alt
        else:
            print(f"ERROR: neither {checkpoint_path} nor {alt} found")
            return False

    model = TinyImitationNN(state_dim=80, hidden_dim=32, action_dim=95)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    print(f"  Architecture: 80 -> 32 -> 95")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randn(1, 80)

    output_path = str(OUTPUT_DIR / "tiny_nn.onnx")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["state_features"],
        output_names=["logits"],
        dynamic_axes={
            "state_features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"  Exported to {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX validation passed")

    verify_match(model, output_path, state_dim=80)
    return True


def verify_match(pytorch_model, onnx_path, state_dim=80, n_tests=5):
    """Verify ONNX output matches PyTorch output on random inputs."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    max_diff = 0.0
    for _ in range(n_tests):
        test_input = np.random.randn(1, state_dim).astype(np.float32)

        with torch.no_grad():
            pt_out = pytorch_model(torch.from_numpy(test_input)).numpy()

        onnx_out = session.run(None, {input_name: test_input})[0]

        diff = np.abs(pt_out - onnx_out).max()
        max_diff = max(max_diff, diff)

    print(f"  Max output difference (PyTorch vs ONNX): {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  MATCH: outputs are equivalent")
    else:
        print("  WARNING: outputs diverge — check model loading")


if __name__ == "__main__":
    ok1 = export_full_dqn()
    ok2 = export_tiny_nn()

    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("Both models exported successfully.")
        print(f"\nCopy to Flutter project:")
        print(f"  cp models/onnx/dqn_full.onnx  <flutter>/assets/models/")
        print(f"  cp models/onnx/tiny_nn.onnx   <flutter>/assets/models/")
    else:
        print("Some exports failed — check errors above.")
    print("=" * 60)
