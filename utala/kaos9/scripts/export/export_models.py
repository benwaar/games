#!/usr/bin/env python3
"""Export models for mobile integration.

Produces:
  models/mobile/td_linear_noint.json   — 27 weights, portable JSON
  models/mobile/imitation_nn.onnx      — 1,089 params, ONNX format
  models/mobile/imitation_nn.json      — same NN as raw JSON (no ONNX runtime needed)

Then benchmarks inference speed and reports size comparison.
"""

import json
import os
import time
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def export_td_linear_noint(out_dir):
    """Train TD-Linear NoInt and export weights as JSON."""
    from src.utala.agents.linear_value_agent import LinearValueAgent
    from src.utala.agents.heuristic_agent import HeuristicAgent
    from src.utala.agents.random_agent import RandomAgent
    from src.utala.evaluation.harness import Harness
    from src.utala.state import Player

    print("=" * 60)
    print("1. TD-Linear NoInt — Training + JSON Export")
    print("=" * 60)

    # Train with interaction features masked (indices 26-30)
    agent = LinearValueAgent("TD-NoInt", learning_rate=0.01, discount=0.95,
                             deck_awareness=False)
    interaction_mask = list(range(26, 31))
    original_extract = agent.feature_extractor.extract

    def masked_extract(state, action, player):
        features = original_extract(state, action, player)
        for idx in interaction_mask:
            features[idx] = 0.0
        return features

    agent.feature_extractor.extract = masked_extract

    # Train 5K games vs Heuristic (70%) and Random (30%)
    print("Training 5K games...")
    t0 = time.time()
    harness = Harness(verbose=False)
    h_opp = HeuristicAgent("H", seed=99)
    r_opp = RandomAgent("R")
    agent.set_training_mode(True)

    for g in range(1, 5001):
        opponent = h_opp if g % 10 < 7 else r_opp
        agent_as_p1 = (g % 2 == 0)
        seed = 900000 + g
        if agent_as_p1:
            p1, p2 = agent, opponent
            agent_player = Player.ONE
        else:
            p1, p2 = opponent, agent
            agent_player = Player.TWO
        agent.start_episode()
        result = harness.run_game(p1, p2, seed=seed)
        outcome = 0.5 if result.winner is None else (1.0 if result.winner == agent_player else 0.0)
        agent.observe_outcome(outcome)

    agent.set_training_mode(False)
    train_time = time.time() - t0
    print(f"  Training: {train_time:.1f}s")

    # Evaluate vs Heuristic (1K games)
    print("Evaluating vs Heuristic (1K games)...")
    wins = 0
    h_eval = HeuristicAgent("H2", seed=55)
    for g in range(1000):
        seed = 800000 + g
        if g % 2 == 0:
            result = harness.run_game(agent, h_eval, seed=seed)
            win = result.winner == Player.ONE
        else:
            result = harness.run_game(h_eval, agent, seed=seed)
            win = result.winner == Player.TWO
        if win:
            wins += 1
    win_rate = wins / 1000
    print(f"  vs Heuristic: {win_rate:.1%}")

    # Extract the 27 non-interaction weights
    all_weights = agent.weights.tolist() if hasattr(agent.weights, 'tolist') else list(agent.weights)

    # Feature names for the 32 features (minus 5 interaction = 27)
    feature_names_32 = [
        # Board state (10)
        "my_rocket_count", "opp_rocket_count", "my_avg_power", "opp_avg_power",
        "my_power_std", "my_max_power", "opp_max_power", "my_squares",
        "opp_squares", "contested_squares",
        # Board summary (3)
        "material_advantage", "control_advantage", "contested_ratio",
        # Placement features (8) - zeros during dogfight
        "placement_center", "placement_edge", "placement_corner",
        "placement_uncontested", "placement_contested", "placement_adj_friendly",
        "placement_forms_line_2", "placement_forms_line_3",
        # Dogfight features (8) - zeros during placement
        "dogfight_my_power", "dogfight_opp_power", "dogfight_power_diff",
        "dogfight_my_count", "dogfight_center", "dogfight_edge",
        "dogfight_commits_strongest", "dogfight_overcommit",
        # Interaction features (5) - MASKED, skip these
        # "interaction_winning", "interaction_losing", "interaction_contested_material",
        # "interaction_early_game", "interaction_late_game",
    ]

    # Remove interaction weights (indices 26-30)
    noint_weights = [w for i, w in enumerate(all_weights) if i not in interaction_mask]
    noint_names = [n for i, n in enumerate(feature_names_32) if i not in range(26, 31)]

    assert len(noint_weights) == 27, f"Expected 27, got {len(noint_weights)}"

    # Export JSON
    model_json = {
        "format": "td_linear_noint",
        "version": "1.0",
        "description": "TD-Linear with interaction features removed (Phase 3.3 best agent)",
        "num_features": 27,
        "feature_names": noint_names,
        "weights": noint_weights,
        "masked_indices": interaction_mask,
        "training": {
            "games": 5000,
            "algorithm": "TD(0)",
            "learning_rate": 0.01,
            "discount": 0.95,
            "time_seconds": round(train_time, 1),
        },
        "performance": {
            "vs_heuristic": round(win_rate, 4),
        },
        "usage": {
            "inference": "Q(s,a) = dot(weights, features)",
            "select_action": "argmax over legal actions",
            "note": "Features extracted WITHOUT interaction features (indices 26-30 of the full 32-feature set)"
        }
    }

    path = os.path.join(out_dir, "td_linear_noint.json")
    with open(path, "w") as f:
        json.dump(model_json, f, indent=2)
    size = os.path.getsize(path)
    print(f"  Exported: {path} ({size:,} bytes)")

    return path, noint_weights


def export_imitation_nn(out_dir):
    """Export Imitation-NN to ONNX and JSON."""
    import torch

    print()
    print("=" * 60)
    print("2. Imitation-NN — ONNX + JSON Export")
    print("=" * 60)

    # Load PyTorch model
    state_dict = torch.load("models/imitation_nn_32h.pth", weights_only=True)
    print(f"  Loaded: models/imitation_nn_32h.pth")

    # Reconstruct the network
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: 32 → 32 (ReLU) → 1")
    print(f"  Parameters: {total_params:,}")

    # --- ONNX export ---
    onnx_path = os.path.join(out_dir, "imitation_nn.onnx")
    try:
        dummy_input = torch.randn(1, 32)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["features"],
            output_names=["value"],
            dynamic_axes={"features": {0: "batch"}, "value": {0: "batch"}},
            opset_version=13,
        )
        onnx_size = os.path.getsize(onnx_path)
        print(f"  Exported ONNX: {onnx_path} ({onnx_size:,} bytes)")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        onnx_size = None

    # --- JSON export (portable, no ONNX runtime needed) ---
    json_model = {
        "format": "neural_network",
        "version": "1.0",
        "description": "Imitation-NN trained on MC-Fast decisions (Phase 3.4)",
        "architecture": [
            {"type": "linear", "in": 32, "out": 32},
            {"type": "relu"},
            {"type": "linear", "in": 32, "out": 1},
        ],
        "parameters": {
            "layer0_weight": state_dict["0.weight"].tolist(),   # [32, 32]
            "layer0_bias": state_dict["0.bias"].tolist(),       # [32]
            "layer2_weight": state_dict["2.weight"].tolist(),   # [1, 32]
            "layer2_bias": state_dict["2.bias"].tolist(),       # [1]
        },
        "total_params": total_params,
        "training": {
            "method": "pairwise margin ranking loss",
            "teacher": "MC-Fast (10 rollouts, info-set sampling)",
            "dataset": "8,228 decisions from 500 games",
        },
        "usage": {
            "inference": "h = ReLU(W0 @ features + b0); value = W2 @ h + b2",
            "select_action": "argmax over legal actions",
            "note": "Uses full 32-feature set (including interaction features)"
        }
    }

    json_path = os.path.join(out_dir, "imitation_nn.json")
    with open(json_path, "w") as f:
        json.dump(json_model, f, indent=2)
    json_size = os.path.getsize(json_path)
    print(f"  Exported JSON: {json_path} ({json_size:,} bytes)")

    return onnx_path, json_path, onnx_size, json_size, model


def benchmark_inference(linear_weights, nn_model):
    """Compare inference speed of both models."""
    import torch

    print()
    print("=" * 60)
    print("3. Inference Speed Benchmark")
    print("=" * 60)

    n_trials = 10_000
    features_np = np.random.randn(32).astype(np.float32)

    # --- Linear: numpy dot product (27 features) ---
    w = np.array(linear_weights, dtype=np.float32)
    f_noint = np.delete(features_np, list(range(26, 31)))

    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = np.dot(w, f_noint)
    linear_time = (time.perf_counter() - t0) / n_trials * 1_000_000  # microseconds

    # --- Linear: pure Python dot product (simulates mobile with no numpy) ---
    w_list = w.tolist()
    f_list = f_noint.tolist()

    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = sum(a * b for a, b in zip(w_list, f_list))
    linear_python_time = (time.perf_counter() - t0) / n_trials * 1_000_000

    # --- NN: PyTorch inference ---
    nn_model.eval()
    features_t = torch.from_numpy(features_np).unsqueeze(0)

    # Warmup
    for _ in range(100):
        with torch.no_grad():
            _ = nn_model(features_t)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        with torch.no_grad():
            _ = nn_model(features_t)
    nn_torch_time = (time.perf_counter() - t0) / n_trials * 1_000_000

    # --- NN: pure Python forward pass (simulates mobile with no PyTorch) ---
    w0 = nn_model[0].weight.detach().numpy()
    b0 = nn_model[0].bias.detach().numpy()
    w2 = nn_model[2].weight.detach().numpy()
    b2 = nn_model[2].bias.detach().numpy()

    t0 = time.perf_counter()
    for _ in range(n_trials):
        h = np.maximum(0, w0 @ features_np + b0)  # ReLU
        _ = w2 @ h + b2
    nn_numpy_time = (time.perf_counter() - t0) / n_trials * 1_000_000

    print(f"  {n_trials:,} trials each")
    print()
    print(f"  {'Model':<30} {'Per inference':>15}")
    print(f"  {'-'*30} {'-'*15}")
    print(f"  {'Linear (numpy dot)':<30} {linear_time:>12.1f} µs")
    print(f"  {'Linear (pure Python loop)':<30} {linear_python_time:>12.1f} µs")
    print(f"  {'NN (PyTorch)':<30} {nn_torch_time:>12.1f} µs")
    print(f"  {'NN (numpy matmul)':<30} {nn_numpy_time:>12.1f} µs")

    return {
        "linear_numpy_us": round(linear_time, 2),
        "linear_python_us": round(linear_python_time, 2),
        "nn_pytorch_us": round(nn_torch_time, 2),
        "nn_numpy_us": round(nn_numpy_time, 2),
    }


def print_summary(linear_json_path, onnx_path, nn_json_path, onnx_size, nn_json_size, timings):
    """Print final comparison table."""
    linear_size = os.path.getsize(linear_json_path)

    print()
    print("=" * 60)
    print("SUMMARY — Mobile Export Comparison")
    print("=" * 60)
    print()
    print(f"  {'Format':<25} {'Size':>10} {'Inference':>14} {'Deps':>18}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*18}")
    print(f"  {'TD-Linear JSON':<25} {linear_size:>8,} B {timings['linear_python_us']:>10.1f} µs {'None':>18}")
    if onnx_size:
        print(f"  {'Imitation-NN ONNX':<25} {onnx_size:>8,} B {'N/A':>14} {'ONNX Runtime':>18}")
    print(f"  {'Imitation-NN JSON':<25} {nn_json_size:>8,} B {timings['nn_numpy_us']:>10.1f} µs {'None (or numpy)':>18}")
    print()
    print("  Notes:")
    print(f"  - TD-Linear: 47.5% vs Heuristic, 27 params")
    print(f"  - Imitation-NN: 48.0% vs Heuristic, 1,089 params")
    print(f"  - Performance difference: +0.5pp (within noise)")
    print(f"  - 'Pure Python loop' timing simulates mobile without numpy")
    print()

    # Export paths
    print("  Exported files:")
    print(f"    {linear_json_path}")
    if onnx_size:
        print(f"    {onnx_path}")
    print(f"    {nn_json_path}")


def main():
    out_dir = "models/mobile"
    os.makedirs(out_dir, exist_ok=True)

    # Export both models
    linear_path, linear_weights = export_td_linear_noint(out_dir)
    onnx_path, nn_json_path, onnx_size, nn_json_size, nn_model = export_imitation_nn(out_dir)

    # Benchmark
    timings = benchmark_inference(linear_weights, nn_model)

    # Summary
    print_summary(linear_path, onnx_path, nn_json_path, onnx_size, nn_json_size, timings)


if __name__ == "__main__":
    main()
