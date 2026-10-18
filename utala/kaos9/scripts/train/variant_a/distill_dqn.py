#!/usr/bin/env python3
"""
Phase 5.2 — Distill DQN into Production Model

Steps:
  1. Load improved DQN teacher  (results/dqn_v2d/dqn_v2_best.pth)
  2. Generate demonstration dataset: teacher plays games, record (state_80, action)
  3. Train student models via behavioral cloning (cross-entropy):
       - TinyNN  : 80 → 32 → 95  (~5.7K params, 1 hidden layer)
       - Linear  : 80 → 95        (~7.7K params, no hidden layer — fastest inference)
  4. Evaluate all students vs Heuristic + Random (200-game eval)
  5. Export best model weights as JSON  (ready for Flutter in 5.4)

Pass criteria: student achieves >=45% vs Heuristic at <5K params.
              (TinyNN just clears 5K; Linear is included for speed comparison.)

Dataset is saved to disk so re-runs can skip generation with --skip-gen.
Usage:
  python scripts/train/variant_a/distill_dqn.py
  python scripts/train/variant_a/distill_dqn.py --skip-gen   # reuse saved dataset
"""

import sys
import time
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, 'src')

from utala.deep_learning.dqn_agent import DQNAgent
from utala.learning.dqn_features import DQNFeatureExtractor
from utala.agents.base import Agent
from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.evaluation.harness import Harness
from utala.state import GameConfig, Player

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEACHER_PATH = Path("results/dqn_v2d/dqn_v2_best.pth")
OUTPUT_DIR   = Path("results/distill_v1")
DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"

CONFIG        = GameConfig(fixed_dogfight_order=False)
STATE_DIM     = 80
ACTION_DIM    = 95

# Dataset generation
GEN_GAMES     = 5000
GEN_SEED_BASE = 800000

# Training
TRAIN_EPOCHS  = 50
BATCH_SIZE    = 256
LR            = 0.001

# Evaluation
EVAL_GAMES    = 200
EVAL_SEED     = 900000


# ---------------------------------------------------------------------------
# Step 1: Recording wrapper — captures teacher decisions without interfering
# ---------------------------------------------------------------------------

class RecordingDQNAgent:
    """
    Wraps a DQN agent in eval mode and records every (state_features, action)
    decision. The DQN's own feature extractor is used so features are identical
    to what the teacher saw.
    """

    def __init__(self, dqn: DQNAgent):
        self.dqn     = dqn
        self.dqn.set_training(False)
        self.name    = dqn.name
        self.dataset: list[dict] = []

    def select_action(self, state, legal_actions, player):
        # Extract features with the same extractor the teacher uses
        features = self.dqn.feature_extractor.extract(state, player)
        # Teacher chooses greedy action (epsilon=0, training=False)
        action   = self.dqn.select_action(state, legal_actions, player)
        self.dataset.append({'features': features.tolist(), 'action': action})
        return action

    def game_start(self, player, seed=None):
        self.dqn.game_start(player, seed)

    def game_end(self, state, winner):
        pass   # DQN.game_end is a no-op in eval mode anyway


def generate_dataset(teacher_path: Path, num_games: int, seed_base: int) -> list[dict]:
    """Play games with the DQN teacher and collect every decision."""
    if not teacher_path.exists():
        sys.exit(f"ERROR: teacher checkpoint not found at {teacher_path}\n"
                 f"       Run Phase 5.1 training first.")

    print(f"Loading teacher from {teacher_path} ...")
    dqn = DQNAgent.load(str(teacher_path))
    print(f"  Architecture: {dqn.state_dim}-dim state, {dqn.action_dim} actions, "
          f"hidden={dqn.q_network.hidden_dim}")
    print(f"  Teacher params: {sum(p.numel() for p in dqn.q_network.parameters()):,}")
    print()

    recorder  = RecordingDQNAgent(dqn)
    heuristic = HeuristicAgent("Heuristic-Gen", seed=42, config=CONFIG)
    rand_opp  = RandomAgent("Random-Gen", seed=42)
    harness   = Harness(verbose=False, config=CONFIG)

    random.seed(42)
    t0 = time.time()

    for g in range(num_games):
        seed     = seed_base + g
        # 70% vs Heuristic, 30% vs Random (mirrors teacher's training opponent mix)
        opponent = heuristic if random.random() < 0.7 else rand_opp
        as_p1    = (g % 2 == 0)

        if as_p1:
            harness.run_game(recorder, opponent, seed=seed)
        else:
            harness.run_game(opponent, recorder, seed=seed)

        if (g + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{g+1}/{num_games}] {(g+1)/elapsed:.1f} g/s | "
                  f"{len(recorder.dataset)} decisions recorded")

    elapsed = time.time() - t0
    print(f"  Done: {len(recorder.dataset)} decisions from {num_games} games ({elapsed:.0f}s)")
    return recorder.dataset


def save_dataset(dataset: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')
    print(f"Dataset saved to {path}  ({len(dataset)} examples)")


def load_dataset(path: Path) -> list[dict]:
    dataset = []
    with open(path) as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    print(f"Loaded {len(dataset)} examples from {path}")
    return dataset


# ---------------------------------------------------------------------------
# Step 2: Student model definitions
# ---------------------------------------------------------------------------

class TinyImitationNN(nn.Module):
    """
    Tiny 2-layer NN: 80 → 32 → 95.
    Input:  80-dim DQN state features
    Output: 95 logits (one per action; argmax over legal actions at inference)

    Params: 80×32 + 32 + 32×95 + 95 = 5,727
    """

    def __init__(self, state_dim: int = 80, hidden_dim: int = 32, action_dim: int = 95):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class LinearImitation(nn.Module):
    """
    Single linear layer: 80 → 95.
    No hidden layer — fastest possible inference, useful Pareto reference point.

    Params: 80×95 + 95 = 7,695
    """

    def __init__(self, state_dim: int = 80, action_dim: int = 95):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Step 3: Behavioral cloning training
# ---------------------------------------------------------------------------

def train_student(
    model: nn.Module,
    dataset: list[dict],
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    label: str = "student",
) -> nn.Module:
    """Train model via cross-entropy: predict teacher's action from state features."""
    X = torch.tensor([ex['features'] for ex in dataset], dtype=torch.float32)
    Y = torch.tensor([ex['action']   for ex in dataset], dtype=torch.long)

    n        = len(dataset)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training {label}  ({n_params:,} params, {n} examples, {epochs} epochs) ...")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    t0        = time.time()

    for epoch in range(epochs):
        perm       = torch.randperm(n)
        total_loss = 0.0
        batches    = 0

        for i in range(0, n, batch_size):
            idx  = perm[i : i + batch_size]
            logits = model(X[idx])
            loss   = loss_fn(logits, Y[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches    += 1

        avg_loss = total_loss / batches
        with torch.no_grad():
            preds    = model(X).argmax(dim=1)
            accuracy = (preds == Y).float().mean().item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  acc={accuracy:.1%}")

    print(f"  Done ({time.time()-t0:.0f}s)")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Step 4: Agent wrappers for evaluation
# ---------------------------------------------------------------------------

class ImitationNNAgent(Agent):
    """
    Production-ready agent backed by TinyImitationNN (or any nn.Module with the
    same 80-dim input / 95-dim output signature).

    Uses DQNFeatureExtractor — same 80-dim state representation as the teacher.
    Selects argmax of logits over legal actions.
    """

    def __init__(self, name: str, model: nn.Module, config: GameConfig | None = None):
        super().__init__(name)
        self.model             = model
        self.model.eval()
        self.feature_extractor = DQNFeatureExtractor(config=config or CONFIG)
        self.action_dim        = ACTION_DIM

    def select_action(self, state, legal_actions: list[int], player: Player) -> int:
        features     = self.feature_extractor.extract(state, player)
        features_t   = torch.FloatTensor(features)
        legal_mask   = torch.zeros(self.action_dim, dtype=torch.bool)
        legal_mask[legal_actions] = True

        with torch.no_grad():
            logits = self.model(features_t)
            logits[~legal_mask] = float('-inf')
            action = logits.argmax().item()

        return int(action)


# ---------------------------------------------------------------------------
# Step 5: Evaluation helpers
# ---------------------------------------------------------------------------

class TimingWrapper:
    """Measures per-call inference time for an agent."""

    def __init__(self, agent):
        self.agent       = agent
        self.name        = agent.name
        self.total_time  = 0.0
        self.total_calls = 0

    def select_action(self, state, legal_actions, player):
        t0     = time.perf_counter()
        action = self.agent.select_action(state, legal_actions, player)
        self.total_time  += time.perf_counter() - t0
        self.total_calls += 1
        return action

    def game_start(self, player, seed=None):
        if hasattr(self.agent, 'game_start'):
            self.agent.game_start(player, seed)

    def game_end(self, state, winner):
        if hasattr(self.agent, 'game_end'):
            self.agent.game_end(state, winner)

    @property
    def avg_ms(self) -> float:
        return (self.total_time / self.total_calls * 1000) if self.total_calls else 0.0


def evaluate(agent, num_games: int = EVAL_GAMES, seed_base: int = EVAL_SEED) -> dict:
    """Evaluate agent vs Heuristic and Random.  Returns {opp: win_rate}."""
    harness = Harness(verbose=False, config=CONFIG)
    results = {}

    for opp_name, opp in [
        ("Heuristic", HeuristicAgent("Heuristic-Eval", config=CONFIG)),
        ("Random",    RandomAgent("Random-Eval")),
    ]:
        timed = TimingWrapper(agent)
        wins  = 0

        for i in range(num_games):
            seed    = seed_base + i
            as_p1   = (i < num_games // 2)

            if as_p1:
                result = harness.run_game(timed, opp, seed=seed)
                if result.winner == Player.ONE:
                    wins += 1
            else:
                result = harness.run_game(opp, timed, seed=seed)
                if result.winner == Player.TWO:
                    wins += 1

        win_rate = wins / num_games
        results[opp_name] = {
            'win_rate': win_rate,
            'wins':     wins,
            'avg_ms':   timed.avg_ms,
        }
        print(f"    vs {opp_name:<12} {win_rate:5.1%}  ({wins}/{num_games})  "
              f"{timed.avg_ms:.3f} ms/decision")

    return results


# ---------------------------------------------------------------------------
# Step 6: JSON export
# ---------------------------------------------------------------------------

def export_model_json(model: nn.Module, path: Path, metadata: dict):
    """
    Export model weights to JSON for Flutter integration (Phase 5.4).

    Format:
      {
        "metadata": { ... },
        "layers": [
          {"name": "fc1", "weight": [[...]], "bias": [...]},
          ...
        ]
      }
    """
    layers = []
    for name, param in model.named_parameters():
        layers.append({
            'name':  name,
            'shape': list(param.shape),
            'data':  param.detach().numpy().tolist(),
        })

    payload = {'metadata': metadata, 'layers': layers}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f)

    size_kb = path.stat().st_size / 1024
    print(f"  Exported {path}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip dataset generation; load from disk instead')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 5.2 — DISTILL DQN INTO PRODUCTION MODEL")
    print("=" * 72)
    print()

    # ── Step 1: Dataset ──────────────────────────────────────────────────────
    if args.skip_gen and DATASET_PATH.exists():
        print("Skipping dataset generation (--skip-gen).")
        dataset = load_dataset(DATASET_PATH)
    else:
        print(f"Generating dataset: {GEN_GAMES} games ...")
        dataset = generate_dataset(TEACHER_PATH, GEN_GAMES, GEN_SEED_BASE)
        save_dataset(dataset, DATASET_PATH)

    print(f"  Dataset size: {len(dataset):,} examples")
    print()

    # ── Step 2: Train students ───────────────────────────────────────────────
    tiny_nn  = TinyImitationNN(STATE_DIM, hidden_dim=32, action_dim=ACTION_DIM)
    linear   = LinearImitation(STATE_DIM, action_dim=ACTION_DIM)

    tiny_nn  = train_student(tiny_nn,  dataset, label="TinyNN  (80→32→95)")
    print()
    linear   = train_student(linear,   dataset, label="Linear  (80→95)   ")
    print()

    # ── Step 3: Build agents ─────────────────────────────────────────────────
    tiny_nn_agent = ImitationNNAgent("TinyNN-Distilled",  tiny_nn,  config=CONFIG)
    linear_agent  = ImitationNNAgent("Linear-Distilled",  linear,   config=CONFIG)

    # Also load teacher for baseline comparison
    print(f"Loading teacher for comparison ...")
    teacher_dqn = DQNAgent.load(str(TEACHER_PATH))
    teacher_dqn.set_training(False)
    print()

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    print("=" * 72)
    print(f"EVALUATION ({EVAL_GAMES} games each)")
    print("=" * 72)
    print()

    agents = [
        ("DQN Teacher",       teacher_dqn,   sum(p.numel() for p in teacher_dqn.q_network.parameters())),
        ("TinyNN (80→32→95)", tiny_nn_agent, sum(p.numel() for p in tiny_nn.parameters())),
        ("Linear (80→95)",    linear_agent,  sum(p.numel() for p in linear.parameters())),
    ]

    eval_results = []
    for label, agent, n_params in agents:
        print(f"  {label}  ({n_params:,} params)")
        r = evaluate(agent)
        eval_results.append((label, n_params, r))
        print()

    # ── Step 5: Summary table ────────────────────────────────────────────────
    print("=" * 72)
    print("PARETO SUMMARY")
    print("=" * 72)
    print()
    print(f"{'Agent':<24} {'Params':>7}  {'vs Heuristic':>13}  {'vs Random':>10}  {'ms/dec':>8}")
    print("-" * 68)

    for label, n_params, r in eval_results:
        h  = r['Heuristic']['win_rate']
        ra = r['Random']['win_rate']
        ms = r['Heuristic']['avg_ms']
        print(f"{label:<24} {n_params:>7,}  {h:>12.1%}  {ra:>10.1%}  {ms:>8.3f}")

    print()

    # Pass/fail check against plan criteria (>=45% vs Heuristic at <5K params)
    tiny_heur = eval_results[1][2]['Heuristic']['win_rate']
    tiny_params = eval_results[1][1]
    if tiny_heur >= 0.45 and tiny_params < 5000:
        print(f"PASS: TinyNN achieves {tiny_heur:.1%} vs Heuristic at {tiny_params:,} params")
    elif tiny_heur >= 0.45:
        print(f"PARTIAL PASS: TinyNN achieves {tiny_heur:.1%} vs Heuristic "
              f"but has {tiny_params:,} params (target <5K)")
    else:
        print(f"MISS: TinyNN {tiny_heur:.1%} vs Heuristic (target >=45%)")

    print()

    # ── Step 6: Export JSON ──────────────────────────────────────────────────
    print("Exporting model weights ...")

    tiny_meta = {
        'phase':      '5.2',
        'model':      'TinyImitationNN',
        'teacher':    str(TEACHER_PATH),
        'state_dim':  STATE_DIM,
        'hidden_dim': 32,
        'action_dim': ACTION_DIM,
        'n_params':   tiny_params,
        'vs_heuristic': round(tiny_heur, 4),
        'architecture': f'{STATE_DIM} -> 32 -> {ACTION_DIM}',
        'activation':   'relu',
        'dataset_size': len(dataset),
        'gen_games':    GEN_GAMES,
        'train_epochs': TRAIN_EPOCHS,
    }
    export_model_json(tiny_nn, OUTPUT_DIR / "tiny_nn.json", tiny_meta)

    linear_heur = eval_results[2][2]['Heuristic']['win_rate']
    linear_params = eval_results[2][1]
    linear_meta = {
        'phase':      '5.2',
        'model':      'LinearImitation',
        'teacher':    str(TEACHER_PATH),
        'state_dim':  STATE_DIM,
        'action_dim': ACTION_DIM,
        'n_params':   linear_params,
        'vs_heuristic': round(linear_heur, 4),
        'architecture': f'{STATE_DIM} -> {ACTION_DIM} (linear)',
        'dataset_size': len(dataset),
        'gen_games':    GEN_GAMES,
        'train_epochs': TRAIN_EPOCHS,
    }
    export_model_json(linear, OUTPUT_DIR / "linear.json", linear_meta)

    # Also save PyTorch checkpoints
    torch.save(tiny_nn.state_dict(), OUTPUT_DIR / "tiny_nn.pth")
    torch.save(linear.state_dict(),  OUTPUT_DIR / "linear.pth")
    print(f"  PyTorch checkpoints saved to {OUTPUT_DIR}/")
    print()

    print("=" * 72)
    print("Phase 5.2 complete.")
    print(f"  Best student JSON:  {OUTPUT_DIR}/tiny_nn.json")
    print(f"  Use in Flutter:     implement 80→32→95 forward pass in Dart (Phase 5.4)")
    print("=" * 72)


if __name__ == "__main__":
    main()
