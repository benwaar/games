#!/usr/bin/env python3
"""
Phase 3.4 — Imitation Learning + Pareto Frontier

1. Generate MC-Fast demonstration dataset (state_features, chosen_action)
2. Train linear imitation model via supervised learning
3. Train tiny neural net imitation model
4. Evaluate all agents on Pareto frontier (strength vs compute)
"""

import time
import json
import numpy as np
import os

from src.utala.agents.monte_carlo_agent import FastMonteCarloAgent
from src.utala.agents.linear_value_agent import LinearValueAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.evaluation.harness import Harness
from src.utala.state import Player
from src.utala.learning.state_action_features import StateActionFeatureExtractor


EVAL_GAMES = 200


def p(msg=""):
    print(msg, flush=True)


# ────────────────────────────────────────────────────────────
# Step 1: Generate MC demonstration dataset
# ────────────────────────────────────────────────────────────

class RecordingMCAgent:
    """Wraps MC agent and records (state, legal_actions, chosen) with features."""

    def __init__(self, mc_agent, extractor):
        self.mc = mc_agent
        self.extractor = extractor
        self.name = mc_agent.name
        self.dataset = []

    def select_action(self, state, legal_actions, player):
        chosen = self.mc.select_action(state, legal_actions, player)

        # Extract features for all legal actions
        features_per_action = {}
        for a in legal_actions:
            f = self.extractor.extract(state, a, player)
            features_per_action[a] = f.tolist()

        self.dataset.append({
            'chosen': chosen,
            'legal': legal_actions,
            'features': {str(a): features_per_action[a] for a in legal_actions},
        })
        return chosen

    def game_start(self, player, seed):
        if hasattr(self.mc, 'game_start'):
            self.mc.game_start(player, seed)

    def game_end(self, state, winner):
        if hasattr(self.mc, 'game_end'):
            self.mc.game_end(state, winner)

    def __str__(self):
        return self.name


def generate_mc_dataset(num_games=5000, seed_base=500000):
    """Play MC-Fast games, record (features, action) for every decision."""
    p(f"Generating MC-Fast dataset ({num_games} games)...")

    mc = FastMonteCarloAgent("MC-Teacher", seed=42)
    extractor = StateActionFeatureExtractor(deck_awareness=False)
    recorder = RecordingMCAgent(mc, extractor)

    heuristic = HeuristicAgent("H-Opp", seed=99)
    rand_opp = RandomAgent("R-Opp")
    harness = Harness(verbose=False)

    t0 = time.time()

    for g in range(num_games):
        opponent = heuristic if g % 10 < 7 else rand_opp
        mc_as_p1 = (g % 2 == 0)
        seed = seed_base + g

        if mc_as_p1:
            harness.run_game(recorder, opponent, seed=seed)
        else:
            harness.run_game(opponent, recorder, seed=seed)

        if (g + 1) % 50 == 0:
            elapsed = time.time() - t0
            p(f"  [{g+1}/{num_games}] {(g+1)/elapsed:.1f} games/sec | {len(recorder.dataset)} decisions")

    p(f"  Done: {len(recorder.dataset)} MC decisions from {num_games} games ({time.time()-t0:.0f}s)")
    return recorder.dataset


# ────────────────────────────────────────────────────────────
# Step 2: Train linear imitation model
# ────────────────────────────────────────────────────────────

def train_linear_imitation(dataset, num_features=32, learning_rate=0.01, epochs=10):
    """
    Train a linear model to predict MC's chosen action.

    For each decision point, we want: Q(s, chosen_action) > Q(s, other_actions).
    Use a pairwise ranking loss: for each (chosen, other) pair,
    push weights so w·f(chosen) > w·f(other).
    """
    p(f"Training linear imitation ({len(dataset)} examples, {epochs} epochs)...")

    weights = np.zeros(num_features, dtype=np.float64)
    t0 = time.time()

    for epoch in range(epochs):
        np.random.shuffle(dataset)
        updates = 0
        correct = 0
        total = 0

        for example in dataset:
            chosen = example['chosen']
            legal = example['legal']
            features = example['features']

            if len(legal) <= 1:
                continue

            chosen_f = np.array(features[str(chosen)], dtype=np.float64)
            chosen_score = np.dot(weights, chosen_f)

            for a in legal:
                if a == chosen:
                    continue
                total += 1
                other_f = np.array(features[str(a)], dtype=np.float64)
                other_score = np.dot(weights, other_f)

                if chosen_score > other_score:
                    correct += 1
                else:
                    # Pairwise update: push chosen up, other down
                    weights += learning_rate * (chosen_f - other_f)
                    updates += 1
                    chosen_score = np.dot(weights, chosen_f)  # recompute

        accuracy = correct / total if total > 0 else 0
        p(f"  Epoch {epoch+1}/{epochs}: accuracy={accuracy:.1%}, updates={updates}")

    p(f"  Training done ({time.time()-t0:.0f}s)")
    return weights.astype(np.float32)


# ────────────────────────────────────────────────────────────
# Step 3: Train tiny neural net imitation
# ────────────────────────────────────────────────────────────

def train_nn_imitation(dataset, num_features=32, hidden_dim=32, epochs=20, lr=0.001):
    """Train a tiny 2-layer NN scorer: features -> score."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        p("  PyTorch not available — skipping NN imitation")
        return None

    p(f"Training tiny NN imitation ({hidden_dim}-unit hidden, {epochs} epochs)...")

    # Build pairwise training data: (chosen_features, other_features) -> chosen should score higher
    chosen_list = []
    other_list = []
    for example in dataset:
        chosen = example['chosen']
        legal = example['legal']
        features = example['features']
        if len(legal) <= 1:
            continue
        chosen_f = np.array(features[str(chosen)], dtype=np.float32)
        for a in legal:
            if a == chosen:
                continue
            other_f = np.array(features[str(a)], dtype=np.float32)
            chosen_list.append(chosen_f)
            other_list.append(other_f)

    X_chosen = torch.tensor(np.array(chosen_list), dtype=torch.float32)
    X_other = torch.tensor(np.array(other_list), dtype=torch.float32)

    p(f"  Training pairs: {len(X_chosen)}")

    # Tiny scorer network: features -> score
    model = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    batch_size = 256

    for epoch in range(epochs):
        perm = torch.randperm(len(X_chosen))
        total_loss = 0
        batches = 0

        for i in range(0, len(X_chosen), batch_size):
            idx = perm[i:i+batch_size]
            s_chosen = model(X_chosen[idx])
            s_other = model(X_other[idx])

            # Margin ranking loss: chosen should score > other by margin
            loss = torch.clamp(1.0 - (s_chosen - s_other), min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        with torch.no_grad():
            acc = (model(X_chosen) > model(X_other)).float().mean().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            p(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={acc:.1%}")

    p(f"  Training done ({time.time()-t0:.0f}s)")
    return model


# ────────────────────────────────────────────────────────────
# Step 4: Imitation agents for evaluation
# ────────────────────────────────────────────────────────────

class LinearImitationAgent:
    """Agent that uses imitation-learned linear weights."""

    def __init__(self, name, weights):
        self.name = name
        self.weights = weights
        self.extractor = StateActionFeatureExtractor(deck_awareness=False)

    def select_action(self, state, legal_actions, player):
        best_action = legal_actions[0]
        best_score = float('-inf')
        for a in legal_actions:
            f = self.extractor.extract(state, a, player)
            score = float(np.dot(self.weights, f))
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    def start_episode(self):
        pass

    def observe_outcome(self, outcome):
        pass


class NNImitationAgent:
    """Agent that uses imitation-learned tiny NN."""

    def __init__(self, name, model):
        import torch
        self.name = name
        self.model = model
        self.model.eval()
        self.extractor = StateActionFeatureExtractor(deck_awareness=False)

    def select_action(self, state, legal_actions, player):
        import torch
        # Score each action
        features = []
        for a in legal_actions:
            f = self.extractor.extract(state, a, player)
            features.append(f)

        features_t = torch.tensor(np.array(features), dtype=torch.float32)
        with torch.no_grad():
            scores = self.model(features_t).squeeze(-1)

        best_idx = scores.argmax().item()
        return legal_actions[best_idx]

    def start_episode(self):
        pass

    def observe_outcome(self, outcome):
        pass


# ────────────────────────────────────────────────────────────
# Step 5: Evaluate all agents (Pareto frontier)
# ────────────────────────────────────────────────────────────

def evaluate_agent(agent, seed_offset=0):
    """Evaluate agent vs Heuristic and Random."""
    harness = Harness(verbose=False)
    results = {}

    for opp_name, opp in [("Heuristic", HeuristicAgent("HE", seed=77)),
                           ("Random", RandomAgent("RE"))]:
        wins = 0
        for i in range(EVAL_GAMES):
            agent_as_p1 = i < EVAL_GAMES // 2
            seed = 700000 + seed_offset + i
            if agent_as_p1:
                p1, p2, ap = agent, opp, Player.ONE
            else:
                p1, p2, ap = opp, agent, Player.TWO

            if hasattr(agent, 'start_episode'):
                agent.start_episode()
            if hasattr(agent, 'set_training_mode'):
                agent.set_training_mode(False)

            result = harness.run_game(p1, p2, seed=seed)
            if result.winner == ap:
                wins += 1
        results[opp_name] = wins / EVAL_GAMES
    return results


class TimingWrapper:
    """Wraps an agent and measures time spent in select_action."""

    def __init__(self, agent):
        self.agent = agent
        self.name = agent.name if hasattr(agent, 'name') else str(agent)
        self.total_time = 0.0
        self.total_calls = 0

    def select_action(self, state, legal_actions, player):
        t0 = time.perf_counter()
        result = self.agent.select_action(state, legal_actions, player)
        self.total_time += time.perf_counter() - t0
        self.total_calls += 1
        return result

    def game_start(self, player, seed):
        if hasattr(self.agent, 'game_start'):
            self.agent.game_start(player, seed)

    def game_end(self, state, winner):
        if hasattr(self.agent, 'game_end'):
            self.agent.game_end(state, winner)

    def __str__(self):
        return self.name

    @property
    def avg_ms(self):
        return (self.total_time / self.total_calls * 1000) if self.total_calls > 0 else 0


def timed_eval(agent, seed_offset=0):
    """Evaluate with timing using harness."""
    harness = Harness(verbose=False)
    opp = HeuristicAgent("HE-Time", seed=77)

    if hasattr(agent, 'set_training_mode'):
        agent.set_training_mode(False)

    timed = TimingWrapper(agent)
    wins = 0

    for i in range(EVAL_GAMES):
        agent_as_p1 = i < EVAL_GAMES // 2
        seed = 600000 + seed_offset + i

        if hasattr(agent, 'start_episode'):
            agent.start_episode()

        if agent_as_p1:
            result = harness.run_game(timed, opp, seed=seed)
            if result.winner == Player.ONE:
                wins += 1
        else:
            result = harness.run_game(opp, timed, seed=seed)
            if result.winner == Player.TWO:
                wins += 1

    return wins / EVAL_GAMES, timed.avg_ms


def main():
    p("=" * 72)
    p("Phase 3.4 — Imitation Learning + Pareto Frontier")
    p("=" * 72)
    p()

    # ── Step 1: Generate dataset ──
    dataset = generate_mc_dataset(num_games=500)
    p()

    num_features = 32

    # ── Step 2: Train linear imitation ──
    linear_weights = train_linear_imitation(dataset, num_features=num_features, epochs=15)
    linear_agent = LinearImitationAgent("Imitation-Linear", linear_weights)
    p()

    # ── Step 3: Train NN imitation ──
    nn_model = train_nn_imitation(dataset, num_features=num_features, hidden_dim=32, epochs=20)
    nn_agent = NNImitationAgent("Imitation-NN", nn_model) if nn_model else None
    p()

    # ── Step 4: Evaluate all agents ──
    p("=" * 72)
    p("EVALUATION — All Agents vs Heuristic")
    p("=" * 72)
    p()

    agents = [
        ("Random", RandomAgent("Random")),
        ("Heuristic", HeuristicAgent("Heuristic", seed=55)),
    ]

    # TD-trained linear (best from 3.3: no interaction features)
    td_agent = LinearValueAgent("TD-NoInt", learning_rate=0.01, discount=0.95,
                                epsilon=0.1, seed=42, deck_awareness=False)
    # Quick train
    p("Training TD-NoInt baseline (5K games)...")
    harness = Harness(verbose=False)
    h_opp = HeuristicAgent("H", seed=99)
    r_opp = RandomAgent("R")
    interaction_mask = list(range(26, 31))
    original_extract = td_agent.feature_extractor.extract
    def masked_extract(state, action, player):
        features = original_extract(state, action, player)
        for idx in interaction_mask:
            features[idx] = 0.0
        return features
    td_agent.feature_extractor.extract = masked_extract
    td_agent.set_training_mode(True)
    for g in range(1, 5001):
        opponent = h_opp if g % 10 < 7 else r_opp
        agent_as_p1 = (g % 2 == 0)
        seed = 900000 + g
        if agent_as_p1:
            p1, p2, ap = td_agent, opponent, Player.ONE
        else:
            p1, p2, ap = opponent, td_agent, Player.TWO
        td_agent.start_episode()
        result = harness.run_game(p1, p2, seed=seed)
        outcome = 0.5 if result.winner is None else (1.0 if result.winner == ap else 0.0)
        td_agent.observe_outcome(outcome)
    p("  Done")
    p()

    agents.append(("TD-Linear (NoInt)", td_agent))
    agents.append(("Imitation-Linear", linear_agent))
    if nn_agent:
        agents.append(("Imitation-NN (32h)", nn_agent))

    # MC-Fast (slow but strong)
    agents.append(("MC-Fast (10 rollouts)", FastMonteCarloAgent("MC-Fast", seed=42)))

    # Evaluate with timing
    p("─" * 72)
    p("Timed evaluation vs Heuristic")
    p("─" * 72)
    p()

    pareto = []
    for name, agent in agents:
        if name == "Heuristic":
            # Can't evaluate Heuristic vs itself meaningfully
            # Use a different seed
            wr, ms = timed_eval(HeuristicAgent("H2", seed=123), seed_offset=0)
            pareto.append((name, wr, ms))
            p(f"  {name:<25} {wr:5.1%} vs H  |  {ms:8.2f} ms/decision")
        else:
            wr, ms = timed_eval(agent, seed_offset=len(pareto) * 10000)
            pareto.append((name, wr, ms))
            p(f"  {name:<25} {wr:5.1%} vs H  |  {ms:8.2f} ms/decision")

    p()

    # Also get vs Random for full picture
    p("─" * 72)
    p("Full evaluation")
    p("─" * 72)
    p()
    p(f"{'Agent':<25}{'vs Heuristic':>14}{'ms/decision':>14}")
    p("-" * 53)
    for name, wr, ms in sorted(pareto, key=lambda x: -x[1]):
        p(f"{name:<25}{wr:>13.1%}{ms:>13.2f}")
    p()

    # ── Pareto analysis ──
    p("=" * 72)
    p("PARETO FRONTIER (Strength vs Speed)")
    p("=" * 72)
    p()
    p("Agents on the Pareto frontier (no other agent is both stronger AND faster):")
    p()

    sorted_agents = sorted(pareto, key=lambda x: x[2])  # sort by speed
    frontier = []
    best_wr = -1
    for name, wr, ms in sorted_agents:
        if wr > best_wr:
            frontier.append((name, wr, ms))
            best_wr = wr

    for name, wr, ms in frontier:
        p(f"  {name:<25} {wr:5.1%} vs H  |  {ms:.2f} ms/decision")
    p()

    # Save imitation weights
    os.makedirs("models", exist_ok=True)
    np.save("models/imitation_linear_weights.npy", linear_weights)
    p(f"Linear imitation weights saved to models/imitation_linear_weights.npy")

    if nn_model:
        import torch
        torch.save(nn_model.state_dict(), "models/imitation_nn_32h.pth")
        p(f"NN imitation model saved to models/imitation_nn_32h.pth")
        param_count = sum(p.numel() for p in nn_model.parameters())
        p(f"NN parameters: {param_count}")
    p()


if __name__ == "__main__":
    main()
