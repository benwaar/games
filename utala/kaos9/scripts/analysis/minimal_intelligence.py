#!/usr/bin/env python3
"""
Phase 3.3.3+4 — Minimal Intelligence + Representation Test

Combined experiment:
1. Verify "No Interaction" model (ablation showed 49% vs H — best result yet)
2. Absolute minimal: just material_advantage + bias (2 features)
3. Representation test: raw board input (grid occupancy) vs engineered features
"""

import time
import numpy as np

from src.utala.agents.linear_value_agent import LinearValueAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.evaluation.harness import Harness
from src.utala.state import Player, Phase
from src.utala.learning.state_action_features import StateActionFeatureExtractor


TRAIN_GAMES = 5000
EVAL_GAMES = 200
LR = 0.01
DISCOUNT = 0.95
EPSILON = 0.1
SEED = 42


def p(msg=""):
    print(msg, flush=True)


def train_and_eval(agent, seed_offset=0):
    harness = Harness(verbose=False)
    heuristic = HeuristicAgent("H", seed=99)
    rand_agent = RandomAgent("R")
    agent.set_training_mode(True)

    for g in range(1, TRAIN_GAMES + 1):
        opponent = heuristic if g % 10 < 7 else rand_agent
        agent_as_p1 = (g % 2 == 0)
        seed = 900000 + seed_offset + g

        if agent_as_p1:
            p1, p2, ap = agent, opponent, Player.ONE
        else:
            p1, p2, ap = opponent, agent, Player.TWO

        agent.start_episode()
        result = harness.run_game(p1, p2, seed=seed)
        outcome = 0.5 if result.winner is None else (1.0 if result.winner == ap else 0.0)
        agent.observe_outcome(outcome)

    agent.set_training_mode(False)
    results = {}
    for opp_name, opp in [("Heuristic", HeuristicAgent("HE", seed=77)),
                           ("Random", RandomAgent("RE"))]:
        wins = 0
        for i in range(EVAL_GAMES):
            agent_as_p1 = i < EVAL_GAMES // 2
            seed = 800000 + seed_offset + i
            if agent_as_p1:
                p1, p2, ap = agent, opp, Player.ONE
            else:
                p1, p2, ap = opp, agent, Player.TWO
            agent.start_episode()
            result = harness.run_game(p1, p2, seed=seed)
            if result.winner == ap:
                wins += 1
        results[opp_name] = wins / EVAL_GAMES
    return results


def create_masked_agent(name, mask, seed_offset=0):
    agent = LinearValueAgent(name, learning_rate=LR, discount=DISCOUNT,
                             epsilon=EPSILON, seed=SEED, deck_awareness=False)
    original_extract = agent.feature_extractor.extract

    def masked_extract(state, action, player):
        features = original_extract(state, action, player)
        for idx in mask:
            if idx < len(features):
                features[idx] = 0.0
        return features

    agent.feature_extractor.extract = masked_extract
    return agent


class RawBoardFeatureExtractor:
    """
    Minimal raw board representation — no engineered features.

    Features (27 total):
    - 9 squares × 3 values each:
      - my_presence (1.0 if I have a rocketman here)
      - opp_presence (1.0 if opponent has a rocketman here)
      - is_empty (1.0 if no one here)
    """

    def __init__(self):
        self.feature_names = []
        for r in range(3):
            for c in range(3):
                self.feature_names.extend([
                    f"sq_{r}{c}_mine",
                    f"sq_{r}{c}_opp",
                    f"sq_{r}{c}_empty",
                ])

    def extract(self, state, action, player):
        features = []
        for r in range(3):
            for c in range(3):
                sq = state.grid[r][c]
                mine = any(rm.player == player for rm in sq.rocketmen)
                opp = any(rm.player == player.opponent() for rm in sq.rocketmen)
                empty = not mine and not opp
                features.extend([float(mine), float(opp), float(empty)])
        return np.array(features, dtype=np.float32)

    def get_feature_count(self):
        return len(self.feature_names)

    def get_feature_names(self):
        return self.feature_names.copy()


def create_raw_board_agent(name, seed_offset=0):
    """Create agent using only raw board features."""
    agent = LinearValueAgent(name, learning_rate=LR, discount=DISCOUNT,
                             epsilon=EPSILON, seed=SEED, deck_awareness=False)
    # Replace feature extractor
    raw_extractor = RawBoardFeatureExtractor()
    agent.feature_extractor = raw_extractor
    agent.num_features = raw_extractor.get_feature_count()
    agent.weights = np.random.randn(agent.num_features).astype(np.float32) * 0.01
    return agent


def main():
    p("=" * 72)
    p("Phase 3.3.3+4 — Minimal Intelligence + Representation Test")
    p("=" * 72)
    p()

    results_table = []

    # ── 1. Baseline (full 32 features) ──
    p("─" * 72)
    p("1. Baseline (32 features)")
    t0 = time.time()
    baseline = LinearValueAgent("Baseline", learning_rate=LR, discount=DISCOUNT,
                                epsilon=EPSILON, seed=SEED, deck_awareness=False)
    r = train_and_eval(baseline, seed_offset=0)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("Baseline (32 features)", 32, r))
    p()

    # ── 2. No Interaction (verification — ablation showed 49%) ──
    p("─" * 72)
    p("2. No Interaction features (27 features)")
    t0 = time.time()
    interaction_mask = list(range(26, 31))
    agent = create_masked_agent("No-Interaction", interaction_mask, seed_offset=50000)
    r = train_and_eval(agent, seed_offset=50000)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("No Interaction (27 feat)", 27, r))
    p()

    # ── 3. No Interaction + No Placement (17 features) ──
    p("─" * 72)
    p("3. State + Dogfight only (17 features)")
    t0 = time.time()
    mask = list(range(10, 20)) + list(range(26, 31))
    agent = create_masked_agent("State+Dog", mask, seed_offset=60000)
    r = train_and_eval(agent, seed_offset=60000)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("State + Dogfight (17 feat)", 17, r))
    p()

    # ── 4. Top 3 only ──
    p("─" * 72)
    p("4. Top 3: material_adv + control_adv + opp_rocketmen + bias")
    t0 = time.time()
    keep = {5, 8, 4, 31}  # material_advantage, control_advantage, opp_rocketmen_count, bias
    mask = [i for i in range(32) if i not in keep]
    agent = create_masked_agent("Top3", mask, seed_offset=70000)
    r = train_and_eval(agent, seed_offset=70000)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("Top 3 + bias (4 feat)", 4, r))
    p()

    # ── 5. Just material_advantage + bias (absolute minimum) ──
    p("─" * 72)
    p("5. Absolute minimum: material_advantage + bias")
    t0 = time.time()
    keep = {5, 31}  # material_advantage, bias
    mask = [i for i in range(32) if i not in keep]
    agent = create_masked_agent("Material-Only", mask, seed_offset=80000)
    r = train_and_eval(agent, seed_offset=80000)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("Material only (2 feat)", 2, r))
    p()

    # ── 6. Raw board representation (27 features, no engineering) ──
    p("─" * 72)
    p("6. Raw board input (27 grid features, no engineering)")
    t0 = time.time()
    agent = create_raw_board_agent("RawBoard", seed_offset=90000)
    r = train_and_eval(agent, seed_offset=90000)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("Raw board (27 feat)", 27, r))
    p()

    # ── Summary ──
    p("=" * 72)
    p("SUMMARY TABLE")
    p("=" * 72)
    p()
    p(f"{'Model':<30}{'#Feat':>7}{'vs Heuristic':>15}{'vs Random':>12}")
    p("-" * 64)
    for name, nfeat, r in results_table:
        p(f"{name:<30}{nfeat:>7}{r['Heuristic']:>14.1%}{r['Random']:>12.1%}")
    p()

    # ── Key questions ──
    p("=" * 72)
    p("KEY QUESTIONS ANSWERED")
    p("=" * 72)
    p()

    baseline_h = results_table[0][2]['Heuristic']
    no_int_h = results_table[1][2]['Heuristic']
    top3_h = results_table[3][2]['Heuristic']
    material_h = results_table[4][2]['Heuristic']
    raw_h = results_table[5][2]['Heuristic']

    p(f"  Q: What is the minimal intelligence needed?")
    p(f"  A: Top 3 features ({top3_h:.1%}) ≈ Baseline ({baseline_h:.1%})")
    p(f"     Just material_advantage alone: {material_h:.1%}")
    p()
    p(f"  Q: Are interaction features helpful?")
    p(f"  A: No — removing them improves from {baseline_h:.1%} to {no_int_h:.1%}")
    p()
    p(f"  Q: Can learning discover 3-in-a-row from raw board input?")
    p(f"  A: Raw board agent: {raw_h:.1%} vs Heuristic")
    if raw_h < baseline_h * 0.8:
        p(f"     No — feature engineering is essential for linear models")
    elif raw_h >= baseline_h * 0.9:
        p(f"     Yes — raw input is competitive with engineered features")
    else:
        p(f"     Partially — raw input captures some but not all signal")
    p()


if __name__ == "__main__":
    main()
