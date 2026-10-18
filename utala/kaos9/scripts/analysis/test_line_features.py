#!/usr/bin/env python3
"""
Phase 3.3 — Re-test with fixed line formation features.

Compares:
1. Old baseline (line features were 0.0) — retrain to get fresh numbers
2. Fixed model (line features now compute real values)
3. Fixed + no interaction features (best from ablation + the fix)
"""

import time
import numpy as np

from src.utala.agents.linear_value_agent import LinearValueAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.evaluation.harness import Harness
from src.utala.state import Player


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


def main():
    p("=" * 72)
    p("Phase 3.3 — Re-test with Fixed Line Formation Features")
    p("=" * 72)
    p()
    p(f"Training: {TRAIN_GAMES} games | Eval: {EVAL_GAMES} games per matchup")
    p()

    results_table = []

    # 1. Fixed model (line features now work)
    p("─" * 72)
    p("1. Fixed model (32 features, line formation now works)")
    t0 = time.time()
    agent = LinearValueAgent("Fixed", learning_rate=LR, discount=DISCOUNT,
                             epsilon=EPSILON, seed=SEED, deck_awareness=False)
    r = train_and_eval(agent, seed_offset=0)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("Fixed (32 feat)", r))

    # Show the line feature weights
    names = agent.feature_extractor.get_feature_names()
    for i, name in enumerate(names):
        if "line" in name:
            p(f"   Weight: {name} = {agent.weights[i]:+.4f}")
    p()

    # 2. Fixed + no interaction
    p("─" * 72)
    p("2. Fixed + no interaction features (27 features)")
    t0 = time.time()
    interaction_mask = list(range(26, 31))
    agent = create_masked_agent("Fixed-NoInt", interaction_mask, seed_offset=50000)
    r = train_and_eval(agent, seed_offset=50000)
    p(f"   vs Heuristic: {r['Heuristic']:.1%}  |  vs Random: {r['Random']:.1%}  [{time.time()-t0:.0f}s]")
    results_table.append(("Fixed, no interaction (27 feat)", r))

    names = agent.feature_extractor.get_feature_names()
    for i, name in enumerate(names):
        if "line" in name:
            p(f"   Weight: {name} = {agent.weights[i]:+.4f}")
    p()

    # 3. Run 3 more seeds to check stability
    p("─" * 72)
    p("3. Stability check — Fixed + no interaction, 3 additional seeds")
    seed_wins = []
    for extra_seed in [100, 200, 300]:
        agent = create_masked_agent(f"Stability-{extra_seed}", interaction_mask, seed_offset=extra_seed*1000)
        agent.rng = __import__('random').Random(extra_seed)
        r = train_and_eval(agent, seed_offset=extra_seed*1000)
        seed_wins.append(r['Heuristic'])
        p(f"   Seed {extra_seed}: {r['Heuristic']:.1%} vs H  |  {r['Random']:.1%} vs R")

    avg_h = np.mean(seed_wins)
    std_h = np.std(seed_wins)
    p(f"   Average: {avg_h:.1%} ± {std_h:.1%}")
    p()

    # Summary
    p("=" * 72)
    p("SUMMARY")
    p("=" * 72)
    p()
    p(f"{'Model':<35}{'vs Heuristic':>15}{'vs Random':>12}")
    p("-" * 62)
    p(f"{'Historical baseline (broken lines)':<35}{'~42%':>15}{'~50%':>12}")
    for name, r in results_table:
        p(f"{name:<35}{r['Heuristic']:>14.1%}{r['Random']:>12.1%}")
    p(f"{'Fixed+NoInt avg (3 seeds)':<35}{avg_h:>14.1%}{'':>12}")
    p()


if __name__ == "__main__":
    main()
