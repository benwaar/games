#!/usr/bin/env python3
"""
Phase 3.2: Deck Awareness Training & Evaluation.

Trains two Linear Value agents side-by-side:
- Baseline: existing features (no deck awareness)
- Deck-Aware: existing features + deck awareness features

Both use identical hyperparameters, opponent mix, and seeds.
Evaluates head-to-head and against common opponents.
"""

import sys
sys.path.insert(0, 'src')

import random
import time
import json
from pathlib import Path

from utala.agents.linear_value_agent import LinearValueAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.random_agent import RandomAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness
from utala.state import Player

# Config
NUM_TRAINING_GAMES = 20000
EVAL_INTERVAL = 5000
EVAL_GAMES = 200
SEED = 42
LR = 0.01
DISCOUNT = 0.95
EPSILON = 0.1

OUTPUT_DIR = Path("results/deck_aware")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def p(msg=""):
    import builtins
    builtins.print(msg, flush=True)


def evaluate_pair(baseline, deck_aware, game_num):
    """Evaluate both agents against common opponents and each other."""
    p(f"\n{'='*60}")
    p(f"Evaluation at {game_num} training games")
    p(f"{'='*60}")

    harness = Harness(verbose=False)
    heuristic = HeuristicAgent("Heuristic-Eval")
    random_agent = RandomAgent("Random-Eval")

    results = {"game_num": game_num}

    for agent, label in [(baseline, "Baseline"), (deck_aware, "DeckAware")]:
        agent.set_training_mode(False)

        # vs Heuristic
        r = harness.run_balanced_match(agent, heuristic, EVAL_GAMES, starting_seed=SEED + game_num)
        wr_h = r.player_one_wins / EVAL_GAMES
        results[f"{label}_vs_heuristic"] = wr_h

        # vs Random
        r = harness.run_balanced_match(agent, random_agent, EVAL_GAMES, starting_seed=SEED + game_num + 1000)
        wr_r = r.player_one_wins / EVAL_GAMES
        results[f"{label}_vs_random"] = wr_r

        agent.set_training_mode(True)
        p(f"  {label:>10s}: {wr_h:.1%} vs Heuristic, {wr_r:.1%} vs Random")

    # Head-to-head
    baseline.set_training_mode(False)
    deck_aware.set_training_mode(False)
    r = harness.run_balanced_match(deck_aware, baseline, EVAL_GAMES, starting_seed=SEED + game_num + 2000)
    da_wr = r.player_one_wins / EVAL_GAMES
    results["DeckAware_vs_Baseline"] = da_wr
    p(f"  {'H2H':>10s}: DeckAware {da_wr:.1%} vs Baseline")
    baseline.set_training_mode(True)
    deck_aware.set_training_mode(True)

    return results


def main():
    p("=" * 60)
    p("Phase 3.2: Deck Awareness Training")
    p("=" * 60)
    p()

    # Create agents
    baseline = LinearValueAgent("Baseline", learning_rate=LR, discount=DISCOUNT,
                                epsilon=EPSILON, seed=SEED, deck_awareness=False)
    deck_aware = LinearValueAgent("DeckAware", learning_rate=LR, discount=DISCOUNT,
                                  epsilon=EPSILON, seed=SEED, deck_awareness=True)

    p(f"Baseline features:   {baseline.num_features}")
    p(f"DeckAware features:  {deck_aware.num_features}")
    p(f"New deck features:   {deck_aware.num_features - baseline.num_features}")
    p(f"Training games:      {NUM_TRAINING_GAMES:,}")
    p()

    # Training opponents
    heuristic = HeuristicAgent("Heuristic-Train")
    random_agent = RandomAgent("Random-Train")

    # Initial eval
    eval_results = []
    eval_results.append(evaluate_pair(baseline, deck_aware, 0))

    # Training loop
    p(f"\n{'='*60}")
    p("Training...")
    p(f"{'='*60}")

    rng = random.Random(SEED)
    start_time = time.time()

    for game_num in range(1, NUM_TRAINING_GAMES + 1):
        # Select opponent
        opponent = heuristic if rng.random() < 0.7 else random_agent

        # Train both agents against same opponent with same seed
        game_seed = SEED + game_num

        for agent in [baseline, deck_aware]:
            harness = Harness(verbose=False)

            # Alternate who goes first (same for both agents)
            if game_num % 2 == 0:
                result = harness.run_game(agent, opponent, seed=game_seed)
                winner = result.winner
                outcome = 1.0 if winner == Player.ONE else (0.5 if winner is None else 0.0)
            else:
                result = harness.run_game(opponent, agent, seed=game_seed)
                winner = result.winner
                outcome = 1.0 if winner == Player.TWO else (0.5 if winner is None else 0.0)

            agent.observe_outcome(outcome)

        # Progress
        if game_num % 2000 == 0:
            elapsed = time.time() - start_time
            gps = game_num / elapsed
            eta = (NUM_TRAINING_GAMES - game_num) / gps
            p(f"  [{game_num:5d}/{NUM_TRAINING_GAMES}] {gps:.0f} games/sec | ETA: {eta/60:.1f} min")

        # Evaluate
        if game_num % EVAL_INTERVAL == 0:
            eval_results.append(evaluate_pair(baseline, deck_aware, game_num))

    # Final evaluation
    p(f"\n{'='*60}")
    p("Final Evaluation")
    p(f"{'='*60}")
    final = evaluate_pair(baseline, deck_aware, NUM_TRAINING_GAMES)
    eval_results.append(final)

    # Save results
    eval_path = OUTPUT_DIR / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    p(f"\nResults saved: {eval_path}")

    # Summary
    p(f"\n{'='*60}")
    p("SUMMARY")
    p(f"{'='*60}")
    p()
    p(f"{'Metric':<30s} {'Baseline':>10s} {'DeckAware':>10s} {'Delta':>8s}")
    p("-" * 60)

    for key in ["vs_heuristic", "vs_random"]:
        b = final[f"Baseline_{key}"]
        d = final[f"DeckAware_{key}"]
        delta = d - b
        p(f"{key:<30s} {b:>9.1%} {d:>9.1%} {delta:>+7.1%}")

    da_h2h = final["DeckAware_vs_Baseline"]
    p(f"{'Head-to-head (DeckAware wins)':<30s} {'':>10s} {da_h2h:>9.1%}")
    p()

    if da_h2h > 0.53:
        p("Deck awareness provides a measurable advantage.")
    elif da_h2h > 0.47:
        p("Deck awareness shows no significant difference.")
    else:
        p("Deck awareness may be hurting performance (more features, same data).")
    p()


if __name__ == "__main__":
    main()
