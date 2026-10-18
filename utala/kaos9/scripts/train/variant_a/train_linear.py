#!/usr/bin/env python3
"""
Phase 4 Variant A — Train Linear Value Agent with Choosable Dogfight Order

Same hyperparams as Phase 2 baseline, just with GameConfig(fixed_dogfight_order=False).

Baseline comparison (Phase 2, fixed order):
  TD-Linear vs Heuristic: ~47.5% (after 5000 games, 27 features)
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, 'src')

from utala.agents.linear_value_agent import LinearValueAgent
from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.evaluation.harness import Harness
from utala.state import GameConfig, Player

# Variant A config
CONFIG = GameConfig(fixed_dogfight_order=False)

# Training configuration (same as Phase 2)
NUM_TRAINING_GAMES = 5000
EVAL_INTERVAL = 500
EVAL_GAMES = 100
LEARNING_RATE = 0.01
DISCOUNT = 0.95
EPSILON = 0.1
SEED_BASE = 900000


def play_training_game(harness, agent, opponent, agent_as_p1, seed):
    """Play one training game and return outcome from agent's perspective."""
    if agent_as_p1:
        p1, p2, agent_player = agent, opponent, Player.ONE
    else:
        p1, p2, agent_player = opponent, agent, Player.TWO

    agent.start_episode()
    result = harness.run_game(p1, p2, seed=seed)

    if result.winner is None:
        outcome = 0.5
    elif result.winner == agent_player:
        outcome = 1.0
    else:
        outcome = 0.0

    agent.observe_outcome(outcome)
    return outcome


def evaluate_agent(harness, agent, num_games, seed_base):
    """Evaluate agent against baselines."""
    agent.set_training_mode(False)

    opponents = {
        'Random': RandomAgent("Random-Eval"),
        'Heuristic': HeuristicAgent("Heuristic-Eval", config=CONFIG)
    }

    results = {}
    for opp_name, opponent in opponents.items():
        wins = draws = losses = 0

        for i in range(num_games):
            agent_as_p1 = (i < num_games // 2)
            outcome = play_training_game(harness, agent, opponent, agent_as_p1, seed_base + i)

            if outcome == 1.0:
                wins += 1
            elif outcome == 0.5:
                draws += 1
            else:
                losses += 1

        win_rate = wins / num_games
        results[opp_name] = {'wins': wins, 'draws': draws, 'losses': losses, 'win_rate': win_rate}
        print(f"  vs {opp_name:10s}: {win_rate:.1%} ({wins}-{draws}-{losses})")

    agent.set_training_mode(True)
    return results


def main():
    print("=" * 80)
    print("VARIANT A: LINEAR VALUE AGENT TRAINING")
    print("Choosable Dogfight Order (fixed_dogfight_order=False)")
    print("=" * 80)
    print()
    print(f"Config: {NUM_TRAINING_GAMES} games, alpha={LEARNING_RATE}, "
          f"gamma={DISCOUNT}, epsilon={EPSILON}")
    print()

    agent = LinearValueAgent(
        name="LinearValue-VariantA",
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon=EPSILON,
        seed=42,
        config=CONFIG
    )

    print(f"Features: {agent.num_features}")
    print()

    harness = Harness(verbose=False, config=CONFIG)
    random_agent = RandomAgent("Random-Train")
    heuristic_agent = HeuristicAgent("Heuristic-Train", config=CONFIG)

    eval_history = []
    start_time = time.time()

    # Initial evaluation
    print("Initial evaluation:")
    initial = evaluate_agent(harness, agent, EVAL_GAMES, SEED_BASE)
    eval_history.append({'games': 0, 'results': initial})
    print()

    # Training loop
    for game_num in range(1, NUM_TRAINING_GAMES + 1):
        opponent = heuristic_agent if game_num % 10 < 7 else random_agent
        agent_as_p1 = (game_num % 2 == 0)
        play_training_game(harness, agent, opponent, agent_as_p1, SEED_BASE + game_num)

        if game_num % 100 == 0:
            stats = agent.get_training_stats()
            elapsed = time.time() - start_time
            print(f"Game {game_num:5d}/{NUM_TRAINING_GAMES} | "
                  f"Updates: {stats['weight_updates']:6d} | "
                  f"Avg weight: {stats['avg_weight_magnitude']:.4f} | "
                  f"{game_num/elapsed:.1f} games/sec")

        if game_num % EVAL_INTERVAL == 0:
            print()
            print(f"Evaluation after {game_num} games:")
            results = evaluate_agent(harness, agent, EVAL_GAMES, SEED_BASE + 100000 + game_num)
            eval_history.append({'games': game_num, 'results': results})
            print()

    # Final evaluation
    print("=" * 80)
    print("FINAL EVALUATION (200 games)")
    print("=" * 80)
    print()

    final = evaluate_agent(harness, agent, EVAL_GAMES * 2, SEED_BASE + 200000)
    elapsed = time.time() - start_time

    print()
    print(f"Training complete: {elapsed/60:.1f} minutes")
    print()

    # Progression table
    print(f"{'Games':<10} {'vs Random':<12} {'vs Heuristic':<15}")
    print("-" * 40)
    for entry in eval_history:
        g = entry['games']
        r = entry['results']['Random']['win_rate']
        h = entry['results']['Heuristic']['win_rate']
        print(f"{g:<10} {r:>6.1%}       {h:>6.1%}")

    print(f"{'FINAL':<10} {final['Random']['win_rate']:>6.1%}       {final['Heuristic']['win_rate']:>6.1%}")
    print()

    # Comparison with baseline
    baseline_heur = 47.5
    actual_heur = final['Heuristic']['win_rate'] * 100
    print(f"vs Heuristic: {actual_heur:.1f}% (baseline: {baseline_heur}%, delta: {actual_heur - baseline_heur:+.1f}%)")
    print()

    # Top weights
    print("Top 10 learned weights:")
    for feature, weight in agent.get_top_weights(10):
        print(f"  {feature:35s}: {weight:8.4f}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
