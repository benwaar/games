#!/usr/bin/env python3
"""
Phase 4 Variant A — Train DQN Agent with Bluffing Awareness

Curriculum training:
  Stage 1 (0-10K):    70% Heuristic + 30% Random, ε: 1.0 → 0.1
  Stage 2 (10K-30K):  50% Heuristic + 50% self-play, ε: 0.1 → 0.05
  Stage 3 (30K-50K):  80% self-play + 20% Heuristic, ε: 0.05 → 0.01

Baseline comparison:
  Phase 3.1 DQN (fixed order): 31% vs Heuristic (FAILED)
  TD-Linear (Variant A): 35.5% vs Heuristic
"""

import sys
import time
import random

sys.path.insert(0, 'src')

from utala.deep_learning.dqn_agent import DQNAgent
from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.evaluation.harness import Harness
from utala.state import GameConfig, Player

# Variant A config
CONFIG = GameConfig(fixed_dogfight_order=False)

# Training
TOTAL_GAMES = 50000
EVAL_INTERVAL = 2500
EVAL_GAMES = 100
SEED_BASE = 500000

# Curriculum boundaries
STAGE_1_END = 10000
STAGE_2_END = 30000

# DQN hyperparameters
LEARNING_RATE = 0.001
DISCOUNT = 0.95
REPLAY_CAPACITY = 30000
BATCH_SIZE = 64
TARGET_UPDATE = 1000
HIDDEN_DIM = 128
N_STEP = 3


def get_opponent(game_num, agents, dqn_agent):
    """Select opponent based on curriculum stage."""
    if game_num <= STAGE_1_END:
        # Stage 1: 70% Heuristic, 30% Random
        return agents['Heuristic'] if random.random() < 0.7 else agents['Random']
    elif game_num <= STAGE_2_END:
        # Stage 2: 50% Heuristic, 50% self-play
        return agents['Heuristic'] if random.random() < 0.5 else dqn_agent
    else:
        # Stage 3: 80% self-play, 20% Heuristic
        return dqn_agent if random.random() < 0.8 else agents['Heuristic']


def get_epsilon(game_num):
    """Compute epsilon for current training stage."""
    if game_num <= STAGE_1_END:
        # 1.0 → 0.1 over Stage 1
        progress = game_num / STAGE_1_END
        return 1.0 - 0.9 * progress
    elif game_num <= STAGE_2_END:
        # 0.1 → 0.05 over Stage 2
        progress = (game_num - STAGE_1_END) / (STAGE_2_END - STAGE_1_END)
        return 0.1 - 0.05 * progress
    else:
        # 0.05 → 0.01 over Stage 3
        progress = (game_num - STAGE_2_END) / (TOTAL_GAMES - STAGE_2_END)
        return 0.05 - 0.04 * progress


def evaluate(harness, agent, opponents, num_games, seed_base):
    """Evaluate agent against baselines."""
    agent.set_training(False)
    results = {}

    for opp_name, opponent in opponents.items():
        wins = draws = losses = 0

        for i in range(num_games):
            agent_as_p1 = (i < num_games // 2)
            seed = seed_base + i

            if agent_as_p1:
                result = harness.run_game(agent, opponent, seed=seed)
                if result.winner == Player.ONE:
                    wins += 1
                elif result.winner is None:
                    draws += 1
                else:
                    losses += 1
            else:
                result = harness.run_game(opponent, agent, seed=seed)
                if result.winner == Player.TWO:
                    wins += 1
                elif result.winner is None:
                    draws += 1
                else:
                    losses += 1

        win_rate = wins / num_games
        results[opp_name] = {'wins': wins, 'draws': draws, 'losses': losses, 'win_rate': win_rate}
        print(f"  vs {opp_name:10s}: {win_rate:.1%} ({wins}-{draws}-{losses})")

    agent.set_training(True)
    return results


def main():
    print("=" * 80)
    print("VARIANT A: DQN TRAINING WITH BLUFFING AWARENESS")
    print("Choosable Dogfight Order (fixed_dogfight_order=False)")
    print("=" * 80)
    print()
    print(f"Config: {TOTAL_GAMES} games, LR={LEARNING_RATE}, γ={DISCOUNT}, "
          f"n-step={N_STEP}, replay={REPLAY_CAPACITY}")
    print(f"Curriculum: Stage 1 (0-{STAGE_1_END}), "
          f"Stage 2 ({STAGE_1_END}-{STAGE_2_END}), "
          f"Stage 3 ({STAGE_2_END}-{TOTAL_GAMES})")
    print()

    # Create DQN agent
    agent = DQNAgent(
        name="DQN-VariantA",
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1.0,  # We manage epsilon externally
        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        hidden_dim=HIDDEN_DIM,
        n_step=N_STEP,
        seed=42,
        config=CONFIG,
    )

    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print(f"Network params: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print()

    harness = Harness(verbose=False, config=CONFIG)

    opponents = {
        'Random': RandomAgent("Random-Train", seed=42),
        'Heuristic': HeuristicAgent("Heuristic-Train", seed=42, config=CONFIG),
    }

    eval_opponents = {
        'Random': RandomAgent("Random-Eval"),
        'Heuristic': HeuristicAgent("Heuristic-Eval", config=CONFIG),
    }

    eval_history = []
    start_time = time.time()
    random.seed(42)

    # Initial evaluation
    print("Initial evaluation:")
    initial = evaluate(harness, agent, eval_opponents, EVAL_GAMES, SEED_BASE)
    eval_history.append({'games': 0, 'results': initial})
    print()

    # Training loop
    for game_num in range(1, TOTAL_GAMES + 1):
        # Set epsilon from curriculum
        agent.epsilon = get_epsilon(game_num)

        # Select opponent
        opponent = get_opponent(game_num, opponents, agent)
        agent_as_p1 = (game_num % 2 == 0)
        seed = SEED_BASE + game_num

        if agent_as_p1:
            harness.run_game(agent, opponent, seed=seed)
        else:
            harness.run_game(opponent, agent, seed=seed)

        # Progress logging
        if game_num % 500 == 0:
            stats = agent.get_stats()
            elapsed = time.time() - start_time
            stage = 1 if game_num <= STAGE_1_END else (2 if game_num <= STAGE_2_END else 3)
            print(f"Game {game_num:6d}/{TOTAL_GAMES} | "
                  f"Stage {stage} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"Loss={stats['avg_loss_recent']:.4f} | "
                  f"Buffer={stats['buffer_size']:6d} | "
                  f"{game_num/elapsed:.1f} g/s")

        # Evaluation checkpoints
        if game_num % EVAL_INTERVAL == 0:
            print()
            print(f"Evaluation after {game_num} games (Stage {'1' if game_num <= STAGE_1_END else '2' if game_num <= STAGE_2_END else '3'}):")
            results = evaluate(harness, agent, eval_opponents, EVAL_GAMES, SEED_BASE + 100000 + game_num)
            eval_history.append({'games': game_num, 'results': results})
            print()

    # Final evaluation
    print("=" * 80)
    print("FINAL EVALUATION (200 games)")
    print("=" * 80)
    print()

    final = evaluate(harness, agent, eval_opponents, EVAL_GAMES * 2, SEED_BASE + 200000)
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

    # Comparison
    linear_baseline = 35.5
    dqn_old_baseline = 31.0
    actual_heur = final['Heuristic']['win_rate'] * 100
    print(f"vs Heuristic: {actual_heur:.1f}% (TD-Linear: {linear_baseline}%, Old DQN: {dqn_old_baseline}%)")
    print()

    if actual_heur > 40:
        print("PASS: DQN beats Linear baseline (>40% vs Heuristic)")
    elif actual_heur > linear_baseline:
        print("MARGINAL: DQN improves on Linear but below 40% target")
    else:
        print("FAIL: DQN does not improve on Linear baseline")

    print("=" * 80)


if __name__ == "__main__":
    main()
