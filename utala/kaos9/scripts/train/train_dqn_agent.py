"""
Train DQN Agent - Phase 3.1

Implements curriculum learning:
- Stage 1 (0-20K): 70% Heuristic, 30% Random
- Stage 2 (20K-50K): 50% Heuristic, 30% Linear, 20% self-play
- Stage 3 (50K-100K): 60% self-play, 20% Heuristic, 20% Linear

Target: 60-70% win rate vs Heuristic (Phase 2 was 42%)
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

from src.utala.deep_learning.dqn_agent import DQNAgent
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.agents.linear_value_agent import LinearValueAgent
from src.utala.evaluation.harness import Harness
from src.utala.state import Player

# Training configuration
TOTAL_GAMES = 100000
EVAL_INTERVAL = 5000      # Evaluate every 5K games
EVAL_GAMES = 200          # 200 games per evaluation
CHECKPOINT_INTERVAL = 10000  # Save every 10K games

# Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995    # Decays to ~0.01 after 50K games
REPLAY_CAPACITY = 20000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 1000
HIDDEN_DIM = 128

# Curriculum stages
STAGE_1_END = 20000
STAGE_2_END = 50000

SEED = 42

OUTPUT_DIR = Path("results/dqn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_opponent_mix(games_completed: int):
    """
    Get opponent distribution based on curriculum stage.

    Returns:
        List of (agent, probability) tuples
    """
    if games_completed < STAGE_1_END:
        # Stage 1: Learn basics vs Heuristic and Random
        return [
            ("heuristic", 0.7),
            ("random", 0.3),
        ]
    elif games_completed < STAGE_2_END:
        # Stage 2: Face stronger opponents
        return [
            ("heuristic", 0.5),
            ("linear", 0.3),
            ("self", 0.2),
        ]
    else:
        # Stage 3: Master through self-play
        return [
            ("self", 0.6),
            ("heuristic", 0.2),
            ("linear", 0.2),
        ]


def select_opponent(games_completed: int, opponent_pool: dict, rng) -> tuple:
    """
    Select opponent based on curriculum stage.

    Args:
        games_completed: Number of games completed
        opponent_pool: Dict of opponent agents
        rng: Random number generator

    Returns:
        (opponent_agent, opponent_name)
    """
    mix = get_opponent_mix(games_completed)

    # Sample opponent type
    types = [t for t, _ in mix]
    probs = [p for _, p in mix]
    opponent_type = rng.choices(types, weights=probs)[0]

    if opponent_type == "self":
        # Self-play: opponent is a copy of current agent (frozen)
        return opponent_pool["dqn"], "DQN-self"
    else:
        return opponent_pool[opponent_type], opponent_pool[opponent_type].name


def evaluate_agent(agent: DQNAgent, opponents: dict, game_num: int, harness: Harness):
    """Evaluate agent against all baselines."""
    print(f"\n{'='*70}")
    print(f"Evaluation at {game_num:,} training games")
    print(f"{'='*70}")

    # Disable training for evaluation
    agent.set_training(False)

    results = {}

    # Evaluate vs Heuristic (primary metric)
    print(f"\nEvaluating vs Heuristic ({EVAL_GAMES} games)...")
    heuristic = opponents["heuristic"]
    result = harness.run_balanced_match(
        agent_one=agent,
        agent_two=heuristic,
        num_games=EVAL_GAMES,
        starting_seed=SEED + game_num
    )
    win_rate_heuristic = result.player_one_wins / EVAL_GAMES
    print(f"  DQN wins: {result.player_one_wins}/{EVAL_GAMES} ({win_rate_heuristic:.1%})")
    print(f"  Heuristic wins: {result.player_two_wins}/{EVAL_GAMES}")
    print(f"  Draws: {result.draws}/{EVAL_GAMES}")
    results["vs_heuristic"] = win_rate_heuristic

    # Evaluate vs Random
    print(f"\nEvaluating vs Random ({EVAL_GAMES} games)...")
    random_agent = opponents["random"]
    result = harness.run_balanced_match(
        agent_one=agent,
        agent_two=random_agent,
        num_games=EVAL_GAMES,
        starting_seed=SEED + game_num + 1000
    )
    win_rate_random = result.player_one_wins / EVAL_GAMES
    print(f"  DQN wins: {result.player_one_wins}/{EVAL_GAMES} ({win_rate_random:.1%})")
    results["vs_random"] = win_rate_random

    # Evaluate vs Linear Value (Phase 2 baseline)
    print(f"\nEvaluating vs Linear Value ({EVAL_GAMES} games)...")
    linear = opponents["linear"]
    result = harness.run_balanced_match(
        agent_one=agent,
        agent_two=linear,
        num_games=EVAL_GAMES,
        starting_seed=SEED + game_num + 2000
    )
    win_rate_linear = result.player_one_wins / EVAL_GAMES
    print(f"  DQN wins: {result.player_one_wins}/{EVAL_GAMES} ({win_rate_linear:.1%})")
    results["vs_linear"] = win_rate_linear

    # Agent statistics
    stats = agent.get_stats()
    print(f"\nDQN Agent Statistics:")
    print(f"  Episodes: {stats['episodes']:,}")
    print(f"  Steps: {stats['steps']:,}")
    print(f"  Epsilon: {stats['epsilon']:.4f}")
    print(f"  Buffer: {stats['buffer_size']:,} / {REPLAY_CAPACITY:,} ({stats['buffer_utilization']:.1%})")
    print(f"  Avg Loss (recent 100): {stats['avg_loss_recent']:.4f}")
    print(f"  Avg Reward/Episode: {stats['avg_reward_per_episode']:.3f}")

    # Re-enable training
    agent.set_training(True)

    print(f"{'='*70}\n")

    return {
        "game_num": game_num,
        "win_rates": results,
        "stats": stats,
    }


def main():
    print("="*70)
    print("Training DQN Agent - Phase 3.1")
    print("="*70)
    print(f"Total games: {TOTAL_GAMES:,}")
    print(f"Evaluation: every {EVAL_INTERVAL:,} games")
    print(f"Checkpoints: every {CHECKPOINT_INTERVAL:,} games")
    print()
    print("Hyperparameters:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Discount: {DISCOUNT}")
    print(f"  Epsilon: {EPSILON_START} → {EPSILON_END} (decay: {EPSILON_DECAY})")
    print(f"  Replay buffer: {REPLAY_CAPACITY:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Target update freq: {TARGET_UPDATE_FREQ}")
    print()
    print("Curriculum:")
    print(f"  Stage 1 (0-{STAGE_1_END//1000}K): 70% Heuristic, 30% Random")
    print(f"  Stage 2 ({STAGE_1_END//1000}K-{STAGE_2_END//1000}K): 50% Heuristic, 30% Linear, 20% self-play")
    print(f"  Stage 3 ({STAGE_2_END//1000}K-{TOTAL_GAMES//1000}K): 60% self-play, 20% Heuristic, 20% Linear")
    print()

    # Create DQN agent
    agent = DQNAgent(
        name="DQN-v1",
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        hidden_dim=HIDDEN_DIM,
        seed=SEED,
    )

    print(f"DQN agent created:")
    print(f"  State dim: {agent.state_dim}")
    print(f"  Action dim: {agent.action_dim}")
    print(f"  Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print()

    # Create opponent pool
    # Load Linear Value agent from Phase 2
    linear_agent = LinearValueAgent(name="LinearValue")
    try:
        linear_agent.load("models/LinearValue-v1-checkpoint-5000_20260324_090050.json")
    except FileNotFoundError:
        print("Warning: Linear Value model not found, will skip Linear opponent")
        linear_agent = None

    opponent_pool = {
        "heuristic": HeuristicAgent("Heuristic"),
        "random": RandomAgent("Random"),
        "linear": linear_agent if linear_agent else HeuristicAgent("Heuristic-fallback"),
        "dqn": agent,  # For self-play
    }

    print("Opponent pool ready:")
    for name, opp in opponent_pool.items():
        if name != "dqn":
            print(f"  {name}: {opp.name}")
    print()

    # Create evaluation harness
    harness = Harness(verbose=False)

    # Initial evaluation
    eval_results = []
    initial_eval = evaluate_agent(agent, opponent_pool, 0, harness)
    eval_results.append(initial_eval)

    # Training loop
    print("="*70)
    print("Starting Training")
    print("="*70)

    import random
    rng = random.Random(SEED)

    games_completed = 0
    next_eval = EVAL_INTERVAL
    next_checkpoint = CHECKPOINT_INTERVAL

    start_time = time.time()
    stage_start_time = start_time
    current_stage = 1

    while games_completed < TOTAL_GAMES:
        # Check for stage transitions
        if games_completed == STAGE_1_END:
            current_stage = 2
            stage_start_time = time.time()
            print(f"\n{'='*70}")
            print(f"STAGE 2 START (Game {games_completed:,})")
            print(f"Opponent mix: 50% Heuristic, 30% Linear, 20% self-play")
            print(f"{'='*70}\n")
        elif games_completed == STAGE_2_END:
            current_stage = 3
            stage_start_time = time.time()
            print(f"\n{'='*70}")
            print(f"STAGE 3 START (Game {games_completed:,})")
            print(f"Opponent mix: 60% self-play, 20% Heuristic, 20% Linear")
            print(f"{'='*70}\n")

        # Select opponent
        opponent, opp_name = select_opponent(games_completed, opponent_pool, rng)

        # Decide who goes first
        if rng.random() < 0.5:
            agents = [agent, opponent]
            agent_is_p1 = True
        else:
            agents = [opponent, agent]
            agent_is_p1 = False

        # Play game
        game_result = harness.run_game(
            agent_one=agents[0],
            agent_two=agents[1],
            seed=SEED + games_completed
        )

        games_completed += 1

        # Progress update
        if games_completed % 1000 == 0:
            elapsed = time.time() - start_time
            games_per_sec = games_completed / elapsed
            eta_sec = (TOTAL_GAMES - games_completed) / games_per_sec

            stats = agent.get_stats()
            print(f"  [{games_completed:6,}/{TOTAL_GAMES:,}] "
                  f"({games_completed/TOTAL_GAMES*100:5.1f}%) "
                  f"| Stage {current_stage} "
                  f"| ε={stats['epsilon']:.4f} "
                  f"| Buf={stats['buffer_size']:5,} "
                  f"| Loss={stats['avg_loss_recent']:.4f} "
                  f"| {games_per_sec:.1f} g/s "
                  f"| ETA: {eta_sec/60:.0f}m")

        # Evaluation checkpoint
        if games_completed >= next_eval:
            eval_result = evaluate_agent(agent, opponent_pool, games_completed, harness)
            eval_results.append(eval_result)
            next_eval += EVAL_INTERVAL

            # Save evaluation results
            eval_path = OUTPUT_DIR / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2, default=str)

        # Save checkpoint
        if games_completed >= next_checkpoint:
            checkpoint_path = OUTPUT_DIR / f"dqn_checkpoint_{games_completed}.pth"
            agent.save(str(checkpoint_path))
            print(f"\n💾 Saved checkpoint: {checkpoint_path}")
            next_checkpoint += CHECKPOINT_INTERVAL

    # Final evaluation
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")

    final_eval = evaluate_agent(agent, opponent_pool, TOTAL_GAMES, harness)
    eval_results.append(final_eval)

    # Save final agent
    final_path = OUTPUT_DIR / "dqn_final.pth"
    agent.save(str(final_path))
    print(f"\n💾 Final agent saved: {final_path}")

    # Save evaluation results
    eval_path = OUTPUT_DIR / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"💾 Evaluation results saved: {eval_path}")

    # Training summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"Games per second: {TOTAL_GAMES/total_time:.1f}")
    print()
    print("Performance:")
    print(f"  Initial vs Heuristic: {eval_results[0]['win_rates']['vs_heuristic']:.1%}")
    print(f"  Final vs Heuristic:   {eval_results[-1]['win_rates']['vs_heuristic']:.1%}")
    improvement = eval_results[-1]['win_rates']['vs_heuristic'] - eval_results[0]['win_rates']['vs_heuristic']
    print(f"  Improvement: {improvement*100:+.1f} percentage points")
    print()
    print(f"  Initial vs Linear (Phase 2): {eval_results[0]['win_rates']['vs_linear']:.1%}")
    print(f"  Final vs Linear (Phase 2):   {eval_results[-1]['win_rates']['vs_linear']:.1%}")
    print()
    print("Target: 60-70% vs Heuristic (Phase 2 was 42%)")
    if eval_results[-1]['win_rates']['vs_heuristic'] >= 0.60:
        print("✅ TARGET ACHIEVED!")
    else:
        gap = 0.60 - eval_results[-1]['win_rates']['vs_heuristic']
        print(f"⚠️  Target not reached (gap: {gap*100:.1f} pp)")
    print()


if __name__ == "__main__":
    main()
