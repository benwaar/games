#!/usr/bin/env python3
"""
Train Policy Network Agent using REINFORCE.

This script trains a 2-layer MLP using policy gradient learning.
"""

import time
from datetime import datetime

from src.utala.agents.policy_network_agent import PolicyNetworkAgent
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.evaluation.harness import Harness


def train_policy_network(
    num_episodes: int = 5000,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    discount: float = 0.99,
    opponent_mix: float = 0.7,
    eval_interval: int = 500,
    seed: int = 900000
):
    """
    Train Policy Network agent via self-play.

    Args:
        num_episodes: Number of training games
        hidden_size: Number of hidden units
        learning_rate: Learning rate for gradient descent
        discount: Discount factor for returns
        opponent_mix: Fraction of games vs Heuristic (vs Random)
        eval_interval: Evaluate every N episodes
        seed: Random seed
    """
    print("="*80)
    print("POLICY NETWORK TRAINING (REINFORCE)")
    print("="*80)
    print()

    # Create agent
    agent = PolicyNetworkAgent(
        name="PolicyNetwork-v1",
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        discount=discount,
        seed=seed
    )

    # Create opponents
    heuristic = HeuristicAgent("Heuristic")
    random_agent = RandomAgent("Random")

    print(f"Configuration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Discount factor: {discount}")
    print(f"  Training episodes: {num_episodes}")
    print(f"  Opponent mix: {opponent_mix:.0%} Heuristic, {1-opponent_mix:.0%} Random")
    print(f"  Eval interval: every {eval_interval} episodes")
    print(f"  Seed: {seed}")
    print()

    # Training loop
    import random as py_random
    rng = py_random.Random(seed)
    harness = Harness(verbose=False)
    agent.set_training_mode(True)

    start_time = time.time()
    print("Training...")
    print()

    eval_results = []

    for episode in range(1, num_episodes + 1):
        # Select opponent
        if rng.random() < opponent_mix:
            opponent = heuristic
        else:
            opponent = random_agent

        # Randomize sides
        if rng.random() < 0.5:
            result = harness.run_game(agent, opponent, seed + episode)
            outcome = 1.0 if result.winner == agent.name else (0.5 if result.winner == "Draw" else 0.0)
        else:
            result = harness.run_game(opponent, agent, seed + episode)
            outcome = 1.0 if result.winner == agent.name else (0.5 if result.winner == "Draw" else 0.0)

        # Update policy
        agent.observe_outcome(outcome)

        # Periodic evaluation
        if episode % eval_interval == 0:
            elapsed = time.time() - start_time
            stats = agent.get_stats()

            print(f"Episode {episode}/{num_episodes} ({elapsed/60:.1f}m)")
            print(f"  Baseline: {stats['baseline']:.3f}")
            print(f"  Total updates: {stats['total_updates']}")

            # Quick evaluation
            agent.set_training_mode(False)
            eval_result = harness.run_match(agent, heuristic, 20, seed + 100000 + episode)
            win_rate = eval_result.player_one_wins / 20
            print(f"  vs Heuristic (20 games): {win_rate:.1%} win rate")
            agent.set_training_mode(True)

            eval_results.append({
                'episode': episode,
                'baseline': stats['baseline'],
                'win_rate': win_rate
            })
            print()

    training_time = time.time() - start_time

    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Time: {training_time/60:.1f} minutes")
    print(f"Episodes: {num_episodes}")
    print(f"Updates: {agent.total_updates}")
    print()

    # Final evaluation
    print("Final Evaluation...")
    agent.set_training_mode(False)

    # vs Random
    result_random = harness.run_balanced_match(agent, random_agent, 100, seed + 200000)
    win_rate_random = result_random.player_one_wins / 100

    # vs Heuristic
    result_heuristic = harness.run_balanced_match(agent, heuristic, 100, seed + 300000)
    win_rate_heuristic = result_heuristic.player_one_wins / 100

    print(f"  vs Random (100 games): {win_rate_random:.1%}")
    print(f"  vs Heuristic (100 games): {win_rate_heuristic:.1%}")
    print()

    # Save model
    print("Saving model...")
    model_path = agent.save(
        agent_name="PolicyNetwork-v1",
        agent_type="policy_network",
        version="1.0",
        hyperparameters={
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'discount': discount,
            'opponent_mix': opponent_mix,
            'episodes_trained': num_episodes,
            'training_time_minutes': training_time / 60
        },
        performance={
            'vs_random': win_rate_random,
            'vs_heuristic': win_rate_heuristic,
            'total_updates': agent.total_updates,
            'eval_history': eval_results
        }
    )

    print(f"Model saved to: {model_path}")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"PolicyNetwork-v1 Performance:")
    print(f"  vs Random:    {win_rate_random:.1%}")
    print(f"  vs Heuristic: {win_rate_heuristic:.1%}")
    print()

    if win_rate_heuristic >= 0.55:
        print("✓ SUCCESS: Beats Heuristic baseline!")
    elif win_rate_heuristic >= 0.45:
        print("~ PROMISING: Competitive with Heuristic")
    else:
        print("✗ NEEDS WORK: Below Heuristic performance")

    print()
    print("Recommendations:")
    if win_rate_heuristic < 0.45:
        print("  - Train longer (10K-20K episodes)")
        print("  - Tune learning rate (try 0.0001-0.01)")
        print("  - Increase hidden size (128 units)")
        print("  - Add entropy bonus for exploration")
    elif win_rate_heuristic < 0.55:
        print("  - Train longer for convergence")
        print("  - Try different network architecture")
        print("  - Implement experience replay")
    else:
        print("  - Excellent! Try even stronger opponents (MC-Fast)")
        print("  - Consider curriculum learning")

    return agent


if __name__ == "__main__":
    import sys

    # Parse command line args
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001

    agent = train_policy_network(
        num_episodes=num_episodes,
        learning_rate=learning_rate
    )
