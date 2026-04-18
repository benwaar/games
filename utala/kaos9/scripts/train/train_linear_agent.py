#!/usr/bin/env python3
"""
Train Linear Value Agent with TD Learning.

Trains the agent through self-play and vs baseline opponents,
using temporal difference learning to update feature weights.
"""

import sys
import time
from datetime import datetime

from src.utala.agents.linear_value_agent import LinearValueAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.evaluation.harness import Harness
from src.utala.state import Player

# Training configuration
NUM_TRAINING_GAMES = 5000
EVAL_INTERVAL = 500  # Evaluate every N games
EVAL_GAMES = 100     # Games per evaluation
CHECKPOINT_INTERVAL = 1000

LEARNING_RATE = 0.01
DISCOUNT = 0.95
EPSILON = 0.1
SEED_BASE = 900000


def play_training_game(
    harness: Harness,
    agent: LinearValueAgent,
    opponent,
    agent_as_player_one: bool,
    seed: int
) -> float:
    """
    Play one training game and return outcome from agent's perspective.

    Args:
        harness: Game harness
        agent: Learning agent
        opponent: Opponent agent
        agent_as_player_one: Whether agent plays as P1
        seed: Random seed

    Returns:
        Outcome: 1.0=win, 0.5=draw, 0.0=loss (from agent's perspective)
    """
    # Set up players
    if agent_as_player_one:
        player_one = agent
        player_two = opponent
        agent_player = Player.ONE
    else:
        player_one = opponent
        player_two = agent
        agent_player = Player.TWO

    # Start episode
    agent.start_episode()

    # Play game
    result = harness.run_game(player_one, player_two, seed=seed)

    # Determine outcome from agent's perspective
    if result.winner is None:
        outcome = 0.5  # Draw
    elif result.winner == agent_player:
        outcome = 1.0  # Win
    else:
        outcome = 0.0  # Loss

    # Let agent observe outcome for learning
    agent.observe_outcome(outcome)

    return outcome


def evaluate_agent(harness: Harness, agent: LinearValueAgent, num_games: int, seed_base: int) -> dict:
    """
    Evaluate agent against baseline opponents.

    Args:
        harness: Game harness
        agent: Agent to evaluate
        num_games: Number of games per matchup (balanced)
        seed_base: Base seed for evaluation games

    Returns:
        Dict with win rates vs each opponent
    """
    # Disable training during evaluation
    agent.set_training_mode(False)

    opponents = {
        'Random': RandomAgent("Random-Eval"),
        'Heuristic': HeuristicAgent("Heuristic-Eval")
    }

    results = {}

    for opp_name, opponent in opponents.items():
        wins = 0
        draws = 0
        losses = 0

        # Play balanced games (half as P1, half as P2)
        for i in range(num_games):
            agent_as_p1 = (i < num_games // 2)
            seed = seed_base + i

            outcome = play_training_game(harness, agent, opponent, agent_as_p1, seed)

            if outcome == 1.0:
                wins += 1
            elif outcome == 0.5:
                draws += 1
            else:
                losses += 1

        win_rate = wins / num_games
        results[opp_name] = {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate
        }

        print(f"  vs {opp_name:10s}: {win_rate:.1%} ({wins}-{draws}-{losses})")

    # Re-enable training
    agent.set_training_mode(True)

    return results


def train_linear_agent():
    """Main training loop."""
    print("="*80)
    print("LINEAR VALUE AGENT TRAINING")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  Training games: {NUM_TRAINING_GAMES}")
    print(f"  Learning rate (α): {LEARNING_RATE}")
    print(f"  Discount (γ): {DISCOUNT}")
    print(f"  Exploration (ε): {EPSILON}")
    print(f"  Eval interval: {EVAL_INTERVAL} games")
    print()

    # Create agent
    agent = LinearValueAgent(
        name="LinearValue-v1",
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon=EPSILON,
        seed=42
    )

    print(f"Agent initialized:")
    print(f"  Features: {agent.num_features}")
    print(f"  Initial weights: mean={agent.weights.mean():.4f}, std={agent.weights.std():.4f}")
    print()

    # Create harness (non-verbose for training)
    harness = Harness(verbose=False)

    # Create training opponents
    random_agent = RandomAgent("Random-Train")
    heuristic_agent = HeuristicAgent("Heuristic-Train")

    # Training history
    eval_history = []
    start_time = time.time()

    # Initial evaluation
    print("Initial evaluation (before training):")
    initial_results = evaluate_agent(harness, agent, EVAL_GAMES, SEED_BASE)
    eval_history.append({
        'games': 0,
        'results': initial_results,
        'training_stats': agent.get_training_stats()
    })
    print()

    # Training loop
    print("Starting training...")
    print()

    for game_num in range(1, NUM_TRAINING_GAMES + 1):
        # Alternate opponents (70% vs Heuristic, 30% vs Random for better learning)
        if game_num % 10 < 7:
            opponent = heuristic_agent
        else:
            opponent = random_agent

        # Alternate who plays first
        agent_as_p1 = (game_num % 2 == 0)

        # Play training game
        seed = SEED_BASE + game_num
        outcome = play_training_game(harness, agent, opponent, agent_as_p1, seed)

        # Progress update
        if game_num % 100 == 0:
            stats = agent.get_training_stats()
            elapsed = time.time() - start_time
            games_per_sec = game_num / elapsed

            print(f"Game {game_num:5d}/{NUM_TRAINING_GAMES} | "
                  f"Updates: {stats['weight_updates']:6d} | "
                  f"Avg weight: {stats['avg_weight_magnitude']:.4f} | "
                  f"{games_per_sec:.1f} games/sec")

        # Periodic evaluation
        if game_num % EVAL_INTERVAL == 0:
            print()
            print(f"Evaluation after {game_num} training games:")
            results = evaluate_agent(harness, agent, EVAL_GAMES, SEED_BASE + 100000 + game_num)
            eval_history.append({
                'games': game_num,
                'results': results,
                'training_stats': agent.get_training_stats()
            })

            # Show top learned weights
            print()
            print("Top learned weights:")
            for feature, weight in agent.get_top_weights(10):
                print(f"  {feature:30s}: {weight:8.4f}")
            print()

        # Save checkpoint
        if game_num % CHECKPOINT_INTERVAL == 0:
            checkpoint_name = f"LinearValue-v1-checkpoint-{game_num}"
            stats = agent.get_training_stats()

            filepath = agent.save(
                agent_name=checkpoint_name,
                agent_type="linear_value",
                version="1.0",
                hyperparameters={
                    'learning_rate': LEARNING_RATE,
                    'discount': DISCOUNT,
                    'epsilon': EPSILON,
                    'training_games': game_num
                },
                performance={
                    'training_games': game_num,
                    'weight_updates': stats['weight_updates']
                }
            )
            print(f"Checkpoint saved: {filepath}")
            print()

    # Final evaluation
    print()
    print("="*80)
    print("FINAL EVALUATION")
    print("="*80)
    print()

    final_results = evaluate_agent(harness, agent, EVAL_GAMES * 2, SEED_BASE + 200000)

    print()
    print("Training complete!")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"  Games trained: {agent.games_trained}")
    print(f"  Weight updates: {agent.total_weight_updates}")
    print()

    # Save final model
    final_name = f"LinearValue-v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filepath = agent.save(
        agent_name=final_name,
        agent_type="linear_value",
        version="1.0",
        hyperparameters={
            'learning_rate': LEARNING_RATE,
            'discount': DISCOUNT,
            'epsilon': EPSILON,
            'training_games': NUM_TRAINING_GAMES
        },
        performance={
            'vs_random': final_results['Random']['win_rate'],
            'vs_heuristic': final_results['Heuristic']['win_rate']
        }
    )

    print(f"Final model saved: {filepath}")
    print()

    # Print training progression
    print("Training progression:")
    print(f"{'Games':<10} {'vs Random':<12} {'vs Heuristic':<15} {'Avg Weight':<12}")
    print("-" * 50)
    for entry in eval_history:
        games = entry['games']
        random_wr = entry['results']['Random']['win_rate']
        heuristic_wr = entry['results']['Heuristic']['win_rate']
        avg_weight = entry['training_stats']['avg_weight_magnitude']

        print(f"{games:<10} {random_wr:>6.1%}       {heuristic_wr:>6.1%}          {avg_weight:>6.4f}")

    print()
    print("Top 15 learned weights:")
    for feature, weight in agent.get_top_weights(15):
        print(f"  {feature:35s}: {weight:8.4f}")


if __name__ == "__main__":
    train_linear_agent()
