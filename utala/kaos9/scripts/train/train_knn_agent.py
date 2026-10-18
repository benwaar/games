#!/usr/bin/env python3
"""
Train Associative Memory (k-NN) Agent.

Generates training data from baseline agents and trains k-NN agent.
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.associative_memory_agent import AssociativeMemoryAgent
from utala.learning.training_data import TrainingDataGenerator
from utala.evaluation.harness import Harness


def generate_training_data(num_games: int, starting_seed: int):
    """Generate training data from baseline agents."""
    print("=" * 80)
    print("GENERATING TRAINING DATA")
    print("=" * 80)

    # Use Heuristic vs Random as training opponents
    # This gives the k-NN agent varied experiences
    heuristic = HeuristicAgent("Heuristic", seed=starting_seed)
    random_agent = RandomAgent("Random", seed=starting_seed + 1000)

    print(f"\nTraining opponents: {heuristic.name} vs {random_agent.name}")
    print(f"Games: {num_games}")
    print(f"Starting seed: {starting_seed}\n")

    generator = TrainingDataGenerator()

    # Generate data from both matchups for diversity
    print("Generating from Heuristic vs Random...")
    examples_1 = generator.generate_batch(
        heuristic, random_agent,
        num_games // 2,
        starting_seed
    )

    print("\nGenerating from Random vs Heuristic...")
    examples_2 = generator.generate_batch(
        random_agent, heuristic,
        num_games // 2,
        starting_seed + num_games // 2
    )

    all_examples = examples_1 + examples_2

    print(f"\nTotal training examples: {len(all_examples)}")

    # Save to file
    output_path = "data/training/knn_training_data.jsonl"
    metadata = {
        'agent_one': heuristic.name,
        'agent_two': random_agent.name,
        'num_games': num_games,
        'num_examples': len(all_examples),
        'starting_seed': starting_seed
    }
    generator.save_training_data(all_examples, output_path, metadata)

    return all_examples


def train_agent(training_examples, k: int = 20):
    """Train k-NN agent on training data."""
    print("\n" + "=" * 80)
    print("TRAINING K-NN AGENT")
    print("=" * 80)

    agent = AssociativeMemoryAgent(
        name="AssociativeMemory-v1",
        k=k,
        distance_metric="euclidean",
        epsilon=0.05,
        seed=42
    )

    agent.train(training_examples)

    stats = agent.get_memory_stats()
    print(f"\nMemory statistics:")
    print(f"  Size: {stats['size']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Draw rate: {stats['draw_rate']:.1%}")
    print(f"  Loss rate: {stats['loss_rate']:.1%}")

    return agent


def evaluate_agent(agent, num_games: int = 50):
    """Quick evaluation against baselines."""
    print("\n" + "=" * 80)
    print("QUICK EVALUATION")
    print("=" * 80)

    seed_offset = 900000
    harness = Harness(verbose=False)

    # Test vs Random
    print(f"\n{agent.name} vs Random ({num_games} games)...")
    random_agent = RandomAgent("Random", seed=seed_offset)
    result_random = harness.run_match(agent, random_agent, num_games, seed_offset)

    win_rate_random = result_random.player_one_wins / num_games
    print(f"  Win rate: {win_rate_random:.1%}")

    # Test vs Heuristic
    print(f"\n{agent.name} vs Heuristic ({num_games} games)...")
    heuristic = HeuristicAgent("Heuristic", seed=seed_offset + 10000)
    result_heuristic = harness.run_match(agent, heuristic, num_games, seed_offset + 20000)

    win_rate_heuristic = result_heuristic.player_one_wins / num_games
    print(f"  Win rate: {win_rate_heuristic:.1%}")

    return {
        'vs_random': win_rate_random,
        'vs_heuristic': win_rate_heuristic
    }


def save_agent(agent, performance):
    """Save trained agent to JSON."""
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    filepath = agent.save(
        agent_name=agent.name,
        agent_type="associative_memory",
        version="1.0",
        hyperparameters={
            'k': agent.k,
            'distance_metric': agent.distance_metric,
            'epsilon': agent.epsilon
        },
        performance=performance
    )

    print(f"Model saved to: {filepath}")
    return filepath


def main():
    """Main training pipeline."""
    # Configuration
    NUM_TRAINING_GAMES = 1000
    K_NEIGHBORS = 20
    EVAL_GAMES = 50
    STARTING_SEED = 500000

    print("=" * 80)
    print("K-NN AGENT TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Training games: {NUM_TRAINING_GAMES}")
    print(f"  k (neighbors): {K_NEIGHBORS}")
    print(f"  Evaluation games: {EVAL_GAMES}")
    print(f"  Starting seed: {STARTING_SEED}")

    # Step 1: Generate training data
    training_examples = generate_training_data(NUM_TRAINING_GAMES, STARTING_SEED)

    # Step 2: Train agent
    agent = train_agent(training_examples, k=K_NEIGHBORS)

    # Step 3: Quick evaluation
    performance = evaluate_agent(agent, num_games=EVAL_GAMES)

    # Step 4: Save trained model
    save_agent(agent, performance)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nAgent: {agent}")
    print(f"Performance:")
    print(f"  vs Random: {performance['vs_random']:.1%}")
    print(f"  vs Heuristic: {performance['vs_heuristic']:.1%}")
    print("\nNext steps:")
    print("  - Run full evaluation: python eval_knn_agent.py")
    print("  - Tune hyperparameters (try different k values)")
    print("  - Generate more training data for better performance")


if __name__ == '__main__':
    main()
