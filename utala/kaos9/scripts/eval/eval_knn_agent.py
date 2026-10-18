#!/usr/bin/env python3
"""
Comprehensive evaluation of k-NN agent.

Runs full balanced matches against all Phase 1 baselines.
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.agents.associative_memory_agent import AssociativeMemoryAgent
from utala.evaluation.harness import Harness


def load_trained_agent(filepath: str):
    """Load a trained k-NN agent from file."""
    agent = AssociativeMemoryAgent(
        name="AssociativeMemory-v1",
        k=20,
        epsilon=0.0,  # No exploration during evaluation
        seed=42
    )
    agent.load(filepath)
    print(f"Loaded agent: {agent}")
    return agent


def run_balanced_evaluation(agent, opponent, opponent_name, num_games, seed):
    """Run balanced evaluation (agent plays both sides) with progress updates."""
    print(f"\n{'='*80}")
    print(f"{agent.name} vs {opponent_name}")
    print(f"{'='*80}")
    print(f"Running {num_games} games (balanced: {num_games//2} as P1, {num_games//2} as P2)...")

    harness = Harness(verbose=False)

    # Run in chunks to show progress
    chunk_size = 10
    agent_wins = 0
    opponent_wins = 0
    draws = 0

    for i in range(0, num_games, chunk_size):
        games_this_chunk = min(chunk_size, num_games - i)
        half_chunk = games_this_chunk // 2

        # First half: agent as P1
        result1 = harness.run_match(agent, opponent, half_chunk, seed + i)
        agent_wins += result1.player_one_wins
        opponent_wins += result1.player_two_wins
        draws += result1.draws

        # Second half: agent as P2
        result2 = harness.run_match(opponent, agent, games_this_chunk - half_chunk, seed + i + half_chunk)
        agent_wins += result2.player_two_wins
        opponent_wins += result2.player_one_wins
        draws += result2.draws

        # Progress update
        games_so_far = i + games_this_chunk
        current_win_rate = agent_wins / games_so_far if games_so_far > 0 else 0.0
        print(f"  Progress: {games_so_far}/{num_games} games | Win rate: {current_win_rate:.1%} ({agent_wins}-{opponent_wins}-{draws})")

    print(f"\nFinal Results:")
    print(f"  {agent.name} wins: {agent_wins} ({agent_wins/num_games:.1%})")
    print(f"  {opponent_name} wins: {opponent_wins} ({opponent_wins/num_games:.1%})")
    print(f"  Draws: {draws} ({draws/num_games:.1%})")

    return {
        'agent_wins': agent_wins,
        'opponent_wins': opponent_wins,
        'draws': draws,
        'win_rate': agent_wins / num_games
    }


def main():
    import glob
    import os

    print("="*80)
    print("K-NN AGENT COMPREHENSIVE EVALUATION")
    print("="*80)

    # Find most recent model
    model_files = glob.glob("models/AssociativeMemory-v1_*.json")
    if not model_files:
        print("ERROR: No trained model found. Run train_knn_agent.py first.")
        return

    latest_model = max(model_files, key=os.path.getctime)
    print(f"\nLoading model: {latest_model}")

    # Load agent
    agent = load_trained_agent(latest_model)

    # Configuration
    NUM_GAMES = 100
    SEED_BASE = 800000

    print(f"\nEvaluation configuration:")
    print(f"  Games per matchup: {NUM_GAMES} (balanced)")
    print(f"  Seed base: {SEED_BASE}")

    results = {}

    # Evaluate vs Random
    print("\n" + "="*80)
    print("MATCHUP 1: VS RANDOM")
    print("="*80)
    random_agent = RandomAgent("Random", seed=SEED_BASE)
    results['random'] = run_balanced_evaluation(
        agent, random_agent, "Random",
        NUM_GAMES, SEED_BASE + 100000
    )

    # Evaluate vs Heuristic
    print("\n" + "="*80)
    print("MATCHUP 2: VS HEURISTIC")
    print("="*80)
    heuristic = HeuristicAgent("Heuristic", seed=SEED_BASE + 10000)
    results['heuristic'] = run_balanced_evaluation(
        agent, heuristic, "Heuristic",
        NUM_GAMES, SEED_BASE + 200000
    )

    # Evaluate vs MC-Fast
    print("\n" + "="*80)
    print("MATCHUP 3: VS MC-FAST")
    print("="*80)
    mc_fast = FastMonteCarloAgent("MC-Fast", seed=SEED_BASE + 20000)
    results['mc_fast'] = run_balanced_evaluation(
        agent, mc_fast, "MC-Fast",
        NUM_GAMES, SEED_BASE + 300000
    )

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\n{agent.name} Performance:\n")
    print(f"  vs Random:    {results['random']['win_rate']:.1%} ({results['random']['agent_wins']}-{results['random']['opponent_wins']}-{results['random']['draws']})")
    print(f"  vs Heuristic: {results['heuristic']['win_rate']:.1%} ({results['heuristic']['agent_wins']}-{results['heuristic']['opponent_wins']}-{results['heuristic']['draws']})")
    print(f"  vs MC-Fast:   {results['mc_fast']['win_rate']:.1%} ({results['mc_fast']['agent_wins']}-{results['mc_fast']['opponent_wins']}-{results['mc_fast']['draws']})")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    if results['heuristic']['win_rate'] > 0.55:
        print("\n✓ SUCCESS: Beats Heuristic baseline (Phase 2 minimum goal)")
    elif results['heuristic']['win_rate'] > 0.45:
        print("\n~ COMPETITIVE: Roughly equal to Heuristic (needs improvement)")
    else:
        print("\n✗ NEEDS WORK: Loses to Heuristic (below baseline)")

    print("\nRecommendations:")
    if results['heuristic']['win_rate'] < 0.55:
        print("  - Generate more training data (try 5000+ games)")
        print("  - Train on higher-quality games (Heuristic vs Heuristic)")
        print("  - Tune k parameter (try k=10, k=50)")
        print("  - Try Manhattan distance metric")
        print("  - Prune memory to keep only high-quality examples")


if __name__ == '__main__':
    main()
