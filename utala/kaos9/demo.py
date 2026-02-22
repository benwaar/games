#!/usr/bin/env python3
"""
Comprehensive demo for utala: kaos 9.

Showcases all baseline agents and their relative strengths:
- Random: Baseline control
- Heuristic: Strategic decision-making
- Monte Carlo: Look-ahead search with rollouts
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness


def main():
    print("=" * 70)
    print("utala: kaos 9 - Comprehensive Agent Demo")
    print("=" * 70)
    print()
    print("Phase 1 Baseline Agents:")
    print("  • Random:      Uniform random action selection")
    print("  • Heuristic:   Strategic placement and dogfight decisions")
    print("  • Monte Carlo: Look-ahead search with random rollouts")
    print()
    print("=" * 70)
    print()

    # Create agents
    random = RandomAgent("Random", seed=42)
    heuristic = HeuristicAgent("Heuristic", seed=123)
    monte_carlo = FastMonteCarloAgent("MonteCarlo", seed=456)

    # Create evaluation harness
    harness = Harness(verbose=False)

    # Demonstrate agent hierarchy with tournament-style matchups
    print("Running 5-game matches between all agent pairs...")
    print("(Note: Monte Carlo is slower due to rollout simulations)")
    print()

    matches = [
        (random, heuristic, "Random vs Heuristic"),
        (random, monte_carlo, "Random vs Monte Carlo"),
        (heuristic, monte_carlo, "Heuristic vs Monte Carlo"),
    ]

    results = []
    for i, (agent1, agent2, description) in enumerate(matches):
        seed = 4000 + (i * 100)
        print(f"{i+1}. {description}...")
        match_result = harness.run_match(agent1, agent2, num_games=5, starting_seed=seed)
        results.append((description, match_result))
        print(f"   P1 wins: {match_result.player_one_wins}/5, P2 wins: {match_result.player_two_wins}/5, Draws: {match_result.draws}/5")
        print()

    # Show detailed results
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    print()

    for description, match_result in results:
        print(match_result)
        print()

    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Expected hierarchy: Monte Carlo ≥ Heuristic > Random")
    print()
    print("• Heuristic should beat Random consistently (strategic advantage)")
    print("• Monte Carlo should beat Random consistently (search advantage)")
    print("• Heuristic vs Monte Carlo depends on rollout depth and heuristic quality")
    print()
    print("These baselines establish the skill gradient for Phase 2 learning agents.")
    print()
    print("=" * 70)
    print()
    print("To see detailed game play:")
    print("  • ./run.sh demo_random.py     - Random vs Random")
    print("  • ./run.sh demo_heuristic.py  - Heuristic vs Random (verbose)")
    print("  • ./run.sh demo_montecarlo.py - Monte Carlo vs others")


if __name__ == "__main__":
    main()
