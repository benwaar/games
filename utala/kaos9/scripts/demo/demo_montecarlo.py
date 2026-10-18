#!/usr/bin/env python3
"""
Demo script for the Monte Carlo agent in utala: kaos 9.

Shows the Monte Carlo agent's look-ahead evaluation:
- Simulates random rollouts from each action
- Selects actions with best win rate
- Demonstrates stronger play through search
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.monte_carlo_agent import MonteCarloAgent, FastMonteCarloAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.random_agent import RandomAgent
from utala.evaluation.harness import Harness


def main():
    print("utala: kaos 9 - Monte Carlo Agent Demo")
    print("=" * 50)
    print()

    # Create agents
    mc_agent = FastMonteCarloAgent("FastMC", seed=42)
    heuristic = HeuristicAgent("Heuristic", seed=123)
    random = RandomAgent("Random", seed=456)

    # Create evaluation harness
    harness = Harness(verbose=True)

    print("Monte Carlo vs Heuristic - Single Game")
    print("(Using FastMC with 20 rollouts per action)")
    print()

    # Run a single game
    result = harness.run_game(mc_agent, heuristic, seed=3000)

    print()
    print("=" * 50)
    print("Game completed!")
    print(f"Winner: {result.winner}")
    print(f"Turns: {result.num_turns}")
    print()

    # Run matches to compare performance
    print("Running 10-game matches to compare agent strength...")
    print()

    # MC vs Random
    print("Monte Carlo vs Random:")
    match1 = harness.run_match(mc_agent, random, num_games=10, starting_seed=3000)
    print(match1)
    print()

    # MC vs Heuristic
    print("Monte Carlo vs Heuristic:")
    match2 = harness.run_match(mc_agent, heuristic, num_games=10, starting_seed=3100)
    print(match2)
    print()

    print("=" * 50)
    print("Monte Carlo uses random rollouts to evaluate positions.")
    print("It should beat Random consistently and compete with Heuristic.")
    print()
    print("Note: For stronger play, use StrongMonteCarloAgent (100 rollouts),")
    print("but it will be significantly slower.")


if __name__ == "__main__":
    main()
