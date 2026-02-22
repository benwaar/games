#!/usr/bin/env python3
"""
Demo script for the Heuristic agent in utala: kaos 9.

Shows the heuristic agent's strategic decision-making:
- Prioritizes center and edge positions
- Places stronger rocketmen in better positions
- Uses rockets when stronger, flares when weaker
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.random_agent import RandomAgent
from utala.evaluation.harness import Harness


def main():
    print("utala: kaos 9 - Heuristic Agent Demo")
    print("=" * 50)
    print()

    # Create agents
    heuristic = HeuristicAgent("Heuristic", seed=42)
    random = RandomAgent("Random", seed=123)

    # Create evaluation harness
    harness = Harness(verbose=True)

    print("Heuristic vs Random - Single Game")
    print()

    # Run a single game
    result = harness.run_game(heuristic, random, seed=2000)

    print()
    print("=" * 50)
    print("Game completed!")
    print(f"Winner: {result.winner}")
    print(f"Turns: {result.num_turns}")
    print()

    # Run a match to see performance
    print("Running a 20-game match: Heuristic vs Random...")
    match_result = harness.run_match(heuristic, random, num_games=20, starting_seed=2000)

    print()
    print("=" * 50)
    print(match_result)
    print()
    print("The heuristic agent should win significantly more than 50%")
    print("demonstrating that strategic decision-making matters.")


if __name__ == "__main__":
    main()
