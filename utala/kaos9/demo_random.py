#!/usr/bin/env python3
"""
Demo script for utala: kaos 9.

Tests the game engine with random agents.
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.evaluation.harness import Harness


def main():
    print("utala: kaos 9 - Demo")
    print("=" * 50)
    print()

    # Create two random agents
    agent1 = RandomAgent("Random-A", seed=42)
    agent2 = RandomAgent("Random-B", seed=123)

    # Create evaluation harness
    harness = Harness(verbose=True)

    print("Running a single game...")
    print()

    # Run a single game
    result = harness.run_game(agent1, agent2, seed=1337)

    print()
    print("=" * 50)
    print("Game completed!")
    print(f"Winner: {result.winner}")
    print(f"Turns: {result.num_turns}")
    print()

    # Run a match
    print("Running a 10-game match...")
    match_result = harness.run_match(agent1, agent2, num_games=10, starting_seed=1000)

    print()
    print("=" * 50)
    print(match_result)


if __name__ == "__main__":
    main()
