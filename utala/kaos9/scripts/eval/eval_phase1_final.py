#!/usr/bin/env python3
"""
Phase 1 Final Evaluation - MC-Fast (Strategic) Baseline

100 games each matchup to establish final Phase 1 baseline numbers.
MC-Fast now includes strategic dogfight evaluation by default.
"""

import sys
import time
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness


def evaluate_balanced(agent_name, agent, opponent_name, opponent, num_games, seed):
    """Run balanced evaluation (half as P1, half as P2)."""
    print(f"\n{agent_name} vs {opponent_name} ({num_games} games, balanced)...")
    print(f"  Running...", end=" ", flush=True)

    harness = Harness(verbose=False)
    start_time = time.time()

    # First half: agent as P1
    result1 = harness.run_match(agent, opponent, num_games // 2, seed)

    # Second half: agent as P2 (swapped)
    result2 = harness.run_match(opponent, agent, num_games // 2, seed + num_games // 2)

    elapsed = time.time() - start_time

    # Aggregate from agent's perspective
    agent_wins = result1.player_one_wins + result2.player_two_wins
    opp_wins = result1.player_two_wins + result2.player_one_wins
    draws = result1.draws + result2.draws

    win_rate = agent_wins / num_games * 100

    print(f"Done! ({elapsed:.1f}s)")
    print(f"  {agent_name}: {agent_wins} wins ({win_rate:.1f}%)")
    print(f"  {opponent_name}: {opp_wins} wins ({100 - win_rate - (draws/num_games*100):.1f}%)")
    print(f"  Draws: {draws} ({draws/num_games*100:.1f}%)")

    return {
        'win_rate': win_rate,
        'wins': agent_wins,
        'losses': opp_wins,
        'draws': draws,
        'time': elapsed
    }


def main():
    print("=" * 80)
    print("PHASE 1 FINAL EVALUATION - MC-Fast Baseline")
    print("=" * 80)
    print()
    print("MC-Fast now includes strategic dogfight evaluation by default.")
    print()
    print("Testing:")
    print("  1. MC-Fast vs Random (100 games)")
    print("  2. MC-Fast vs Heuristic (100 games)")
    print()
    print("=" * 80)

    num_games = 100
    seed = 600000

    # Test 1: MC-Fast vs Random
    print("\n" + "=" * 80)
    print("TEST 1: MC-FAST VS RANDOM")
    print("=" * 80)

    mc_fast = FastMonteCarloAgent("MC-Fast", seed=seed)
    random_agent = RandomAgent("Random", seed=seed + 10000)

    vs_random = evaluate_balanced(
        "MC-Fast", mc_fast,
        "Random", random_agent,
        num_games, seed + 100000
    )

    # Test 2: MC-Fast vs Heuristic
    print("\n" + "=" * 80)
    print("TEST 2: MC-FAST VS HEURISTIC")
    print("=" * 80)

    mc_fast2 = FastMonteCarloAgent("MC-Fast", seed=seed + 20000)
    heuristic = HeuristicAgent("Heuristic", seed=seed + 30000)

    vs_heuristic = evaluate_balanced(
        "MC-Fast", mc_fast2,
        "Heuristic", heuristic,
        num_games, seed + 200000
    )

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 1 FINAL RESULTS")
    print("=" * 80)
    print()

    print(f"MC-Fast (10 rollouts, strategic dogfights):")
    print(f"  vs Random:    {vs_random['wins']}/{num_games} ({vs_random['win_rate']:.1f}%)")
    print(f"  vs Heuristic: {vs_heuristic['wins']}/{num_games} ({vs_heuristic['win_rate']:.1f}%)")
    print()

    print(f"Performance:")
    print(f"  vs Random:    {vs_random['time']:.1f}s ({num_games/vs_random['time']*60:.1f} games/min)")
    print(f"  vs Heuristic: {vs_heuristic['time']:.1f}s ({num_games/vs_heuristic['time']*60:.1f} games/min)")
    print()

    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Skill ladder
    heur_vs_random = 65.0  # From previous evals
    print(f"Skill Ladder (vs Random):")
    print(f"  Random:     baseline")
    print(f"  Heuristic:  {heur_vs_random:.1f}%")
    print(f"  MC-Fast:    {vs_random['win_rate']:.1f}%")
    print()

    if vs_random['win_rate'] >= heur_vs_random:
        print(f"✓ MC-Fast matches or exceeds Heuristic vs Random")
    print()

    print(f"Head-to-Head:")
    print(f"  MC-Fast:    {vs_heuristic['win_rate']:.1f}% vs Heuristic")
    print(f"  Heuristic:  {100 - vs_heuristic['win_rate'] - (vs_heuristic['draws']/num_games*100):.1f}% vs MC-Fast")
    print()

    if vs_heuristic['win_rate'] > 55:
        print(f"✓ MC-Fast beats Heuristic head-to-head ({vs_heuristic['win_rate']:.1f}%)")
        print(f"  → Strategic dogfight evaluation is critical")
    print()

    print("=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print()
    print(f"**Baseline Agent**: MC-Fast (10 rollouts)")
    print(f"  - Strategic placement evaluation (via rollouts)")
    print(f"  - Strategic dogfight evaluation (via rollouts)")
    print(f"  - vs Random: {vs_random['win_rate']:.1f}%")
    print(f"  - vs Heuristic: {vs_heuristic['win_rate']:.1f}%")
    print(f"  - Performance: ~{num_games/vs_random['time']*60:.1f} games/min")
    print()
    print(f"**Phase 2 Target**: Learning agents should aim for >{vs_random['win_rate']:.0f}% vs Random")
    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
