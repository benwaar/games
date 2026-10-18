#!/usr/bin/env python3
"""
Comprehensive v1.8 evaluation across all baseline agents.

Tests:
1. Self-play FPA for each agent (balance)
2. Cross-play skill ladder (Random < Heuristic < MC)
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness
from utala.state import Player


def run_self_play(agent_class, agent_name, num_games, seed):
    """Run self-play FPA test."""
    harness = Harness(verbose=False)

    agent1 = agent_class(f"{agent_name}-P1", seed=seed)
    agent2 = agent_class(f"{agent_name}-P2", seed=seed + 1000)

    result = harness.run_match(agent1, agent2, num_games, seed)

    fpa = (result.player_one_wins - result.player_two_wins) / result.num_games * 100

    return {
        'p1_wins': result.player_one_wins,
        'p2_wins': result.player_two_wins,
        'draws': result.draws,
        'fpa': fpa
    }


def run_cross_play(agent1_class, agent1_name, agent2_class, agent2_name, num_games, seed):
    """Run balanced cross-play."""
    harness = Harness(verbose=False)

    agent1 = agent1_class(agent1_name, seed=seed)
    agent2 = agent2_class(agent2_name, seed=seed + 1000)

    # First half: agent1 as P1, agent2 as P2
    result1 = harness.run_match(agent1, agent2, num_games // 2, seed)

    # Second half: agent2 as P1, agent1 as P2 (swapped)
    result2 = harness.run_match(agent2, agent1, num_games // 2, seed + num_games // 2)

    # Aggregate from agent1's perspective
    agent1_wins = result1.player_one_wins + result2.player_two_wins
    agent2_wins = result1.player_two_wins + result2.player_one_wins
    draws = result1.draws + result2.draws

    win_rate = agent1_wins / num_games * 100

    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'win_rate': win_rate
    }


def main():
    print()
    print("=" * 80)
    print("v1.8 COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print()
    print("Testing:")
    print("  - Random agent (baseline)")
    print("  - Heuristic agent (v1.8 with 3-in-a-row + joker awareness)")
    print("  - MC-Fast agent (10 rollouts)")
    print()
    print("=" * 80)
    print()

    num_games = 200
    seed = 120000

    agents = [
        (RandomAgent, "Random"),
        (HeuristicAgent, "Heuristic"),
        (FastMonteCarloAgent, "MC-Fast"),
    ]

    # Part 1: Self-play FPA
    print("PART 1: Self-Play FPA (Balance Test)")
    print("-" * 80)
    print()

    fpa_results = {}

    for agent_class, agent_name in agents:
        print(f"Testing {agent_name} self-play ({num_games} games)...")
        result = run_self_play(agent_class, agent_name, num_games, seed)
        fpa_results[agent_name] = result
        seed += 10000

        print(f"  P1: {result['p1_wins']}, P2: {result['p2_wins']}, Draws: {result['draws']}")
        print(f"  FPA: {result['fpa']:+.1f}%")
        print()

    print("-" * 80)
    print()
    print("FPA SUMMARY:")
    print()
    print(f"{'Agent':<20} {'FPA':<12} {'Status':<20}")
    print("-" * 80)

    for agent_name in [name for _, name in agents]:
        fpa = fpa_results[agent_name]['fpa']
        if abs(fpa) < 3:
            status = "✓ Excellent"
        elif abs(fpa) < 5:
            status = "✓ Good"
        elif abs(fpa) < 8:
            status = "~ Acceptable"
        else:
            status = "✗ High"
        print(f"{agent_name:<20} {fpa:>+6.1f}%     {status:<20}")

    avg_fpa = sum(r['fpa'] for r in fpa_results.values()) / len(fpa_results)
    print()
    print(f"Average FPA: {avg_fpa:+.1f}%")
    print()

    print("=" * 80)
    print()

    # Part 2: Cross-play skill ladder
    print("PART 2: Cross-Play Skill Ladder")
    print("-" * 80)
    print()

    matchups = [
        ("Heuristic", "Random", "Heuristic should beat Random"),
        ("MC-Fast", "Random", "MC should beat Random"),
        ("MC-Fast", "Heuristic", "MC should beat Heuristic"),
    ]

    cross_results = {}

    for agent1_name, agent2_name, description in matchups:
        agent1_class = next(cls for cls, name in agents if name == agent1_name)
        agent2_class = next(cls for cls, name in agents if name == agent2_name)

        print(f"{agent1_name} vs {agent2_name} ({num_games} games, balanced)...")
        result = run_cross_play(
            agent1_class, agent1_name,
            agent2_class, agent2_name,
            num_games, seed
        )
        cross_results[f"{agent1_name}_vs_{agent2_name}"] = result
        seed += 10000

        print(f"  {agent1_name}: {result['agent1_wins']} ({result['win_rate']:.1f}%)")
        print(f"  {agent2_name}: {result['agent2_wins']} ({100 - result['win_rate'] - result['draws']/num_games*100:.1f}%)")
        print(f"  Draws: {result['draws']}")
        print()

    print("-" * 80)
    print()
    print("SKILL LADDER:")
    print()

    heur_vs_rand = cross_results['Heuristic_vs_Random']['win_rate']
    mc_vs_rand = cross_results['MC-Fast_vs_Random']['win_rate']
    mc_vs_heur = cross_results['MC-Fast_vs_Heuristic']['win_rate']

    print(f"Heuristic vs Random: {heur_vs_rand:.1f}% win rate")
    if heur_vs_rand >= 60:
        print("  ✓ Heuristic shows strategic advantage")
    elif heur_vs_rand >= 55:
        print("  ~ Heuristic has modest advantage")
    else:
        print("  ✗ Heuristic needs improvement")
    print()

    print(f"MC-Fast vs Random: {mc_vs_rand:.1f}% win rate")
    if mc_vs_rand >= 70:
        print("  ✓ MC dominates Random")
    elif mc_vs_rand >= 60:
        print("  ✓ MC beats Random consistently")
    else:
        print("  ~ MC advantage is modest")
    print()

    print(f"MC-Fast vs Heuristic: {mc_vs_heur:.1f}% win rate")
    if mc_vs_heur >= 60:
        print("  ✓ MC clearly stronger than Heuristic")
    elif mc_vs_heur >= 55:
        print("  ✓ MC has advantage over Heuristic")
    elif mc_vs_heur >= 50:
        print("  ~ MC and Heuristic are roughly equal")
    else:
        print("  ✗ Heuristic unexpectedly stronger than MC")
    print()

    print("=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    print()

    # Check skill differentiation
    clear_ladder = (heur_vs_rand >= 55 and mc_vs_rand >= 60 and mc_vs_heur >= 55)

    if clear_ladder and abs(avg_fpa) < 8:
        print("✓✓✓ EXCELLENT: v1.8 is successful")
        print("    - Clear skill differentiation (Random < Heuristic < MC)")
        print("    - Acceptable game balance (FPA < 8%)")
    elif clear_ladder:
        print("✓✓ GOOD: v1.8 improvements working")
        print("    - Clear skill differentiation")
        print(f"    - FPA needs work ({avg_fpa:+.1f}%)")
    elif abs(avg_fpa) < 8:
        print("✓ PARTIAL: v1.8 has progress")
        print("    - Game balance is acceptable")
        print("    - Skill differentiation needs work")
    else:
        print("~ v1.8 still needs tuning")

    print()
    print(f"Key metrics:")
    print(f"  Average FPA: {avg_fpa:+.1f}%")
    print(f"  Heuristic vs Random: {heur_vs_rand:.1f}%")
    print(f"  MC vs Heuristic: {mc_vs_heur:.1f}%")
    print()
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
