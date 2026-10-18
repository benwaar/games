#!/usr/bin/env python3
"""
Test evaluation harness capabilities for Phase 1.

Verifies:
1. Self-play - same agent vs itself
2. Cross-play - different agents vs each other
3. Tournament metrics - aggregated statistics
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness


def test_self_play():
    """Test self-play: same agent vs itself."""
    print("=" * 70)
    print("TEST 1: SELF-PLAY")
    print("=" * 70)
    print()
    print("Testing Random vs Random (same agent, different seeds)...")
    print()

    harness = Harness(verbose=False)

    # Create two instances of the same agent with different seeds
    random1 = RandomAgent("Random-A", seed=100)
    random2 = RandomAgent("Random-B", seed=200)

    result = harness.run_match(random1, random2, num_games=10, starting_seed=5000)

    print(f"Games: {result.num_games}")
    print(f"Random-A wins: {result.player_one_wins} ({result.player_one_win_rate:.1%})")
    print(f"Random-B wins: {result.player_two_wins} ({result.player_two_win_rate:.1%})")
    print(f"Draws: {result.draws} ({result.draw_rate:.1%})")
    print()

    # For self-play with truly random agents, expect roughly equal distribution
    # (though with low sample size there will be variance)
    if result.num_games == 10:
        print("✓ Self-play supported - same agent type vs itself")
    else:
        print("✗ Self-play failed")
        return False

    print()
    return True


def test_cross_play():
    """Test cross-play: different agents vs each other."""
    print("=" * 70)
    print("TEST 2: CROSS-PLAY")
    print("=" * 70)
    print()
    print("Testing different agent types against each other...")
    print()

    harness = Harness(verbose=False)

    # Test various cross-play matchups
    random = RandomAgent("Random")
    heuristic = HeuristicAgent("Heuristic")
    monte_carlo = FastMonteCarloAgent("MonteCarlo")

    matchups = [
        (random, heuristic, "Random vs Heuristic"),
        (random, monte_carlo, "Random vs MonteCarlo"),
        (heuristic, monte_carlo, "Heuristic vs MonteCarlo"),
    ]

    results = []
    for agent1, agent2, description in matchups:
        seed = 6000 + len(results) * 100
        print(f"{description}:")
        result = harness.run_match(agent1, agent2, num_games=5, starting_seed=seed)
        results.append(result)

        print(f"  {agent1.name}: {result.player_one_wins}/5 wins ({result.player_one_win_rate:.1%})")
        print(f"  {agent2.name}: {result.player_two_wins}/5 wins ({result.player_two_win_rate:.1%})")
        print(f"  Draws: {result.draws}/5 ({result.draw_rate:.1%})")
        print()

    if len(results) == 3 and all(r.num_games == 5 for r in results):
        print("✓ Cross-play supported - different agent types compete")
    else:
        print("✗ Cross-play failed")
        return False

    print()
    return True


def test_tournament_metrics():
    """Test tournament metrics: aggregated statistics."""
    print("=" * 70)
    print("TEST 3: TOURNAMENT METRICS")
    print("=" * 70)
    print()
    print("Testing tournament-style aggregation and metrics...")
    print()

    harness = Harness(verbose=False)

    # Create a small tournament
    random = RandomAgent("Random")
    heuristic = HeuristicAgent("Heuristic")
    monte_carlo = FastMonteCarloAgent("MonteCarlo")

    agents = [random, heuristic, monte_carlo]

    # Round-robin tournament
    tournament_results = []
    game_count = 0

    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i < j:  # Only play each matchup once
                seed = 7000 + game_count * 100
                result = harness.run_match(agent1, agent2, num_games=5, starting_seed=seed)
                tournament_results.append({
                    'agents': (agent1.name, agent2.name),
                    'result': result
                })
                game_count += 1

    print(f"Tournament completed: {len(tournament_results)} matchups")
    print()

    # Calculate tournament metrics
    agent_stats = {agent.name: {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0} for agent in agents}

    for match in tournament_results:
        agent1_name, agent2_name = match['agents']
        result = match['result']

        agent_stats[agent1_name]['games'] += result.num_games
        agent_stats[agent2_name]['games'] += result.num_games

        agent_stats[agent1_name]['wins'] += result.player_one_wins
        agent_stats[agent1_name]['losses'] += result.player_two_wins
        agent_stats[agent1_name]['draws'] += result.draws

        agent_stats[agent2_name]['wins'] += result.player_two_wins
        agent_stats[agent2_name]['losses'] += result.player_one_wins
        agent_stats[agent2_name]['draws'] += result.draws

    # Display tournament standings
    print("Tournament Standings:")
    print("-" * 70)
    print(f"{'Agent':<15} {'Games':<8} {'Wins':<8} {'Losses':<8} {'Draws':<8} {'Win%':<8}")
    print("-" * 70)

    for agent_name in sorted(agent_stats.keys(),
                             key=lambda x: (agent_stats[x]['wins'], -agent_stats[x]['losses']),
                             reverse=True):
        stats = agent_stats[agent_name]
        win_rate = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
        print(f"{agent_name:<15} {stats['games']:<8} {stats['wins']:<8} "
              f"{stats['losses']:<8} {stats['draws']:<8} {win_rate:>6.1f}%")

    print()

    # Verify metrics are tracked correctly
    total_games = sum(stats['games'] for stats in agent_stats.values()) // 2  # Divide by 2 since each game counts twice
    expected_games = len(tournament_results) * 5

    if total_games == expected_games:
        print(f"✓ Tournament metrics verified - {expected_games} total games tracked")
    else:
        print(f"✗ Tournament metrics failed - expected {expected_games}, got {total_games}")
        return False

    # Check that we have per-game results
    has_game_results = all('result' in match and
                          len(match['result'].game_results) == 5
                          for match in tournament_results)

    if has_game_results:
        print("✓ Per-game results captured for detailed analysis")
    else:
        print("✗ Per-game results missing")
        return False

    print()
    return True


def main():
    """Run all harness capability tests."""
    print()
    print("=" * 70)
    print("PHASE 1 EVALUATION HARNESS TEST")
    print("Testing: self-play, cross-play, tournament metrics")
    print("=" * 70)
    print()

    results = []

    # Test 1: Self-play
    results.append(("Self-play", test_self_play()))

    # Test 2: Cross-play
    results.append(("Cross-play", test_cross_play()))

    # Test 3: Tournament metrics
    results.append(("Tournament metrics", test_tournament_metrics()))

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print()
    print("=" * 70)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        print()
        print("Evaluation harness meets Phase 1 requirements:")
        print("  • Self-play: agents can play against themselves")
        print("  • Cross-play: different agents compete against each other")
        print("  • Tournament metrics: win/loss/draw rates, per-game results")
        print()
        print("Ready for Phase 1 checkpoint evaluation!")
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("Evaluation harness needs fixes before checkpoint.")

    print("=" * 70)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
