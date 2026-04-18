#!/usr/bin/env python3
"""
MC Scaling Evaluation - Final Phase 1 Checkpoint

Comprehensive study of Monte Carlo agent scaling to establish:
1. Skill ceiling with increased compute
2. Diminishing returns curve
3. Game strategic depth assessment
4. Target baseline for Phase 2 learning agents
"""

import sys
import time
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import MonteCarloAgent
from utala.evaluation.harness import Harness
from utala.state import Player


def run_self_play_fpa(num_rollouts, num_games, seed):
    """Run self-play FPA test."""
    harness = Harness(verbose=False)

    agent1 = MonteCarloAgent(f"MC-{num_rollouts}-P1", num_rollouts=num_rollouts, seed=seed)
    agent2 = MonteCarloAgent(f"MC-{num_rollouts}-P2", num_rollouts=num_rollouts, seed=seed + 1000)

    result = harness.run_match(agent1, agent2, num_games, seed)

    fpa = (result.player_one_wins - result.player_two_wins) / result.num_games * 100

    return {
        'p1_wins': result.player_one_wins,
        'p2_wins': result.player_two_wins,
        'draws': result.draws,
        'fpa': fpa
    }


def run_vs_baseline(num_rollouts, baseline_class, baseline_name, num_games, seed):
    """Run MC vs baseline (Random or Heuristic) - balanced positions."""
    harness = Harness(verbose=False)

    mc_agent = MonteCarloAgent(f"MC-{num_rollouts}", num_rollouts=num_rollouts, seed=seed)
    baseline = baseline_class(baseline_name, seed=seed + 1000)

    # First half: MC as P1
    result1 = harness.run_match(mc_agent, baseline, num_games // 2, seed)

    # Second half: MC as P2 (swapped)
    result2 = harness.run_match(baseline, mc_agent, num_games // 2, seed + num_games // 2)

    # Aggregate from MC's perspective
    mc_wins = result1.player_one_wins + result2.player_two_wins
    baseline_wins = result1.player_two_wins + result2.player_one_wins
    draws = result1.draws + result2.draws

    win_rate = mc_wins / num_games * 100

    return {
        'mc_wins': mc_wins,
        'baseline_wins': baseline_wins,
        'draws': draws,
        'win_rate': win_rate
    }


def run_head_to_head(rollouts_a, rollouts_b, num_games, seed):
    """Run MC-A vs MC-B direct comparison - balanced positions."""
    harness = Harness(verbose=False)

    agent_a = MonteCarloAgent(f"MC-{rollouts_a}", num_rollouts=rollouts_a, seed=seed)
    agent_b = MonteCarloAgent(f"MC-{rollouts_b}", num_rollouts=rollouts_b, seed=seed + 1000)

    # First half: A as P1
    result1 = harness.run_match(agent_a, agent_b, num_games // 2, seed)

    # Second half: A as P2 (swapped)
    result2 = harness.run_match(agent_b, agent_a, num_games // 2, seed + num_games // 2)

    # Aggregate from agent_a's perspective
    a_wins = result1.player_one_wins + result2.player_two_wins
    b_wins = result1.player_two_wins + result2.player_one_wins
    draws = result1.draws + result2.draws

    win_rate = a_wins / num_games * 100

    return {
        'a_wins': a_wins,
        'b_wins': b_wins,
        'draws': draws,
        'win_rate': win_rate
    }


def measure_performance(num_rollouts, num_games, seed):
    """Measure time per game for performance metrics."""
    harness = Harness(verbose=False)

    agent1 = MonteCarloAgent(f"MC-{num_rollouts}-A", num_rollouts=num_rollouts, seed=seed)
    agent2 = MonteCarloAgent(f"MC-{num_rollouts}-B", num_rollouts=num_rollouts, seed=seed + 1000)

    start_time = time.time()
    harness.run_match(agent1, agent2, num_games, seed)
    elapsed = time.time() - start_time

    time_per_game = elapsed / num_games

    return time_per_game


def main():
    print()
    print("=" * 80)
    print("MC SCALING EVALUATION - Final Phase 1 Checkpoint")
    print("=" * 80)
    print()
    print("Testing Monte Carlo agents with different rollout counts:")
    print("  10, 50, 100, 200, 500 rollouts")
    print()
    print("Baseline comparisons:")
    print("  - Random agent (absolute skill measure)")
    print("  - Heuristic agent (competitive benchmark)")
    print()
    print("=" * 80)
    print()

    # Configuration
    ROLLOUT_LEVELS = [10, 50, 100, 200, 500]
    GAMES_FPA = 100  # For FPA sanity check
    GAMES_BASELINE = 100  # For vs Random/Heuristic
    GAMES_H2H = 100  # For head-to-head
    GAMES_PERF = 20  # For performance measurement
    seed = 200000

    # Results storage
    fpa_results = {}
    vs_random_results = {}
    vs_heuristic_results = {}
    h2h_results = {}
    perf_results = {}

    # Phase 1: Self-Play FPA (Sanity Check)
    print("PHASE 1: Self-Play FPA (Sanity Check)")
    print("-" * 80)
    print()

    for num_rollouts in ROLLOUT_LEVELS:
        print(f"Testing MC-{num_rollouts} self-play ({GAMES_FPA} games)...")
        result = run_self_play_fpa(num_rollouts, GAMES_FPA, seed)
        fpa_results[num_rollouts] = result
        seed += 10000

        print(f"  P1: {result['p1_wins']}, P2: {result['p2_wins']}, Draws: {result['draws']}")
        print(f"  FPA: {result['fpa']:+.1f}%")
        print()

    print()

    # Phase 2: Absolute Skill (vs Random)
    print("PHASE 2: Absolute Skill (vs Random)")
    print("-" * 80)
    print()

    for num_rollouts in ROLLOUT_LEVELS:
        print(f"Testing MC-{num_rollouts} vs Random ({GAMES_BASELINE} games, balanced)...")
        result = run_vs_baseline(num_rollouts, RandomAgent, "Random", GAMES_BASELINE, seed)
        vs_random_results[num_rollouts] = result
        seed += 10000

        print(f"  MC-{num_rollouts}: {result['mc_wins']} ({result['win_rate']:.1f}%)")
        print(f"  Random: {result['baseline_wins']}")
        print(f"  Draws: {result['draws']}")
        print()

    print()

    # Phase 3: Competitive Benchmark (vs Heuristic)
    print("PHASE 3: Competitive Benchmark (vs Heuristic)")
    print("-" * 80)
    print()

    for num_rollouts in ROLLOUT_LEVELS:
        print(f"Testing MC-{num_rollouts} vs Heuristic ({GAMES_BASELINE} games, balanced)...")
        result = run_vs_baseline(num_rollouts, HeuristicAgent, "Heuristic", GAMES_BASELINE, seed)
        vs_heuristic_results[num_rollouts] = result
        seed += 10000

        print(f"  MC-{num_rollouts}: {result['mc_wins']} ({result['win_rate']:.1f}%)")
        print(f"  Heuristic: {result['baseline_wins']}")
        print(f"  Draws: {result['draws']}")
        print()

    print()

    # Phase 4: Scaling Curve (Head-to-Head)
    print("PHASE 4: Scaling Curve (Head-to-Head)")
    print("-" * 80)
    print()

    for i in range(len(ROLLOUT_LEVELS) - 1):
        rollouts_a = ROLLOUT_LEVELS[i]
        rollouts_b = ROLLOUT_LEVELS[i + 1]

        print(f"Testing MC-{rollouts_a} vs MC-{rollouts_b} ({GAMES_H2H} games, balanced)...")
        result = run_head_to_head(rollouts_a, rollouts_b, GAMES_H2H, seed)
        h2h_results[f"{rollouts_a}_vs_{rollouts_b}"] = result
        seed += 10000

        print(f"  MC-{rollouts_a}: {result['a_wins']} ({result['win_rate']:.1f}%)")
        print(f"  MC-{rollouts_b}: {result['b_wins']} ({100 - result['win_rate'] - result['draws']/GAMES_H2H*100:.1f}%)")
        print(f"  Draws: {result['draws']}")
        print()

    print()

    # Phase 5: Performance Metrics
    print("PHASE 5: Performance Metrics")
    print("-" * 80)
    print()

    for num_rollouts in ROLLOUT_LEVELS:
        print(f"Measuring MC-{num_rollouts} performance ({GAMES_PERF} games)...")
        time_per_game = measure_performance(num_rollouts, GAMES_PERF, seed)
        perf_results[num_rollouts] = time_per_game
        seed += 10000

        print(f"  Time per game: {time_per_game:.3f}s")
        print()

    print()
    print("=" * 80)
    print("COMPREHENSIVE RESULTS")
    print("=" * 80)
    print()

    # Results table
    print(f"{'Rollouts':<12} {'vs Random':<12} {'vs Heuristic':<15} {'Time/Game':<12} {'FPA':<10}")
    print("-" * 80)

    for num_rollouts in ROLLOUT_LEVELS:
        vs_rand = vs_random_results[num_rollouts]['win_rate']
        vs_heur = vs_heuristic_results[num_rollouts]['win_rate']
        time_pg = perf_results[num_rollouts]
        fpa = fpa_results[num_rollouts]['fpa']

        print(f"{num_rollouts:<12} {vs_rand:>6.1f}%      {vs_heur:>7.1f}%        {time_pg:>6.3f}s      {fpa:>+5.1f}%")

    print()
    print("-" * 80)
    print("SCALING CURVE (Head-to-Head Win Rates)")
    print("-" * 80)
    print()

    for i in range(len(ROLLOUT_LEVELS) - 1):
        rollouts_a = ROLLOUT_LEVELS[i]
        rollouts_b = ROLLOUT_LEVELS[i + 1]
        key = f"{rollouts_a}_vs_{rollouts_b}"
        win_rate = h2h_results[key]['win_rate']

        improvement = "=" if abs(win_rate - 50) < 5 else ("+" if win_rate < 50 else "++")
        print(f"  MC-{rollouts_a:>3} vs MC-{rollouts_b:>3}: {win_rate:>5.1f}% (MC-{rollouts_a} perspective) [{improvement}]")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Skill ceiling
    max_vs_random = max(r['win_rate'] for r in vs_random_results.values())
    max_vs_heuristic = max(r['win_rate'] for r in vs_heuristic_results.values())

    print(f"Skill Ceiling:")
    print(f"  Best MC vs Random: {max_vs_random:.1f}%")
    print(f"  Best MC vs Heuristic: {max_vs_heuristic:.1f}%")
    print(f"  Reference: Heuristic vs Random = 65%")
    print()

    # Diminishing returns
    print("Diminishing Returns:")
    for i in range(len(ROLLOUT_LEVELS) - 1):
        rollouts_a = ROLLOUT_LEVELS[i]
        rollouts_b = ROLLOUT_LEVELS[i + 1]
        gain = vs_random_results[rollouts_b]['win_rate'] - vs_random_results[rollouts_a]['win_rate']
        print(f"  {rollouts_a:>3} → {rollouts_b:>3} rollouts: {gain:>+5.1f} points vs Random")
    print()

    # Strategic depth assessment
    total_gain = vs_random_results[ROLLOUT_LEVELS[-1]]['win_rate'] - vs_random_results[ROLLOUT_LEVELS[0]]['win_rate']
    late_gain = vs_random_results[ROLLOUT_LEVELS[-1]]['win_rate'] - vs_random_results[ROLLOUT_LEVELS[-2]]['win_rate']

    print("Strategic Depth Assessment:")
    print(f"  Total gain (10 → 500): {total_gain:+.1f} points")
    print(f"  Late-stage gain (200 → 500): {late_gain:+.1f} points")

    if late_gain > 2.0:
        depth = "Excellent - continued scaling suggests rich strategy space"
    elif late_gain > 0.5:
        depth = "Good - modest gains still available with more search"
    else:
        depth = "Limited - plateau reached, suggests skill ceiling"

    print(f"  Assessment: {depth}")
    print()

    # Phase 2 baseline
    mc50_vs_random = vs_random_results[50]['win_rate']
    print(f"Phase 2 Baseline Target:")
    print(f"  MC-50 achieves {mc50_vs_random:.1f}% vs Random")
    print(f"  Learning agents should aim to beat this consistently")
    print(f"  Stretch goal: Match or beat best MC ({max_vs_random:.1f}% vs Random)")
    print()

    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
