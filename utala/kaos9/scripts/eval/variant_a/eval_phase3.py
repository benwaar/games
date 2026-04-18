#!/usr/bin/env python3
"""
Phase 4 Variant A — Key Phase 3 Metrics

Re-runs the most important Phase 3 evaluations with choosable dogfight order.

Baseline comparison (Phase 3, fixed order):
  MC-Fair vs MC-Perfect: 28% (perfect info worth +12%)
  MC-Fair vs Heuristic: 54%
  MC-Perfect vs Heuristic: 66%
"""

import sys
import time
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness
from utala.state import GameConfig

CONFIG = GameConfig(fixed_dogfight_order=False)


def run_matchup(harness, agent1, agent2, num_games=50, seed_offset=0):
    """Run balanced matchup and return win rates."""
    half = num_games // 2
    r1 = harness.run_match(agent1, agent2, half, starting_seed=1000 + seed_offset)
    r2 = harness.run_match(agent2, agent1, half, starting_seed=2000 + seed_offset)

    a1_wins = r1.player_one_wins + r2.player_two_wins
    a1_losses = r1.player_two_wins + r2.player_one_wins
    a1_draws = r1.draws + r2.draws
    total = a1_wins + a1_losses + a1_draws

    return {
        'wins': a1_wins,
        'losses': a1_losses,
        'draws': a1_draws,
        'win_rate': a1_wins / total if total > 0 else 0.0
    }


def main():
    print("=" * 70)
    print("VARIANT A: PHASE 3 KEY METRICS")
    print("Choosable Dogfight Order (fixed_dogfight_order=False)")
    print("=" * 70)
    print()

    harness = Harness(verbose=False, config=CONFIG)
    num_games = 50

    # --- Information set impact (Phase 3.2 replay) ---

    print("=" * 70)
    print("INFORMATION SET IMPACT (Phase 3.2 replay)")
    print("=" * 70)
    print()

    mc_fair = FastMonteCarloAgent("MC-Fair", seed=42, use_information_sets=True)
    mc_perfect = FastMonteCarloAgent("MC-Perfect", seed=42, use_information_sets=False)
    heuristic = HeuristicAgent("Heuristic", seed=42, config=CONFIG)

    start = time.time()

    print(f"[1/5] MC-Fair vs MC-Perfect (head-to-head)...", end=" ", flush=True)
    r_h2h = run_matchup(harness, mc_fair, mc_perfect, num_games, seed_offset=100)
    print(f"MC-Fair: {r_h2h['win_rate']:.1%}")

    print(f"[2/5] MC-Fair vs Heuristic...", end=" ", flush=True)
    r_fair_heur = run_matchup(harness, mc_fair, heuristic, num_games, seed_offset=200)
    print(f"MC-Fair: {r_fair_heur['win_rate']:.1%}")

    print(f"[3/5] MC-Perfect vs Heuristic...", end=" ", flush=True)
    r_perf_heur = run_matchup(harness, mc_perfect, heuristic, num_games, seed_offset=300)
    print(f"MC-Perfect: {r_perf_heur['win_rate']:.1%}")

    print(f"[4/5] MC-Fair vs Random...", end=" ", flush=True)
    rand = RandomAgent("Random", seed=42)
    r_fair_rand = run_matchup(harness, mc_fair, rand, num_games, seed_offset=400)
    print(f"MC-Fair: {r_fair_rand['win_rate']:.1%}")

    print(f"[5/5] MC-Perfect vs Random...", end=" ", flush=True)
    rand2 = RandomAgent("Random", seed=43)
    r_perf_rand = run_matchup(harness, mc_perfect, rand2, num_games, seed_offset=500)
    print(f"MC-Perfect: {r_perf_rand['win_rate']:.1%}")

    elapsed = time.time() - start

    # --- Summary ---

    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Matchup':<35} {'Variant A':>10} {'Baseline':>10} {'Delta':>8}")
    print("-" * 65)
    print(f"{'MC-Fair vs MC-Perfect':<35} {r_h2h['win_rate']:>9.1%} {'28.0%':>10} {r_h2h['win_rate']*100 - 28.0:>+7.1f}%")
    print(f"{'MC-Fair vs Heuristic':<35} {r_fair_heur['win_rate']:>9.1%} {'54.0%':>10} {r_fair_heur['win_rate']*100 - 54.0:>+7.1f}%")
    print(f"{'MC-Perfect vs Heuristic':<35} {r_perf_heur['win_rate']:>9.1%} {'66.0%':>10} {r_perf_heur['win_rate']*100 - 66.0:>+7.1f}%")
    print(f"{'MC-Fair vs Random':<35} {r_fair_rand['win_rate']:>9.1%} {'58.0%':>10} {r_fair_rand['win_rate']*100 - 58.0:>+7.1f}%")
    print(f"{'MC-Perfect vs Random':<35} {r_perf_rand['win_rate']:>9.1%} {'56.0%':>10} {r_perf_rand['win_rate']*100 - 56.0:>+7.1f}%")
    print()

    # Hidden info gap
    info_gap = r_perf_heur['win_rate'] - r_fair_heur['win_rate']
    baseline_gap = 0.66 - 0.54  # 12% in baseline
    print(f"Hidden info gap (vs Heuristic): {info_gap:.1%} (baseline: {baseline_gap:.1%})")
    if info_gap > baseline_gap:
        print("  -> Hidden info MORE valuable with choosable order")
    elif info_gap < baseline_gap * 0.5:
        print("  -> Hidden info LESS valuable with choosable order")
    else:
        print("  -> Hidden info gap similar to baseline")
    print()

    print(f"Time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
