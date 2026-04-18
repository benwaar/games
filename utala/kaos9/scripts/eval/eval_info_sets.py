#!/usr/bin/env python3
"""
Phase 3.2 evaluation: Impact of information set sampling on MC agents.

Compares MC agents with perfect info vs incomplete info (information sets).
Uses FastMC (10 rollouts) for practical runtime.
"""

import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness
from utala.state import Player


def run_matchup(harness, agent1, agent2, num_games=100):
    """Run balanced matchup and return win rates."""
    half = num_games // 2
    result_p1 = harness.run_match(agent1, agent2, num_games=half, starting_seed=1000)
    result_p2 = harness.run_match(agent2, agent1, num_games=half, starting_seed=2000)

    a1_wins = result_p1.player_one_wins + result_p2.player_two_wins
    a1_losses = result_p1.player_two_wins + result_p2.player_one_wins
    a1_draws = result_p1.draws + result_p2.draws
    total = a1_wins + a1_losses + a1_draws

    return {
        'wins': a1_wins,
        'losses': a1_losses,
        'draws': a1_draws,
        'win_rate': a1_wins / total if total > 0 else 0.0
    }


import builtins as _b
def p(msg=""):
    """Print with flush for non-TTY output."""
    _b.print(msg, flush=True)


def main():
    p("=" * 70)
    p("Phase 3.2: Information Set Impact Evaluation")
    p("=" * 70)

    harness = Harness(verbose=False)
    num_games = 50

    # Create agents
    mc_fair = FastMonteCarloAgent("MC-Fair", seed=42, use_information_sets=True)
    mc_perfect = FastMonteCarloAgent("MC-Perfect", seed=42, use_information_sets=False)
    random_agent = RandomAgent("Random", seed=42)
    heuristic = HeuristicAgent("Heuristic", seed=42)

    p(f"\nGames per matchup: {num_games} (balanced)")
    p()

    # 1. Direct comparison: MC-Fair vs MC-Perfect
    p("[1/5] MC-Fair vs MC-Perfect (head-to-head)...")
    r = run_matchup(harness, mc_fair, mc_perfect, num_games)
    p(f"  MC-Fair: {r['win_rate']:.1%} ({r['wins']}W-{r['losses']}L-{r['draws']}D)")
    p()

    # 2. Both vs Random
    p("[2/5] MC-Fair vs Random...")
    r_fair_rand = run_matchup(harness, mc_fair, random_agent, num_games)
    p(f"  MC-Fair: {r_fair_rand['win_rate']:.1%}")

    p("[3/5] MC-Perfect vs Random...")
    r_perf_rand = run_matchup(harness, mc_perfect, random_agent, num_games)
    p(f"  MC-Perfect: {r_perf_rand['win_rate']:.1%}")
    p()

    # 3. Both vs Heuristic
    p("[4/5] MC-Fair vs Heuristic...")
    r_fair_heur = run_matchup(harness, mc_fair, heuristic, num_games)
    p(f"  MC-Fair: {r_fair_heur['win_rate']:.1%}")

    p("[5/5] MC-Perfect vs Heuristic...")
    r_perf_heur = run_matchup(harness, mc_perfect, heuristic, num_games)
    p(f"  MC-Perfect: {r_perf_heur['win_rate']:.1%}")
    p()

    # Summary
    p("=" * 70)
    p("SUMMARY: Impact of Information Sets")
    p("=" * 70)
    p()
    p(f"{'Matchup':<30} {'MC-Fair':>10} {'MC-Perfect':>12} {'Delta':>8}")
    p("-" * 62)
    p(f"{'vs Random':<30} {r_fair_rand['win_rate']:>9.1%} {r_perf_rand['win_rate']:>11.1%} {r_perf_rand['win_rate'] - r_fair_rand['win_rate']:>+7.1%}")
    p(f"{'vs Heuristic':<30} {r_fair_heur['win_rate']:>9.1%} {r_perf_heur['win_rate']:>11.1%} {r_perf_heur['win_rate'] - r_fair_heur['win_rate']:>+7.1%}")
    p(f"{'Head-to-head (Fair wins)':<30} {r['win_rate']:>9.1%}")
    p()

    if r_perf_rand['win_rate'] - r_fair_rand['win_rate'] > 0.05:
        p("Finding: Perfect info gives MC a meaningful advantage.")
        p("Prior MC benchmarks were inflated — fair comparison is more accurate.")
    else:
        p("Finding: Hidden information has minimal impact on MC strength.")
        p("Prior results are approximately valid.")
    p()


if __name__ == "__main__":
    main()
