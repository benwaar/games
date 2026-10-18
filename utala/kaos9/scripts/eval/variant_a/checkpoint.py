#!/usr/bin/env python3
"""
Phase 4 Variant A Checkpoint — Choosable Dogfight Order

Re-runs Phase 1 baseline evaluation with GameConfig(fixed_dogfight_order=False).
Does skill still beat luck? Is the skill hierarchy preserved?

Baseline comparison (Phase 1, fixed order):
  Heuristic vs Random: ~65%
  MC-Fast vs Random: ~79%
  MC-Fast vs Heuristic: ~72%
"""

import sys
import time
sys.path.insert(0, 'src')

from dataclasses import dataclass
from typing import Dict, Tuple

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.evaluation.harness import Harness
from utala.state import GameConfig

# Variant A config
CONFIG = GameConfig(fixed_dogfight_order=False)


def evaluate_balanced(harness, agent_name, agent, opponent_name, opponent, num_games, seed):
    """Run balanced evaluation (half as P1, half as P2)."""
    print(f"\n{agent_name} vs {opponent_name} ({num_games} games, balanced)...")
    print(f"  Running...", end=" ", flush=True)

    start_time = time.time()
    result1 = harness.run_match(agent, opponent, num_games // 2, seed)
    result2 = harness.run_match(opponent, agent, num_games // 2, seed + num_games // 2)
    elapsed = time.time() - start_time

    agent_wins = result1.player_one_wins + result2.player_two_wins
    opp_wins = result1.player_two_wins + result2.player_one_wins
    draws = result1.draws + result2.draws
    win_rate = agent_wins / num_games * 100

    print(f"Done! ({elapsed:.1f}s)")
    print(f"  {agent_name}: {agent_wins} wins ({win_rate:.1f}%)")
    print(f"  {opponent_name}: {opp_wins} wins ({opp_wins/num_games*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/num_games*100:.1f}%)")

    return {
        'win_rate': win_rate,
        'wins': agent_wins,
        'losses': opp_wins,
        'draws': draws,
        'time': elapsed
    }


def run_tournament(harness, agents, games_per_matchup=20):
    """Run round-robin tournament."""
    agent_names = [a.name for a in agents]
    wins = {n: 0 for n in agent_names}
    losses = {n: 0 for n in agent_names}
    draws = {n: 0 for n in agent_names}
    head_to_head = {}

    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents):
            if i >= j:
                continue

            print(f"  {a1.name} vs {a2.name}...", end=" ", flush=True)

            r1 = harness.run_match(a1, a2, games_per_matchup // 2, 1000 + i * 100 + j)
            r2 = harness.run_match(a2, a1, games_per_matchup // 2, 2000 + i * 100 + j)

            a1_w = r1.player_one_wins + r2.player_two_wins
            a1_l = r1.player_two_wins + r2.player_one_wins
            a1_d = r1.draws + r2.draws

            wins[a1.name] += a1_w
            losses[a1.name] += a1_l
            draws[a1.name] += a1_d
            wins[a2.name] += a1_l
            losses[a2.name] += a1_w
            draws[a2.name] += a1_d

            head_to_head[(a1.name, a2.name)] = (a1_w, a1_l, a1_d)
            head_to_head[(a2.name, a1.name)] = (a1_l, a1_w, a1_d)

            print(f"{a1_w}-{a1_l}-{a1_d}")

    return agent_names, wins, losses, draws, head_to_head


def main():
    print("=" * 80)
    print("PHASE 4 VARIANT A CHECKPOINT")
    print("Choosable Dogfight Order (fixed_dogfight_order=False)")
    print("=" * 80)
    print()

    harness = Harness(verbose=False, config=CONFIG)
    num_games = 100
    seed = 700000

    # --- Key matchups ---

    print("=" * 80)
    print("TEST 1: HEURISTIC VS RANDOM")
    print("=" * 80)

    heur = HeuristicAgent("Heuristic", seed=seed, config=CONFIG)
    rand = RandomAgent("Random", seed=seed + 10000)

    heur_vs_random = evaluate_balanced(harness, "Heuristic", heur, "Random", rand, num_games, seed)

    print("\n" + "=" * 80)
    print("TEST 2: MC-FAST VS RANDOM")
    print("=" * 80)

    mc = FastMonteCarloAgent("MC-Fast", seed=seed + 20000)
    rand2 = RandomAgent("Random", seed=seed + 30000)

    mc_vs_random = evaluate_balanced(harness, "MC-Fast", mc, "Random", rand2, num_games, seed + 100000)

    print("\n" + "=" * 80)
    print("TEST 3: MC-FAST VS HEURISTIC")
    print("=" * 80)

    mc2 = FastMonteCarloAgent("MC-Fast", seed=seed + 40000)
    heur2 = HeuristicAgent("Heuristic", seed=seed + 50000, config=CONFIG)

    mc_vs_heur = evaluate_balanced(harness, "MC-Fast", mc2, "Heuristic", heur2, num_games, seed + 200000)

    # --- Tournament ---

    print("\n" + "=" * 80)
    print("ROUND-ROBIN TOURNAMENT (20 games per matchup)")
    print("=" * 80)

    agents = [
        RandomAgent("Random", seed=42),
        HeuristicAgent("Heuristic", seed=42, config=CONFIG),
        FastMonteCarloAgent("FastMC", seed=42),
    ]

    names, wins, losses, draws, h2h = run_tournament(harness, agents, games_per_matchup=20)

    # Print standings
    print()
    print(f"{'Agent':<15} {'Wins':>6} {'Losses':>8} {'Draws':>7} {'Win%':>7}")
    print("-" * 45)
    for name in sorted(names, key=lambda n: wins[n] / max(wins[n] + losses[n] + draws[n], 1), reverse=True):
        total = wins[name] + losses[name] + draws[name]
        wr = wins[name] / total * 100 if total else 0
        print(f"{name:<15} {wins[name]:>6} {losses[name]:>8} {draws[name]:>7} {wr:>6.1f}%")

    # --- Summary ---

    print("\n" + "=" * 80)
    print("VARIANT A CHECKPOINT RESULTS")
    print("=" * 80)
    print()
    print(f"{'Matchup':<30} {'Variant A':>12} {'Baseline':>12} {'Delta':>8}")
    print("-" * 65)
    print(f"{'Heuristic vs Random':<30} {heur_vs_random['win_rate']:>11.1f}% {'~65.0':>11}% {heur_vs_random['win_rate'] - 65.0:>+7.1f}%")
    print(f"{'MC-Fast vs Random':<30} {mc_vs_random['win_rate']:>11.1f}% {'~79.0':>11}% {mc_vs_random['win_rate'] - 79.0:>+7.1f}%")
    print(f"{'MC-Fast vs Heuristic':<30} {mc_vs_heur['win_rate']:>11.1f}% {'~72.0':>11}% {mc_vs_heur['win_rate'] - 72.0:>+7.1f}%")
    print()

    # Pass/fail criteria (Variant A — adapted for changed game dynamics)
    checks = []

    check = heur_vs_random['win_rate'] > 50
    checks.append(check)
    print(f"{'PASS' if check else 'FAIL'}: Strategy beats luck — Heuristic > 50% vs Random ({heur_vs_random['win_rate']:.1f}%)")

    # Tournament order: at least one strategic agent > Random
    sorted_tournament = sorted(names, key=lambda n: wins[n] / max(wins[n] + losses[n] + draws[n], 1), reverse=True)
    check = sorted_tournament[-1] == "Random"
    checks.append(check)
    print(f"{'PASS' if check else 'FAIL'}: Random is weakest in tournament (actual weakest: {sorted_tournament[-1]})")

    draw_pct = (heur_vs_random['draws'] + mc_vs_random['draws'] + mc_vs_heur['draws']) / (3 * num_games) * 100
    check = draw_pct < 10
    checks.append(check)
    print(f"{'PASS' if check else 'FAIL'}: Draw rate < 10% (actual: {draw_pct:.1f}%)")

    print()
    if all(checks):
        print("CHECKPOINT PASSED — Variant A shows skill expression")
    else:
        print("CHECKPOINT ISSUES — review results above")

    print("=" * 80)


if __name__ == "__main__":
    main()
