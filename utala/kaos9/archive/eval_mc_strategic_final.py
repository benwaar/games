#!/usr/bin/env python3
"""
Final evaluation of MC-strategic as the new Phase 1 baseline.

100 games each configuration to establish robust baseline numbers.
"""

import sys
import time
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import MonteCarloAgent
from utala.evaluation.harness import Harness


def evaluate_vs_opponent(agent_name, agent, opponent_name, opponent, num_games, seed):
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
    print(f"  {opponent_name}: {opp_wins} wins")
    print(f"  Draws: {draws}")

    return win_rate, agent_wins, opp_wins, draws, elapsed


def main():
    print("=" * 80)
    print("MC-STRATEGIC FINAL EVALUATION (100 games per matchup)")
    print("=" * 80)
    print()
    print("Establishing baseline numbers for Phase 1 completion:")
    print("  1. MC-strategic vs Random")
    print("  2. MC-strategic vs Heuristic v1.8")
    print("  3. MC-baseline vs Random (reference)")
    print("  4. MC-baseline vs Heuristic (reference)")
    print()
    print("=" * 80)

    num_games = 100
    seed = 500000

    # Create agents
    mc_strategic = MonteCarloAgent(
        "MC-strategic",
        num_rollouts=10,
        seed=seed,
        evaluate_dogfights=True
    )

    mc_baseline = MonteCarloAgent(
        "MC-baseline",
        num_rollouts=10,
        seed=seed + 10000
    )

    random_agent = RandomAgent("Random", seed=seed + 20000)
    heuristic = HeuristicAgent("Heuristic", seed=seed + 30000)

    # Test 1: MC-strategic vs Random
    print("\nTEST 1: MC-STRATEGIC VS RANDOM")
    print("-" * 80)
    strat_vs_rand_wr, strat_vs_rand_w, _, _, strat_rand_time = evaluate_vs_opponent(
        "MC-strategic", mc_strategic,
        "Random", RandomAgent("Random-2", seed=seed + 40000),
        num_games, seed + 100000
    )

    # Test 2: MC-strategic vs Heuristic
    print("\nTEST 2: MC-STRATEGIC VS HEURISTIC")
    print("-" * 80)
    strat_vs_heur_wr, strat_vs_heur_w, heur_vs_strat_w, _, strat_heur_time = evaluate_vs_opponent(
        "MC-strategic", MonteCarloAgent("MC-strategic-2", num_rollouts=10,
                                        seed=seed + 50000, evaluate_dogfights=True),
        "Heuristic", HeuristicAgent("Heuristic-2", seed=seed + 60000),
        num_games, seed + 200000
    )

    # Test 3: MC-baseline vs Random (reference)
    print("\nTEST 3: MC-BASELINE VS RANDOM (Reference)")
    print("-" * 80)
    base_vs_rand_wr, base_vs_rand_w, _, _, base_rand_time = evaluate_vs_opponent(
        "MC-baseline", mc_baseline,
        "Random", RandomAgent("Random-3", seed=seed + 70000),
        num_games, seed + 300000
    )

    # Test 4: MC-baseline vs Heuristic (reference)
    print("\nTEST 4: MC-BASELINE VS HEURISTIC (Reference)")
    print("-" * 80)
    base_vs_heur_wr, base_vs_heur_w, heur_vs_base_w, _, base_heur_time = evaluate_vs_opponent(
        "MC-baseline", MonteCarloAgent("MC-baseline-2", num_rollouts=10,
                                       seed=seed + 80000),
        "Heuristic", HeuristicAgent("Heuristic-3", seed=seed + 90000),
        num_games, seed + 400000
    )

    # Summary
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()

    print(f"{'Agent':<20} {'vs Random':<15} {'vs Heuristic':<15} {'Description':<30}")
    print("-" * 80)
    print(f"{'MC-Strategic (10)':<20} {strat_vs_rand_wr:>6.1f}%        {strat_vs_heur_wr:>6.1f}%        {'Strategic placement + dogfights':<30}")
    print(f"{'MC-baseline (10)':<20} {base_vs_rand_wr:>6.1f}%        {base_vs_heur_wr:>6.1f}%        {'Smart placement, random dogfights':<30}")
    print()

    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # vs Random comparison
    rand_improvement = strat_vs_rand_wr - base_vs_rand_wr
    print(f"1. vs Random:")
    print(f"   MC-strategic: {strat_vs_rand_wr:.1f}%")
    print(f"   MC-baseline:  {base_vs_rand_wr:.1f}%")
    print(f"   Improvement: {rand_improvement:+.1f} percentage points")
    if abs(rand_improvement) < 2:
        print(f"   → Placement dominates vs Random (dogfights less critical)")
    print()

    # vs Heuristic comparison
    heur_improvement = strat_vs_heur_wr - base_vs_heur_wr
    print(f"2. vs Heuristic:")
    print(f"   MC-strategic: {strat_vs_heur_wr:.1f}%")
    print(f"   MC-baseline:  {base_vs_heur_wr:.1f}%")
    print(f"   Improvement: {heur_improvement:+.1f} percentage points")
    if heur_improvement > 20:
        print(f"   → HUGE improvement! Strategic dogfights are critical vs strategic opponents")
    elif heur_improvement > 10:
        print(f"   → Significant improvement from strategic dogfight evaluation")
    print()

    # Skill ladder
    print(f"3. Skill Ladder:")
    print(f"   Random baseline")
    print(f"   MC-baseline: {base_vs_rand_wr:.1f}% vs Random")
    print(f"   Heuristic: {100 - base_vs_heur_wr:.1f}% vs MC-baseline " +
          f"(~{100 - strat_vs_heur_wr:.1f}% vs MC-strategic)")
    print(f"   MC-strategic: {strat_vs_heur_wr:.1f}% vs Heuristic (NEW CEILING)")
    print()

    # Performance
    print(f"4. Performance:")
    print(f"   MC-strategic: {strat_rand_time:.1f}s for 100 games vs Random " +
          f"({100/strat_rand_time*60:.1f} games/min)")
    print(f"   MC-baseline:  {base_rand_time:.1f}s for 100 games vs Random " +
          f"({100/base_rand_time*60:.1f} games/min)")
    time_mult = strat_rand_time / base_rand_time
    print(f"   Cost: {time_mult:.2f}x slower (strategic dogfight evaluation)")
    print()

    print("=" * 80)
    print("PHASE 1 BASELINE (NEW)")
    print("=" * 80)
    print()
    print(f"**MC-Strategic (10 rollouts)**")
    print(f"  - vs Random: {strat_vs_rand_wr:.1f}%")
    print(f"  - vs Heuristic: {strat_vs_heur_wr:.1f}%")
    print(f"  - Evaluates placement AND dogfight actions via rollouts")
    print(f"  - Performance: ~{100/strat_rand_time*60:.1f} games/min")
    print()
    print(f"**Phase 2 Target**: Learning agents should aim for >{strat_vs_rand_wr:.1f}% vs Random")
    print()

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
