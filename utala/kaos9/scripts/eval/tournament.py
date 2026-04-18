#!/usr/bin/env python3
"""
Tournament evaluation script for utala: kaos 9.

Runs round-robin tournament between all baseline agents and reports:
- Overall win rates
- Head-to-head matchups
- Skill expression analysis
"""

import sys
sys.path.insert(0, 'src')

from dataclasses import dataclass
from typing import Dict, Tuple

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent, MonteCarloAgent
from utala.evaluation.harness import Harness
from utala.state import Player


@dataclass
class TournamentResult:
    """Results from a round-robin tournament."""
    agent_names: list[str]
    wins: Dict[str, int]
    losses: Dict[str, int]
    draws: Dict[str, int]
    head_to_head: Dict[Tuple[str, str], Tuple[int, int, int]]  # (wins, losses, draws)

    def total_games(self, agent_name: str) -> int:
        """Total games played by an agent."""
        return self.wins[agent_name] + self.losses[agent_name] + self.draws[agent_name]

    def win_rate(self, agent_name: str) -> float:
        """Win rate for an agent."""
        total = self.total_games(agent_name)
        return self.wins[agent_name] / total if total > 0 else 0.0

    def print_summary(self):
        """Print tournament summary."""
        print("\n" + "=" * 70)
        print("TOURNAMENT RESULTS")
        print("=" * 70)
        print()

        # Overall standings
        print("Overall Standings:")
        print("-" * 70)
        print(f"{'Agent':<20} {'Wins':>8} {'Losses':>8} {'Draws':>8} {'Win Rate':>10}")
        print("-" * 70)

        # Sort by win rate
        sorted_agents = sorted(
            self.agent_names,
            key=lambda name: self.win_rate(name),
            reverse=True
        )

        for agent_name in sorted_agents:
            wins = self.wins[agent_name]
            losses = self.losses[agent_name]
            draws = self.draws[agent_name]
            win_rate = self.win_rate(agent_name)
            print(f"{agent_name:<20} {wins:>8} {losses:>8} {draws:>8} {win_rate:>9.1%}")

        print()

        # Head-to-head results
        print("Head-to-Head Results (Row vs Column):")
        print("-" * 70)

        # Print header
        header = "Agent".ljust(20)
        for agent_name in sorted_agents:
            header += agent_name[:10].center(12)
        print(header)
        print("-" * 70)

        # Print matrix
        for agent1 in sorted_agents:
            row = agent1[:20].ljust(20)
            for agent2 in sorted_agents:
                if agent1 == agent2:
                    row += "-".center(12)
                else:
                    wins, losses, draws = self.head_to_head.get((agent1, agent2), (0, 0, 0))
                    total = wins + losses + draws
                    win_rate = wins / total if total > 0 else 0.0
                    row += f"{win_rate:.1%}".center(12)
            print(row)

        print()


def run_tournament(
    agents: list,
    games_per_matchup: int = 20,
    verbose: bool = False
) -> TournamentResult:
    """
    Run a round-robin tournament between agents.

    Args:
        agents: List of agents to compete
        games_per_matchup: Number of games per matchup (played in both directions)
        verbose: If True, print game-by-game results

    Returns:
        TournamentResult with all statistics
    """
    harness = Harness(verbose=verbose)

    # Initialize results
    agent_names = [agent.name for agent in agents]
    wins = {name: 0 for name in agent_names}
    losses = {name: 0 for name in agent_names}
    draws = {name: 0 for name in agent_names}
    head_to_head = {}

    print(f"\nRunning tournament with {len(agents)} agents...")
    print(f"Games per matchup: {games_per_matchup} (x2 for both player positions)")
    print()

    # Run round-robin
    total_matchups = len(agents) * (len(agents) - 1) // 2
    matchup_num = 0

    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i >= j:
                continue

            matchup_num += 1
            print(f"[{matchup_num}/{total_matchups}] {agent1.name} vs {agent2.name}...", end=" ", flush=True)

            # Play games with agent1 as P1
            result_as_p1 = harness.run_match(
                agent1, agent2, num_games=games_per_matchup // 2, starting_seed=1000 + i * 100 + j
            )

            # Play games with agent1 as P2
            result_as_p2 = harness.run_match(
                agent2, agent1, num_games=games_per_matchup // 2, starting_seed=2000 + i * 100 + j
            )

            # Aggregate results for agent1
            agent1_wins = result_as_p1.player_one_wins + result_as_p2.player_two_wins
            agent1_losses = result_as_p1.player_two_wins + result_as_p2.player_one_wins
            agent1_draws = result_as_p1.draws + result_as_p2.draws

            # Update standings
            wins[agent1.name] += agent1_wins
            losses[agent1.name] += agent1_losses
            draws[agent1.name] += agent1_draws

            wins[agent2.name] += agent1_losses
            losses[agent2.name] += agent1_wins
            draws[agent2.name] += agent1_draws

            # Record head-to-head
            head_to_head[(agent1.name, agent2.name)] = (agent1_wins, agent1_losses, agent1_draws)
            head_to_head[(agent2.name, agent1.name)] = (agent1_losses, agent1_wins, agent1_draws)

            print(f"{agent1.name} {agent1_wins}-{agent1_losses}-{agent1_draws} {agent2.name}")

    return TournamentResult(
        agent_names=agent_names,
        wins=wins,
        losses=losses,
        draws=draws,
        head_to_head=head_to_head
    )


def analyze_skill_expression(result: TournamentResult):
    """
    Analyze whether the game shows skill expression.

    Skill expression means: better agents consistently beat weaker agents.
    """
    print("=" * 70)
    print("SKILL EXPRESSION ANALYSIS")
    print("=" * 70)
    print()

    # Sort agents by win rate
    sorted_agents = sorted(
        result.agent_names,
        key=lambda name: result.win_rate(name),
        reverse=True
    )

    print("Expected skill hierarchy (based on design):")
    print("  1. Monte Carlo (looks ahead)")
    print("  2. Heuristic (uses strategy)")
    print("  3. Random (no strategy)")
    print()

    print("Actual results:")
    for i, agent_name in enumerate(sorted_agents, 1):
        win_rate = result.win_rate(agent_name)
        print(f"  {i}. {agent_name:<20} (win rate: {win_rate:.1%})")
    print()

    # Check if skill hierarchy is respected
    print("Skill expression test:")

    # Random should lose to both Heuristic and MC
    random_agents = [name for name in result.agent_names if "Random" in name]
    heuristic_agents = [name for name in result.agent_names if "Heuristic" in name]
    mc_agents = [name for name in result.agent_names if "MC" in name or "Monte" in name]

    tests_passed = 0
    tests_total = 0

    if random_agents and heuristic_agents:
        for rand in random_agents:
            for heur in heuristic_agents:
                if (heur, rand) in result.head_to_head:
                    wins, losses, _ = result.head_to_head[(heur, rand)]
                    total = wins + losses
                    if total > 0:
                        heur_win_rate = wins / total
                        tests_total += 1
                        if heur_win_rate > 0.55:  # Should win >55% of time
                            tests_passed += 1
                            print(f"  ✓ {heur} beats {rand} ({heur_win_rate:.1%})")
                        else:
                            print(f"  ✗ {heur} does NOT consistently beat {rand} ({heur_win_rate:.1%})")

    if random_agents and mc_agents:
        for rand in random_agents:
            for mc in mc_agents:
                if (mc, rand) in result.head_to_head:
                    wins, losses, _ = result.head_to_head[(mc, rand)]
                    total = wins + losses
                    if total > 0:
                        mc_win_rate = wins / total
                        tests_total += 1
                        if mc_win_rate > 0.55:
                            tests_passed += 1
                            print(f"  ✓ {mc} beats {rand} ({mc_win_rate:.1%})")
                        else:
                            print(f"  ✗ {mc} does NOT consistently beat {rand} ({mc_win_rate:.1%})")

    if heuristic_agents and mc_agents:
        for heur in heuristic_agents:
            for mc in mc_agents:
                if (mc, heur) in result.head_to_head:
                    wins, losses, _ = result.head_to_head[(mc, heur)]
                    total = wins + losses
                    if total > 0:
                        mc_win_rate = wins / total
                        tests_total += 1
                        if mc_win_rate > 0.52:  # Smaller margin (both are good)
                            tests_passed += 1
                            print(f"  ✓ {mc} beats {heur} ({mc_win_rate:.1%})")
                        else:
                            print(f"  ~ {mc} vs {heur} is close ({mc_win_rate:.1%})")

    print()
    print(f"Skill expression: {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        print("✓ Game shows clear skill expression!")
    elif tests_passed >= tests_total * 0.7:
        print("~ Game shows moderate skill expression")
    else:
        print("✗ Game may not show sufficient skill expression")

    print()


def main():
    """Run Phase 1 baseline tournament."""
    print("=" * 70)
    print("utala: kaos 9 - Phase 1 Baseline Tournament")
    print("=" * 70)

    # Create agents
    agents = [
        RandomAgent("Random", seed=42),
        HeuristicAgent("Heuristic", seed=42),
        FastMonteCarloAgent("FastMC", seed=42),
        MonteCarloAgent("MonteCarlo-50", seed=42),
    ]

    print(f"\nAgents: {', '.join(agent.name for agent in agents)}")

    # Run tournament
    result = run_tournament(agents, games_per_matchup=20, verbose=False)

    # Print results
    result.print_summary()

    # Analyze skill expression
    analyze_skill_expression(result)

    print("=" * 70)
    print("Tournament complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
