"""
Extended metrics collection for Phase 2 evaluation.

Tracks detailed game statistics beyond just win/loss:
- Weapon usage patterns
- Rocket effectiveness
- Comeback rates
- First-player advantage
- Decision time performance
"""

from dataclasses import dataclass, field
from typing import List, Dict
import time

from ..state import Player, GameState
from ..actions import Action, ActionType


@dataclass
class GameMetrics:
    """Detailed metrics for a single game."""

    # Basic outcome
    winner: Player | None
    num_turns: int
    seed: int

    # Weapon usage
    weapons_used_p1: int = 0
    weapons_used_p2: int = 0
    rockets_fired_p1: int = 0  # Offensive use
    rockets_fired_p2: int = 0
    flares_deployed_p1: int = 0  # Defensive use
    flares_deployed_p2: int = 0

    # Rocket effectiveness
    rockets_hit: int = 0  # Rocket hit (no flare)
    rockets_deflected: int = 0  # Rocket blocked by flare
    dogfights_both_removed: int = 0  # Both rocketmen removed

    # Comeback tracking
    max_advantage_p1: int = 0  # Max squares controlled lead
    max_advantage_p2: int = 0
    final_squares_p1: int = 0
    final_squares_p2: int = 0

    # Decision time (for performance profiling)
    total_decision_time_p1: float = 0.0  # seconds
    total_decision_time_p2: float = 0.0
    num_decisions_p1: int = 0
    num_decisions_p2: int = 0

    @property
    def avg_decision_time_p1(self) -> float:
        """Average decision time for P1 (milliseconds)."""
        if self.num_decisions_p1 == 0:
            return 0.0
        return (self.total_decision_time_p1 / self.num_decisions_p1) * 1000

    @property
    def avg_decision_time_p2(self) -> float:
        """Average decision time for P2 (milliseconds)."""
        if self.num_decisions_p2 == 0:
            return 0.0
        return (self.total_decision_time_p2 / self.num_decisions_p2) * 1000

    @property
    def comeback_happened(self) -> bool:
        """Did the underdog (in squares controlled) come back to win?"""
        if self.winner is None:
            return False

        # Check if winner was ever significantly behind
        if self.winner == Player.ONE:
            return self.max_advantage_p2 >= 2  # P2 was ahead by 2+ squares
        else:
            return self.max_advantage_p1 >= 2


@dataclass
class DetailedMatchResult:
    """Extended match results with detailed metrics."""

    player_one_agent: str
    player_two_agent: str
    num_games: int

    # Basic outcomes
    player_one_wins: int = 0
    player_two_wins: int = 0
    draws: int = 0

    # Detailed metrics per game
    game_metrics: List[GameMetrics] = field(default_factory=list)

    @property
    def first_player_advantage(self) -> float:
        """First player win rate (should be ~52-56% for balanced game)."""
        if self.num_games == 0:
            return 0.0
        return self.player_one_wins / self.num_games

    @property
    def avg_weapons_used_p1(self) -> float:
        """Average weapons used by P1 per game."""
        if not self.game_metrics:
            return 0.0
        return sum(m.weapons_used_p1 for m in self.game_metrics) / len(self.game_metrics)

    @property
    def avg_weapons_used_p2(self) -> float:
        """Average weapons used by P2 per game."""
        if not self.game_metrics:
            return 0.0
        return sum(m.weapons_used_p2 for m in self.game_metrics) / len(self.game_metrics)

    @property
    def rocket_hit_rate(self) -> float:
        """Rate at which rockets hit (not deflected by flare)."""
        total_rockets = sum(m.rockets_hit + m.rockets_deflected for m in self.game_metrics)
        if total_rockets == 0:
            return 0.0
        total_hits = sum(m.rockets_hit for m in self.game_metrics)
        return total_hits / total_rockets

    @property
    def wipeout_rate(self) -> float:
        """Rate of dogfights where both rocketmen were removed."""
        total_dogfights = sum(
            m.weapons_used_p1 + m.weapons_used_p2
            for m in self.game_metrics
        )
        if total_dogfights == 0:
            return 0.0
        total_wipeouts = sum(m.dogfights_both_removed for m in self.game_metrics)
        return total_wipeouts / total_dogfights

    @property
    def comeback_rate(self) -> float:
        """Rate at which underdogs (2+ squares behind) come back to win."""
        if not self.game_metrics:
            return 0.0
        comebacks = sum(1 for m in self.game_metrics if m.comeback_happened)
        return comebacks / len(self.game_metrics)

    @property
    def avg_decision_time_p1(self) -> float:
        """Average decision time for P1 across all games (ms)."""
        if not self.game_metrics:
            return 0.0
        return sum(m.avg_decision_time_p1 for m in self.game_metrics) / len(self.game_metrics)

    @property
    def avg_decision_time_p2(self) -> float:
        """Average decision time for P2 across all games (ms)."""
        if not self.game_metrics:
            return 0.0
        return sum(m.avg_decision_time_p2 for m in self.game_metrics) / len(self.game_metrics)

    def summary(self) -> str:
        """Generate detailed summary report."""
        lines = [
            f"=== DETAILED MATCH RESULTS ===",
            f"{self.player_one_agent} vs {self.player_two_agent}",
            f"Games: {self.num_games}",
            f"",
            f"OUTCOMES:",
            f"  P1 Wins: {self.player_one_wins} ({self.player_one_wins/self.num_games:.1%})",
            f"  P2 Wins: {self.player_two_wins} ({self.player_two_wins/self.num_games:.1%})",
            f"  Draws: {self.draws} ({self.draws/self.num_games:.1%})",
            f"  First-player advantage: {self.first_player_advantage:.1%}",
            f"",
            f"WEAPON USAGE:",
            f"  P1 avg: {self.avg_weapons_used_p1:.2f} weapons/game",
            f"  P2 avg: {self.avg_weapons_used_p2:.2f} weapons/game",
            f"  Rocket hit rate: {self.rocket_hit_rate:.1%}",
            f"  Wipeout rate: {self.wipeout_rate:.1%}",
            f"",
            f"GAME DYNAMICS:",
            f"  Comeback rate: {self.comeback_rate:.1%}",
            f"",
            f"PERFORMANCE:",
            f"  P1 avg decision time: {self.avg_decision_time_p1:.1f}ms",
            f"  P2 avg decision time: {self.avg_decision_time_p2:.1f}ms",
        ]
        return "\n".join(lines)


class MetricsCollector:
    """
    Collects detailed metrics during game execution.

    This is a stateful helper that tracks events during a single game
    and produces a GameMetrics object at the end.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.weapons_used = {Player.ONE: 0, Player.TWO: 0}
        self.rockets_fired = {Player.ONE: 0, Player.TWO: 0}
        self.flares_deployed = {Player.ONE: 0, Player.TWO: 0}

        self.rockets_hit = 0
        self.rockets_deflected = 0
        self.dogfights_both_removed = 0

        self.max_advantage = {Player.ONE: 0, Player.TWO: 0}

        self.decision_times = {Player.ONE: [], Player.TWO: []}

        self.turn_count = 0
        self.winner = None
        self.final_state = None

    def record_decision_time(self, player: Player, duration: float):
        """Record time taken for a decision (in seconds)."""
        self.decision_times[player].append(duration)

    def record_weapon_use(self, player: Player, is_rocket: bool):
        """Record weapon usage (rocket vs flare determined by context)."""
        self.weapons_used[player] += 1
        if is_rocket:
            self.rockets_fired[player] += 1
        else:
            self.flares_deployed[player] += 1

    def record_rocket_outcome(self, hit: bool):
        """Record whether a rocket hit or was deflected."""
        if hit:
            self.rockets_hit += 1
        else:
            self.rockets_deflected += 1

    def record_dogfight_wipeout(self):
        """Record a dogfight where both rocketmen were removed."""
        self.dogfights_both_removed += 1

    def update_square_advantage(self, state: GameState):
        """Track maximum square control advantage for comeback detection."""
        p1_squares = state.count_controlled_squares(Player.ONE)
        p2_squares = state.count_controlled_squares(Player.TWO)

        advantage_p1 = p1_squares - p2_squares
        advantage_p2 = p2_squares - p1_squares

        self.max_advantage[Player.ONE] = max(self.max_advantage[Player.ONE], advantage_p1)
        self.max_advantage[Player.TWO] = max(self.max_advantage[Player.TWO], advantage_p2)

    def finalize(self, winner: Player | None, final_state: GameState) -> GameMetrics:
        """Create final GameMetrics object."""
        self.winner = winner
        self.final_state = final_state

        return GameMetrics(
            winner=winner,
            num_turns=final_state.turn_number,
            seed=self.seed,
            weapons_used_p1=self.weapons_used[Player.ONE],
            weapons_used_p2=self.weapons_used[Player.TWO],
            rockets_fired_p1=self.rockets_fired[Player.ONE],
            rockets_fired_p2=self.rockets_fired[Player.TWO],
            flares_deployed_p1=self.flares_deployed[Player.ONE],
            flares_deployed_p2=self.flares_deployed[Player.TWO],
            rockets_hit=self.rockets_hit,
            rockets_deflected=self.rockets_deflected,
            dogfights_both_removed=self.dogfights_both_removed,
            max_advantage_p1=self.max_advantage[Player.ONE],
            max_advantage_p2=self.max_advantage[Player.TWO],
            final_squares_p1=final_state.count_controlled_squares(Player.ONE),
            final_squares_p2=final_state.count_controlled_squares(Player.TWO),
            total_decision_time_p1=sum(self.decision_times[Player.ONE]),
            total_decision_time_p2=sum(self.decision_times[Player.TWO]),
            num_decisions_p1=len(self.decision_times[Player.ONE]),
            num_decisions_p2=len(self.decision_times[Player.TWO]),
        )
