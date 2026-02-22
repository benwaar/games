"""
Evaluation harness for utala: kaos 9.

Runs games between agents and collects statistics.
"""

from dataclasses import dataclass, field

from ..agents.base import Agent
from ..engine import GameEngine
from ..state import Player


@dataclass
class GameResult:
    """Result of a single game."""
    winner: Player | None
    num_turns: int
    seed: int
    player_one_agent: str
    player_two_agent: str


@dataclass
class MatchResult:
    """Result of a match (multiple games)."""
    player_one_agent: str
    player_two_agent: str
    num_games: int
    player_one_wins: int = 0
    player_two_wins: int = 0
    draws: int = 0
    game_results: list[GameResult] = field(default_factory=list)

    @property
    def player_one_win_rate(self) -> float:
        """Win rate for player one."""
        return self.player_one_wins / self.num_games if self.num_games > 0 else 0.0

    @property
    def player_two_win_rate(self) -> float:
        """Win rate for player two."""
        return self.player_two_wins / self.num_games if self.num_games > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Draw rate."""
        return self.draws / self.num_games if self.num_games > 0 else 0.0

    def __repr__(self) -> str:
        lines = [
            f"Match Results: {self.player_one_agent} vs {self.player_two_agent}",
            f"Games: {self.num_games}",
            f"P1 Wins: {self.player_one_wins} ({self.player_one_win_rate:.1%})",
            f"P2 Wins: {self.player_two_wins} ({self.player_two_win_rate:.1%})",
            f"Draws: {self.draws} ({self.draw_rate:.1%})",
        ]
        return "\n".join(lines)


class Harness:
    """
    Evaluation harness for running games and matches.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the harness.

        Args:
            verbose: If True, print game progress
        """
        self.verbose = verbose

    def run_game(
        self,
        agent_one: Agent,
        agent_two: Agent,
        seed: int | None = None
    ) -> GameResult:
        """
        Run a single game between two agents.

        Args:
            agent_one: Agent playing as Player ONE
            agent_two: Agent playing as Player TWO
            seed: RNG seed for deterministic games

        Returns:
            GameResult with outcome
        """
        engine = GameEngine(seed=seed)

        # Notify agents of game start
        agent_one.game_start(Player.ONE, seed)
        agent_two.game_start(Player.TWO, seed)

        if self.verbose:
            print(f"\n=== Game Start (seed={engine.seed}) ===")
            print(f"Player 1: {agent_one}")
            print(f"Player 2: {agent_two}")

        # Play until game ends
        while not engine.is_game_over():
            state = engine.get_state_copy()

            if state.phase.value == "placement":
                # Placement phase: alternate turns
                current_player = state.current_player
                agent = agent_one if current_player == Player.ONE else agent_two

                legal_actions = engine.get_legal_actions()
                action_idx = agent.select_action(state, legal_actions, current_player)

                if self.verbose:
                    action = engine.action_space.get_action(action_idx)
                    print(f"Turn {state.turn_number}: P{current_player.value + 1} {action}")

                engine.apply_action(action_idx)

            elif state.phase.value == "dogfights":
                # Dogfight phase: turn-based action/reaction
                engine.begin_current_dogfight()

                # v1.3: Get fresh state after begin_current_dogfight() to show revealed cards
                state = engine.get_state_copy()

                if self.verbose:
                    dogfight_pos = engine.get_current_dogfight_square()
                    print(f"Dogfight @ {dogfight_pos}:")

                # Collect actions turn by turn
                while not engine.is_dogfight_complete():
                    current_player = engine.get_dogfight_current_actor()
                    agent = agent_one if current_player == Player.ONE else agent_two

                    legal_actions = engine.get_dogfight_legal_actions_for_player(current_player)
                    action_idx = agent.select_action(state, legal_actions, current_player)

                    if self.verbose:
                        action = engine.action_space.get_action(action_idx)
                        print(f"  P{current_player.value + 1} plays: {action}")

                    engine.apply_dogfight_turn_action(current_player, action_idx)

                # Resolve the dogfight
                engine.finish_current_dogfight()

        # Game ended
        winner = engine.get_winner()

        if self.verbose:
            print("\n=== Game Over ===")
            print(f"Winner: {'P1' if winner == Player.ONE else 'P2' if winner == Player.TWO else 'Draw'}")
            print(engine.state)

        # Notify agents
        final_state = engine.get_state_copy()
        agent_one.game_end(final_state, winner)
        agent_two.game_end(final_state, winner)

        return GameResult(
            winner=winner,
            num_turns=engine.state.turn_number,
            seed=engine.seed,
            player_one_agent=agent_one.name,
            player_two_agent=agent_two.name
        )

    def run_match(
        self,
        agent_one: Agent,
        agent_two: Agent,
        num_games: int,
        starting_seed: int | None = None
    ) -> MatchResult:
        """
        Run multiple games between two agents.

        Args:
            agent_one: Agent playing as Player ONE
            agent_two: Agent playing as Player TWO
            num_games: Number of games to play
            starting_seed: Starting seed (increments for each game)

        Returns:
            MatchResult with aggregated statistics
        """
        result = MatchResult(
            player_one_agent=agent_one.name,
            player_two_agent=agent_two.name,
            num_games=num_games
        )

        for i in range(num_games):
            seed = starting_seed + i if starting_seed is not None else None
            game_result = self.run_game(agent_one, agent_two, seed)

            result.game_results.append(game_result)

            if game_result.winner == Player.ONE:
                result.player_one_wins += 1
            elif game_result.winner == Player.TWO:
                result.player_two_wins += 1
            else:
                result.draws += 1

        if self.verbose:
            print(f"\n{result}")

        return result

    def run_balanced_match(
        self,
        agent_one: Agent,
        agent_two: Agent,
        num_games: int,
        starting_seed: int | None = None
    ) -> MatchResult:
        """
        Run a balanced match between two agents.

        Runs num_games//2 with agent_one as P1, and num_games//2 with agent_two as P1.
        This controls for first player advantage by giving each agent equal time in each position.

        Args:
            agent_one: First agent
            agent_two: Second agent
            num_games: Total number of games (must be even)
            starting_seed: Starting seed (increments for each game)

        Returns:
            MatchResult with balanced statistics (agent_one's perspective)
        """
        if num_games % 2 != 0:
            raise ValueError("num_games must be even for balanced match")

        half_games = num_games // 2

        # First half: agent_one as P1, agent_two as P2
        result_1 = self.run_match(agent_one, agent_two, half_games, starting_seed)

        # Second half: agent_two as P1, agent_one as P2 (swapped positions)
        seed_offset = starting_seed + half_games if starting_seed is not None else None
        result_2 = self.run_match(agent_two, agent_one, half_games, seed_offset)

        # Aggregate results from agent_one's perspective
        # result_1: agent_one is P1, so agent_one wins = player_one_wins
        # result_2: agent_one is P2, so agent_one wins = player_two_wins
        agent_one_wins = result_1.player_one_wins + result_2.player_two_wins
        agent_two_wins = result_1.player_two_wins + result_2.player_one_wins
        draws = result_1.draws + result_2.draws

        # Combine game results
        all_game_results = result_1.game_results + result_2.game_results

        # Create balanced result
        balanced_result = MatchResult(
            player_one_agent=agent_one.name,
            player_two_agent=agent_two.name,
            num_games=num_games,
            player_one_wins=agent_one_wins,
            player_two_wins=agent_two_wins,
            draws=draws,
            game_results=all_game_results
        )

        if self.verbose:
            print("\n=== BALANCED MATCH ===")
            print(f"Position-controlled: {half_games} games each as P1/P2")
            print(f"\n{balanced_result}")

        return balanced_result
