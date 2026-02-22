"""
Game state representation for utala: kaos 9.

Architecture principle: State is immutable and owned by the engine.
Agents receive copies for observation only.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class Player(IntEnum):
    """Player identifiers."""
    ONE = 0
    TWO = 1

    def opponent(self) -> 'Player':
        """Return the opponent player."""
        return Player.TWO if self == Player.ONE else Player.ONE


class Phase(Enum):
    """Game phases."""
    PLACEMENT = "placement"
    DOGFIGHTS = "dogfights"
    ENDED = "ended"


class CardType(Enum):
    """Card types in the game."""
    # Rocketmen (2-10)
    ROCKETMAN_2 = 2
    ROCKETMAN_3 = 3
    ROCKETMAN_4 = 4
    ROCKETMAN_5 = 5
    ROCKETMAN_6 = 6
    ROCKETMAN_7 = 7
    ROCKETMAN_8 = 8
    ROCKETMAN_9 = 9
    ROCKETMAN_10 = 10

    # Weapons (Rockets)
    ROCKET_ACE = "A"
    ROCKET_KING = "K"

    # Defenses (Flares)
    FLARE_QUEEN = "Q"
    FLARE_JACK = "J"

    # Kaos cards (1-9)
    KAOS_1 = 1
    KAOS_2 = 2
    KAOS_3 = 3
    KAOS_4 = 4
    KAOS_5 = 5
    KAOS_6 = 6
    KAOS_7 = 7
    KAOS_8 = 8
    KAOS_9 = 9


# Useful card groups
ROCKETMEN = [CardType.ROCKETMAN_2, CardType.ROCKETMAN_3, CardType.ROCKETMAN_4,
             CardType.ROCKETMAN_5, CardType.ROCKETMAN_6, CardType.ROCKETMAN_7,
             CardType.ROCKETMAN_8, CardType.ROCKETMAN_9, CardType.ROCKETMAN_10]

ROCKETS = [CardType.ROCKET_ACE, CardType.ROCKET_KING]
FLARES = [CardType.FLARE_QUEEN, CardType.FLARE_JACK]
KAOS_CARDS = [CardType.KAOS_1, CardType.KAOS_2, CardType.KAOS_3, CardType.KAOS_4,
              CardType.KAOS_5, CardType.KAOS_6, CardType.KAOS_7, CardType.KAOS_8,
              CardType.KAOS_9]


@dataclass(frozen=True)
class DogfightContext:
    """
    Context information for current dogfight turn (v1.4+).

    Exposes minimal information agents need to make strategic weapon decisions:
    - rocket_in_play tells if weapon will be offensive (None) or defensive (set)
    - underdog/other show power relationship
    """
    position: tuple[int, int]  # Grid position of dogfight
    underdog: Player  # Player with lower power (acts first)
    other: Player  # Other player
    rocket_in_play: Player | None  # Which player played a rocket (if any)
    # None = no rocket yet, weapon plays offensively
    # Set = rocket in play, weapon plays defensively


@dataclass(frozen=True)
class Rocketman:
    """A rocketman placed on the grid."""
    player: Player
    power: int  # 2-10
    face_down: bool = False  # For Level 2 hidden information

    def __repr__(self) -> str:
        if self.face_down:
            return f"P{self.player.value + 1}:??"
        return f"P{self.player.value + 1}:{self.power}"


@dataclass
class GridSquare:
    """A square on the 3x3 grid."""
    rocketmen: list[Rocketman] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.rocketmen) == 0

    @property
    def is_contested(self) -> bool:
        """True if both players have a rocketman here."""
        return len(self.rocketmen) == 2

    @property
    def is_controlled(self) -> bool:
        """True if exactly one player has a rocketman here."""
        return len(self.rocketmen) == 1

    @property
    def controller(self) -> Player | None:
        """Return the controlling player, or None if contested or empty."""
        if self.is_controlled:
            return self.rocketmen[0].player
        return None

    def __repr__(self) -> str:
        if self.is_empty:
            return "[ ]"
        if self.is_controlled:
            return f"[{self.rocketmen[0]}]"
        return f"[{self.rocketmen[0]} vs {self.rocketmen[1]}]"


@dataclass
class PlayerResources:
    """Resources available to a player (v1.4: dual-purpose weapons)."""
    # Rocketmen in hand (not yet placed)
    rocketmen: list[int] = field(default_factory=lambda: list(range(2, 11)))

    # v1.4: Unified weapons list (all dual-purpose: any can be Rocket or Flare)
    weapons: list[str] = field(default_factory=lambda: ["A", "K", "Q", "J"])

    # Kaos deck (1-9, shuffled)
    kaos_deck: list[int] = field(default_factory=list)
    kaos_discard: list[int] = field(default_factory=list)

    def has_rocketman(self, power: int) -> bool:
        """Check if player has a rocketman of given power."""
        return power in self.rocketmen

    def has_weapon(self) -> bool:
        """Check if player has any weapons."""
        return len(self.weapons) > 0

    def remaining_kaos_cards(self) -> int:
        """Number of cards remaining in Kaos deck."""
        return len(self.kaos_deck)


@dataclass
class GameState:
    """Complete game state for utala: kaos 9."""

    # Grid state (indexed as grid[row][col], 0-indexed)
    grid: list[list[GridSquare]] = field(default_factory=lambda:
        [[GridSquare() for _ in range(3)] for _ in range(3)])

    # Player resources
    player_resources: dict[Player, PlayerResources] = field(default_factory=lambda: {
        Player.ONE: PlayerResources(),
        Player.TWO: PlayerResources()
    })

    # Game flow
    phase: Phase = Phase.PLACEMENT
    current_player: Player = Player.ONE
    turn_number: int = 0

    # Dogfight tracking
    dogfight_order: list[tuple[int, int]] = field(default_factory=list)  # (row, col) pairs
    current_dogfight_index: int = 0
    dogfight_context: DogfightContext | None = None  # Current dogfight turn context (v1.4+)
    joker_holder: Player = Player.TWO  # v1.8: Tie-breaker token (P2 starts to balance first-move)

    # Winner tracking
    winner: Player | None = None
    game_over: bool = False

    # RNG state (owned by engine)
    rng_seed: int | None = None

    def get_square(self, row: int, col: int) -> GridSquare:
        """Get a square from the grid."""
        return self.grid[row][col]

    def get_resources(self, player: Player) -> PlayerResources:
        """Get resources for a player."""
        return self.player_resources[player]

    def count_controlled_squares(self, player: Player) -> int:
        """Count how many squares a player controls."""
        count = 0
        for row in range(3):
            for col in range(3):
                square = self.grid[row][col]
                if square.controller == player:
                    count += 1
        return count

    def check_three_in_row(self, player: Player) -> bool:
        """Check if player has three rocketmen in a row."""
        # Check rows
        for row in range(3):
            if all(self.grid[row][col].controller == player for col in range(3)):
                return True

        # Check columns
        for col in range(3):
            if all(self.grid[row][col].controller == player for row in range(3)):
                return True

        # Check diagonals
        if all(self.grid[i][i].controller == player for i in range(3)):
            return True
        return all(self.grid[i][2-i].controller == player for i in range(3))

    def __repr__(self) -> str:
        """Pretty print the game state."""
        lines = [f"=== utala: kaos 9 - {self.phase.value} ==="]
        lines.append(f"Turn {self.turn_number}, Current Player: P{self.current_player.value + 1}")
        lines.append("")

        # Grid
        for row in range(3):
            line = " ".join(str(self.grid[row][col]) for col in range(3))
            lines.append(line)

        lines.append("")

        # Player resources
        for player in [Player.ONE, Player.TWO]:
            res = self.player_resources[player]
            lines.append(f"Player {player.value + 1}:")
            lines.append(f"  Rocketmen: {res.rocketmen}")
            lines.append(f"  Weapons: {res.weapons}")
            lines.append(f"  Kaos deck: {len(res.kaos_deck)} cards, Discard: {res.kaos_discard}")

        if self.game_over:
            winner_str = f"P{self.winner.value + 1}" if self.winner else "Draw"
            lines.append(f"\n*** Game Over - Winner: {winner_str} ***")

        return "\n".join(lines)
