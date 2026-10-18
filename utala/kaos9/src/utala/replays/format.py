"""
Replay format v1 for utala: kaos 9.

Format: seed + action list
Deterministic replay: same seed + same actions = same game outcome.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..state import Player


@dataclass
class ReplayMetadata:
    """Metadata for a replay."""
    format_version: str = "v1"
    rules_version: str = "1.8"  # Game rules version (1.8 = joker token + immediate 3-in-a-row)
    game_variant: str = "level1"  # level1, level2, etc.
    player_one_name: str = "Player 1"
    player_two_name: str = "Player 2"
    timestamp: str | None = None


@dataclass
class ReplayV1:
    """
    Replay format v1.

    Contains:
    - RNG seed
    - Action history (player, action_index pairs)
    - Metadata
    """
    seed: int
    actions: list[tuple[int, int]]  # (player_id, action_index)
    metadata: ReplayMetadata
    winner: int | None = None  # 0 for P1, 1 for P2, None for draw

    def to_dict(self) -> dict:
        """Convert replay to dictionary."""
        return {
            "format_version": self.metadata.format_version,
            "seed": self.seed,
            "actions": [[player_id, action_idx] for player_id, action_idx in self.actions],
            "metadata": asdict(self.metadata),
            "winner": self.winner
        }

    def to_json(self) -> str:
        """Serialize replay to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: Path):
        """Save replay to a file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> 'ReplayV1':
        """Load replay from dictionary."""
        metadata = ReplayMetadata(**data.get("metadata", {}))
        actions = [tuple(a) for a in data["actions"]]
        return cls(
            seed=data["seed"],
            actions=actions,
            metadata=metadata,
            winner=data.get("winner")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ReplayV1':
        """Deserialize replay from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, filepath: Path) -> 'ReplayV1':
        """Load replay from a file."""
        with open(filepath) as f:
            return cls.from_json(f.read())


def create_replay_from_game(engine, metadata: ReplayMetadata | None = None) -> ReplayV1:
    """
    Create a replay from a completed game.

    Args:
        engine: GameEngine instance with completed game
        metadata: Optional metadata. If None, uses defaults.

    Returns:
        ReplayV1 instance
    """
    if metadata is None:
        metadata = ReplayMetadata()

    # Convert action history to replay format
    actions = [(int(player), action_idx) for player, action_idx in engine.action_history]

    # Get winner
    winner = None
    if engine.state.winner is not None:
        winner = int(engine.state.winner)

    return ReplayV1(
        seed=engine.seed,
        actions=actions,
        metadata=metadata,
        winner=winner
    )


def replay_game(replay: ReplayV1):
    """
    Replay a game from a replay file.

    Returns the final game engine state.

    Note: Replays recorded with v1.2+ rules use turn-based dogfights.
    Each dogfight action is recorded individually in the action history.
    """
    from ..engine import GameEngine

    engine = GameEngine(seed=replay.seed)
    action_iter = iter(replay.actions)

    for player_id, action_idx in action_iter:
        player = Player(player_id)

        if engine.state.phase.value == "placement":
            # During placement, actions alternate
            if player == engine.state.current_player:
                engine.apply_action(action_idx)
            else:
                raise ValueError(f"Replay error: expected {engine.state.current_player}, got {player}")

        elif engine.state.phase.value == "dogfights":
            # Turn-based dogfights (v1.2+)
            # Need to begin dogfight if not already started
            if engine.current_dogfight is None:
                engine.begin_current_dogfight()

            # Apply this turn's action
            current_actor = engine.get_dogfight_current_actor()
            if player == current_actor:
                engine.apply_dogfight_turn_action(player, action_idx)
            else:
                raise ValueError(f"Replay error: expected {current_actor}, got {player}")

            # If dogfight is complete, resolve it
            if engine.is_dogfight_complete():
                engine.finish_current_dogfight()

    return engine
