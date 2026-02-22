"""
Action enumeration and masking for utala: kaos 9.

Architecture principle: The action space is fixed and fully enumerated.
Illegal actions are masked, never removed. Agents propose action indices,
the engine validates and applies them.
"""

from dataclasses import dataclass
from enum import Enum, auto

from .state import GameState, Phase, Player


class ActionType(Enum):
    """Types of actions in the game."""
    PLACE_ROCKETMAN = auto()  # Place a rocketman during placement phase
    PLAY_WEAPON = auto()       # Play a weapon (dual-purpose: Rocket or Flare based on context)
    PASS = auto()              # Pass (play nothing) during dogfight


@dataclass(frozen=True)
class Action:
    """An action in the game."""
    action_type: ActionType
    # For PLACE_ROCKETMAN: rocketman power, grid position
    rocketman_power: int | None = None
    row: int | None = None
    col: int | None = None
    # For PLAY_ROCKET/PLAY_FLARE: which card (index in list)
    card_index: int | None = None

    def __repr__(self) -> str:
        if self.action_type == ActionType.PLACE_ROCKETMAN:
            return f"PLACE({self.rocketman_power} @ [{self.row},{self.col}])"
        elif self.action_type == ActionType.PLAY_WEAPON:
            return f"WEAPON[{self.card_index}]"
        else:  # PASS
            return "PASS"


class ActionSpace:
    """
    Fixed action space for utala: kaos 9.

    The action space is enumerated once and remains constant throughout the game.
    Each action has a unique index. Illegal actions are masked, not removed.
    """

    def __init__(self):
        """Initialize the complete action space."""
        self.actions: list[Action] = []
        self._build_action_space()

    def _build_action_space(self):
        """Build the complete fixed action space."""
        # Placement actions: 9 rocketmen × 9 grid positions = 81 actions
        for power in range(2, 11):  # Rocketmen 2-10
            for row in range(3):
                for col in range(3):
                    action = Action(
                        action_type=ActionType.PLACE_ROCKETMAN,
                        rocketman_power=power,
                        row=row,
                        col=col
                    )
                    self.actions.append(action)

        # Dogfight actions:
        # - Play Weapon (4 dual-purpose weapons: A, K, Q, J) = 4 actions
        #   Context determines role: offensive (Rocket) or defensive (Flare)
        for card_idx in range(4):
            action = Action(
                action_type=ActionType.PLAY_WEAPON,
                card_index=card_idx
            )
            self.actions.append(action)

        # - Pass = 1 action
        self.actions.append(Action(action_type=ActionType.PASS))

        # Total: 81 + 4 + 1 = 86 actions

    def size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def get_action(self, action_index: int) -> Action:
        """Get an action by its index."""
        return self.actions[action_index]

    def get_legal_actions_mask(self, state: GameState, player: Player) -> list[bool]:
        """
        Return a boolean mask indicating which actions are legal.

        Architecture principle: The mask is computed by the engine,
        never by the agent. The mask has the same length as the action space.
        """
        mask = [False] * self.size()
        resources = state.get_resources(player)

        if state.phase == Phase.PLACEMENT:
            # During placement: can only place rocketmen we have on legal squares
            for i, action in enumerate(self.actions):
                if action.action_type == ActionType.PLACE_ROCKETMAN:
                    # For placement actions, these fields are always set
                    assert action.rocketman_power is not None
                    assert action.row is not None
                    assert action.col is not None

                    if resources.has_rocketman(action.rocketman_power):
                        # Check if square is legal (doesn't already have player's rocketman)
                        square = state.get_square(action.row, action.col)
                        already_occupied_by_player = any(
                            rm.player == player for rm in square.rocketmen
                        )
                        if not already_occupied_by_player:
                            mask[i] = True

        elif state.phase == Phase.DOGFIGHTS:
            # During dogfights: can play weapons or pass
            # Note: Context (turn sequence) determines weapon role (Rocket vs Flare)
            for i, action in enumerate(self.actions):
                if action.action_type == ActionType.PLAY_WEAPON:
                    assert action.card_index is not None
                    if action.card_index < len(resources.weapons):
                        mask[i] = True
                elif action.action_type == ActionType.PASS:
                    mask[i] = True

        return mask

    def get_legal_actions(self, state: GameState, player: Player) -> list[int]:
        """
        Return a list of legal action indices.

        This is a convenience method. The canonical representation is the mask.
        """
        mask = self.get_legal_actions_mask(state, player)
        return [i for i, legal in enumerate(mask) if legal]

    def action_to_index(self, action: Action) -> int | None:
        """
        Find the index of an action in the action space.
        Returns None if action is not found.
        """
        try:
            return self.actions.index(action)
        except ValueError:
            return None


# Global singleton action space
# (Can be instantiated once and reused across all games)
_ACTION_SPACE = None


def get_action_space() -> ActionSpace:
    """Get the global action space singleton."""
    global _ACTION_SPACE
    if _ACTION_SPACE is None:
        _ACTION_SPACE = ActionSpace()
    return _ACTION_SPACE
