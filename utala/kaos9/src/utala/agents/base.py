"""
Base agent class for utala: kaos 9.

Architecture principle: Agents only propose actions, never apply them.
Agents observe state but do not own it. All randomness lives in the engine.
"""

from abc import ABC, abstractmethod

from ..state import GameState, Player


class Agent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str):
        """
        Initialize an agent.

        Args:
            name: Human-readable name for this agent
        """
        self.name = name

    @abstractmethod
    def select_action(
        self,
        state: GameState,
        legal_actions: list[int],
        player: Player
    ) -> int:
        """
        Select an action given the current state and legal actions.

        Args:
            state: Current game state (read-only copy)
            legal_actions: List of legal action indices
            player: Which player this agent is acting as

        Returns:
            Index of the selected action (must be in legal_actions)

        Note:
            - The agent must NOT modify the state
            - The agent must return an action from legal_actions
            - If the agent violates these, the engine will reject the action
        """
        pass

    def game_start(self, player: Player, seed: int | None = None):
        """
        Called when a new game starts.

        Args:
            player: Which player this agent will be
            seed: Game seed (for reference only; agents must not use RNG)

        Note:
            This is an optional hook. Override in subclasses if needed.
        """
        return  # Optional hook - subclasses can override

    def game_end(self, state: GameState, winner: Player | None):
        """
        Called when the game ends.

        Args:
            state: Final game state
            winner: Winning player, or None for draw

        Note:
            This is an optional hook. Override in subclasses if needed.
        """
        return  # Optional hook - subclasses can override

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
