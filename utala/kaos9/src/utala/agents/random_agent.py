"""
Random baseline agent for utala: kaos 9 v1.4.

Selects uniformly random legal actions.
"""

import random

from ..state import GameState, Player
from .base import Agent


class RandomAgent(Agent):
    """
    Random baseline agent.

    Selects uniformly random legal actions with no strategy.
    Uses explicit RNG seed for deterministic behavior.
    """

    def __init__(self, name: str = "Random", seed: int | None = None):
        """
        Initialize random agent.

        Args:
            name: Agent name
            seed: Optional seed for deterministic randomness
        """
        super().__init__(name)
        self.rng = random.Random(seed)

    def select_action(
        self,
        state: GameState,
        legal_actions: list[int],
        player: Player
    ) -> int:
        """Select uniformly random legal action."""
        return self.rng.choice(legal_actions)
