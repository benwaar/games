"""
Linear Value Agent v2 with Enhanced Features (Phase 2.5).

Major improvements over v1:
- Uses enhanced feature extractor with 49 features (up from 43)
- Proper action decoding for placement moves
- Line detection (2-in-a-row, 3-in-a-row, winning lines)
- Tactical patterns (blocking, forking, threatening)
- Kaos deck tracking

This version should achieve 55%+ win rate vs Heuristic baseline.
"""

import numpy as np
from typing import List, Dict, Tuple
import random

from .base import Agent
from ..state import GameState, Player
from ..learning.serialization import SerializableAgent
from ..learning.state_action_features_enhanced import EnhancedStateActionFeatureExtractor


class LinearValueAgentV2(Agent, SerializableAgent):
    """
    TD learning agent with enhanced linear function approximation.

    Uses richer features (49 total) to better capture tactical patterns.
    """

    def __init__(
        self,
        name: str,
        learning_rate: float = 0.01,
        discount: float = 0.95,
        epsilon: float = 0.1,
        seed: int | None = None
    ):
        """
        Initialize the enhanced linear value agent.

        Args:
            name: Agent name
            learning_rate: Step size for weight updates (α)
            discount: Discount factor for future rewards (γ)
            epsilon: Exploration rate (random action probability)
            seed: Random seed for reproducibility
        """
        super().__init__(name)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

        # Enhanced feature extractor
        self.feature_extractor = EnhancedStateActionFeatureExtractor()
        self.num_features = self.feature_extractor.get_feature_count()

        # Initialize weights to small random values
        self.weights: np.ndarray = np.random.randn(self.num_features).astype(np.float32) * 0.01

        # Training statistics
        self.games_trained = 0
        self.total_weight_updates = 0

        # Trajectory buffer for current episode
        self.trajectory: List[Tuple[np.ndarray, int, float]] = []
        self.is_training = True

        # RNG for epsilon-greedy
        self.rng = random.Random(seed)

    def select_action(
        self,
        state: GameState,
        legal_actions: List[int],
        player: Player
    ) -> int:
        """
        Select action using epsilon-greedy policy over learned Q-values.

        Args:
            state: Current game state
            legal_actions: List of legal action indices
            player: Current player

        Returns:
            Selected action index
        """
        # Epsilon-greedy: explore with probability epsilon
        if self.is_training and self.rng.random() < self.epsilon:
            action = self.rng.choice(legal_actions)
            if self.is_training:
                # Store (features, action, q_value) for later update
                # Use zero features for random actions (won't update weights)
                self.trajectory.append((np.zeros(self.num_features), action, 0.0))
            return action

        # Greedy: select action with highest Q-value
        best_action = None
        best_value = float('-inf')

        for action in legal_actions:
            # Extract features for this state-action pair
            features = self.feature_extractor.extract(state, action, player)

            # Compute Q-value: Q(s,a) = w^T × φ(s,a)
            q_value = float(np.dot(self.weights, features))

            if q_value > best_value:
                best_value = q_value
                best_action = action

        # Store trajectory for training
        if self.is_training and best_action is not None:
            features = self.feature_extractor.extract(state, best_action, player)
            self.trajectory.append((features, best_action, best_value))

        return best_action if best_action is not None else legal_actions[0]

    def observe_outcome(self, outcome: float):
        """
        Update weights based on game outcome using TD(0).

        Backpropagates the final outcome through the trajectory:
        - For each (s,a) pair, target = next_value
        - TD error: δ = target - Q(s,a)
        - Weight update: w ← w + α × δ × φ(s,a)

        Args:
            outcome: Final reward (+1 for win, 0 for loss, 0.5 for draw)
        """
        if not self.is_training:
            return

        # Backpropagate outcome through trajectory
        next_value = outcome

        for features, action, q_value in reversed(self.trajectory):
            # Skip zero-feature entries (random exploration)
            if np.sum(features) == 0:
                continue

            # TD error: δ = target - Q(s,a)
            td_error = next_value - q_value

            # Update weights: w ← w + α × δ × φ(s,a)
            self.weights += self.learning_rate * td_error * features
            self.total_weight_updates += 1

            # Bootstrap: next_value = γ × Q(s,a)
            next_value = self.discount * q_value

        # Clear trajectory for next game
        self.trajectory.clear()
        self.games_trained += 1

    def reset_episode(self):
        """Reset trajectory buffer for new episode."""
        self.trajectory.clear()

    def set_training(self, is_training: bool):
        """Enable or disable training mode."""
        self.is_training = is_training

    # === SerializableAgent interface ===

    def save(self, path: str):
        """Save agent state to file."""
        data = {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "weights": self.weights.tolist(),
            "games_trained": self.games_trained,
            "total_weight_updates": self.total_weight_updates,
            "num_features": self.num_features,
        }
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> 'LinearValueAgentV2':
        """Load agent state from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        agent = LinearValueAgentV2(
            name=data["name"],
            learning_rate=data["learning_rate"],
            discount=data["discount"],
            epsilon=data["epsilon"],
        )

        agent.weights = np.array(data["weights"], dtype=np.float32)
        agent.games_trained = data["games_trained"]
        agent.total_weight_updates = data["total_weight_updates"]

        # Verify feature count matches
        if agent.num_features != data["num_features"]:
            raise ValueError(
                f"Feature count mismatch: expected {data['num_features']}, "
                f"got {agent.num_features}"
            )

        return agent

    def get_stats(self) -> Dict[str, any]:
        """Get training statistics."""
        return {
            "games_trained": self.games_trained,
            "total_weight_updates": self.total_weight_updates,
            "num_features": self.num_features,
            "weight_norm": float(np.linalg.norm(self.weights)),
            "weight_mean": float(np.mean(self.weights)),
            "weight_std": float(np.std(self.weights)),
        }

    def explain_action(
        self,
        state: GameState,
        action: int,
        player: Player,
        top_k: int = 10
    ) -> str:
        """
        Explain why this action was chosen by showing top features.

        Args:
            state: Current game state
            action: Selected action
            player: Current player
            top_k: Number of top features to show

        Returns:
            Human-readable explanation
        """
        features = self.feature_extractor.extract(state, action, player)
        q_value = float(np.dot(self.weights, features))

        lines = [f"Action {action} (Q-value: {q_value:.3f})"]
        lines.append(self.feature_extractor.explain_features(features, top_k))

        return "\n".join(lines)
