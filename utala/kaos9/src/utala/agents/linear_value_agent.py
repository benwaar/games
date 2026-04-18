"""
Linear Value Agent with TD Learning.

This agent learns a linear value function Q(s,a) = w^T × φ(s,a) using
temporal difference learning. Unlike k-NN (which memorizes examples),
this agent generalizes by learning feature weights.

Learning approach:
1. Extract features for each (state, action) pair
2. Compute value: Q(s,a) = sum(weights[i] * features[i])
3. After game, backpropagate outcome through trajectory
4. Update weights: w ← w + α × (target - Q(s,a)) × φ(s,a)
"""

import numpy as np
from typing import List, Dict, Tuple
import random

from .base import Agent
from ..state import GameState, GameConfig, Player
from ..learning.serialization import SerializableAgent
from ..learning.state_action_features import StateActionFeatureExtractor


class LinearValueAgent(Agent, SerializableAgent):
    """
    TD learning agent with linear function approximation.

    Learns weights for state-action features to estimate action values.
    Uses epsilon-greedy exploration and updates weights after each game.
    """

    def __init__(
        self,
        name: str,
        learning_rate: float = 0.01,
        discount: float = 0.95,
        epsilon: float = 0.1,
        seed: int | None = None,
        deck_awareness: bool = True,
        config: GameConfig | None = None
    ):
        """
        Initialize the linear value agent.

        Args:
            name: Agent name
            learning_rate: Step size for weight updates (α)
            discount: Discount factor for future rewards (γ)
            epsilon: Exploration rate (random action probability)
            seed: Random seed for reproducibility
            deck_awareness: If True, include deck awareness features (Phase 3.2)
            config: Game config (needed for Variant A dogfight choice features)
        """
        super().__init__(name)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

        # Feature extractor
        self.feature_extractor = StateActionFeatureExtractor(
            deck_awareness=deck_awareness, config=config
        )
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
        best_q_value = float('-inf')
        best_features = None

        for action in legal_actions:
            features = self._extract_state_action_features(state, action, player)
            q_value = self._compute_q_value(features)

            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
                best_features = features

        # Store trajectory for training
        if self.is_training and best_features is not None:
            self.trajectory.append((best_features, best_action, best_q_value))

        return best_action

    def _extract_state_action_features(
        self,
        state: GameState,
        action: int,
        player: Player
    ) -> np.ndarray:
        """
        Extract features for a (state, action) pair.

        Uses StateActionFeatureExtractor for rich feature representation.

        Args:
            state: Game state
            action: Action index
            player: Current player

        Returns:
            Feature vector for this state-action pair
        """
        return self.feature_extractor.extract(state, action, player)

    def _compute_q_value(self, features: np.ndarray) -> float:
        """Compute Q(s,a) = w^T × φ(s,a)"""
        return float(np.dot(self.weights, features))

    def get_top_weights(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k feature weights by magnitude.

        Args:
            top_k: Number of top weights to return

        Returns:
            List of (feature_name, weight) tuples
        """
        feature_names = self.feature_extractor.get_feature_names()
        indices = np.argsort(np.abs(self.weights))[-top_k:][::-1]

        return [(feature_names[idx], float(self.weights[idx])) for idx in indices]

    def observe_outcome(self, outcome: float):
        """
        Called at end of game to backpropagate outcome through trajectory.

        Args:
            outcome: Game outcome from agent's perspective (1.0=win, 0.5=draw, 0.0=loss)
        """
        if not self.is_training or len(self.trajectory) == 0:
            self.trajectory.clear()
            return

        # TD(0) update: work backwards through trajectory
        # Terminal reward is the outcome
        next_value = outcome

        for features, action, q_value in reversed(self.trajectory):
            # Skip zero features (random exploration)
            if np.sum(features) == 0:
                continue

            # TD error: δ = r + γ × V(s') - Q(s,a)
            # For terminal state: r = outcome, V(s') = 0
            # For non-terminal: r = 0, V(s') = max Q(s',a')
            td_error = next_value - q_value

            # Weight update: w ← w + α × δ × φ(s,a)
            self.weights += self.learning_rate * td_error * features
            self.total_weight_updates += 1

            # Next value is discounted current value
            next_value = self.discount * q_value

        # Clear trajectory for next episode
        self.trajectory.clear()
        self.games_trained += 1

    def start_episode(self):
        """Called at start of new game."""
        self.trajectory.clear()

    def set_training_mode(self, is_training: bool):
        """Enable/disable training mode (affects exploration)."""
        self.is_training = is_training

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'games_trained': self.games_trained,
            'weight_updates': self.total_weight_updates,
            'avg_weight_magnitude': float(np.mean(np.abs(self.weights))),
            'max_weight': float(np.max(self.weights)),
            'min_weight': float(np.min(self.weights)),
            'weight_std': float(np.std(self.weights))
        }

    def to_dict(self) -> Dict:
        """Export agent state for serialization."""
        # Get top weights for interpretability
        top_weights = self.get_top_weights(20)

        return {
            'learning_rate': self.learning_rate,
            'discount': self.discount,
            'epsilon': self.epsilon,
            'weights': self.weights.tolist(),
            'num_features': self.num_features,
            'training_stats': self.get_training_stats(),
            'top_weights': [{'feature': name, 'weight': w} for name, w in top_weights]
        }

    def from_dict(self, data: Dict):
        """Import agent state from serialization."""
        self.learning_rate = data['learning_rate']
        self.discount = data['discount']
        self.epsilon = data['epsilon']
        self.num_features = data['num_features']
        self.weights = np.array(data['weights'], dtype=np.float32)

        if 'training_stats' in data:
            stats = data['training_stats']
            self.games_trained = stats.get('games_trained', 0)
            self.total_weight_updates = stats.get('weight_updates', 0)

    def __repr__(self) -> str:
        stats = self.get_training_stats()
        return (
            f"LinearValueAgent({self.name}, "
            f"α={self.learning_rate}, γ={self.discount}, "
            f"games={stats['games_trained']})"
        )
