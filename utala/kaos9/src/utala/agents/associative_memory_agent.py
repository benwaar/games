"""
Associative Memory Agent (k-Nearest Neighbors).

This is a learning agent that stores game experiences and makes decisions
by finding similar past states and voting on actions.

Learning approach:
1. Store (state_features, action, outcome) tuples from training games
2. At decision time, find k most similar past states
3. Weight each neighbor by similarity × outcome
4. Vote on legal actions, select highest-weighted
"""

import numpy as np
from typing import List, Dict
import random

from .base import Agent
from ..state import GameState, Player
from ..learning.features import get_feature_extractor
from ..learning.serialization import SerializableAgent
from ..learning.training_data import TrainingExample


class AssociativeMemoryAgent(Agent, SerializableAgent):
    """
    k-NN learning agent that remembers past game experiences.

    The agent maintains a memory of (features, action, outcome) tuples.
    When selecting an action, it finds k similar past states and votes
    based on similarity and outcome quality.
    """

    def __init__(
        self,
        name: str,
        k: int = 20,
        distance_metric: str = "euclidean",
        epsilon: float = 0.05,
        seed: int | None = None
    ):
        """
        Initialize the associative memory agent.

        Args:
            name: Agent name
            k: Number of nearest neighbors to consider
            distance_metric: Distance metric ("euclidean" or "manhattan")
            epsilon: Exploration rate (random action probability)
            seed: Random seed for reproducibility
        """
        super().__init__(name)
        self.k = k
        self.distance_metric = distance_metric
        self.epsilon = epsilon
        self.feature_extractor = get_feature_extractor()

        # Memory: list of (features, action, outcome)
        self.memory: List[Dict] = []

        # RNG for epsilon-greedy
        self.rng = random.Random(seed)

    def select_action(
        self,
        state: GameState,
        legal_actions: List[int],
        player: Player
    ) -> int:
        """
        Select action by k-NN voting over past experiences.

        Args:
            state: Current game state
            legal_actions: List of legal action indices
            player: Current player

        Returns:
            Selected action index
        """
        # Epsilon-greedy: explore with small probability
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # If no memory yet, choose randomly
        if len(self.memory) == 0:
            return self.rng.choice(legal_actions)

        # Extract features for current state
        current_features = self.feature_extractor.extract(state, player)

        # Find k nearest neighbors
        neighbors = self._find_k_nearest(current_features, min(self.k, len(self.memory)))

        # Vote on actions
        action = self._vote_on_action(neighbors, legal_actions)

        return action

    def _find_k_nearest(self, query_features: np.ndarray, k: int) -> List[Dict]:
        """
        Find k nearest neighbors in memory.

        Args:
            query_features: Feature vector to find neighbors for
            k: Number of neighbors to return

        Returns:
            List of k nearest memory entries, each with added 'distance' field
        """
        # Calculate distances to all memory entries
        distances = []
        for i, mem_entry in enumerate(self.memory):
            mem_features = np.array(mem_entry['features'])
            dist = self._calculate_distance(query_features, mem_features)
            distances.append((i, dist))

        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        nearest_indices = [idx for idx, dist in distances[:k]]

        # Return neighbors with distance information
        neighbors = []
        for idx in nearest_indices:
            neighbor = self.memory[idx].copy()
            neighbor['distance'] = distances[idx][1]
            neighbors.append(neighbor)

        return neighbors

    def _calculate_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate distance between two feature vectors."""
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(features1 - features2))
        elif self.distance_metric == "manhattan":
            return float(np.sum(np.abs(features1 - features2)))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _vote_on_action(self, neighbors: List[Dict], legal_actions: List[int]) -> int:
        """
        Vote on action based on neighbor similarity and outcomes.

        Each neighbor votes for its action, weighted by:
        - Inverse distance (closer states have more weight)
        - Outcome quality (wins have more weight than losses)

        Args:
            neighbors: List of neighbor memory entries (with 'distance' field)
            legal_actions: List of legal action indices

        Returns:
            Best action according to weighted vote
        """
        # Accumulate weighted votes for each action
        action_votes: Dict[int, float] = {action: 0.0 for action in legal_actions}

        for neighbor in neighbors:
            action = neighbor['action']
            outcome = neighbor['outcome']
            distance = neighbor['distance']

            # Skip if this action is not currently legal
            if action not in legal_actions:
                continue

            # Weight = (1 / (distance + small_epsilon)) × outcome
            # Small epsilon prevents division by zero
            distance_weight = 1.0 / (distance + 0.01)
            outcome_weight = outcome  # 1.0=win, 0.5=draw, 0.0=loss

            vote_weight = distance_weight * outcome_weight

            action_votes[action] += vote_weight

        # Select action with highest vote
        # If all votes are 0 (e.g., all neighbors lost with those actions),
        # fall back to random legal action
        max_vote = max(action_votes.values())
        if max_vote == 0.0:
            return self.rng.choice(legal_actions)

        best_action = max(action_votes.items(), key=lambda x: x[1])[0]
        return best_action

    def train(self, training_examples: List[TrainingExample]):
        """
        Train the agent by adding training examples to memory.

        Args:
            training_examples: List of (features, action, outcome) tuples
        """
        print(f"Training {self.name} with {len(training_examples)} examples...")

        for example in training_examples:
            self.memory.append({
                'features': example.features,
                'action': example.action,
                'outcome': example.outcome,
                'turn': example.turn
            })

        print(f"  Memory size: {len(self.memory)} entries")

    def prune_memory(self, max_size: int | None = None, strategy: str = "random"):
        """
        Prune memory to reduce size.

        Strategies:
        - "random": Keep random sample
        - "recent": Keep most recent examples
        - "diverse": Keep diverse examples (maximize coverage)

        Args:
            max_size: Maximum memory size (None = no limit)
            strategy: Pruning strategy
        """
        if max_size is None or len(self.memory) <= max_size:
            return

        print(f"Pruning memory from {len(self.memory)} to {max_size} ({strategy})...")

        if strategy == "random":
            self.memory = self.rng.sample(self.memory, max_size)
        elif strategy == "recent":
            # Sort by turn number and keep most recent
            self.memory.sort(key=lambda x: x['turn'], reverse=True)
            self.memory = self.memory[:max_size]
        elif strategy == "diverse":
            # Simple diversity: sample evenly across feature space
            # (more sophisticated approach would use clustering)
            self.memory = self.rng.sample(self.memory, max_size)
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

    def get_memory_stats(self) -> Dict:
        """Get statistics about the current memory."""
        if len(self.memory) == 0:
            return {
                'size': 0,
                'win_rate': 0.0,
                'avg_turn': 0.0
            }

        outcomes = [m['outcome'] for m in self.memory]
        turns = [m['turn'] for m in self.memory]

        return {
            'size': len(self.memory),
            'win_rate': np.mean([1.0 if o == 1.0 else 0.0 for o in outcomes]),
            'draw_rate': np.mean([1.0 if o == 0.5 else 0.0 for o in outcomes]),
            'loss_rate': np.mean([1.0 if o == 0.0 else 0.0 for o in outcomes]),
            'avg_turn': np.mean(turns)
        }

    def to_dict(self) -> Dict:
        """Export agent state for serialization."""
        return {
            'k': self.k,
            'distance_metric': self.distance_metric,
            'epsilon': self.epsilon,
            'memory': self.memory,
            'memory_stats': self.get_memory_stats()
        }

    def from_dict(self, data: Dict):
        """Import agent state from serialization."""
        self.k = data['k']
        self.distance_metric = data['distance_metric']
        self.epsilon = data['epsilon']
        self.memory = data['memory']

    def __repr__(self) -> str:
        stats = self.get_memory_stats()
        return (
            f"AssociativeMemoryAgent({self.name}, "
            f"k={self.k}, memory={stats['size']}, "
            f"win_rate={stats['win_rate']:.1%})"
        )
