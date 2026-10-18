"""
Policy Network Agent using REINFORCE (Policy Gradient).

This agent uses a simple 2-layer neural network implemented from scratch
(no ML frameworks) to learn a policy: π(a|s) = probability of action a given state s.

Architecture:
- Input layer: state features (53 dimensions)
- Hidden layer: 64 units with ReLU activation
- Output layer: action logits (86 dimensions) with softmax

Learning:
- REINFORCE policy gradient algorithm
- Monte Carlo returns (episode-based learning)
- Baseline subtraction for variance reduction
"""

import numpy as np
from typing import List, Dict
import random
import json

from .base import Agent
from ..state import GameState, Player
from ..learning.features import get_feature_extractor
from ..learning.serialization import SerializableAgent


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def softmax(x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Softmax with optional masking for illegal actions.

    Args:
        x: Logits
        mask: Binary mask (1=legal, 0=illegal). If provided, illegal actions get 0 probability.

    Returns:
        Probability distribution
    """
    # Numerical stability: subtract max
    x_stable = x - np.max(x)

    if mask is not None:
        # Set illegal action logits to very negative value
        x_stable = np.where(mask, x_stable, -1e9)

    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x)


class PolicyNetwork:
    """
    Simple 2-layer neural network for policy learning.

    Architecture: input → hidden (ReLU) → output (softmax)
    """

    def __init__(
        self,
        input_size: int = 53,
        hidden_size: int = 64,
        output_size: int = 86,
        learning_rate: float = 0.001,
        seed: int = 42
    ):
        """
        Initialize network with random weights.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output actions
            learning_rate: Learning rate for gradient descent
            seed: Random seed for weight initialization
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Initialize weights with Xavier/Glorot initialization
        # W1: input → hidden
        limit1 = np.sqrt(6.0 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)

        # W2: hidden → output
        limit2 = np.sqrt(6.0 / (hidden_size + output_size))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

        # Cache for backpropagation
        self.cache = {}

    def forward(self, state_features: np.ndarray, legal_mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass: compute action probabilities.

        Args:
            state_features: Input features (53,)
            legal_mask: Binary mask for legal actions (86,)

        Returns:
            Action probabilities (86,)
        """
        # Input → Hidden
        z1 = state_features @ self.W1 + self.b1
        h1 = relu(z1)

        # Hidden → Output
        z2 = h1 @ self.W2 + self.b2

        # Softmax with masking
        probs = softmax(z2, legal_mask)

        # Cache for backprop
        self.cache = {
            'state_features': state_features,
            'z1': z1,
            'h1': h1,
            'z2': z2,
            'probs': probs,
            'legal_mask': legal_mask
        }

        return probs

    def backward(self, action: int, advantage: float):
        """
        Backward pass: compute gradients using policy gradient.

        Policy gradient theorem:
        ∇J(θ) = E[∇ log π(a|s) × A(s,a)]

        Where A(s,a) is the advantage (return - baseline).

        Args:
            action: Action that was taken
            advantage: Advantage estimate (return - baseline)
        """
        # Retrieve cached values
        state_features = self.cache['state_features']
        z1 = self.cache['z1']
        h1 = self.cache['h1']
        probs = self.cache['probs']

        # Gradient of log π(a|s) w.r.t. output logits
        # d/dz log(softmax(z)[a]) = indicator(a) - softmax(z)
        dz2 = -probs.copy()
        dz2[action] += 1.0

        # Scale by advantage (REINFORCE)
        dz2 *= advantage

        # Backprop through output layer
        dW2 = np.outer(h1, dz2)
        db2 = dz2
        dh1 = dz2 @ self.W2.T

        # Backprop through ReLU
        dz1 = dh1 * relu_derivative(z1)

        # Backprop through hidden layer
        dW1 = np.outer(state_features, dz1)
        db1 = dz1

        # Store gradients
        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }

    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """
        Update weights using gradients (gradient ascent for policy gradient).

        Args:
            gradients: Dictionary of gradients for each parameter
        """
        self.W1 += self.learning_rate * gradients['dW1']
        self.b1 += self.learning_rate * gradients['db1']
        self.W2 += self.learning_rate * gradients['dW2']
        self.b2 += self.learning_rate * gradients['db2']

    def to_dict(self) -> Dict:
        """Export network parameters."""
        return {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }

    def from_dict(self, data: Dict):
        """Import network parameters."""
        self.W1 = np.array(data['W1'])
        self.b1 = np.array(data['b1'])
        self.W2 = np.array(data['W2'])
        self.b2 = np.array(data['b2'])
        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.output_size = data['output_size']
        self.learning_rate = data['learning_rate']


class PolicyNetworkAgent(Agent, SerializableAgent):
    """
    Agent that uses a policy network to learn action selection.

    Uses REINFORCE (Monte Carlo Policy Gradient) algorithm:
    1. Play episode, recording (state, action, reward) trajectory
    2. Compute returns for each step
    3. Update policy to increase probability of good actions
    """

    def __init__(
        self,
        name: str,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        discount: float = 0.99,
        baseline_decay: float = 0.9,
        seed: int = 42
    ):
        """
        Initialize policy network agent.

        Args:
            name: Agent name
            hidden_size: Number of hidden units
            learning_rate: Learning rate for gradient descent
            discount: Discount factor for returns
            baseline_decay: Exponential moving average decay for baseline
            seed: Random seed
        """
        super().__init__(name)

        self.feature_extractor = get_feature_extractor()
        self.network = PolicyNetwork(
            input_size=53,
            hidden_size=hidden_size,
            output_size=86,
            learning_rate=learning_rate,
            seed=seed
        )

        self.discount = discount
        self.baseline = 0.0  # Moving average baseline
        self.baseline_decay = baseline_decay

        # Training state
        self.training_mode = False
        self.trajectory: List[Dict] = []
        self.episode_count = 0
        self.total_updates = 0

        # RNG for action sampling
        self.rng = random.Random(seed)

    def set_training_mode(self, enabled: bool):
        """Enable/disable training mode."""
        self.training_mode = enabled
        if enabled:
            self.trajectory = []

    def select_action(
        self,
        state: GameState,
        legal_actions: List[int],
        player: Player
    ) -> int:
        """
        Select action using policy network.

        Training mode: Sample from policy distribution
        Evaluation mode: Select highest probability action

        Args:
            state: Current game state
            legal_actions: List of legal action indices
            player: Current player

        Returns:
            Selected action index
        """
        # Extract features
        features = self.feature_extractor.extract(state, player)

        # Create legal action mask
        legal_mask = np.zeros(86, dtype=bool)
        legal_mask[legal_actions] = True

        # Forward pass
        action_probs = self.network.forward(features, legal_mask)

        # Select action
        if self.training_mode:
            # Sample from distribution
            action = self.rng.choices(legal_actions, weights=action_probs[legal_actions])[0]

            # Store in trajectory
            self.trajectory.append({
                'features': features,
                'action': action,
                'legal_mask': legal_mask,
                'prob': action_probs[action]
            })
        else:
            # Greedy: select highest probability
            action = legal_actions[np.argmax(action_probs[legal_actions])]

        return action

    def observe_outcome(self, outcome: float):
        """
        Called after game ends. Update policy using REINFORCE.

        Args:
            outcome: Game outcome (1.0=win, 0.5=draw, 0.0=loss)
        """
        if not self.training_mode or len(self.trajectory) == 0:
            return

        # Compute returns (discounted rewards)
        returns = []
        G = outcome  # Terminal reward

        for step in reversed(self.trajectory):
            returns.insert(0, G)
            G = self.discount * G  # Discount for earlier steps

        # Update baseline (moving average of returns)
        mean_return = np.mean(returns)
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * mean_return

        # Accumulate gradients over episode
        accumulated_grads = {
            'dW1': np.zeros_like(self.network.W1),
            'db1': np.zeros_like(self.network.b1),
            'dW2': np.zeros_like(self.network.W2),
            'db2': np.zeros_like(self.network.b2)
        }

        # Update policy for each step
        for step, G in zip(self.trajectory, returns):
            # Compute advantage (return - baseline)
            advantage = G - self.baseline

            # Forward pass (to populate cache)
            self.network.forward(step['features'], step['legal_mask'])

            # Backward pass
            grads = self.network.backward(step['action'], advantage)

            # Accumulate gradients
            for key in accumulated_grads:
                accumulated_grads[key] += grads[key]

        # Update weights
        self.network.update_weights(accumulated_grads)

        # Clear trajectory
        self.trajectory = []
        self.episode_count += 1
        self.total_updates += 1

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'episodes': self.episode_count,
            'total_updates': self.total_updates,
            'baseline': self.baseline,
            'avg_trajectory_length': len(self.trajectory) if self.trajectory else 0
        }

    def to_dict(self) -> Dict:
        """Export agent state."""
        return {
            'network': self.network.to_dict(),
            'discount': self.discount,
            'baseline': self.baseline,
            'baseline_decay': self.baseline_decay,
            'episode_count': self.episode_count,
            'total_updates': self.total_updates
        }

    def from_dict(self, data: Dict):
        """Import agent state."""
        self.network.from_dict(data['network'])
        self.discount = data['discount']
        self.baseline = data['baseline']
        self.baseline_decay = data['baseline_decay']
        self.episode_count = data.get('episode_count', 0)
        self.total_updates = data.get('total_updates', 0)

    def __repr__(self) -> str:
        return (
            f"PolicyNetworkAgent({self.name}, "
            f"hidden={self.network.hidden_size}, "
            f"episodes={self.episode_count})"
        )
