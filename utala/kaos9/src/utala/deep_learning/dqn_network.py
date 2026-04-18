"""
Deep Q-Network architecture for utala: kaos 9.

A simple feedforward network that maps state features to Q-values for each action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for estimating action values.

    Architecture:
        Input: 53-dim state features
        Hidden: 128 -> 128 (ReLU activations)
        Output: 86-dim Q-values (one per action)

    The network outputs Q(s,a) for all actions simultaneously,
    allowing efficient action selection via argmax.
    """

    def __init__(self, state_dim: int = 53, action_dim: int = 86, hidden_dim: int = 128):
        """
        Initialize the Q-network.

        Args:
            state_dim: Dimensionality of state features (default: 53)
            action_dim: Number of actions (default: 86)
            hidden_dim: Size of hidden layers (default: 128)
        """
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights using He initialization (good for ReLU)
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-values for all actions.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # No activation on output (Q-values can be any real number)
        return q_values

    def get_action(self, state: torch.Tensor, legal_actions_mask: torch.Tensor) -> int:
        """
        Select best action for a single state using greedy policy.

        Args:
            state: State tensor of shape (state_dim,)
            legal_actions_mask: Boolean tensor of shape (action_dim,)
                                True for legal actions, False for illegal

        Returns:
            Action index (int)
        """
        with torch.no_grad():
            q_values = self.forward(state)

            # Mask illegal actions with large negative value
            masked_q = q_values.clone()
            masked_q[~legal_actions_mask] = float('-inf')

            # Select action with highest Q-value
            action = torch.argmax(masked_q).item()

        return action

    def get_max_q_value(self, state: torch.Tensor, legal_actions_mask: torch.Tensor) -> float:
        """
        Get maximum Q-value among legal actions.

        Args:
            state: State tensor of shape (state_dim,)
            legal_actions_mask: Boolean tensor of shape (action_dim,)

        Returns:
            Maximum Q-value (float)
        """
        with torch.no_grad():
            q_values = self.forward(state)
            masked_q = q_values.clone()
            masked_q[~legal_actions_mask] = float('-inf')
            max_q = torch.max(masked_q).item()
        return max_q


def create_dqn_network(state_dim: int = 53, action_dim: int = 86, hidden_dim: int = 128) -> DQNNetwork:
    """
    Factory function to create a DQN network.

    Args:
        state_dim: State feature dimensionality
        action_dim: Number of actions
        hidden_dim: Hidden layer size

    Returns:
        Initialized DQNNetwork
    """
    return DQNNetwork(state_dim, action_dim, hidden_dim)


if __name__ == "__main__":
    # Test the network
    print("Testing DQN Network...")

    # Create network
    net = create_dqn_network()
    print(f"Network created: {net.state_dim} -> {net.hidden_dim} -> {net.hidden_dim} -> {net.action_dim}")

    # Test forward pass
    batch_size = 4
    state = torch.randn(batch_size, 53)
    q_values = net(state)
    print(f"Forward pass: input shape {state.shape} -> output shape {q_values.shape}")

    # Test single action selection
    single_state = torch.randn(53)
    legal_mask = torch.ones(86, dtype=torch.bool)
    legal_mask[10:20] = False  # Make some actions illegal

    action = net.get_action(single_state, legal_mask)
    max_q = net.get_max_q_value(single_state, legal_mask)
    print(f"Selected action: {action}, Max Q-value: {max_q:.3f}")

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n✓ DQN Network test passed!")
