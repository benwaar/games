"""
Experience Replay Buffer for DQN.

Stores transitions (s, a, r, s', done) and samples random minibatches
to break temporal correlation in training data.
"""

import numpy as np
from typing import Tuple, List
from collections import deque


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.

    Implements circular buffer (FIFO): when full, oldest transitions are evicted.
    """

    def __init__(self, capacity: int, state_dim: int = 53):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimensionality of state features
        """
        self.capacity = capacity
        self.state_dim = state_dim

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.position = 0  # Next position to write
        self.size = 0      # Current number of stored transitions

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state features (state_dim,)
            action: Action index taken
            reward: Reward received
            next_state: Next state features (state_dim,)
            done: Whether episode ended
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random minibatch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each is a numpy array of shape (batch_size, ...)
        """
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} transitions, only {self.size} available")

        # Sample random indices
        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough transitions for training."""
        return self.size >= min_size

    def clear(self):
        """Clear all transitions from buffer."""
        self.position = 0
        self.size = 0

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
            }

        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity,
            "avg_reward": np.mean(self.rewards[:self.size]),
            "done_rate": np.mean(self.dones[:self.size]),
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with prioritized sampling (optional enhancement).

    Samples transitions with probability proportional to TD error.
    More important transitions are sampled more often.

    Note: This is a stretch feature - implement only if basic DQN works well.
    """

    def __init__(self, capacity: int, state_dim: int = 53, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            state_dim: State feature dimensionality
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(capacity, state_dim)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add transition with maximum priority."""
        super().push(state, action, reward, next_state, done)
        # New transitions get max priority (will be updated after training)
        self.priorities[self.position - 1] = self.max_priority

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """
        Sample transitions with priority-based probability.

        Args:
            batch_size: Number of transitions
            beta: Importance sampling exponent (annealed from 0.4 to 1.0)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
            weights are importance sampling weights to correct bias
        """
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} transitions, only {self.size} available")

        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize by max weight

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of transitions to update
            td_errors: Absolute TD errors for these transitions
        """
        priorities = np.abs(td_errors) + 1e-6  # Small epsilon to avoid zero priority
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


if __name__ == "__main__":
    # Test replay buffer
    print("Testing Replay Buffer...")

    buffer = ReplayBuffer(capacity=10, state_dim=53)

    # Add some transitions
    for i in range(15):
        state = np.random.randn(53).astype(np.float32)
        action = i % 86
        reward = np.random.randn()
        next_state = np.random.randn(53).astype(np.float32)
        done = (i % 5 == 4)  # Every 5th transition is terminal

        buffer.push(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)} / {buffer.capacity}")
    print(f"Stats: {buffer.get_stats()}")

    # Sample a minibatch
    if buffer.is_ready(min_size=4):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)
        print(f"\nSampled minibatch:")
        print(f"  States shape: {states.shape}")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")
        print(f"  Dones: {dones}")

    print("\n✓ Replay Buffer test passed!")

    # Test prioritized buffer
    print("\nTesting Prioritized Replay Buffer...")
    pri_buffer = PrioritizedReplayBuffer(capacity=10, state_dim=53)

    for i in range(15):
        state = np.random.randn(53).astype(np.float32)
        action = i % 86
        reward = np.random.randn()
        next_state = np.random.randn(53).astype(np.float32)
        done = (i % 5 == 4)
        pri_buffer.push(state, action, reward, next_state, done)

    if pri_buffer.is_ready(min_size=4):
        result = pri_buffer.sample(batch_size=4, beta=0.4)
        states, actions, rewards, next_states, dones, indices, weights = result
        print(f"Prioritized sample:")
        print(f"  Indices: {indices}")
        print(f"  Importance weights: {weights}")

        # Update priorities
        td_errors = np.abs(np.random.randn(4))
        pri_buffer.update_priorities(indices, td_errors)
        print(f"  Updated priorities for indices {indices}")

    print("\n✓ Prioritized Replay Buffer test passed!")
