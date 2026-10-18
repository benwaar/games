"""
Deep Q-Network (DQN) Agent for utala: kaos 9.

Implements DQN with experience replay and target networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import random
import json
from pathlib import Path

from .dqn_network import DQNNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from ..agents.base import Agent
from ..state import GameState, GameConfig, Player
from ..actions import get_action_space
from ..learning.dqn_features import DQNFeatureExtractor


class DQNAgent(Agent):
    """
    Deep Q-Network agent with experience replay and target network.

    Key components:
    - Q-network: estimates Q(s,a)
    - Target network: provides stable TD targets
    - Replay buffer: stores and samples transitions
    - Epsilon-greedy: balances exploration/exploitation
    """

    def __init__(
        self,
        name: str = "DQN",
        learning_rate: float = 0.001,
        discount: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 20000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        hidden_dim: int = 128,
        n_step: int = 1,
        seed: int | None = None,
        config: GameConfig | None = None,
        polyak_tau: float = 0.0,
        use_prioritized_replay: bool = False,
        prioritized_alpha: float = 0.6,
    ):
        """
        Initialize DQN agent.

        Args:
            name: Agent name
            learning_rate: Learning rate for Adam optimizer
            discount: Discount factor (γ)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay factor per step
            replay_capacity: Replay buffer size
            batch_size: Minibatch size for training
            target_update_freq: Steps between target network updates
            hidden_dim: Hidden layer size
            n_step: N-step returns (1=standard, 3=recommended for sparse rewards)
            seed: Random seed
            config: Game config (Variant A needs 95 actions)
            polyak_tau: Soft target update coefficient (0.0 = hard copy every target_update_freq steps,
                        e.g. 0.01 = slow Polyak average applied every step)
            use_prioritized_replay: Use PrioritizedReplayBuffer instead of uniform
            prioritized_alpha: Priority exponent for prioritized replay (0=uniform, 1=full)
        """
        super().__init__(name)

        # Config
        self.config = config or GameConfig()
        self.n_step = n_step

        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Feature extraction (bluffing-aware)
        self.feature_extractor = DQNFeatureExtractor(config=self.config)
        self.state_dim = self.feature_extractor.feature_dim
        self.action_dim = get_action_space(self.config).size()

        # Networks
        self.q_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize target = Q
        self.target_network.eval()  # Target network always in eval mode

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(replay_capacity, self.state_dim, alpha=prioritized_alpha)
            self.prioritized_beta = 0.4  # Annealed toward 1.0 during training
        else:
            self.replay_buffer = ReplayBuffer(replay_capacity, self.state_dim)

        # Target update strategy
        self.polyak_tau = polyak_tau  # 0.0 = hard update, >0 = soft (Polyak) update

        # Training state
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.losses = []

        # Current episode trajectory (for adding to replay buffer)
        self.player = Player.ONE  # Will be set properly in game_start()
        self.episode_states = []
        self.episode_actions = []

        # RNG
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Training mode
        self.training = True

    def game_start(self, player: Player, seed: int | None = None):
        """Called when game starts."""
        self.player = player
        self.episode_states = []
        self.episode_actions = []

    def select_action(
        self,
        state: GameState,
        legal_actions: List[int],
        player: Player
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current game state
            legal_actions: List of legal action indices
            player: Current player

        Returns:
            Selected action index
        """
        # Extract features
        state_features = self.feature_extractor.extract(state, player)

        # Store state for trajectory
        if self.training:
            self.episode_states.append(state_features.copy())

        # Epsilon-greedy exploration
        if self.training and random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            # Greedy: use Q-network
            state_tensor = torch.FloatTensor(state_features)
            legal_mask = torch.zeros(self.action_dim, dtype=torch.bool)
            legal_mask[legal_actions] = True

            action = self.q_network.get_action(state_tensor, legal_mask)

        # Store action for trajectory
        if self.training:
            self.episode_actions.append(action)

        return action

    def game_end(self, final_state: GameState, winner: Player | None):
        """
        Called when game ends. Add episode to replay buffer and train.

        Uses n-step returns to bridge sparse terminal rewards back through
        the trajectory, helping credit assignment over long games.

        Args:
            final_state: Final game state
            winner: Winner (or None for draw)
        """
        if not self.training:
            return

        # Determine reward
        player = self.player  # Set by game_start()
        if winner is None:
            reward = 0.0  # Draw
        elif winner == player:
            reward = 1.0  # Win
        else:
            reward = -1.0  # Loss

        # Build reward vector (sparse: 0 everywhere, reward at end)
        num_transitions = len(self.episode_states) - 1
        rewards = [0.0] * num_transitions
        if num_transitions > 0:
            rewards[-1] = reward

        # Add final state
        final_features = self.feature_extractor.extract(final_state, player)

        # Add transitions with n-step returns
        for i in range(num_transitions):
            state = self.episode_states[i]
            action = self.episode_actions[i]

            # Compute n-step return: R = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1}
            n_step_return = 0.0
            steps_ahead = min(self.n_step, num_transitions - i)
            for k in range(steps_ahead):
                n_step_return += (self.discount ** k) * rewards[i + k]

            # Next state is n steps ahead (or terminal)
            if i + steps_ahead < num_transitions:
                next_state = self.episode_states[i + steps_ahead]
                done = False
            else:
                next_state = final_features
                done = True

            # Store with n-step discount for target computation
            # Target: n_step_return + γ^n * max Q(s_{t+n})
            self.replay_buffer.push(state, action, n_step_return, next_state, done)

        # Train if buffer has enough samples
        if self.replay_buffer.is_ready(self.batch_size):
            loss = self._train_step()
            self.losses.append(loss)

        # Update target network
        if self.polyak_tau > 0.0:
            # Soft (Polyak) update every step: θ_target ← τ·θ + (1-τ)·θ_target
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.polyak_tau * param.data + (1.0 - self.polyak_tau) * target_param.data)
        elif self.steps % self.target_update_freq == 0:
            # Hard update every target_update_freq steps
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Clear episode buffers
        self.episode_states = []
        self.episode_actions = []
        self.episodes += 1
        self.total_reward += reward

    def _train_step(self) -> float:
        """
        Perform one training step on a minibatch.

        Returns:
            Training loss
        """
        # Sample minibatch (uniform or prioritized)
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, is_weights = \
                self.replay_buffer.sample(self.batch_size, beta=self.prioritized_beta)
            is_weights_t = torch.FloatTensor(is_weights)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            is_weights_t = None
            indices = None

        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Current Q-values: Q(s,a)
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values: R_n + γ^n × max_a' Q_target(s_{t+n}, a')
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1)[0]
            n_step_discount = self.discount ** self.n_step
            target_q = rewards_t + (1.0 - dones_t) * n_step_discount * next_q

        # Compute loss (weighted by IS weights for prioritized replay)
        td_errors = current_q - target_q
        if is_weights_t is not None:
            loss = (is_weights_t * td_errors.pow(2)).mean()
            self.replay_buffer.update_priorities(indices, td_errors.detach().abs().numpy())
        else:
            loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps += 1

        return loss.item()

    def set_training(self, training: bool):
        """Enable or disable training mode."""
        self.training = training
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()

    def save(self, path: str):
        """Save agent state to file."""
        save_dict = {
            'name': self.name,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount': self.discount,
                'epsilon_start': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'replay_capacity': self.replay_buffer.capacity,
                'n_step': self.n_step,
                'fixed_dogfight_order': self.config.fixed_dogfight_order,
                'polyak_tau': self.polyak_tau,
                'use_prioritized_replay': self.use_prioritized_replay,
            },
            'training_stats': {
                'steps': self.steps,
                'episodes': self.episodes,
                'total_reward': self.total_reward,
                'avg_loss': np.mean(self.losses[-1000:]) if self.losses else 0.0,
                'buffer_size': len(self.replay_buffer),
            },
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }

        # Save with torch
        torch.save(save_dict, path)

        # Also save metadata as JSON
        metadata_path = Path(path).with_suffix('.json')
        metadata = {
            'name': self.name,
            'agent_type': 'dqn',
            'hyperparameters': save_dict['hyperparameters'],
            'training_stats': save_dict['training_stats'],
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load(path: str) -> 'DQNAgent':
        """Load agent from file."""
        checkpoint = torch.load(path, weights_only=False)

        hp = checkpoint['hyperparameters']
        config = GameConfig(
            fixed_dogfight_order=hp.get('fixed_dogfight_order', True)
        )
        agent = DQNAgent(
            name=checkpoint['name'],
            learning_rate=hp['learning_rate'],
            discount=hp['discount'],
            epsilon_start=hp['epsilon_start'],
            epsilon_end=hp['epsilon_end'],
            epsilon_decay=hp['epsilon_decay'],
            replay_capacity=hp['replay_capacity'],
            batch_size=hp['batch_size'],
            target_update_freq=hp['target_update_freq'],
            polyak_tau=hp.get('polyak_tau', 0.0),
            use_prioritized_replay=hp.get('use_prioritized_replay', False),
            config=config,
        )

        # Load network states
        agent.q_network.load_state_dict(checkpoint['q_network_state'])
        agent.target_network.load_state_dict(checkpoint['target_network_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Load training stats
        stats = checkpoint['training_stats']
        agent.steps = stats['steps']
        agent.episodes = stats['episodes']
        agent.total_reward = stats['total_reward']

        return agent

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.capacity,
            'avg_loss_recent': np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0.0,
            'total_reward': self.total_reward,
            'avg_reward_per_episode': self.total_reward / max(self.episodes, 1),
        }


if __name__ == "__main__":
    from ..state import GameConfig
    print("Testing DQN Agent...")

    # Test with Variant A config
    config = GameConfig(fixed_dogfight_order=False)
    agent = DQNAgent(name="DQN-Test", seed=42, config=config, n_step=3)
    print(f"Agent created: {agent.name}")
    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print(f"N-step: {agent.n_step}")
    print(f"Q-network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")

    print("\n✓ DQN Agent test passed!")
    print("\nReady for training!")
