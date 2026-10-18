#!/usr/bin/env python3
"""
Phase 3.2 DQN Sanity Check: CartPole-v1

Validates that our DQN implementation (DQNNetwork + ReplayBuffer) can solve
a known RL problem. If this fails, the DQN code is buggy. If it passes,
utala's poor DQN results are a real finding about the game, not a bug.

Reuses:
- DQNNetwork from src/utala/deep_learning/dqn_network.py
- ReplayBuffer from src/utala/deep_learning/replay_buffer.py
"""

import sys
sys.path.insert(0, 'src')

import random
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from collections import deque

from utala.deep_learning.dqn_network import DQNNetwork
from utala.deep_learning.replay_buffer import ReplayBuffer

# Hyperparameters (matching our utala DQN where applicable)
SEED = 42
STATE_DIM = 4       # CartPole observation space
ACTION_DIM = 2      # CartPole action space
HIDDEN_DIM = 64     # Smaller than utala (64 vs 128) — CartPole is simpler
LR = 0.001          # Same as utala
DISCOUNT = 0.99     # Higher than utala (0.99 vs 0.95) — CartPole has longer horizons
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Same as utala
REPLAY_CAPACITY = 10000
BATCH_SIZE = 64        # Same as utala
TARGET_UPDATE = 100    # Steps between target network sync
MAX_EPISODES = 500
SOLVED_THRESHOLD = 195  # Standard CartPole-v1 solved criterion (avg over 100 episodes)


def main():
    print("=" * 60, flush=True)
    print("Phase 3.2: DQN Sanity Check — CartPole-v1", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Environment
    env = gym.make('CartPole-v1')

    # Reuse our DQN components
    q_net = DQNNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    target_net = DQNNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_CAPACITY, STATE_DIM)

    epsilon = EPSILON_START
    steps = 0
    rewards_history = deque(maxlen=100)
    solved = False
    solved_episode = None

    print(f"Network: {STATE_DIM} -> {HIDDEN_DIM} -> {HIDDEN_DIM} -> {ACTION_DIM}", flush=True)
    print(f"Params: {sum(p.numel() for p in q_net.parameters()):,}", flush=True)
    print(f"Target: avg reward >= {SOLVED_THRESHOLD} over 100 episodes", flush=True)
    print(flush=True)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state)
                    q_values = q_net(state_t)
                    action = q_values.argmax().item()

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition
            replay.push(
                np.array(state, dtype=np.float32),
                action,
                reward,
                np.array(next_state, dtype=np.float32),
                done
            )

            # Train
            if replay.is_ready(BATCH_SIZE):
                states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones)

                # Current Q
                current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # Target Q
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(dim=1)[0]
                    target_q = rewards_t + (1.0 - dones_t) * DISCOUNT * next_q

                # Loss and optimize
                loss = torch.nn.functional.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()

                steps += 1

                # Update target network
                if steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(q_net.state_dict())

            state = next_state
            if done:
                break

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history)

        # Progress report
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1:>4d}  |  Avg reward: {avg_reward:>6.1f}  |  Epsilon: {epsilon:.3f}  |  Steps: {steps}", flush=True)

        # Check solved
        if len(rewards_history) >= 100 and avg_reward >= SOLVED_THRESHOLD and not solved:
            solved = True
            solved_episode = episode + 1
            print(f"\n*** SOLVED at episode {solved_episode}! Avg reward: {avg_reward:.1f} ***\n", flush=True)

    env.close()

    # Final verdict
    final_avg = np.mean(rewards_history)
    print(flush=True)
    print("=" * 60, flush=True)
    print("VERDICT", flush=True)
    print("=" * 60, flush=True)

    if solved:
        print(f"PASS: DQN solved CartPole at episode {solved_episode}", flush=True)
        print(f"Final avg reward: {final_avg:.1f}", flush=True)
        print(flush=True)
        print("DQN implementation is correct.", flush=True)
        print("Utala's poor DQN results are a real finding about the game.", flush=True)
    else:
        print(f"FAIL: DQN did not solve CartPole in {MAX_EPISODES} episodes", flush=True)
        print(f"Final avg reward: {final_avg:.1f} (needed {SOLVED_THRESHOLD})", flush=True)
        print(flush=True)
        print("DQN implementation may have bugs.", flush=True)
        print("Investigate before trusting utala DQN results.", flush=True)

    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
