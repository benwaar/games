"""
Training data generation pipeline for learning agents.

Generates (state, action, outcome) tuples from self-play games.
Data is saved in compact JSONL format for efficient storage and loading.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np

from ..agents.base import Agent
from ..engine import GameEngine
from ..state import Player, GameState
from .features import get_feature_extractor


@dataclass
class TrainingExample:
    """A single training example: (state features, action, outcome)."""

    features: List[float]  # State feature vector
    action: int  # Action index taken
    player: int  # Player who took the action (0 or 1)
    turn: int  # Turn number
    outcome: float  # Game outcome from player's perspective (1.0=win, 0.5=draw, 0.0=loss)
    seed: int  # Game seed for reproducibility

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @staticmethod
    def from_dict(data):
        """Create from dictionary."""
        return TrainingExample(**data)


class TrainingDataGenerator:
    """
    Generates training data from agent self-play.

    Records every decision point during games and labels outcomes.
    """

    def __init__(self):
        self.feature_extractor = get_feature_extractor()

    def generate_game_data(
        self,
        agent_one: Agent,
        agent_two: Agent,
        seed: int | None = None
    ) -> List[TrainingExample]:
        """
        Generate training data from a single game.

        Args:
            agent_one: Agent playing as Player ONE
            agent_two: Agent playing as Player TWO
            seed: RNG seed for reproducibility

        Returns:
            List of TrainingExample objects (one per decision point)
        """
        engine = GameEngine(seed=seed)
        examples = []

        # Notify agents
        agent_one.game_start(Player.ONE, engine.seed)
        agent_two.game_start(Player.TWO, engine.seed)

        # Play game and record decisions
        while not engine.is_game_over():
            state = engine.get_state_copy()

            if state.phase.value == "placement":
                # Placement phase
                current_player = state.current_player
                agent = agent_one if current_player == Player.ONE else agent_two

                # Extract features
                features = self.feature_extractor.extract(state, current_player)

                # Get agent's action
                legal_actions = engine.get_legal_actions()
                action_idx = agent.select_action(state, legal_actions, current_player)

                # Record decision (outcome filled in later)
                examples.append({
                    'features': features.tolist(),
                    'action': action_idx,
                    'player': current_player.value,
                    'turn': state.turn_number,
                    'seed': engine.seed,
                    'outcome': None  # Filled after game ends
                })

                engine.apply_action(action_idx)

            elif state.phase.value == "dogfights":
                # Dogfight phase
                engine.begin_current_dogfight()
                state = engine.get_state_copy()

                # Collect turn-by-turn dogfight actions
                while not engine.is_dogfight_complete():
                    current_player = engine.get_dogfight_current_actor()
                    agent = agent_one if current_player == Player.ONE else agent_two

                    # Extract features
                    features = self.feature_extractor.extract(state, current_player)

                    # Get agent's action
                    legal_actions = engine.get_dogfight_legal_actions_for_player(current_player)
                    action_idx = agent.select_action(state, legal_actions, current_player)

                    # Record decision
                    examples.append({
                        'features': features.tolist(),
                        'action': action_idx,
                        'player': current_player.value,
                        'turn': state.turn_number,
                        'seed': engine.seed,
                        'outcome': None
                    })

                    engine.apply_dogfight_turn_action(current_player, action_idx)

                engine.finish_current_dogfight()

        # Game ended - label all examples with outcome
        winner = engine.get_winner()

        training_examples = []
        for ex_dict in examples:
            player = Player(ex_dict['player'])

            # Assign outcome from player's perspective
            if winner is None:
                outcome = 0.5  # Draw
            elif winner == player:
                outcome = 1.0  # Win
            else:
                outcome = 0.0  # Loss

            training_examples.append(TrainingExample(
                features=ex_dict['features'],
                action=ex_dict['action'],
                player=ex_dict['player'],
                turn=ex_dict['turn'],
                outcome=outcome,
                seed=ex_dict['seed']
            ))

        # Notify agents
        final_state = engine.get_state_copy()
        agent_one.game_end(final_state, winner)
        agent_two.game_end(final_state, winner)

        return training_examples

    def generate_batch(
        self,
        agent_one: Agent,
        agent_two: Agent,
        num_games: int,
        starting_seed: int
    ) -> List[TrainingExample]:
        """
        Generate training data from multiple games.

        Args:
            agent_one: First agent
            agent_two: Second agent
            num_games: Number of games to play
            starting_seed: Starting seed (incremented per game)

        Returns:
            List of all TrainingExamples from all games
        """
        all_examples = []

        for i in range(num_games):
            seed = starting_seed + i
            game_examples = self.generate_game_data(agent_one, agent_two, seed)
            all_examples.extend(game_examples)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_games} games ({len(all_examples)} examples)")

        return all_examples

    def save_training_data(
        self,
        examples: List[TrainingExample],
        filepath: str,
        metadata: dict | None = None
    ):
        """
        Save training data to JSONL file.

        Args:
            examples: List of training examples
            filepath: Output file path
            metadata: Optional metadata (agent names, version, etc.)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write metadata as first line (with special marker)
        with open(filepath, 'w') as f:
            if metadata is not None:
                f.write(json.dumps({'__metadata__': metadata}) + '\n')

            # Write examples as JSONL
            for example in examples:
                f.write(json.dumps(example.to_dict()) + '\n')

        print(f"Saved {len(examples)} training examples to: {filepath}")

    def load_training_data(
        self,
        filepath: str
    ) -> Tuple[List[TrainingExample], dict | None]:
        """
        Load training data from JSONL file.

        Args:
            filepath: Input file path

        Returns:
            Tuple of (examples, metadata)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training data file not found: {filepath}")

        examples = []
        metadata = None

        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line.strip())

                # Check for metadata line
                if '__metadata__' in data:
                    metadata = data['__metadata__']
                    continue

                examples.append(TrainingExample.from_dict(data))

        print(f"Loaded {len(examples)} training examples from: {filepath}")
        if metadata:
            print(f"  Metadata: {metadata}")

        return examples, metadata


def generate_training_dataset(
    agent_one: Agent,
    agent_two: Agent,
    num_games: int,
    output_path: str,
    starting_seed: int = 100000,
    metadata: dict | None = None
):
    """
    Convenience function to generate and save a training dataset.

    Args:
        agent_one: First agent
        agent_two: Second agent
        num_games: Number of games to generate
        output_path: Where to save the data
        starting_seed: Starting RNG seed
        metadata: Optional metadata to include
    """
    print(f"Generating training data: {num_games} games")
    print(f"  P1: {agent_one.name}")
    print(f"  P2: {agent_two.name}")
    print(f"  Output: {output_path}")

    generator = TrainingDataGenerator()
    examples = generator.generate_batch(agent_one, agent_two, num_games, starting_seed)

    # Add agent info to metadata
    if metadata is None:
        metadata = {}
    metadata.update({
        'agent_one': agent_one.name,
        'agent_two': agent_two.name,
        'num_games': num_games,
        'starting_seed': starting_seed,
        'num_examples': len(examples)
    })

    generator.save_training_data(examples, output_path, metadata)

    return examples
