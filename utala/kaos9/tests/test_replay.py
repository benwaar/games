"""
Unit tests for replay format.
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from utala.replays.format import ReplayV1, ReplayMetadata, create_replay_from_game
from utala.engine import GameEngine
from utala.agents.random_agent import RandomAgent
from utala.state import Player


class TestReplayMetadata(unittest.TestCase):
    """Test ReplayMetadata."""

    def test_metadata_creation(self):
        """Test creating metadata."""
        metadata = ReplayMetadata(
            game_variant="level1",
            player_one_name="Alice",
            player_two_name="Bob"
        )

        self.assertEqual(metadata.format_version, "v1")
        self.assertEqual(metadata.game_variant, "level1")
        self.assertEqual(metadata.player_one_name, "Alice")
        self.assertEqual(metadata.player_two_name, "Bob")

    def test_default_metadata(self):
        """Test default metadata values."""
        metadata = ReplayMetadata()

        self.assertEqual(metadata.format_version, "v1")
        self.assertEqual(metadata.game_variant, "level1")


class TestReplayV1(unittest.TestCase):
    """Test ReplayV1 format."""

    def test_replay_creation(self):
        """Test creating a replay."""
        replay = ReplayV1(
            seed=1337,
            actions=[(0, 5), (1, 10)],
            metadata=ReplayMetadata(),
            winner=0
        )

        self.assertEqual(replay.seed, 1337)
        self.assertEqual(len(replay.actions), 2)
        self.assertEqual(replay.winner, 0)

    def test_to_dict(self):
        """Test converting replay to dictionary."""
        replay = ReplayV1(
            seed=42,
            actions=[(0, 1), (1, 2)],
            metadata=ReplayMetadata(),
            winner=None
        )

        data = replay.to_dict()

        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["actions"], [[0, 1], [1, 2]])
        self.assertIsNone(data["winner"])
        self.assertIn("metadata", data)

    def test_to_json(self):
        """Test converting replay to JSON."""
        replay = ReplayV1(
            seed=99,
            actions=[(0, 5)],
            metadata=ReplayMetadata(),
            winner=1
        )

        json_str = replay.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["seed"], 99)
        self.assertEqual(parsed["winner"], 1)

    def test_from_dict(self):
        """Test loading replay from dictionary."""
        data = {
            "format_version": "v1",
            "seed": 123,
            "actions": [[0, 10], [1, 20]],
            "metadata": {
                "format_version": "v1",
                "game_variant": "level1",
                "player_one_name": "Player 1",
                "player_two_name": "Player 2",
                "timestamp": None
            },
            "winner": 0
        }

        replay = ReplayV1.from_dict(data)

        self.assertEqual(replay.seed, 123)
        self.assertEqual(len(replay.actions), 2)
        self.assertEqual(replay.actions[0], (0, 10))
        self.assertEqual(replay.winner, 0)

    def test_from_json(self):
        """Test loading replay from JSON."""
        json_str = '''
        {
            "format_version": "v1",
            "seed": 456,
            "actions": [[0, 5], [1, 6]],
            "metadata": {
                "format_version": "v1",
                "game_variant": "level1",
                "player_one_name": "Player 1",
                "player_two_name": "Player 2",
                "timestamp": null
            },
            "winner": 1
        }
        '''

        replay = ReplayV1.from_json(json_str)

        self.assertEqual(replay.seed, 456)
        self.assertEqual(len(replay.actions), 2)
        self.assertEqual(replay.winner, 1)

    def test_roundtrip(self):
        """Test replay survives serialization roundtrip."""
        original = ReplayV1(
            seed=7777,
            actions=[(0, 1), (1, 2), (0, 3)],
            metadata=ReplayMetadata(
                player_one_name="Alice",
                player_two_name="Bob"
            ),
            winner=0
        )

        # Convert to JSON and back
        json_str = original.to_json()
        restored = ReplayV1.from_json(json_str)

        self.assertEqual(original.seed, restored.seed)
        self.assertEqual(original.actions, restored.actions)
        self.assertEqual(original.winner, restored.winner)
        self.assertEqual(original.metadata.player_one_name,
                        restored.metadata.player_one_name)

    def test_save_and_load(self):
        """Test saving and loading replay from file."""
        replay = ReplayV1(
            seed=8888,
            actions=[(0, 10), (1, 20)],
            metadata=ReplayMetadata(),
            winner=1
        )

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_replay.json"

            replay.save(filepath)
            self.assertTrue(filepath.exists())

            # Load back
            loaded = ReplayV1.load(filepath)

            self.assertEqual(replay.seed, loaded.seed)
            self.assertEqual(replay.actions, loaded.actions)
            self.assertEqual(replay.winner, loaded.winner)


class TestCreateReplayFromGame(unittest.TestCase):
    """Test creating replays from completed games."""

    def test_create_replay_from_completed_game(self):
        """Test creating replay from a completed game."""
        # Play a quick game
        engine = GameEngine(seed=42)
        agent1 = RandomAgent("P1", seed=100)
        agent2 = RandomAgent("P2", seed=101)

        # Play through placement
        while engine.state.phase.value == "placement":
            state = engine.get_state_copy()
            current = state.current_player
            agent = agent1 if current == Player.ONE else agent2
            legal = engine.get_legal_actions()
            action = agent.select_action(state, legal, current)
            engine.apply_action(action)

        # Play through dogfights
        while engine.state.phase.value == "dogfights":
            state = engine.get_state_copy()
            legal_p1 = engine.get_legal_actions(Player.ONE)
            legal_p2 = engine.get_legal_actions(Player.TWO)

            action_p1 = agent1.select_action(state, legal_p1, Player.ONE)
            action_p2 = agent2.select_action(state, legal_p2, Player.TWO)

            engine.apply_dogfight_actions(action_p1, action_p2)

        # Create replay
        metadata = ReplayMetadata(
            player_one_name="TestPlayer1",
            player_two_name="TestPlayer2"
        )
        replay = create_replay_from_game(engine, metadata)

        # Verify replay
        self.assertEqual(replay.seed, engine.seed)
        self.assertGreater(len(replay.actions), 0)
        self.assertIn(replay.winner, [0, 1, None])

        # Check actions are properly formatted
        for player_id, action_idx in replay.actions:
            self.assertIn(player_id, [0, 1])
            self.assertIsInstance(action_idx, int)

    def test_replay_captures_all_actions(self):
        """Test replay captures all actions from game."""
        engine = GameEngine(seed=123)
        agent = RandomAgent("Test", seed=42)

        # Play through placement and count actions
        action_count = 0
        while engine.state.phase.value == "placement":
            legal = engine.get_legal_actions()
            action = agent.select_action(engine.get_state_copy(), legal,
                                        engine.state.current_player)
            engine.apply_action(action)
            action_count += 1

        # Create replay
        replay = create_replay_from_game(engine)

        # Replay should have recorded all placement actions
        self.assertEqual(len(replay.actions), action_count)


if __name__ == '__main__':
    unittest.main()
