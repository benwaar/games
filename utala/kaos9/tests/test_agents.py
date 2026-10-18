"""
Unit tests for agents.
"""

import unittest
import sys
sys.path.insert(0, 'src')

from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent, MonteCarloAgent
from utala.engine import GameEngine
from utala.state import Player, GameState


class TestRandomAgent(unittest.TestCase):
    """Test RandomAgent."""

    def test_agent_creation(self):
        """Test creating a random agent."""
        agent = RandomAgent("TestRandom", seed=42)
        self.assertEqual(agent.name, "TestRandom")

    def test_select_action(self):
        """Test agent can select an action."""
        agent = RandomAgent("TestRandom", seed=42)
        engine = GameEngine(seed=100)

        state = engine.get_state_copy()
        legal_actions = engine.get_legal_actions()

        action = agent.select_action(state, legal_actions, Player.ONE)

        # Should return a legal action
        self.assertIn(action, legal_actions)

    def test_deterministic_with_seed(self):
        """Test agent is deterministic with same seed."""
        agent1 = RandomAgent("Test1", seed=1337)
        agent2 = RandomAgent("Test2", seed=1337)

        engine = GameEngine(seed=100)
        state = engine.get_state_copy()
        legal_actions = engine.get_legal_actions()

        action1 = agent1.select_action(state, legal_actions, Player.ONE)
        action2 = agent2.select_action(state, legal_actions, Player.ONE)

        self.assertEqual(action1, action2)

    def test_plays_full_game(self):
        """Test agent can play a full game."""
        agent1 = RandomAgent("Random1", seed=42)
        agent2 = RandomAgent("Random2", seed=43)
        engine = GameEngine(seed=100)

        # Play through placement
        while engine.state.phase.value == "placement":
            state = engine.get_state_copy()
            current = state.current_player
            agent = agent1 if current == Player.ONE else agent2
            legal = engine.get_legal_actions()
            action = agent.select_action(state, legal, current)
            engine.apply_action(action)

        # Play through dogfights (turn-based)
        while engine.state.phase.value == "dogfights":
            engine.begin_current_dogfight()
            while not engine.is_dogfight_complete():
                current = engine.get_dogfight_current_actor()
                state = engine.get_state_copy()
                agent = agent1 if current == Player.ONE else agent2
                legal = engine.get_dogfight_legal_actions_for_player(current)
                action = agent.select_action(state, legal, current)
                engine.apply_dogfight_turn_action(current, action)
            engine.finish_current_dogfight()

        # Game should end successfully
        self.assertTrue(engine.is_game_over())


class TestHeuristicAgent(unittest.TestCase):
    """Test HeuristicAgent."""

    def test_agent_creation(self):
        """Test creating a heuristic agent."""
        agent = HeuristicAgent("TestHeur", seed=42)
        self.assertEqual(agent.name, "TestHeur")

    def test_select_action(self):
        """Test agent can select an action."""
        agent = HeuristicAgent("TestHeur", seed=42)
        engine = GameEngine(seed=100)

        state = engine.get_state_copy()
        legal_actions = engine.get_legal_actions()

        action = agent.select_action(state, legal_actions, Player.ONE)

        # Should return a legal action
        self.assertIn(action, legal_actions)

    def test_prefers_center(self):
        """Test heuristic agent prefers center square."""
        agent = HeuristicAgent("TestHeur", seed=42)
        engine = GameEngine(seed=100)

        # Give agent a choice including center
        state = engine.get_state_copy()
        legal_actions = engine.get_legal_actions(Player.ONE)

        # Run multiple times to see if center is frequently chosen
        center_count = 0
        trials = 20

        for i in range(trials):
            agent_with_seed = HeuristicAgent(f"Test{i}", seed=100 + i)
            action_idx = agent_with_seed.select_action(state, legal_actions, Player.ONE)
            action = engine.action_space.get_action(action_idx)

            if action.row == 1 and action.col == 1:
                center_count += 1

        # Heuristic should prefer center more often than random (which would be ~1/81)
        # With 20 trials, expect several center placements
        self.assertGreater(center_count, 1)

    def test_plays_full_game(self):
        """Test agent can play a full game."""
        agent1 = HeuristicAgent("Heur1", seed=42)
        agent2 = HeuristicAgent("Heur2", seed=43)
        engine = GameEngine(seed=100)

        # Play through placement
        while engine.state.phase.value == "placement":
            state = engine.get_state_copy()
            current = state.current_player
            agent = agent1 if current == Player.ONE else agent2
            legal = engine.get_legal_actions()
            action = agent.select_action(state, legal, current)
            engine.apply_action(action)

        # Play through dogfights (turn-based)
        while engine.state.phase.value == "dogfights":
            engine.begin_current_dogfight()
            while not engine.is_dogfight_complete():
                current = engine.get_dogfight_current_actor()
                state = engine.get_state_copy()
                agent = agent1 if current == Player.ONE else agent2
                legal = engine.get_dogfight_legal_actions_for_player(current)
                action = agent.select_action(state, legal, current)
                engine.apply_dogfight_turn_action(current, action)
            engine.finish_current_dogfight()

        # Game should end successfully
        self.assertTrue(engine.is_game_over())


class TestMonteCarloAgent(unittest.TestCase):
    """Test MonteCarloAgent."""

    def test_agent_creation(self):
        """Test creating a Monte Carlo agent."""
        agent = FastMonteCarloAgent("TestMC", seed=42)
        self.assertEqual(agent.name, "TestMC")
        self.assertEqual(agent.num_rollouts, 10)

    def test_select_action(self):
        """Test agent can select an action."""
        agent = FastMonteCarloAgent("TestMC", seed=42)
        engine = GameEngine(seed=100)

        state = engine.get_state_copy()
        legal_actions = engine.get_legal_actions()

        # Limit legal actions to speed up test
        limited_actions = legal_actions[:5]

        action = agent.select_action(state, limited_actions, Player.ONE)

        # Should return a legal action
        self.assertIn(action, limited_actions)

    def test_plays_full_game(self):
        """Test agent can play a full game."""
        # Use very few rollouts for speed
        agent1 = MonteCarloAgent("MC1", num_rollouts=5, seed=42)
        agent2 = RandomAgent("Random2", seed=43)
        engine = GameEngine(seed=100)

        # Play through placement
        while engine.state.phase.value == "placement":
            state = engine.get_state_copy()
            current = state.current_player
            agent = agent1 if current == Player.ONE else agent2
            legal = engine.get_legal_actions()
            action = agent.select_action(state, legal, current)
            engine.apply_action(action)

        # Play through dogfights (turn-based)
        while engine.state.phase.value == "dogfights":
            engine.begin_current_dogfight()
            while not engine.is_dogfight_complete():
                current = engine.get_dogfight_current_actor()
                state = engine.get_state_copy()
                agent = agent1 if current == Player.ONE else agent2
                legal = engine.get_dogfight_legal_actions_for_player(current)
                action = agent.select_action(state, legal, current)
                engine.apply_dogfight_turn_action(current, action)
            engine.finish_current_dogfight()

        # Game should end successfully
        self.assertTrue(engine.is_game_over())


class TestAgentInterface(unittest.TestCase):
    """Test agent interface compliance."""

    def test_all_agents_have_required_methods(self):
        """Test all agents implement required methods."""
        agents = [
            RandomAgent("R"),
            HeuristicAgent("H"),
            FastMonteCarloAgent("MC")
        ]

        for agent in agents:
            self.assertTrue(hasattr(agent, 'select_action'))
            self.assertTrue(hasattr(agent, 'game_start'))
            self.assertTrue(hasattr(agent, 'game_end'))
            self.assertTrue(hasattr(agent, 'name'))

    def test_agent_callbacks(self):
        """Test agent callbacks are called."""
        agent = RandomAgent("Test")

        # Should not raise exception
        agent.game_start(Player.ONE, seed=42)
        state = GameState()
        agent.game_end(state, None)


if __name__ == '__main__':
    unittest.main()
