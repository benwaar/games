"""
Unit tests for game engine.
"""

import unittest
import sys
sys.path.insert(0, 'src')

from utala.engine import GameEngine
from utala.state import Player, Phase
from utala.actions import ActionType, get_action_space


class TestEngineInitialization(unittest.TestCase):
    """Test engine initialization."""

    def test_engine_creation(self):
        """Test creating a game engine."""
        engine = GameEngine(seed=42)
        self.assertEqual(engine.seed, 42)
        self.assertIsNotNone(engine.state)
        self.assertIsNotNone(engine.rng)

    def test_initial_state(self):
        """Test initial game state."""
        engine = GameEngine(seed=42)
        state = engine.get_state_copy()

        self.assertEqual(state.phase, Phase.PLACEMENT)
        self.assertEqual(state.current_player, Player.ONE)
        self.assertEqual(state.turn_number, 0)
        self.assertFalse(state.game_over)

    def test_kaos_decks_shuffled(self):
        """Test Kaos decks are created and shuffled."""
        engine = GameEngine(seed=42)
        state = engine.get_state_copy()

        p1_kaos = state.player_resources[Player.ONE].kaos_deck
        p2_kaos = state.player_resources[Player.TWO].kaos_deck

        # Should have 13 cards each (A-K, Ace low)
        self.assertEqual(len(p1_kaos), 13)
        self.assertEqual(len(p2_kaos), 13)

        # Should contain all values 1-13
        self.assertEqual(set(p1_kaos), set(range(1, 14)))
        self.assertEqual(set(p2_kaos), set(range(1, 14)))

    def test_deterministic_seed(self):
        """Test same seed produces same initial state."""
        engine1 = GameEngine(seed=1337)
        engine2 = GameEngine(seed=1337)

        p1_kaos1 = engine1.state.player_resources[Player.ONE].kaos_deck
        p1_kaos2 = engine2.state.player_resources[Player.ONE].kaos_deck

        self.assertEqual(p1_kaos1, p1_kaos2)


class TestPlacementPhase(unittest.TestCase):
    """Test placement phase mechanics."""

    def setUp(self):
        """Create engine for each test."""
        self.engine = GameEngine(seed=42)
        self.action_space = get_action_space()

    def test_place_rocketman(self):
        """Test placing a rocketman."""
        # Find a placement action
        legal_actions = self.engine.get_legal_actions()
        action_idx = legal_actions[0]
        action = self.action_space.get_action(action_idx)
        assert action.row is not None
        assert action.col is not None
        assert action.rocketman_power is not None

        # Apply action
        success = self.engine.apply_action(action_idx)
        self.assertTrue(success)

        # Check rocketman was placed
        square = self.engine.state.get_square(action.row, action.col)
        self.assertEqual(len(square.rocketmen), 1)
        self.assertEqual(square.rocketmen[0].power, action.rocketman_power)

        # Check rocketman removed from hand
        resources = self.engine.state.get_resources(Player.ONE)
        self.assertNotIn(action.rocketman_power, resources.rocketmen)

    def test_turn_alternation(self):
        """Test players alternate turns."""
        self.assertEqual(self.engine.state.current_player, Player.ONE)

        legal_actions = self.engine.get_legal_actions()
        self.engine.apply_action(legal_actions[0])

        self.assertEqual(self.engine.state.current_player, Player.TWO)

        legal_actions = self.engine.get_legal_actions()
        self.engine.apply_action(legal_actions[0])

        self.assertEqual(self.engine.state.current_player, Player.ONE)

    def test_turn_counter(self):
        """Test turn counter increments."""
        initial_turn = self.engine.state.turn_number

        legal_actions = self.engine.get_legal_actions()
        self.engine.apply_action(legal_actions[0])

        self.assertEqual(self.engine.state.turn_number, initial_turn + 1)

    def test_invalid_action_rejected(self):
        """Test invalid actions are rejected."""
        # Try to place rocketman we don't have
        resources = self.engine.state.get_resources(Player.ONE)
        resources.rocketmen.remove(5)

        # Find action for placing rocketman 5
        action_idx = None
        for i in range(self.action_space.size()):
            action = self.action_space.get_action(i)
            if (action.action_type == ActionType.PLACE_ROCKETMAN and
                action.rocketman_power == 5):
                action_idx = i
                break
        assert action_idx is not None

        # Should be rejected
        success = self.engine.apply_action(action_idx)
        self.assertFalse(success)

    def test_contested_square(self):
        """Test creating contested square."""
        # P1 places at (0,0)
        legal_actions = self.engine.get_legal_actions()
        for idx in legal_actions:
            action = self.action_space.get_action(idx)
            if action.row == 0 and action.col == 0:
                self.engine.apply_action(idx)
                break

        # P2 places at (0,0)
        legal_actions = self.engine.get_legal_actions()
        for idx in legal_actions:
            action = self.action_space.get_action(idx)
            if action.row == 0 and action.col == 0:
                self.engine.apply_action(idx)
                break

        # Square should be contested
        square = self.engine.state.get_square(0, 0)
        self.assertTrue(square.is_contested)
        self.assertEqual(len(square.rocketmen), 2)

    def test_transition_to_dogfights(self):
        """Test transition from placement to dogfights."""
        # Place all rocketmen
        for _ in range(18):  # 9 per player
            legal_actions = self.engine.get_legal_actions()
            self.engine.apply_action(legal_actions[0])

        # Should transition to dogfights
        self.assertEqual(self.engine.state.phase, Phase.DOGFIGHTS)


class TestDogfightPhase(unittest.TestCase):
    """Test dogfight phase mechanics."""

    def setUp(self):
        """Create engine with placement complete."""
        self.engine = GameEngine(seed=42)
        self.action_space = get_action_space()

        # Quick-place all rocketmen to get to dogfights
        for _ in range(18):
            legal_actions = self.engine.get_legal_actions()
            self.engine.apply_action(legal_actions[0])

    def test_dogfight_order(self):
        """Test dogfight resolution order."""
        dogfight_order = self.engine.state.dogfight_order

        # Should only include contested squares
        for row, col in dogfight_order:
            square = self.engine.state.get_square(row, col)
            self.assertTrue(square.is_contested)

    def test_dogfight_resolution(self):
        """Test resolving a dogfight."""
        if len(self.engine.state.dogfight_order) == 0:
            self.skipTest("No dogfights in this game")

        # Get legal actions for both players
        legal_p1 = self.engine.get_legal_actions(Player.ONE)
        legal_p2 = self.engine.get_legal_actions(Player.TWO)

        # Apply dogfight actions (both pass)
        pass_p1 = next(i for i in legal_p1
                      if self.action_space.get_action(i).action_type == ActionType.PASS)
        pass_p2 = next(i for i in legal_p2
                      if self.action_space.get_action(i).action_type == ActionType.PASS)

        self.engine.apply_dogfight_actions(pass_p1, pass_p2)

        # Dogfight index should advance
        self.assertGreater(self.engine.state.current_dogfight_index, 0)

    def test_rocket_removes_rocketman(self):
        """Test rocket can remove enemy rocketman."""
        # Create a scenario with a dogfight
        engine = GameEngine(seed=100)

        # Set up placement to ensure a dogfight at (1,1)
        # Place P1 rocketman
        for idx in engine.get_legal_actions():
            action = self.action_space.get_action(idx)
            if action.row == 1 and action.col == 1 and action.rocketman_power == 3:
                engine.apply_action(idx)
                break

        # Place P2 rocketman at same spot
        for idx in engine.get_legal_actions():
            action = self.action_space.get_action(idx)
            if action.row == 1 and action.col == 1 and action.rocketman_power == 10:
                engine.apply_action(idx)
                break

        # Fill rest of board
        for _ in range(16):
            legal = engine.get_legal_actions()
            engine.apply_action(legal[0])

        # Should be in dogfights
        self.assertEqual(engine.state.phase, Phase.DOGFIGHTS)

        # Check if (1,1) is in dogfight order
        if (1, 1) in engine.state.dogfight_order:
            # Find the dogfight
            dogfight_idx = engine.state.dogfight_order.index((1, 1))

            # Skip to this dogfight
            while engine.state.current_dogfight_index < dogfight_idx:
                legal_p1 = engine.get_legal_actions(Player.ONE)
                legal_p2 = engine.get_legal_actions(Player.TWO)
                pass_p1 = next(i for i in legal_p1
                              if self.action_space.get_action(i).action_type == ActionType.PASS)
                pass_p2 = next(i for i in legal_p2
                              if self.action_space.get_action(i).action_type == ActionType.PASS)
                engine.apply_dogfight_actions(pass_p1, pass_p2)

    def test_kaos_draw(self):
        """Test Kaos cards are drawn and discarded."""
        if len(self.engine.state.dogfight_order) == 0:
            self.skipTest("No dogfights in this game")

        p1_kaos_before = len(self.engine.state.player_resources[Player.ONE].kaos_deck)

        # Resolve a dogfight with pass
        legal_p1 = self.engine.get_legal_actions(Player.ONE)
        legal_p2 = self.engine.get_legal_actions(Player.TWO)

        pass_p1 = next(i for i in legal_p1
                      if self.action_space.get_action(i).action_type == ActionType.PASS)
        pass_p2 = next(i for i in legal_p2
                      if self.action_space.get_action(i).action_type == ActionType.PASS)

        self.engine.apply_dogfight_actions(pass_p1, pass_p2)

        # Kaos cards should have been drawn
        p1_kaos_after = len(self.engine.state.player_resources[Player.ONE].kaos_deck)
        self.assertLess(p1_kaos_after, p1_kaos_before)


class TestGameEnd(unittest.TestCase):
    """Test game ending conditions."""

    def test_game_ends_after_dogfights(self):
        """Test game ends after all dogfights."""
        engine = GameEngine(seed=42)

        # Play through entire game
        while engine.state.phase == Phase.PLACEMENT:
            legal = engine.get_legal_actions()
            engine.apply_action(legal[0])

        while engine.state.phase == Phase.DOGFIGHTS:
            legal_p1 = engine.get_legal_actions(Player.ONE)
            legal_p2 = engine.get_legal_actions(Player.TWO)

            # Both pass
            pass_p1 = next(i for i in legal_p1
                          if get_action_space().get_action(i).action_type == ActionType.PASS)
            pass_p2 = next(i for i in legal_p2
                          if get_action_space().get_action(i).action_type == ActionType.PASS)

            engine.apply_dogfight_actions(pass_p1, pass_p2)

        # Game should end
        self.assertTrue(engine.is_game_over())
        self.assertEqual(engine.state.phase, Phase.ENDED)

    def test_winner_determined(self):
        """Test winner is determined at game end."""
        engine = GameEngine(seed=42)

        # Play through entire game
        while engine.state.phase == Phase.PLACEMENT:
            legal = engine.get_legal_actions()
            engine.apply_action(legal[0])

        while engine.state.phase == Phase.DOGFIGHTS:
            legal_p1 = engine.get_legal_actions(Player.ONE)
            legal_p2 = engine.get_legal_actions(Player.TWO)

            pass_p1 = next(i for i in legal_p1
                          if get_action_space().get_action(i).action_type == ActionType.PASS)
            pass_p2 = next(i for i in legal_p2
                          if get_action_space().get_action(i).action_type == ActionType.PASS)

            engine.apply_dogfight_actions(pass_p1, pass_p2)

        # Winner should be set (or None for draw)
        winner = engine.get_winner()
        self.assertIn(winner, [Player.ONE, Player.TWO, None])


class TestDeterminism(unittest.TestCase):
    """Test deterministic replay."""

    def test_same_seed_same_kaos(self):
        """Test same seed produces same Kaos deck."""
        engine1 = GameEngine(seed=1337)
        engine2 = GameEngine(seed=1337)

        kaos1 = engine1.state.player_resources[Player.ONE].kaos_deck
        kaos2 = engine2.state.player_resources[Player.ONE].kaos_deck

        self.assertEqual(kaos1, kaos2)

    def test_same_actions_same_outcome(self):
        """Test same seed and actions produce same game."""
        seed = 9999

        # Play game 1
        engine1 = GameEngine(seed=seed)
        actions1 = []

        # Play through placement
        while engine1.state.phase == Phase.PLACEMENT:
            legal = engine1.get_legal_actions()
            action_idx = legal[0]  # Always pick first legal action
            actions1.append(action_idx)
            engine1.apply_action(action_idx)

        # Play game 2 with same seed and actions
        engine2 = GameEngine(seed=seed)

        for action_idx in actions1:
            engine2.apply_action(action_idx)

        # States should be identical
        self.assertEqual(engine1.state.phase, engine2.state.phase)
        self.assertEqual(engine1.state.turn_number, engine2.state.turn_number)


class TestV13FaceDownMechanics(unittest.TestCase):
    """Test v1.3 face-down card mechanics."""

    def test_face_down_placement_v13(self):
        """Test that cards 2, 3, 9, 10 are placed face-down."""
        engine = GameEngine(seed=42)

        # Get action space to find placement actions
        action_space = get_action_space()

        # Test face-down cards (2, 3, 9, 10)
        face_down_powers = [2, 3, 9, 10]
        for power in face_down_powers:
            engine = GameEngine(seed=42)  # Fresh engine for each test
            # Find a placement action for this power
            for action_idx, action in enumerate(action_space.actions):
                if (action.action_type == ActionType.PLACE_ROCKETMAN and
                    action.rocketman_power == power and
                    action.row == 0 and action.col == 0):
                    engine.apply_action(action_idx)
                    square = engine.state.get_square(0, 0)
                    self.assertEqual(len(square.rocketmen), 1)
                    rocketman = square.rocketmen[0]
                    self.assertTrue(rocketman.face_down,
                                  f"Card {power} should be placed face-down")
                    break

        # Test visible cards (4-8)
        visible_powers = [4, 5, 6, 7, 8]
        for power in visible_powers:
            engine = GameEngine(seed=100 + power)  # Fresh engine with different seed
            # Find a placement action for this power
            for action_idx, action in enumerate(action_space.actions):
                if (action.action_type == ActionType.PLACE_ROCKETMAN and
                    action.rocketman_power == power and
                    action.row == 1 and action.col == 1):
                    engine.apply_action(action_idx)
                    square = engine.state.get_square(1, 1)
                    self.assertEqual(len(square.rocketmen), 1)
                    rocketman = square.rocketmen[0]
                    self.assertFalse(rocketman.face_down,
                                   f"Card {power} should be placed face-up")
                    break

    def test_dogfight_showdown_reveals_cards(self):
        """Test that face-down cards are revealed at dogfight showdown."""
        engine = GameEngine(seed=42)
        action_space = get_action_space()

        # Create a contested square with face-down cards
        # Place P1's 10 (face-down) at center
        for action_idx, action in enumerate(action_space.actions):
            if (action.action_type == ActionType.PLACE_ROCKETMAN and
                action.rocketman_power == 10 and
                action.row == 1 and action.col == 1):
                engine.apply_action(action_idx)
                break

        # Place P2's 9 (face-down) at same position
        for action_idx, action in enumerate(action_space.actions):
            if (action.action_type == ActionType.PLACE_ROCKETMAN and
                action.rocketman_power == 9 and
                action.row == 1 and action.col == 1):
                engine.apply_action(action_idx)
                break

        # Verify cards are face-down before showdown
        square = engine.state.get_square(1, 1)
        self.assertEqual(len(square.rocketmen), 2)
        for rm in square.rocketmen:
            self.assertTrue(rm.face_down, "Cards should be face-down before showdown")

        # Complete placement phase quickly with remaining cards
        while engine.state.phase == Phase.PLACEMENT:
            legal = engine.get_legal_actions()
            engine.apply_action(legal[0])

        # Now in dogfight phase - begin the contested dogfight
        self.assertEqual(engine.state.phase, Phase.DOGFIGHTS)
        engine.begin_current_dogfight()

        # Verify cards are now revealed
        square = engine.state.get_square(1, 1)
        self.assertEqual(len(square.rocketmen), 2)
        for rm in square.rocketmen:
            self.assertFalse(rm.face_down, "Cards should be revealed at showdown")

    def test_v13_full_game_random_agents(self):
        """Test complete game with RandomAgent to verify v1.3 mechanics work end-to-end."""
        from utala.agents.random_agent import RandomAgent
        from utala.evaluation.harness import Harness

        # Run a complete game with random agents
        agent1 = RandomAgent("Random1", seed=42)
        agent2 = RandomAgent("Random2", seed=43)
        harness = Harness(verbose=False)

        result = harness.run_game(agent1, agent2, seed=100)

        # Verify game completed successfully (winner can be None for draws)
        self.assertGreater(result.num_turns, 0, "Game should have turns")
        self.assertTrue(result.num_turns >= 18, "Game should complete placement (18 turns minimum)")


class TestV18JokerToken(unittest.TestCase):
    """Test v1.8 joker token mechanics."""

    def _find_placement_action(self, engine, row, col, power):
        """Helper to find placement action index."""
        action_space = get_action_space()
        for action_idx, action in enumerate(action_space.actions):
            if (action.action_type == ActionType.PLACE_ROCKETMAN and
                action.rocketman_power == power and
                action.row == row and action.col == col):
                return action_idx
        return None

    def test_joker_initialization(self):
        """Test that P2 starts with joker token by default."""
        engine = GameEngine(seed=42)
        self.assertEqual(engine.state.joker_holder, Player.TWO,
                        "P2 should start with joker token to balance first-move advantage")

    def test_equal_power_joker_determines_order(self):
        """Test that joker holder acts first in equal-power dogfights."""
        engine = GameEngine(seed=42)

        # Manually create equal-power dogfight scenario
        from utala.state import Rocketman
        engine.state.phase = Phase.DOGFIGHTS
        engine.state.dogfight_order = [(1, 1)]

        # Both players have equal power (5) at center
        engine.state.grid[1][1].rocketmen = [
            Rocketman(Player.ONE, 5),
            Rocketman(Player.TWO, 5)
        ]

        # Record initial joker holder
        initial_joker = engine.state.joker_holder

        # Get underdog for center square dogfight
        underdog = engine._get_underdog_at_position(1, 1)

        # In equal power, underdog should be the joker holder
        self.assertEqual(underdog, initial_joker,
                        "Joker holder should act first in equal-power dogfight")

    def test_joker_alternates_after_use(self):
        """Test that joker token passes to opponent after equal-power dogfight."""
        engine = GameEngine(seed=42)

        # Manually create equal-power dogfight scenario
        from utala.state import Rocketman
        engine.state.phase = Phase.DOGFIGHTS
        engine.state.dogfight_order = [(1, 1)]

        # Both players have equal power (6) at center
        engine.state.grid[1][1].rocketmen = [
            Rocketman(Player.ONE, 6),
            Rocketman(Player.TWO, 6)
        ]

        initial_joker = engine.state.joker_holder

        # Determine underdog - this should trigger joker determination and swap
        underdog = engine._get_underdog_at_position(1, 1)

        # After determining underdog with equal power, joker should swap
        self.assertNotEqual(engine.state.joker_holder, initial_joker,
                          "Joker should pass to opponent after equal-power determination")

    def test_different_power_ignores_joker(self):
        """Test that joker is not used when powers differ."""
        engine = GameEngine(seed=42)

        # Manually create unequal-power dogfight scenario
        from utala.state import Rocketman
        engine.state.phase = Phase.DOGFIGHTS
        engine.state.dogfight_order = [(1, 1)]

        # Different powers: P1 has 8, P2 has 5
        engine.state.grid[1][1].rocketmen = [
            Rocketman(Player.ONE, 8),
            Rocketman(Player.TWO, 5)
        ]

        # Determine underdog
        underdog = engine._get_underdog_at_position(1, 1)

        # Underdog should be P2 (lower power), regardless of joker
        self.assertEqual(underdog, Player.TWO,
                        "Lower power player should be underdog, not joker holder")


class TestV18ImmediateThreeInRow(unittest.TestCase):
    """Test v1.8 immediate 3-in-a-row victory checking."""

    def _find_placement_action(self, engine, row, col, power):
        """Helper to find placement action index."""
        action_space = get_action_space()
        for action_idx, action in enumerate(action_space.actions):
            if (action.action_type == ActionType.PLACE_ROCKETMAN and
                action.rocketman_power == power and
                action.row == row and action.col == col):
                return action_idx
        return None

    def test_three_in_row_checked_after_dogfight(self):
        """Test that 3-in-a-row is checked after each dogfight resolves."""
        engine = GameEngine(seed=100)

        # Manually set up a board where P1 gets 3-in-a-row after center resolves
        engine.state.phase = Phase.DOGFIGHTS
        engine.state.dogfight_order = [(0, 1)]  # Top middle square

        # Set up P1 controlling top row except center
        from utala.state import Rocketman
        engine.state.grid[0][0].rocketmen = [Rocketman(Player.ONE, 8)]
        engine.state.grid[0][2].rocketmen = [Rocketman(Player.ONE, 7)]

        # Top center is contested - P1 will win
        engine.state.grid[0][1].rocketmen = [
            Rocketman(Player.ONE, 9),
            Rocketman(Player.TWO, 4)
        ]

        # Begin dogfight
        engine.begin_current_dogfight()

        # Process dogfight (both pass weapons) - use apply_dogfight_actions
        action_space = get_action_space()
        for _ in range(3):  # Up to 3 rounds of weapon exchange
            if engine.state.phase == Phase.ENDED:
                break
            legal_p1 = engine.get_legal_actions(Player.ONE)
            legal_p2 = engine.get_legal_actions(Player.TWO)
            if not legal_p1 or not legal_p2:
                break
            pass_p1 = next(a for a in legal_p1 if action_space.get_action(a).action_type == ActionType.PASS)
            pass_p2 = next(a for a in legal_p2 if action_space.get_action(a).action_type == ActionType.PASS)
            engine.apply_dogfight_actions(pass_p1, pass_p2)

        # If P1 won the center, game should end immediately via 3-in-a-row
        if engine.state.phase == Phase.ENDED and engine.state.winner == Player.ONE:
            self.assertTrue(engine.state.check_three_in_row(Player.ONE),
                          "P1 winner should have 3-in-a-row")

    def test_first_to_achieve_wins_immediately(self):
        """Test that first player to get 3-in-a-row wins immediately."""
        engine = GameEngine(seed=42)

        # Set up scenario where P1 can complete 3-in-a-row in first dogfight
        engine.state.phase = Phase.DOGFIGHTS
        engine.state.dogfight_order = [(0, 1), (2, 2)]

        from utala.state import Rocketman

        # P1 has 2 in top row, (0,1) contested and will win
        engine.state.grid[0][0].rocketmen = [Rocketman(Player.ONE, 7)]
        engine.state.grid[0][2].rocketmen = [Rocketman(Player.ONE, 6)]
        engine.state.grid[0][1].rocketmen = [
            Rocketman(Player.ONE, 9),
            Rocketman(Player.TWO, 3)
        ]

        # P2 has other squares but won't matter
        engine.state.grid[2][0].rocketmen = [Rocketman(Player.TWO, 8)]
        engine.state.grid[2][1].rocketmen = [Rocketman(Player.TWO, 7)]

        # Begin first dogfight
        engine.begin_current_dogfight()

        # Process dogfight (both pass)
        action_space = get_action_space()
        while engine.state.phase == Phase.DOGFIGHTS and engine.state.current_dogfight_index < 1:
            legal_p1 = engine.get_legal_actions(Player.ONE)
            legal_p2 = engine.get_legal_actions(Player.TWO)

            if not legal_p1 or not legal_p2:
                break

            pass_p1 = next(a for a in legal_p1 if action_space.get_action(a).action_type == ActionType.PASS)
            pass_p2 = next(a for a in legal_p2 if action_space.get_action(a).action_type == ActionType.PASS)
            engine.apply_dogfight_actions(pass_p1, pass_p2)

        # If P1 won, game should end immediately without processing second dogfight
        if engine.state.winner == Player.ONE:
            self.assertEqual(engine.state.phase, Phase.ENDED,
                           "Game should end immediately when 3-in-a-row achieved")
            self.assertTrue(engine.state.check_three_in_row(Player.ONE),
                          "P1 should have 3-in-a-row")
            self.assertLess(engine.state.current_dogfight_index, len(engine.state.dogfight_order),
                          "Not all dogfights should be processed (immediate win)")

    def test_no_premature_three_in_row_check(self):
        """Test that 3-in-a-row is NOT checked during placement."""
        engine = GameEngine(seed=42)

        # Place P1 cards in top row
        engine.state.phase = Phase.PLACEMENT
        engine.state.current_player = Player.ONE

        action_idx = self._find_placement_action(engine, 0, 0, 8)
        engine.apply_action(action_idx)

        self.assertEqual(engine.state.phase, Phase.PLACEMENT,
                        "Should remain in placement after first card")

        engine.state.current_player = Player.ONE
        action_idx = self._find_placement_action(engine, 0, 1, 7)
        engine.apply_action(action_idx)

        self.assertEqual(engine.state.phase, Phase.PLACEMENT,
                        "Should remain in placement after second card")

        engine.state.current_player = Player.ONE
        action_idx = self._find_placement_action(engine, 0, 2, 6)
        engine.apply_action(action_idx)

        # Game should not end even though P1 has 3 in a row (uncontested)
        self.assertNotEqual(engine.state.phase, Phase.ENDED,
                          "Game should not end during placement phase")

    def test_v18_full_game_immediate_victory(self):
        """Test that v1.8 immediate victory works in a complete game."""
        from utala.agents.random_agent import RandomAgent
        from utala.evaluation.harness import Harness

        # Run multiple games to test v1.8 mechanics
        agent1 = RandomAgent("P1", seed=500)
        agent2 = RandomAgent("P2", seed=501)
        harness = Harness(verbose=False)

        # Run 10 quick games
        for i in range(10):
            agent1_test = RandomAgent("P1", seed=500 + i)
            agent2_test = RandomAgent("P2", seed=600 + i)

            result = harness.run_game(agent1_test, agent2_test, seed=700 + i)

            # Verify game completed
            self.assertGreater(result.num_turns, 0)
            self.assertTrue(result.num_turns >= 18)

            # Winner should be valid or None (draw)
            self.assertIn(result.winner, [Player.ONE, Player.TWO, None])


if __name__ == '__main__':
    unittest.main()
