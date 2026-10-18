"""
Unit tests for game state representation.
"""

import unittest
import sys
sys.path.insert(0, 'src')

from utala.state import (
    Player, Phase, GameState, Rocketman, GridSquare,
    PlayerResources
)


class TestPlayer(unittest.TestCase):
    """Test Player enum."""

    def test_player_values(self):
        """Test player enum values."""
        self.assertEqual(Player.ONE.value, 0)
        self.assertEqual(Player.TWO.value, 1)

    def test_opponent(self):
        """Test opponent() method."""
        self.assertEqual(Player.ONE.opponent(), Player.TWO)
        self.assertEqual(Player.TWO.opponent(), Player.ONE)


class TestRocketman(unittest.TestCase):
    """Test Rocketman representation."""

    def test_rocketman_creation(self):
        """Test creating rocketmen."""
        rm = Rocketman(player=Player.ONE, power=5)
        self.assertEqual(rm.player, Player.ONE)
        self.assertEqual(rm.power, 5)
        self.assertFalse(rm.face_down)

    def test_face_down_rocketman(self):
        """Test face-down rocketman."""
        rm = Rocketman(player=Player.TWO, power=10, face_down=True)
        self.assertTrue(rm.face_down)
        self.assertIn("??", str(rm))

    def test_rocketman_repr(self):
        """Test string representation."""
        rm = Rocketman(player=Player.ONE, power=7)
        self.assertIn("P1", str(rm))
        self.assertIn("7", str(rm))


class TestGridSquare(unittest.TestCase):
    """Test GridSquare."""

    def test_empty_square(self):
        """Test empty square."""
        square = GridSquare()
        self.assertTrue(square.is_empty)
        self.assertFalse(square.is_contested)
        self.assertFalse(square.is_controlled)
        self.assertIsNone(square.controller)

    def test_controlled_square(self):
        """Test square with one rocketman."""
        square = GridSquare()
        rm = Rocketman(player=Player.ONE, power=5)
        square.rocketmen.append(rm)

        self.assertFalse(square.is_empty)
        self.assertTrue(square.is_controlled)
        self.assertFalse(square.is_contested)
        self.assertEqual(square.controller, Player.ONE)

    def test_contested_square(self):
        """Test square with both players."""
        square = GridSquare()
        rm1 = Rocketman(player=Player.ONE, power=5)
        rm2 = Rocketman(player=Player.TWO, power=7)
        square.rocketmen.extend([rm1, rm2])

        self.assertFalse(square.is_empty)
        self.assertFalse(square.is_controlled)
        self.assertTrue(square.is_contested)
        self.assertIsNone(square.controller)


class TestPlayerResources(unittest.TestCase):
    """Test PlayerResources."""

    def test_initial_resources(self):
        """Test initial resource state."""
        resources = PlayerResources()

        # Should have all 9 rocketmen
        self.assertEqual(len(resources.rocketmen), 9)
        self.assertEqual(resources.rocketmen, list(range(2, 11)))

        # v1.4: Should have 4 dual-purpose weapons
        self.assertEqual(len(resources.weapons), 4)
        self.assertEqual(sorted(resources.weapons), ["A", "J", "K", "Q"])

        # Kaos deck should be empty initially (engine fills it)
        self.assertEqual(len(resources.kaos_deck), 0)
        self.assertEqual(len(resources.kaos_discard), 0)

    def test_has_rocketman(self):
        """Test has_rocketman check."""
        resources = PlayerResources()
        self.assertTrue(resources.has_rocketman(5))
        self.assertTrue(resources.has_rocketman(10))

        # Remove a rocketman
        resources.rocketmen.remove(5)
        self.assertFalse(resources.has_rocketman(5))

    def test_has_weapons(self):
        """Test weapon checks (v1.4: unified dual-purpose weapons)."""
        resources = PlayerResources()
        self.assertTrue(resources.has_weapon())
        self.assertEqual(len(resources.weapons), 4)  # A, K, Q, J

        # Use all weapons
        resources.weapons.clear()
        self.assertFalse(resources.has_weapon())

    def test_remaining_kaos(self):
        """Test Kaos deck counting."""
        resources = PlayerResources()
        resources.kaos_deck = [1, 2, 3, 4, 5]
        self.assertEqual(resources.remaining_kaos_cards(), 5)


class TestGameState(unittest.TestCase):
    """Test GameState."""

    def test_initial_state(self):
        """Test initial game state."""
        state = GameState()

        # Should be in placement phase
        self.assertEqual(state.phase, Phase.PLACEMENT)
        self.assertEqual(state.current_player, Player.ONE)
        self.assertEqual(state.turn_number, 0)
        self.assertFalse(state.game_over)
        self.assertIsNone(state.winner)

    def test_get_square(self):
        """Test getting grid squares."""
        state = GameState()
        square = state.get_square(1, 1)  # Center
        self.assertIsInstance(square, GridSquare)
        self.assertTrue(square.is_empty)

    def test_get_resources(self):
        """Test getting player resources."""
        state = GameState()
        p1_resources = state.get_resources(Player.ONE)
        self.assertIsInstance(p1_resources, PlayerResources)
        self.assertEqual(len(p1_resources.rocketmen), 9)

    def test_count_controlled_squares(self):
        """Test counting controlled squares."""
        state = GameState()

        # Initially no squares controlled
        self.assertEqual(state.count_controlled_squares(Player.ONE), 0)
        self.assertEqual(state.count_controlled_squares(Player.TWO), 0)

        # Add a rocketman for P1
        rm = Rocketman(player=Player.ONE, power=5)
        state.grid[0][0].rocketmen.append(rm)
        self.assertEqual(state.count_controlled_squares(Player.ONE), 1)

        # Add P2 rocketman to same square (contested)
        rm2 = Rocketman(player=Player.TWO, power=6)
        state.grid[0][0].rocketmen.append(rm2)
        self.assertEqual(state.count_controlled_squares(Player.ONE), 0)

    def test_check_three_in_row(self):
        """Test three-in-a-row detection."""
        state = GameState()

        # No three-in-a-row initially
        self.assertFalse(state.check_three_in_row(Player.ONE))

        # Create top row for P1
        for col in range(3):
            rm = Rocketman(player=Player.ONE, power=5)
            state.grid[0][col].rocketmen.append(rm)

        self.assertTrue(state.check_three_in_row(Player.ONE))
        self.assertFalse(state.check_three_in_row(Player.TWO))

    def test_check_diagonal(self):
        """Test diagonal three-in-a-row."""
        state = GameState()

        # Create diagonal for P2
        for i in range(3):
            rm = Rocketman(player=Player.TWO, power=5)
            state.grid[i][i].rocketmen.append(rm)

        self.assertTrue(state.check_three_in_row(Player.TWO))
        self.assertFalse(state.check_three_in_row(Player.ONE))

    def test_check_column(self):
        """Test column three-in-a-row."""
        state = GameState()

        # Create middle column for P1
        for row in range(3):
            rm = Rocketman(player=Player.ONE, power=5)
            state.grid[row][1].rocketmen.append(rm)

        self.assertTrue(state.check_three_in_row(Player.ONE))


if __name__ == '__main__':
    unittest.main()
