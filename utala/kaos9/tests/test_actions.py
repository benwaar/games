"""
Unit tests for action space and masking.
"""

import unittest
import sys
sys.path.insert(0, 'src')

from utala.actions import Action, ActionType, ActionSpace, get_action_space
from utala.state import GameState, Player, Phase, Rocketman


class TestActionType(unittest.TestCase):
    """Test ActionType enum."""

    def test_action_types_exist(self):
        """Test all action types are defined (v1.4)."""
        self.assertIsNotNone(ActionType.PLACE_ROCKETMAN)
        self.assertIsNotNone(ActionType.PLAY_WEAPON)  # v1.4: unified dual-purpose weapon
        self.assertIsNotNone(ActionType.PASS)


class TestAction(unittest.TestCase):
    """Test Action representation."""

    def test_placement_action(self):
        """Test creating a placement action."""
        action = Action(
            action_type=ActionType.PLACE_ROCKETMAN,
            rocketman_power=5,
            row=1,
            col=2
        )
        self.assertEqual(action.action_type, ActionType.PLACE_ROCKETMAN)
        self.assertEqual(action.rocketman_power, 5)
        self.assertEqual(action.row, 1)
        self.assertEqual(action.col, 2)

    def test_weapon_action(self):
        """Test creating a weapon action (v1.4: dual-purpose)."""
        action = Action(
            action_type=ActionType.PLAY_WEAPON,
            card_index=0
        )
        self.assertEqual(action.action_type, ActionType.PLAY_WEAPON)
        self.assertEqual(action.card_index, 0)

        # Test another index
        action2 = Action(
            action_type=ActionType.PLAY_WEAPON,
            card_index=3
        )
        self.assertEqual(action2.action_type, ActionType.PLAY_WEAPON)
        self.assertEqual(action2.card_index, 3)

    def test_pass_action(self):
        """Test creating a pass action."""
        action = Action(action_type=ActionType.PASS)
        self.assertEqual(action.action_type, ActionType.PASS)

    def test_action_repr(self):
        """Test action string representation."""
        action = Action(
            action_type=ActionType.PLACE_ROCKETMAN,
            rocketman_power=7,
            row=0,
            col=0
        )
        repr_str = str(action)
        self.assertIn("PLACE", repr_str)
        self.assertIn("7", repr_str)


class TestActionSpace(unittest.TestCase):
    """Test ActionSpace."""

    def setUp(self):
        """Create action space for testing."""
        self.action_space = ActionSpace()

    def test_action_space_size(self):
        """Test that action space has correct size."""
        # 81 placements + 2 rockets + 2 flares + 1 pass = 86
        self.assertEqual(self.action_space.size(), 86)

    def test_all_placement_actions_present(self):
        """Test all placement actions are in space."""
        placement_actions = [
            a for a in self.action_space.actions
            if a.action_type == ActionType.PLACE_ROCKETMAN
        ]
        # 9 rocketmen × 9 positions = 81
        self.assertEqual(len(placement_actions), 81)

        # Check all combinations exist
        for power in range(2, 11):
            for row in range(3):
                for col in range(3):
                    found = any(
                        a.rocketman_power == power and a.row == row and a.col == col
                        for a in placement_actions
                    )
                    self.assertTrue(found, f"Missing placement: {power} @ [{row},{col}]")

    def test_dogfight_actions_present(self):
        """Test dogfight actions are in space (v1.4: unified weapons)."""
        weapon_actions = [
            a for a in self.action_space.actions
            if a.action_type == ActionType.PLAY_WEAPON
        ]
        pass_actions = [
            a for a in self.action_space.actions
            if a.action_type == ActionType.PASS
        ]

        # v1.4: Should have 4 weapon actions (one per weapon card)
        self.assertEqual(len(weapon_actions), 4)
        self.assertEqual(len(pass_actions), 1)

    def test_get_action(self):
        """Test getting action by index."""
        action = self.action_space.get_action(0)
        self.assertIsInstance(action, Action)

    def test_action_to_index(self):
        """Test finding action index."""
        action = Action(
            action_type=ActionType.PLACE_ROCKETMAN,
            rocketman_power=5,
            row=1,
            col=1
        )
        index = self.action_space.action_to_index(action)
        self.assertIsNotNone(index)
        assert index is not None

        # Verify we can retrieve it
        retrieved = self.action_space.get_action(index)
        self.assertEqual(retrieved, action)

    def test_get_action_space_singleton(self):
        """Test global action space singleton."""
        space1 = get_action_space()
        space2 = get_action_space()
        self.assertIs(space1, space2)  # Same instance


class TestLegalActionMasking(unittest.TestCase):
    """Test legal action masking."""

    def setUp(self):
        """Create action space and state."""
        self.action_space = get_action_space()
        self.state = GameState()

    def test_initial_placement_mask(self):
        """Test legal actions at start of game."""
        mask = self.action_space.get_legal_actions_mask(self.state, Player.ONE)

        self.assertEqual(len(mask), 86)

        # All placement actions should be legal
        placement_count = sum(
            1 for i, legal in enumerate(mask)
            if legal and self.action_space.get_action(i).action_type == ActionType.PLACE_ROCKETMAN
        )
        self.assertEqual(placement_count, 81)

        # No dogfight actions should be legal
        dogfight_count = sum(
            1 for i, legal in enumerate(mask)
            if legal and self.action_space.get_action(i).action_type != ActionType.PLACE_ROCKETMAN
        )
        self.assertEqual(dogfight_count, 0)

    def test_placement_after_placing_rocketman(self):
        """Test mask updates after placing a rocketman."""
        # Remove rocketman 5 from P1
        self.state.player_resources[Player.ONE].rocketmen.remove(5)

        legal_actions = self.action_space.get_legal_actions(self.state, Player.ONE)

        # Should not be able to place rocketman 5 anymore
        for action_idx in legal_actions:
            action = self.action_space.get_action(action_idx)
            if action.action_type == ActionType.PLACE_ROCKETMAN:
                self.assertNotEqual(action.rocketman_power, 5)

    def test_cannot_place_on_own_square(self):
        """Test cannot place on square you already occupy."""
        # Place P1 rocketman at (0,0)
        rm = Rocketman(player=Player.ONE, power=5)
        self.state.grid[0][0].rocketmen.append(rm)

        # P1 should not be able to place at (0,0) anymore
        mask = self.action_space.get_legal_actions_mask(self.state, Player.ONE)

        for i, legal in enumerate(mask):
            if legal:
                action = self.action_space.get_action(i)
                if action.action_type == ActionType.PLACE_ROCKETMAN:
                    if action.row == 0 and action.col == 0:
                        self.fail("P1 should not be able to place at occupied square")

    def test_can_place_on_opponent_square(self):
        """Test can place on square occupied by opponent."""
        # Place P2 rocketman at (0,0)
        rm = Rocketman(player=Player.TWO, power=6)
        self.state.grid[0][0].rocketmen.append(rm)

        # P1 should still be able to place at (0,0)
        legal_actions = self.action_space.get_legal_actions(self.state, Player.ONE)

        can_contest = any(
            self.action_space.get_action(idx).action_type == ActionType.PLACE_ROCKETMAN
            and self.action_space.get_action(idx).row == 0
            and self.action_space.get_action(idx).col == 0
            for idx in legal_actions
        )
        self.assertTrue(can_contest, "P1 should be able to contest opponent's square")

    def test_dogfight_phase_mask(self):
        """Test legal actions during dogfight phase."""
        self.state.phase = Phase.DOGFIGHTS

        mask = self.action_space.get_legal_actions_mask(self.state, Player.ONE)

        # No placement actions should be legal
        placement_count = sum(
            1 for i, legal in enumerate(mask)
            if legal and self.action_space.get_action(i).action_type == ActionType.PLACE_ROCKETMAN
        )
        self.assertEqual(placement_count, 0)

        # v1.4: Should have weapons and pass available
        dogfight_actions = [
            self.action_space.get_action(i)
            for i, legal in enumerate(mask) if legal
        ]

        has_weapons = any(a.action_type == ActionType.PLAY_WEAPON for a in dogfight_actions)
        has_pass = any(a.action_type == ActionType.PASS for a in dogfight_actions)

        self.assertTrue(has_weapons)
        self.assertTrue(has_pass)

    def test_dogfight_no_weapons(self):
        """Test mask when player has no weapons (v1.4)."""
        self.state.phase = Phase.DOGFIGHTS
        self.state.player_resources[Player.ONE].weapons.clear()

        legal_actions = self.action_space.get_legal_actions(self.state, Player.ONE)

        # Should not have weapon actions
        for action_idx in legal_actions:
            action = self.action_space.get_action(action_idx)
            self.assertNotEqual(action.action_type, ActionType.PLAY_WEAPON)

        # Should still have pass
        has_pass = any(
            self.action_space.get_action(idx).action_type == ActionType.PASS
            for idx in legal_actions
        )
        self.assertTrue(has_pass)


if __name__ == '__main__':
    unittest.main()
