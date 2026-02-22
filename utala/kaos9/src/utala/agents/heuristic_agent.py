"""
Heuristic baseline agent for utala: kaos 9 v1.8.

Simple strategic heuristics for placement and turn-based dogfights.
v1.8 update: Added 3-in-a-row awareness and joker consideration.
"""

import random

from ..actions import ActionType, get_action_space
from ..state import GameState, Phase, Player
from .base import Agent


class HeuristicAgent(Agent):
    """
    Heuristic baseline agent using simple strategic rules.

    v1.8 rules:
    - Turn-based dogfights (underdog acts first, joker breaks ties)
    - Dual-purpose weapons (any can be Rocket or Flare, context determines role)
    - Kaos deck 1-9, rocket hit ≥7
    - 3-in-a-row checked after each dogfight (first to achieve wins)

    Placement strategy (v1.8):
    - Prioritize center > edges > corners
    - Place stronger rocketmen in better positions
    - Bonus for contesting opponent squares
    - 3-in-a-row awareness: complete lines, block opponent, setup future lines
    - Handles face-down opponent cards (doesn't peek at hidden info)

    Dogfight strategy (v1.8):
    - Evaluate dogfight importance (winning/losing completes 3-in-a-row)
    - Fight harder for critical squares
    - Consider joker holder in equal-power situations
    - Defending: Use flare if equal/stronger or close fight with spare flares
    - Attacking as underdog: Use rocket when very behind or square is important
    - Attacking as favorite: Pass and win via Kaos resolution
    """

    def __init__(self, name: str = "Heuristic", seed: int | None = None):
        """
        Initialize heuristic agent.

        Args:
            name: Agent name
            seed: Optional seed for tie-breaking randomness
        """
        super().__init__(name)
        self.rng = random.Random(seed)
        self.action_space = get_action_space()

        # Position values: center > edges > corners
        self.position_values = {
            (1, 1): 10,  # center
            (0, 1): 7, (1, 0): 7, (1, 2): 7, (2, 1): 7,  # edges
            (0, 0): 4, (0, 2): 4, (2, 0): 4, (2, 2): 4,  # corners
        }

    def select_action(
        self,
        state: GameState,
        legal_actions: list[int],
        player: Player
    ) -> int:
        """Select action using heuristics."""
        if state.phase == Phase.PLACEMENT:
            return self._select_placement(state, legal_actions, player)
        elif state.phase == Phase.DOGFIGHTS:
            return self._select_dogfight(state, legal_actions, player)
        else:
            return self.rng.choice(legal_actions)

    def _select_placement(
        self,
        state: GameState,
        legal_actions: list[int],
        player: Player
    ) -> int:
        """
        Select placement action.

        Score = position_value + strength_bonus + tactical_bonus + three_in_row_bonus

        v1.3: Agent naturally respects face-down cards. Opponent's cards 2,3,9,10
        are face-down during placement, but this heuristic only checks square
        control status (not power values), so it works correctly without peeking.

        v1.8: Added 3-in-a-row awareness for placement strategy.
        """
        best_score = -float('inf')
        best_actions = []

        for action_idx in legal_actions:
            action = self.action_space.get_action(action_idx)

            if action.action_type != ActionType.PLACE_ROCKETMAN:
                continue

            assert action.row is not None
            assert action.col is not None
            assert action.rocketman_power is not None

            # Base position value
            pos_value = self.position_values.get((action.row, action.col), 0)

            # Strength bonus: stronger rocketmen in better positions
            strength = (action.rocketman_power - 2) / 8.0  # normalize 2-10 to 0-1
            strength_bonus = strength * pos_value * 0.5

            # Tactical bonus: contesting opponent squares
            square = state.get_square(action.row, action.col)
            tactical_bonus = 3.0 if square.is_controlled and square.controller == player.opponent() else 0.0

            # v1.8: Three-in-a-row awareness
            three_bonus = self._evaluate_placement_for_three_in_row(
                state, player, action.row, action.col
            )

            total_score = pos_value + strength_bonus + tactical_bonus + three_bonus

            if total_score > best_score:
                best_score = total_score
                best_actions = [action_idx]
            elif total_score == best_score:
                best_actions.append(action_idx)

        return self.rng.choice(best_actions) if best_actions else self.rng.choice(legal_actions)

    def _select_dogfight(
        self,
        state: GameState,
        legal_actions: list[int],
        player: Player
    ) -> int:
        """
        Select dogfight action for v1.4+ turn-based rules.

        Turn-based flow:
        1. Underdog (weaker) acts first: can play Rocket or Pass
        2. If Rocket played: opponent can play Flare or Pass
        3. If both Pass or Rocket+Flare: go to Kaos resolution

        Strategy (v1.8 updated):
        - Evaluate dogfight importance based on 3-in-a-row potential
        - Consider joker holder in equal-power situations
        - Defending: defend if equal/stronger or close with spare flares
        - Attacking as underdog: rocket when very behind (≤-3), else pass
        - Attacking as favorite: pass (will win via Kaos)
        """
        resources = state.get_resources(player)

        # Get current dogfight position and rocketmen
        if state.current_dogfight_index >= len(state.dogfight_order):
            return self._get_action_by_type(legal_actions, ActionType.PASS)

        row, col = state.dogfight_order[state.current_dogfight_index]
        square = state.get_square(row, col)

        my_rm = next((rm for rm in square.rocketmen if rm.player == player), None)
        opp_rm = next((rm for rm in square.rocketmen if rm.player != player), None)

        if my_rm is None or opp_rm is None:
            return self._get_action_by_type(legal_actions, ActionType.PASS)

        # v1.3: Verify cards are revealed at showdown (before dogfight actions)
        assert not my_rm.face_down, "My rocketman should be revealed at dogfight"
        assert not opp_rm.face_down, "Opponent rocketman should be revealed at dogfight"

        power_diff = my_rm.power - opp_rm.power

        # v1.8: Evaluate strategic importance of this dogfight
        importance = self._evaluate_dogfight_importance(state, player, row, col)

        # v1.8: Check joker holder for equal-power situations
        joker_holder = state.joker_holder
        we_have_joker = (joker_holder == player)

        # v1.4: Check if we can play weapon (all weapons are dual-purpose)
        can_weapon = self._can_play(legal_actions, ActionType.PLAY_WEAPON)

        # v1.4: Context-aware strategic decision
        # Use dogfight_context to determine if weapon will be offensive or defensive
        if can_weapon and resources.has_weapon():
            context = state.dogfight_context
            if context is None:
                # Shouldn't happen during dogfight, fallback to conservative
                return self._get_action_by_type(legal_actions, ActionType.PASS)

            # Determine role from context
            is_defensive = context.rocket_in_play is not None
            weapon_count = len(resources.weapons)

            # v1.8: Critical dogfights (winning/losing completes 3-in-a-row) - always use weapon
            if importance >= 80.0 and can_weapon and resources.has_weapon():
                return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

            if is_defensive:
                # DEFENSIVE CONTEXT: Responding to opponent's Rocket
                # Decision: Defend (spend weapon as Flare) vs Pass (take hit, save weapon)

                # v1.8: Fight harder for important squares
                if importance >= 20.0 and weapon_count >= 2:
                    return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # Always defend when ahead - protect advantage
                if power_diff > 0:
                    return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # Defend when tied or close - contest the square
                if power_diff >= -1:
                    return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # Defend when moderately weak but have spare weapons
                if power_diff == -2 and weapon_count >= 3:
                    return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # When significantly behind, save weapons for offense
                # Taking the hit might be better than spending last weapon defensively
                return self._get_action_by_type(legal_actions, ActionType.PASS)

            else:
                # OFFENSIVE CONTEXT: Can play weapon as Rocket to attack
                # Decision: Attack (spend weapon as Rocket) vs Pass (save weapon)

                # v1.8: Fight harder for important squares
                if importance >= 20.0:
                    return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # Attack when behind - need to catch up
                if power_diff < 0:
                    return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # v1.8: Attack when tied - but consider joker
                if power_diff == 0:
                    # If we have joker, we act first - good position, attack
                    if we_have_joker:
                        return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)
                    # If opponent has joker, they act first - still attack (contest square)
                    else:
                        return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # When ahead, only attack if have spare weapons
                if weapon_count >= 3:
                    return self._get_action_by_type(legal_actions, ActionType.PLAY_WEAPON)

                # When ahead with few weapons, save for defense
                return self._get_action_by_type(legal_actions, ActionType.PASS)

        # No weapons left or can't play - just pass
        else:
            return self._get_action_by_type(legal_actions, ActionType.PASS)

    def _evaluate_placement_for_three_in_row(
        self,
        state: GameState,
        player: Player,
        row: int,
        col: int
    ) -> float:
        """
        Evaluate placement for 3-in-a-row potential (v1.8).

        Returns bonus score for this placement based on:
        - Completing a 3-in-a-row (very high priority)
        - Blocking opponent's 3-in-a-row (high priority)
        - Setting up future 3-in-a-row (moderate priority)
        """
        bonus = 0.0

        # Get all lines that include this position
        lines = self._get_lines_containing(row, col)

        for line_positions in lines:
            # Count our control vs opponent control in this line
            our_count = 0
            opp_count = 0
            empty_count = 0

            for r, c in line_positions:
                sq = state.get_square(r, c)
                if sq.is_controlled:
                    if sq.controller == player:
                        our_count += 1
                    else:
                        opp_count += 1
                elif sq.is_empty:
                    empty_count += 1
                # Contested squares don't help either player

            # COMPLETING: We have 2, opponent has 0, and this completes the line
            if our_count == 2 and opp_count == 0:
                bonus += 50.0  # Huge bonus - winning move!

            # BLOCKING: Opponent has 2, we have 0, block their win
            elif opp_count == 2 and our_count == 0:
                bonus += 30.0  # High bonus - prevent loss!

            # SETTING UP: We have 1, opponent has 0, good setup
            elif our_count == 1 and opp_count == 0:
                bonus += 5.0  # Moderate bonus - building

            # DENYING: Opponent has 1, we have 0, contest the line
            elif opp_count == 1 and our_count == 0:
                bonus += 3.0  # Small bonus - disruption

        return bonus

    def _get_lines_containing(self, row: int, col: int) -> list[list[tuple[int, int]]]:
        """Get all 3-in-a-row lines that contain the given position."""
        lines = []

        # Row containing this position
        lines.append([(row, 0), (row, 1), (row, 2)])

        # Column containing this position
        lines.append([(0, col), (1, col), (2, col)])

        # Main diagonal (if on it)
        if row == col:
            lines.append([(0, 0), (1, 1), (2, 2)])

        # Anti-diagonal (if on it)
        if row + col == 2:
            lines.append([(0, 2), (1, 1), (2, 0)])

        return lines

    def _evaluate_dogfight_importance(
        self,
        state: GameState,
        player: Player,
        row: int,
        col: int
    ) -> float:
        """
        Evaluate how important winning this dogfight is for 3-in-a-row (v1.8).

        Returns bonus for fighting harder based on:
        - Winning completes our 3-in-a-row (critical!)
        - Losing gives opponent 3-in-a-row (critical!)
        - Strategic value for line control
        """
        importance = 0.0

        lines = self._get_lines_containing(row, col)

        for line_positions in lines:
            our_count = 0
            opp_count = 0

            for r, c in line_positions:
                if (r, c) == (row, col):
                    continue  # Skip this square (we're evaluating it)

                sq = state.get_square(r, c)
                if sq.is_controlled:
                    if sq.controller == player:
                        our_count += 1
                    else:
                        opp_count += 1

            # WINNING THIS COMPLETES OUR LINE
            if our_count == 2 and opp_count == 0:
                importance += 100.0  # Critical - win the game!

            # LOSING THIS COMPLETES OPPONENT'S LINE
            elif opp_count == 2 and our_count == 0:
                importance += 80.0  # Critical - prevent loss!

            # Strategic line control
            elif our_count == 1 and opp_count == 0:
                importance += 10.0  # Good - advancing our line
            elif opp_count == 1 and our_count == 0:
                importance += 8.0  # Good - contesting their line

        return importance

    def _can_play(self, legal_actions: list[int], action_type: ActionType) -> bool:
        """Check if action type is legal."""
        return any(
            self.action_space.get_action(idx).action_type == action_type
            for idx in legal_actions
        )

    def _get_action_by_type(self, legal_actions: list[int], action_type: ActionType) -> int:
        """Get action index by type, or random if not found."""
        matching = [
            idx for idx in legal_actions
            if self.action_space.get_action(idx).action_type == action_type
        ]
        return self.rng.choice(matching) if matching else self.rng.choice(legal_actions)
