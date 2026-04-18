"""
Enhanced State-Action Feature Extraction for Linear Value Learning (Phase 2.5).

Major improvements:
- Proper action decoding for placement moves
- Line detection (2-in-a-row, 3-in-a-row, winning lines)
- Kaos deck tracking
- Tactical patterns (blocking, forking, threatening)
- Dogfight power differentials
"""

import numpy as np
from typing import List, Tuple, Optional

from ..state import GameState, Player, Phase


class EnhancedStateActionFeatureExtractor:
    """
    Enhanced feature extractor with richer tactical features.

    Features designed to capture:
    1. Current game situation (state)
    2. Properties of the proposed action
    3. Tactical implications (lines, blocks, threats)
    4. Resource management (Kaos cards)
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of feature names for interpretability."""
        # State features (12)
        self.feature_names.extend([
            "phase_placement",
            "phase_dogfight",
            "turn_normalized",
            "my_rocketmen_count",
            "opp_rocketmen_count",
            "material_advantage",
            "my_squares_controlled",
            "opp_squares_controlled",
            "control_advantage",
            "contested_squares",
            "my_kaos_remaining",
            "opp_kaos_remaining",
        ])

        # Placement action features (18)
        self.feature_names.extend([
            "placement_power_2_4",      # Low power
            "placement_power_5_7",      # Mid power
            "placement_power_8_10",     # High power
            "placement_center",         # Position (1,1)
            "placement_edge",           # Position on edge
            "placement_corner",         # Position on corner
            "placement_contests_square",
            "placement_takes_empty",
            "placement_forms_line_2",   # Creates 2-in-a-row
            "placement_forms_line_3",   # Creates 3-in-a-row (wins!)
            "placement_blocks_opp_line_2",  # Blocks opponent's 2-in-a-row
            "placement_blocks_opp_line_3",  # Blocks opponent's winning move
            "placement_creates_fork",   # Creates multiple threat lines
            "placement_threatens_win",  # One move away from 3-in-a-row
            "placement_power_advantage_here",  # Placing stronger rocketman
            "placement_adjacent_to_mine",  # Near my other pieces
            "placement_adjacent_to_opp",   # Near opponent pieces
            "placement_strategic_value",   # Center > Edge > Corner weight
        ])

        # Dogfight action features (8)
        self.feature_names.extend([
            "dogfight_uses_rocket",
            "dogfight_uses_flare",
            "dogfight_power_advantage",     # My rocketman stronger
            "dogfight_power_disadvantage",  # Opponent stronger
            "dogfight_kaos_high_value",     # Using high Kaos card
            "dogfight_kaos_low_value",      # Using low Kaos card
            "dogfight_center_square",       # Fighting for center
            "dogfight_strategic_square",    # Edge or helps line
        ])

        # Interaction features (10)
        self.feature_names.extend([
            "strong_move_when_winning",
            "defensive_move_when_losing",
            "contests_with_high_power",
            "early_game_aggression",
            "late_game_caution",
            "kaos_conservation",        # Saving Kaos cards
            "material_lead_exploitation",
            "control_lead_exploitation",
            "comeback_desperation",     # Behind and need risks
            "endgame_positioning",      # Last few placements
        ])

        # Bias
        self.feature_names.append("bias")

    def extract(
        self,
        state: GameState,
        action: int,
        player: Player
    ) -> np.ndarray:
        """
        Extract enhanced feature vector for (state, action) pair.

        Args:
            state: Current game state
            action: Action index to evaluate
            player: Player proposing this action

        Returns:
            Feature vector
        """
        features = []

        # === STATE FEATURES ===

        # Game phase
        features.append(1.0 if state.phase == Phase.PLACEMENT else 0.0)
        features.append(1.0 if state.phase == Phase.DOGFIGHTS else 0.0)

        # Turn (normalized)
        features.append(min(state.turn_number / 50.0, 1.0))

        # Material count
        p1_hand = len(state.player_resources[Player.ONE].rocketmen)
        p2_hand = len(state.player_resources[Player.TWO].rocketmen)

        p1_on_board = sum(1 for row in state.grid for sq in row
                         for rm in sq.rocketmen if rm.player == Player.ONE)
        p2_on_board = sum(1 for row in state.grid for sq in row
                         for rm in sq.rocketmen if rm.player == Player.TWO)

        p1_total = p1_hand + p1_on_board
        p2_total = p2_hand + p2_on_board

        my_rockets = p1_total if player == Player.ONE else p2_total
        opp_rockets = p2_total if player == Player.ONE else p1_total

        features.append(my_rockets / 9.0)
        features.append(opp_rockets / 9.0)
        features.append((my_rockets - opp_rockets) / 9.0)

        # Board control
        my_squares = 0
        opp_squares = 0
        contested = 0

        for row in state.grid:
            for sq in row:
                p1_here = any(rm.player == Player.ONE for rm in sq.rocketmen)
                p2_here = any(rm.player == Player.TWO for rm in sq.rocketmen)

                if p1_here and p2_here:
                    contested += 1
                elif p1_here:
                    my_squares += 1 if player == Player.ONE else 0
                    opp_squares += 1 if player == Player.TWO else 0
                elif p2_here:
                    my_squares += 1 if player == Player.TWO else 0
                    opp_squares += 1 if player == Player.ONE else 0

        features.append(my_squares / 9.0)
        features.append(opp_squares / 9.0)
        features.append((my_squares - opp_squares) / 9.0)
        features.append(contested / 9.0)

        # Kaos cards remaining
        p1_kaos = len(state.player_resources[Player.ONE].kaos_deck)
        p2_kaos = len(state.player_resources[Player.TWO].kaos_deck)
        my_kaos = p1_kaos if player == Player.ONE else p2_kaos
        opp_kaos = p2_kaos if player == Player.ONE else p1_kaos

        features.append(my_kaos / 9.0)
        features.append(opp_kaos / 9.0)

        # === ACTION FEATURES ===

        if state.phase == Phase.PLACEMENT:
            self._add_enhanced_placement_features(
                features, state, action, player,
                my_squares, opp_squares, contested
            )
        elif state.phase == Phase.DOGFIGHTS:
            self._add_enhanced_dogfight_features(
                features, state, action, player
            )
        else:
            # Unknown phase - add zeros
            features.extend([0.0] * 26)

        # === INTERACTION FEATURES ===

        material_adv = (my_rockets - opp_rockets) / 9.0
        control_adv = (my_squares - opp_squares) / 9.0
        is_winning = (material_adv + control_adv) > 0.3
        is_losing = (material_adv + control_adv) < -0.3

        features.append(1.0 if is_winning else 0.0)
        features.append(1.0 if is_losing else 0.0)
        features.append(my_rockets / 9.0 if contested > 0 else 0.0)
        features.append(1.0 if state.turn_number < 10 else 0.0)
        features.append(1.0 if state.turn_number > 30 else 0.0)

        # Kaos conservation
        features.append(my_kaos / 9.0 if my_kaos > 6 else 0.0)

        # Exploitation when ahead
        features.append(material_adv if material_adv > 0 else 0.0)
        features.append(control_adv if control_adv > 0 else 0.0)

        # Comeback desperation
        features.append(1.0 if is_losing and state.turn_number > 20 else 0.0)

        # Endgame positioning
        features.append(1.0 if p1_hand + p2_hand <= 4 else 0.0)

        # Bias term
        features.append(1.0)

        return np.array(features, dtype=np.float32)

    def _decode_placement_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode placement action index into (power, row, col).

        Action space: 9 rocketmen × 9 positions = 81 actions (indices 0-80)
        - power: 2-10 (action // 9 + 2)
        - row: 0-2 ((action % 9) // 3)
        - col: 0-2 ((action % 9) % 3)
        """
        if action >= 81:
            return None, None, None  # Not a placement action

        power = (action // 9) + 2  # 2-10
        position = action % 9
        row = position // 3  # 0-2
        col = position % 3   # 0-2

        return power, row, col

    def _check_line_at_position(
        self,
        state: GameState,
        row: int,
        col: int,
        player: Player
    ) -> Tuple[int, int]:
        """
        Check lines through position (row, col) for player AFTER placing there.

        Simulates placing player's piece at (row, col) and counts line lengths.

        Returns:
            (max_line_length, num_lines_of_that_length)
            E.g., (2, 3) means three different 2-in-a-rows AFTER placement
        """
        # All possible lines through this position
        lines_to_check = []

        # Horizontal
        lines_to_check.append([(row, 0), (row, 1), (row, 2)])

        # Vertical
        lines_to_check.append([(0, col), (1, col), (2, col)])

        # Diagonal (if on diagonal)
        if row == col:
            lines_to_check.append([(0, 0), (1, 1), (2, 2)])
        if row + col == 2:
            lines_to_check.append([(0, 2), (1, 1), (2, 0)])

        max_len = 0
        count = 0

        for line in lines_to_check:
            # Count how many in this line belong to player AFTER placing at (row, col)
            player_count = 0
            opp_player = Player.TWO if player == Player.ONE else Player.ONE
            blocked = False

            for r, c in line:
                # Simulate placement: if this is the target square, player owns it
                if r == row and c == col:
                    player_count += 1
                else:
                    sq = state.grid[r][c]
                    has_mine = any(rm.player == player for rm in sq.rocketmen)
                    has_opp = any(rm.player == opp_player for rm in sq.rocketmen)

                    if has_mine:
                        player_count += 1
                    if has_opp:
                        blocked = True

            # Only count non-blocked lines
            if not blocked and player_count > 0:
                if player_count > max_len:
                    max_len = player_count
                    count = 1
                elif player_count == max_len:
                    count += 1

        return max_len, count

    def _check_opponent_threat_at_position(
        self,
        state: GameState,
        row: int,
        col: int,
        player: Player
    ) -> Tuple[bool, bool]:
        """
        Check if placing at (row, col) blocks opponent threats.

        Returns:
            (blocks_2_in_a_row, blocks_3_in_a_row)
        """
        opp_player = Player.TWO if player == Player.ONE else Player.ONE

        # Check lines through this position
        lines_to_check = []

        # Horizontal
        lines_to_check.append([(row, 0), (row, 1), (row, 2)])

        # Vertical
        lines_to_check.append([(0, col), (1, col), (2, col)])

        # Diagonals
        if row == col:
            lines_to_check.append([(0, 0), (1, 1), (2, 2)])
        if row + col == 2:
            lines_to_check.append([(0, 2), (1, 1), (2, 0)])

        blocks_2 = False
        blocks_3 = False

        for line in lines_to_check:
            # Count opponent pieces and empty squares in this line
            opp_count = 0
            empty_or_target = 0  # Empty squares or the target square
            has_my_pieces = False

            for r, c in line:
                if r == row and c == col:
                    # This is where we're placing - it will block the opponent
                    empty_or_target += 1
                else:
                    sq = state.grid[r][c]
                    has_mine = any(rm.player == player for rm in sq.rocketmen)
                    has_opp = any(rm.player == opp_player for rm in sq.rocketmen)
                    is_empty = len(sq.rocketmen) == 0

                    if has_mine:
                        has_my_pieces = True  # Line already blocked by us
                    if has_opp:
                        opp_count += 1
                    if is_empty:
                        empty_or_target += 1

            # Only count as blocking if line wasn't already blocked by our pieces
            if not has_my_pieces:
                # blocks_2: Opponent has 2 pieces, we're filling the gap
                if opp_count == 2 and empty_or_target >= 1:
                    blocks_2 = True

                # blocks_3: Opponent has 2 pieces with exactly 1 square to complete (winning threat)
                if opp_count == 2 and empty_or_target == 1:
                    blocks_3 = True

        return blocks_2, blocks_3

    def _add_enhanced_placement_features(
        self,
        features: List[float],
        state: GameState,
        action: int,
        player: Player,
        my_squares: int,
        opp_squares: int,
        contested: int
    ):
        """Add enhanced features for placement actions."""
        power, row, col = self._decode_placement_action(action)

        if power is None:
            # Not a placement action
            features.extend([0.0] * 18)
            features.extend([0.0] * 8)  # Dogfight features
            return

        # Power indicators
        power_low = 1.0 if power <= 4 else 0.0
        power_mid = 1.0 if 5 <= power <= 7 else 0.0
        power_high = 1.0 if power >= 8 else 0.0

        features.append(power_low)
        features.append(power_mid)
        features.append(power_high)

        # Position strategic value
        is_center = 1.0 if (row == 1 and col == 1) else 0.0
        is_edge = 1.0 if (row == 1 or col == 1) and not (row == 1 and col == 1) else 0.0
        is_corner = 1.0 if (row != 1 and col != 1) else 0.0

        features.append(is_center)
        features.append(is_edge)
        features.append(is_corner)

        # Square state
        square = state.grid[row][col]
        opp_player = Player.TWO if player == Player.ONE else Player.ONE

        has_mine = any(rm.player == player for rm in square.rocketmen)
        has_opp = any(rm.player == opp_player for rm in square.rocketmen)
        is_empty = len(square.rocketmen) == 0

        contests = 1.0 if has_opp and not has_mine else 0.0
        takes_empty = 1.0 if is_empty else 0.0

        features.append(contests)
        features.append(takes_empty)

        # Line formation
        max_len, num_lines = self._check_line_at_position(state, row, col, player)

        forms_2 = 1.0 if max_len == 2 else 0.0
        forms_3 = 1.0 if max_len == 3 else 0.0  # Winning move!

        features.append(forms_2)
        features.append(forms_3)

        # Blocking opponent
        blocks_2, blocks_3 = self._check_opponent_threat_at_position(
            state, row, col, player
        )

        features.append(1.0 if blocks_2 else 0.0)
        features.append(1.0 if blocks_3 else 0.0)  # Blocks opponent win!

        # Fork creation (multiple 2-in-a-rows)
        creates_fork = 1.0 if forms_2 and num_lines >= 2 else 0.0
        features.append(creates_fork)

        # Threatens win (one move from 3-in-a-row)
        threatens_win = 1.0 if max_len == 2 and not blocks_3 else 0.0
        features.append(threatens_win)

        # Power advantage at this square
        if has_opp:
            opp_power = max((rm.power for rm in square.rocketmen if rm.player == opp_player), default=0)
            power_adv = 1.0 if power > opp_power else 0.0
        else:
            power_adv = 0.0
        features.append(power_adv)

        # Adjacency
        adj_mine = 0
        adj_opp = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < 3 and 0 <= c < 3:
                    sq = state.grid[r][c]
                    if any(rm.player == player for rm in sq.rocketmen):
                        adj_mine += 1
                    if any(rm.player == opp_player for rm in sq.rocketmen):
                        adj_opp += 1

        features.append(adj_mine / 8.0)
        features.append(adj_opp / 8.0)

        # Strategic value (center=1.0, edge=0.67, corner=0.33)
        strat_value = 1.0 if is_center else (0.67 if is_edge else 0.33)
        features.append(strat_value)

        # Dogfight features (not applicable)
        features.extend([0.0] * 8)

    def _add_enhanced_dogfight_features(
        self,
        features: List[float],
        state: GameState,
        action: int,
        player: Player
    ):
        """Add enhanced features for dogfight actions."""
        # Placement features (not applicable)
        features.extend([0.0] * 18)

        # Dogfight interpretation (simplified - would need current dogfight context)
        uses_rocket = 1.0 if action < 43 else 0.0  # Approximation
        uses_flare = 1.0 if action >= 43 else 0.0

        features.append(uses_rocket)
        features.append(uses_flare)

        # Power differential (would need current dogfight context)
        # For now, stub with zeros
        features.append(0.0)  # power_advantage
        features.append(0.0)  # power_disadvantage

        # Kaos card value (approximation from action index)
        if action >= 81 and action < 85:
            card_idx = action - 81
            card_value = [14, 13, 12, 11][card_idx]  # A, K, Q, J
            is_high = 1.0 if card_value >= 13 else 0.0
            is_low = 1.0 if card_value <= 11 else 0.0
        else:
            is_high = 0.0
            is_low = 0.0

        features.append(is_high)
        features.append(is_low)

        # Strategic square (would need current dogfight position)
        features.append(0.0)  # center_square
        features.append(0.0)  # strategic_square

    def get_feature_count(self) -> int:
        """Get number of features."""
        return len(self.feature_names)

    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        return self.feature_names.copy()

    def explain_features(self, features: np.ndarray, top_k: int = 10) -> str:
        """
        Explain which features are most significant.

        Args:
            features: Feature vector
            top_k: Number of top features to show

        Returns:
            Human-readable explanation
        """
        indices = np.argsort(np.abs(features))[-top_k:][::-1]

        lines = ["Top features:"]
        for idx in indices:
            if idx < len(self.feature_names):
                name = self.feature_names[idx]
                value = features[idx]
                lines.append(f"  {name:35s}: {value:7.3f}")

        return "\n".join(lines)
