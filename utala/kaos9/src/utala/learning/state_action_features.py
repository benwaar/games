"""
State-Action Feature Extraction for Linear Value Learning.

Extracts rich features from (state, action) pairs that capture strategic
information useful for learning value functions.

Feature categories:
- State features: game phase, material, board control
- Action features: move type, power, position
- Combined features: action quality given state
"""

import numpy as np
from typing import List

from ..state import GameState, GameConfig, Player, Phase
from ..actions import get_action_space, ActionType


class StateActionFeatureExtractor:
    """
    Extracts features for state-action pairs.

    Features are designed to capture:
    1. Current game situation (state)
    2. Properties of the proposed action
    3. Interaction between state and action (how good is this move now?)
    """

    def __init__(self, deck_awareness: bool = True, config: GameConfig | None = None):
        self.deck_awareness = deck_awareness
        self.config = config or GameConfig()
        self.action_space = get_action_space(self.config)
        self.feature_names: List[str] = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of feature names for interpretability."""
        # State features
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
        ])

        # Placement action features
        self.feature_names.extend([
            "placement_power_low",      # 2-4
            "placement_power_mid",      # 5-7
            "placement_power_high",     # 8-10
            "placement_center",
            "placement_edge",
            "placement_corner",
            "placement_contests_square",
            "placement_takes_control",
            "placement_forms_line_2",
            "placement_forms_line_3",
        ])

        # Dogfight action features (repurposed for choice features in Variant A)
        if not self.config.fixed_dogfight_order:
            self.feature_names.extend([
                "choice_win_completes_line",
                "choice_lose_completes_opp_line",
                "choice_advances_our_line",
                "choice_contests_opp_line",
                "choice_power_advantage",
                "choice_position_value",
            ])
        else:
            self.feature_names.extend([
                "dogfight_uses_rocket",
                "dogfight_uses_flare",
                "dogfight_power_diff_positive",  # My rocketman stronger
                "dogfight_power_diff_negative",  # Opponent stronger
                "dogfight_kaos_cards_remaining",
                "dogfight_strategic_square",
            ])

        # Interaction features
        self.feature_names.extend([
            "strong_move_when_winning",
            "defensive_move_when_losing",
            "contests_with_high_power",
            "early_game_aggression",
            "late_game_caution",
        ])

        # Deck awareness features (Phase 3.2)
        if self.deck_awareness:
            self.feature_names.extend([
                "my_high_cards_ratio",
                "my_low_cards_ratio",
                "my_expected_value",
                "my_deck_variance",
                "my_deck_strength",
                "opp_high_cards_ratio",
                "opp_low_cards_ratio",
                "opp_expected_value",
                "opp_deck_variance",
                "opp_deck_strength",
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
        Extract feature vector for (state, action) pair.

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

        # Turn (normalized to [0, 1])
        features.append(min(state.turn_number / 50.0, 1.0))

        # Material count (rocketmen in hand + on board)
        p1_hand = len(state.player_resources[Player.ONE].rocketmen)
        p2_hand = len(state.player_resources[Player.TWO].rocketmen)

        # Count rocketmen on board
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
        features.append((my_rockets - opp_rockets) / 9.0)  # Advantage

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

        # === ACTION FEATURES ===

        if state.phase == Phase.PLACEMENT:
            self._add_placement_features(features, state, action, player)
        elif state.phase == Phase.DOGFIGHTS:
            if state.awaiting_dogfight_choice:
                self._add_dogfight_choice_features(features, state, action, player)
            else:
                self._add_dogfight_features(features, state, action, player)
        else:
            # Unknown phase - add zeros
            features.extend([0.0] * 16)

        # === INTERACTION FEATURES ===

        # Strong move when winning
        material_adv = (my_rockets - opp_rockets) / 9.0
        control_adv = (my_squares - opp_squares) / 9.0
        is_winning = (material_adv + control_adv) > 0.3

        # Heuristics for "strong" vs "defensive" moves
        # (simplified - would need action interpretation)
        features.append(1.0 if is_winning else 0.0)
        features.append(1.0 if not is_winning else 0.0)
        features.append(my_rockets / 9.0 if contested > 0 else 0.0)
        features.append(1.0 if state.turn_number < 10 else 0.0)
        features.append(1.0 if state.turn_number > 30 else 0.0)

        # === DECK AWARENESS FEATURES (Phase 3.2) ===
        if self.deck_awareness:
            self._add_deck_awareness_features(features, state, player)

        # Bias term
        features.append(1.0)

        return np.array(features, dtype=np.float32)

    def _add_placement_features(
        self,
        features: List[float],
        state: GameState,
        action: int,
        player: Player
    ):
        """Add features specific to placement actions."""
        # Try to interpret action as placement
        # Action index structure: position (0-8) × power options

        # Placeholder - would need to decode action properly
        # For now, extract what we can

        # Power indicators (rough approximation from action index)
        power_low = 1.0 if action < 27 else 0.0
        power_mid = 1.0 if 27 <= action < 54 else 0.0
        power_high = 1.0 if action >= 54 else 0.0

        features.append(power_low)
        features.append(power_mid)
        features.append(power_high)

        # Position (approximate from action index)
        position = action % 9
        row = position // 3
        col = position % 3

        is_center = 1.0 if (row == 1 and col == 1) else 0.0
        is_edge = 1.0 if (row == 1 or col == 1) and not (row == 1 and col == 1) else 0.0
        is_corner = 1.0 if (row != 1 and col != 1) else 0.0

        features.append(is_center)
        features.append(is_edge)
        features.append(is_corner)

        # Square state
        if row < 3 and col < 3:
            square = state.grid[row][col]
            p1_here = any(rm.player == Player.ONE for rm in square.rocketmen)
            p2_here = any(rm.player == Player.TWO for rm in square.rocketmen)

            contests = 1.0 if (player == Player.ONE and p2_here) or (player == Player.TWO and p1_here) else 0.0
            takes_control = 1.0 if not p1_here and not p2_here else 0.0
        else:
            contests = 0.0
            takes_control = 0.0

        features.append(contests)
        features.append(takes_control)

        # Line formation potential
        line_2, line_3 = self._check_line_formation(state, row, col, player)
        features.append(line_2)   # forms_line_2: placement creates 2-in-a-row
        features.append(line_3)   # forms_line_3: placement completes 3-in-a-row

        # Dogfight features (not applicable)
        features.extend([0.0] * 6)

    def _add_dogfight_features(
        self,
        features: List[float],
        state: GameState,
        action: int,
        player: Player
    ):
        """Add features specific to dogfight actions."""
        # Placement features (not applicable)
        features.extend([0.0] * 10)

        # Dogfight action interpretation
        # Action encodes: rocket/flare choice, kaos card (if using)

        uses_rocket = 1.0 if action < 43 else 0.0  # Rough approximation
        uses_flare = 1.0 if action >= 43 else 0.0

        features.append(uses_rocket)
        features.append(uses_flare)

        # Power differential (would need current dogfight context)
        # Placeholder
        features.append(0.0)  # power_diff_positive
        features.append(0.0)  # power_diff_negative

        # Kaos cards remaining
        p1_kaos = len(state.player_resources[Player.ONE].kaos_deck)
        p2_kaos = len(state.player_resources[Player.TWO].kaos_deck)
        my_kaos = p1_kaos if player == Player.ONE else p2_kaos

        features.append(my_kaos / 9.0)

        # Strategic square (center/edge/corner of current dogfight)
        # Would need current_dogfight context
        features.append(0.0)

    def _add_dogfight_choice_features(
        self,
        features: List[float],
        state: GameState,
        action: int,
        player: Player
    ):
        """Add features for CHOOSE_DOGFIGHT actions (Variant A)."""
        # Placement features not applicable
        features.extend([0.0] * 10)

        # Decode action to get target square
        action_obj = self.action_space.get_action(action)
        row, col = action_obj.row, action_obj.col

        # Line importance analysis
        win_completes = 0.0
        lose_completes = 0.0
        advances_ours = 0.0
        contests_theirs = 0.0

        for line_positions in self._get_lines_containing(row, col):
            our_count = 0
            opp_count = 0

            for r, c in line_positions:
                if (r, c) == (row, col):
                    continue
                sq = state.grid[r][c]
                if sq.is_controlled:
                    if sq.controller == player:
                        our_count += 1
                    else:
                        opp_count += 1

            if our_count == 2 and opp_count == 0:
                win_completes = 1.0
            elif opp_count == 2 and our_count == 0:
                lose_completes = 1.0
            elif our_count == 1 and opp_count == 0:
                advances_ours = 1.0
            elif opp_count == 1 and our_count == 0:
                contests_theirs = 1.0

        features.append(win_completes)
        features.append(lose_completes)
        features.append(advances_ours)
        features.append(contests_theirs)

        # Power advantage (visible rocketmen only)
        square = state.get_square(row, col)
        my_rm = next((rm for rm in square.rocketmen if rm.player == player), None)
        opp_rm = next((rm for rm in square.rocketmen if rm.player != player), None)

        power_adv = 0.0
        if my_rm and opp_rm and not my_rm.face_down and not opp_rm.face_down:
            power_adv = (my_rm.power - opp_rm.power) / 8.0

        features.append(power_adv)

        # Position value
        is_center = (row == 1 and col == 1)
        is_edge = (row == 1 or col == 1) and not is_center
        pos_value = 1.0 if is_center else (0.7 if is_edge else 0.4)
        features.append(pos_value)

    def _get_lines_containing(self, row: int, col: int) -> list:
        """Get all 3-in-a-row lines containing (row, col)."""
        lines = [[(row, c) for c in range(3)],
                 [(r, col) for r in range(3)]]
        if row == col:
            lines.append([(i, i) for i in range(3)])
        if row + col == 2:
            lines.append([(i, 2 - i) for i in range(3)])
        return lines

    def _check_line_formation(
        self,
        state: GameState,
        row: int,
        col: int,
        player: Player
    ) -> tuple:
        """
        Check if placing at (row, col) forms 2-in-a-row or 3-in-a-row.

        Returns:
            (forms_line_2, forms_line_3): 1.0 if true, 0.0 if false.
            forms_line_3 takes priority (if it completes a line, forms_line_2 = 0).
        """
        # All lines containing this position
        lines = []
        # Row
        lines.append([(row, c) for c in range(3)])
        # Column
        lines.append([(r, col) for r in range(3)])
        # Main diagonal
        if row == col:
            lines.append([(i, i) for i in range(3)])
        # Anti-diagonal
        if row + col == 2:
            lines.append([(i, 2 - i) for i in range(3)])

        best_line = 0  # max friendly count in any line (excluding this square)

        for line in lines:
            friendly = 0
            blocked = False
            for r, c in line:
                if r == row and c == col:
                    continue  # skip the square we're placing into
                sq = state.grid[r][c]
                has_mine = any(rm.player == player for rm in sq.rocketmen)
                has_opp = any(rm.player == player.opponent() for rm in sq.rocketmen)
                if has_mine and not has_opp:
                    friendly += 1
                elif has_opp:
                    blocked = True
            if not blocked:
                best_line = max(best_line, friendly)

        if best_line >= 2:
            return (0.0, 1.0)  # completes 3-in-a-row
        elif best_line == 1:
            return (1.0, 0.0)  # forms 2-in-a-row
        else:
            return (0.0, 0.0)

    def _add_deck_awareness_features(
        self,
        features: List[float],
        state: GameState,
        player: Player
    ):
        """
        Add deck awareness features based on visible information.

        Each player's Kaos deck starts as {1-13}. The discard pile is visible,
        so remaining cards = {1-13} \ discard. From this we compute:
        - High/low card ratios (risk profile)
        - Expected value (mean draw quality)
        - Variance (uncertainty in draws)
        - Deck strength (expected value × remaining count)
        """
        all_cards = set(range(1, 14))

        for p in [player, player.opponent()]:
            resources = state.get_resources(p)
            discard = set(resources.kaos_discard)
            remaining = list(all_cards - discard)

            if len(remaining) == 0:
                # Deck empty and about to reshuffle — treat as full deck
                remaining = list(all_cards)

            n = len(remaining)
            mean_val = sum(remaining) / n
            variance = sum((c - mean_val) ** 2 for c in remaining) / n

            high_count = sum(1 for c in remaining if c >= 8)
            low_count = sum(1 for c in remaining if c <= 5)

            features.append(high_count / n)           # high_cards_ratio
            features.append(low_count / n)             # low_cards_ratio
            features.append(mean_val / 13.0)           # expected_value [0,1]
            features.append(variance / 14.08)          # variance normalized (max var ≈ 14.08)
            features.append((mean_val * n) / 91.0)     # deck_strength (max = 7 * 13 = 91)

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
        # Get top-k features by absolute value
        indices = np.argsort(np.abs(features))[-top_k:][::-1]

        lines = ["Top features:"]
        for idx in indices:
            if idx < len(self.feature_names):
                name = self.feature_names[idx]
                value = features[idx]
                lines.append(f"  {name:30s}: {value:7.3f}")

        return "\n".join(lines)
