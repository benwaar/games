"""
State feature extraction for learning agents.

Architecture principle: All learning agents share the same feature extraction.
This ensures consistency and makes learned models portable.

Feature vector is fixed-dimensional (~50 dims) and normalized to [0, 1] range.
"""

from typing import List
import numpy as np

from ..state import GameState, Player, Phase, GridSquare


class StateFeatureExtractor:
    """
    Extracts fixed-dimensional feature vectors from GameState.

    Feature groups:
    1. Grid occupancy (27 features): 9 positions × 3 states (empty/P1/P2)
    2. Resource counts (6 features): rocketmen, weapons, kaos cards per player
    3. Material balance (3 features): rocketmen, weapons, kaos advantage
    4. Grid control (6 features): controlled squares, contested squares
    5. Phase indicator (2 features): placement=1,0 or dogfights=0,1
    6. Position quality (9 features): center, edges, corners control

    Total: 53 features
    """

    def __init__(self):
        self.feature_dim = 53

    def extract(self, state: GameState, player: Player) -> np.ndarray:
        """
        Extract feature vector from state, from player's perspective.

        Args:
            state: Current game state
            player: Player to extract features for

        Returns:
            Feature vector of shape (53,) with values in [0, 1]
        """
        features = []

        # 1. Grid occupancy (27 features)
        features.extend(self._extract_grid_occupancy(state))

        # 2. Resource counts (6 features)
        features.extend(self._extract_resource_counts(state, player))

        # 3. Material balance (3 features)
        features.extend(self._extract_material_balance(state, player))

        # 4. Grid control (6 features)
        features.extend(self._extract_grid_control(state, player))

        # 5. Phase indicator (2 features)
        features.extend(self._extract_phase(state))

        # 6. Position quality (9 features)
        features.extend(self._extract_position_quality(state, player))

        return np.array(features, dtype=np.float32)

    def _extract_grid_occupancy(self, state: GameState) -> List[float]:
        """
        Grid occupancy: one-hot encoding for each square.
        For each of 9 squares: [empty, P1, P2]
        """
        features = []
        for row in range(3):
            for col in range(3):
                square = state.grid[row][col]
                if square.is_empty:
                    features.extend([1.0, 0.0, 0.0])
                elif square.is_controlled:
                    controller = square.controller
                    if controller == Player.ONE:
                        features.extend([0.0, 1.0, 0.0])
                    else:
                        features.extend([0.0, 0.0, 1.0])
                else:  # contested
                    # For contested squares, both players present
                    features.extend([0.0, 0.5, 0.5])
        return features

    def _extract_resource_counts(self, state: GameState, player: Player) -> List[float]:
        """
        Resource counts for both players (normalized).
        - Rocketmen: 0-9 → [0, 1]
        - Weapons: 0-4 → [0, 1]
        - Kaos cards: 0-9 → [0, 1]
        """
        my_res = state.get_resources(player)
        opp_res = state.get_resources(player.opponent())

        return [
            len(my_res.rocketmen) / 9.0,
            len(my_res.weapons) / 4.0,
            my_res.remaining_kaos_cards() / 9.0,
            len(opp_res.rocketmen) / 9.0,
            len(opp_res.weapons) / 4.0,
            opp_res.remaining_kaos_cards() / 9.0,
        ]

    def _extract_material_balance(self, state: GameState, player: Player) -> List[float]:
        """
        Material advantage (my resources - opponent's resources).
        Normalized to [0, 1] where 0.5 = equal, 0 = max disadvantage, 1 = max advantage.
        """
        my_res = state.get_resources(player)
        opp_res = state.get_resources(player.opponent())

        rocketmen_diff = len(my_res.rocketmen) - len(opp_res.rocketmen)
        weapons_diff = len(my_res.weapons) - len(opp_res.weapons)
        kaos_diff = my_res.remaining_kaos_cards() - opp_res.remaining_kaos_cards()

        return [
            (rocketmen_diff + 9) / 18.0,  # [-9, 9] → [0, 1]
            (weapons_diff + 4) / 8.0,     # [-4, 4] → [0, 1]
            (kaos_diff + 9) / 18.0,       # [-9, 9] → [0, 1]
        ]

    def _extract_grid_control(self, state: GameState, player: Player) -> List[float]:
        """
        Grid control statistics.
        - My controlled squares / 9
        - Opponent controlled squares / 9
        - Contested squares / 9
        - Empty squares / 9
        - Three-in-row threat (binary: do I have 2-in-a-row?)
        - Opponent three-in-row threat
        """
        my_controlled = 0
        opp_controlled = 0
        contested = 0
        empty = 0

        for row in range(3):
            for col in range(3):
                square = state.grid[row][col]
                if square.is_empty:
                    empty += 1
                elif square.is_contested:
                    contested += 1
                elif square.controller == player:
                    my_controlled += 1
                else:
                    opp_controlled += 1

        # Check for 2-in-a-row threats (simplified: just check if close to winning)
        my_threat = 1.0 if self._has_two_in_row(state, player) else 0.0
        opp_threat = 1.0 if self._has_two_in_row(state, player.opponent()) else 0.0

        return [
            my_controlled / 9.0,
            opp_controlled / 9.0,
            contested / 9.0,
            empty / 9.0,
            my_threat,
            opp_threat,
        ]

    def _has_two_in_row(self, state: GameState, player: Player) -> bool:
        """Check if player has two-in-a-row (potential winning threat)."""
        # Check rows
        for row in range(3):
            controlled = sum(1 for col in range(3)
                           if state.grid[row][col].controller == player)
            if controlled >= 2:
                return True

        # Check columns
        for col in range(3):
            controlled = sum(1 for row in range(3)
                           if state.grid[row][col].controller == player)
            if controlled >= 2:
                return True

        # Check diagonals
        diag1 = sum(1 for i in range(3)
                   if state.grid[i][i].controller == player)
        diag2 = sum(1 for i in range(3)
                   if state.grid[i][2-i].controller == player)

        return diag1 >= 2 or diag2 >= 2

    def _extract_phase(self, state: GameState) -> List[float]:
        """
        Phase indicator (one-hot).
        [placement, dogfights]
        (ended phase not used during gameplay)
        """
        if state.phase == Phase.PLACEMENT:
            return [1.0, 0.0]
        elif state.phase == Phase.DOGFIGHTS:
            return [0.0, 1.0]
        else:  # ENDED - shouldn't happen during feature extraction
            return [0.0, 0.0]

    def _extract_position_quality(self, state: GameState, player: Player) -> List[float]:
        """
        Position quality: strategic value of controlled positions.
        - Center square control (most valuable)
        - Edge squares control (4 squares)
        - Corner squares control (4 squares)
        - Opponent center control
        - Opponent edge control
        - Opponent corner control
        - Contested center
        - Contested edges
        - Contested corners
        """
        center = (1, 1)
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]

        def count_controlled(positions, controller):
            return sum(1 for r, c in positions
                      if state.grid[r][c].controller == controller)

        def count_contested(positions):
            return sum(1 for r, c in positions
                      if state.grid[r][c].is_contested)

        my_center = 1.0 if state.grid[1][1].controller == player else 0.0
        my_edges = count_controlled(edges, player) / 4.0
        my_corners = count_controlled(corners, player) / 4.0

        opp_center = 1.0 if state.grid[1][1].controller == player.opponent() else 0.0
        opp_edges = count_controlled(edges, player.opponent()) / 4.0
        opp_corners = count_controlled(corners, player.opponent()) / 4.0

        contested_center = 1.0 if state.grid[1][1].is_contested else 0.0
        contested_edges = count_contested(edges) / 4.0
        contested_corners = count_contested(corners) / 4.0

        return [
            my_center, my_edges, my_corners,
            opp_center, opp_edges, opp_corners,
            contested_center, contested_edges, contested_corners,
        ]

    def feature_names(self) -> List[str]:
        """
        Return human-readable feature names (for interpretability).
        Useful for debugging and understanding learned weights.
        """
        names = []

        # Grid occupancy
        for row in range(3):
            for col in range(3):
                names.extend([
                    f"grid[{row},{col}]_empty",
                    f"grid[{row},{col}]_P1",
                    f"grid[{row},{col}]_P2",
                ])

        # Resource counts
        names.extend([
            "my_rocketmen_count",
            "my_weapons_count",
            "my_kaos_count",
            "opp_rocketmen_count",
            "opp_weapons_count",
            "opp_kaos_count",
        ])

        # Material balance
        names.extend([
            "rocketmen_advantage",
            "weapons_advantage",
            "kaos_advantage",
        ])

        # Grid control
        names.extend([
            "my_controlled_squares",
            "opp_controlled_squares",
            "contested_squares",
            "empty_squares",
            "my_threat_2_in_row",
            "opp_threat_2_in_row",
        ])

        # Phase
        names.extend([
            "phase_placement",
            "phase_dogfights",
        ])

        # Position quality
        names.extend([
            "my_center", "my_edges", "my_corners",
            "opp_center", "opp_edges", "opp_corners",
            "contested_center", "contested_edges", "contested_corners",
        ])

        return names


# Global singleton
_FEATURE_EXTRACTOR = None


def get_feature_extractor() -> StateFeatureExtractor:
    """Get the global feature extractor singleton."""
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        _FEATURE_EXTRACTOR = StateFeatureExtractor()
    return _FEATURE_EXTRACTOR
