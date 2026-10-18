"""
Bluffing-Aware State Feature Extraction for DQN.

Purpose-built for deep learning: richer per-square info with power values
and face-down flags that enable the network to learn spatial patterns
and bluffing strategies.

Key design principle: the agent sees its OWN face-down powers (private info)
but NOT the opponent's. This information asymmetry is what makes bluffing
possible — the agent knows it placed a 2 face-down in the center, and can
learn that opponents tend to avoid contesting face-down cards in strong
positions.
"""

import numpy as np
from typing import List

from ..state import GameState, GameConfig, Player, Phase


class DQNFeatureExtractor:
    """
    Extracts state features for DQN with bluffing awareness.

    Feature groups:
    1. Per-square (54): 9 squares × 6 features (occupancy, powers, face-down)
    2. Resources (6): rocketmen, weapons, kaos per player
    3. Material balance (3): rocketmen, weapons, kaos advantage
    4. Board control (6): controlled, contested, empty, 2-in-a-row threats
    5. Phase + turn (3): placement/dogfight, turn progress
    6. Variant A context (3): awaiting choice, remaining contested, joker
    7. Deck awareness (4): high card ratios, expected values
    8. Bias (1)

    Total: 80 features
    """

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.feature_dim = 80
        self._feature_names: List[str] | None = None

    def extract(self, state: GameState, player: Player) -> np.ndarray:
        """
        Extract feature vector from state, from player's perspective.

        The agent sees its own face-down powers but not the opponent's.
        """
        features = []

        # === PER-SQUARE FEATURES (54) ===
        opp = player.opponent()

        for row in range(3):
            for col in range(3):
                square = state.grid[row][col]

                my_rm = None
                opp_rm = None
                for rm in square.rocketmen:
                    if rm.player == player:
                        my_rm = rm
                    else:
                        opp_rm = rm

                # Occupancy
                features.append(1.0 if my_rm is not None else 0.0)
                features.append(1.0 if opp_rm is not None else 0.0)

                # My power (I always know my own placements)
                features.append(my_rm.power / 10.0 if my_rm else 0.0)

                # Opponent power (only if visible — not face-down)
                if opp_rm and not opp_rm.face_down:
                    features.append(opp_rm.power / 10.0)
                else:
                    features.append(0.0)

                # Face-down flags
                features.append(1.0 if my_rm and my_rm.face_down else 0.0)
                features.append(1.0 if opp_rm and opp_rm.face_down else 0.0)

        # === RESOURCE COUNTS (6) ===
        my_res = state.get_resources(player)
        opp_res = state.get_resources(opp)

        features.append(len(my_res.rocketmen) / 9.0)
        features.append(len(my_res.weapons) / 4.0)
        features.append(my_res.remaining_kaos_cards() / 13.0)
        features.append(len(opp_res.rocketmen) / 9.0)
        features.append(len(opp_res.weapons) / 4.0)
        features.append(opp_res.remaining_kaos_cards() / 13.0)

        # === MATERIAL BALANCE (3) ===
        rm_diff = len(my_res.rocketmen) - len(opp_res.rocketmen)
        wp_diff = len(my_res.weapons) - len(opp_res.weapons)
        kaos_diff = my_res.remaining_kaos_cards() - opp_res.remaining_kaos_cards()

        features.append((rm_diff + 9) / 18.0)
        features.append((wp_diff + 4) / 8.0)
        features.append((kaos_diff + 13) / 26.0)

        # === BOARD CONTROL (6) ===
        my_controlled = 0
        opp_controlled = 0
        contested = 0
        empty = 0

        for row in range(3):
            for col in range(3):
                sq = state.grid[row][col]
                if sq.is_empty:
                    empty += 1
                elif sq.is_contested:
                    contested += 1
                elif sq.controller == player:
                    my_controlled += 1
                else:
                    opp_controlled += 1

        features.append(my_controlled / 9.0)
        features.append(opp_controlled / 9.0)
        features.append(contested / 9.0)
        features.append(empty / 9.0)

        # 2-in-a-row threats
        features.append(1.0 if self._has_two_in_row(state, player) else 0.0)
        features.append(1.0 if self._has_two_in_row(state, opp) else 0.0)

        # === PHASE + TURN (3) ===
        features.append(1.0 if state.phase == Phase.PLACEMENT else 0.0)
        features.append(1.0 if state.phase == Phase.DOGFIGHTS else 0.0)
        features.append(min(state.turn_number / 50.0, 1.0))

        # === VARIANT A CONTEXT (3) ===
        features.append(1.0 if state.awaiting_dogfight_choice else 0.0)
        features.append(len(state.remaining_contested) / 9.0)
        features.append(1.0 if state.joker_holder == player else 0.0)

        # === DECK AWARENESS (4) ===
        all_kaos = set(range(1, 14))

        for p in [player, opp]:
            res = state.get_resources(p)
            discard = set(res.kaos_discard)
            remaining = list(all_kaos - discard)
            if len(remaining) == 0:
                remaining = list(all_kaos)

            n = len(remaining)
            high_count = sum(1 for c in remaining if c >= 8)
            mean_val = sum(remaining) / n

            features.append(high_count / n)
            features.append(mean_val / 13.0)

        # === BIAS (1) ===
        features.append(1.0)

        return np.array(features, dtype=np.float32)

    def _has_two_in_row(self, state: GameState, player: Player) -> bool:
        """Check if player has two-in-a-row (potential winning threat)."""
        lines = [
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)],
        ]
        for line in lines:
            controlled = sum(1 for r, c in line if state.grid[r][c].controller == player)
            if controlled >= 2:
                return True
        return False

    def feature_names(self) -> List[str]:
        """Return human-readable feature names."""
        if self._feature_names is not None:
            return self._feature_names

        names = []

        # Per-square
        for row in range(3):
            for col in range(3):
                prefix = f"sq[{row},{col}]"
                names.extend([
                    f"{prefix}_has_mine",
                    f"{prefix}_has_opp",
                    f"{prefix}_my_power",
                    f"{prefix}_opp_power_visible",
                    f"{prefix}_my_face_down",
                    f"{prefix}_opp_face_down",
                ])

        # Resources
        names.extend([
            "my_rocketmen", "my_weapons", "my_kaos",
            "opp_rocketmen", "opp_weapons", "opp_kaos",
        ])

        # Material balance
        names.extend(["rm_advantage", "wp_advantage", "kaos_advantage"])

        # Board control
        names.extend([
            "my_controlled", "opp_controlled", "contested", "empty",
            "my_2_in_row", "opp_2_in_row",
        ])

        # Phase + turn
        names.extend(["phase_placement", "phase_dogfight", "turn_normalized"])

        # Variant A
        names.extend(["awaiting_choice", "remaining_contested", "has_joker"])

        # Deck awareness
        names.extend([
            "my_high_kaos_ratio", "my_kaos_expected",
            "opp_high_kaos_ratio", "opp_kaos_expected",
        ])

        # Bias
        names.append("bias")

        self._feature_names = names
        return names
