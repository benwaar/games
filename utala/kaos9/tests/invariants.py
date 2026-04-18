"""
Invariant checker for utala: kaos 9.

Verifies critical game invariants after every state transition.
Used as a safety net during Phase 4 rule variant testing.
"""

from utala.actions import get_action_space
from utala.state import GameConfig, GameState, Phase, Player


class InvariantChecker:
    """Checks critical game invariants on a GameState."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.action_space = get_action_space(self.config)

    def check_all(self, state: GameState) -> list[str]:
        """
        Run all invariant checks on a state.

        Returns list of violation descriptions (empty = all passed).
        """
        violations: list[str] = []
        violations.extend(self._check_rocketman_conservation(state))
        violations.extend(self._check_kaos_conservation(state))
        violations.extend(self._check_action_space_size(state))
        violations.extend(self._check_legal_mask_length(state))
        violations.extend(self._check_phase_consistency(state))
        violations.extend(self._check_no_double_occupation(state))
        violations.extend(self._check_contested_squares(state))
        violations.extend(self._check_face_down_validity(state))
        return violations

    def check_transition(
        self, prev_state: GameState, new_state: GameState
    ) -> list[str]:
        """
        Check transition-specific invariants between two states.

        Returns list of violation descriptions (empty = all passed).
        """
        violations: list[str] = []
        violations.extend(self._check_phase_transition(prev_state, new_state))
        violations.extend(self._check_turn_alternation(prev_state, new_state))
        return violations

    def check_terminal(self, state: GameState) -> list[str]:
        """
        Check invariants that apply only to terminal states.

        Returns list of violation descriptions (empty = all passed).
        """
        violations: list[str] = []
        violations.extend(self._check_game_over_consistency(state))
        violations.extend(self._check_three_in_row_winner(state))
        return violations

    # --- Individual invariant checks ---

    def _check_rocketman_conservation(self, state: GameState) -> list[str]:
        """Invariant 1: Rocketman conservation per player.

        During placement: hand + board = total (no elimination yet).
        During dogfights/ended: hand + board <= total (elimination removes rocketmen).
        """
        violations = []
        for player in [Player.ONE, Player.TWO]:
            hand = len(state.get_resources(player).rocketmen)
            board = sum(
                1
                for row in range(3)
                for col in range(3)
                for rm in state.get_square(row, col).rocketmen
                if rm.player == player
            )
            total = self.config.num_rocketmen
            actual = hand + board

            if state.phase == Phase.PLACEMENT:
                if actual != total:
                    violations.append(
                        f"Rocketman conservation (placement): P{player.value + 1} has "
                        f"{actual} (hand={hand}, board={board}), expected {total}"
                    )
            else:
                if actual > total:
                    violations.append(
                        f"Rocketman conservation: P{player.value + 1} has "
                        f"{actual} (hand={hand}, board={board}), exceeds max {total}"
                    )
                if hand > 0:
                    violations.append(
                        f"Rocketman conservation: P{player.value + 1} still has "
                        f"{hand} rocketmen in hand after placement"
                    )
        return violations

    def _check_kaos_conservation(self, state: GameState) -> list[str]:
        """Invariant 2: Kaos card count = deck + discard per player."""
        violations = []
        for player in [Player.ONE, Player.TWO]:
            resources = state.get_resources(player)
            actual = len(resources.kaos_deck) + len(resources.kaos_discard)
            expected = self.config.num_kaos_cards
            if actual != expected:
                violations.append(
                    f"Kaos conservation: P{player.value + 1} has "
                    f"{actual} (deck={len(resources.kaos_deck)}, "
                    f"discard={len(resources.kaos_discard)}), expected {expected}"
                )
        return violations

    def _check_action_space_size(self, state: GameState) -> list[str]:
        """Invariant 3: Action space size is fixed for config."""
        expected = self.action_space.size()
        actual = get_action_space(self.config).size()
        if actual != expected:
            return [f"Action space size changed: {actual} != {expected}"]
        return []

    def _check_legal_mask_length(self, state: GameState) -> list[str]:
        """Invariant 4: Legal mask length matches action space size."""
        violations = []
        if state.phase != Phase.ENDED:
            for player in [Player.ONE, Player.TWO]:
                mask = self.action_space.get_legal_actions_mask(state, player)
                if len(mask) != self.action_space.size():
                    violations.append(
                        f"Legal mask length: P{player.value + 1} mask length "
                        f"{len(mask)} != action space size {self.action_space.size()}"
                    )
        return violations

    def _check_phase_consistency(self, state: GameState) -> list[str]:
        """Invariant 5: Phase is a valid Phase enum value."""
        if state.phase not in (Phase.PLACEMENT, Phase.DOGFIGHTS, Phase.ENDED):
            return [f"Invalid phase: {state.phase}"]
        return []

    def _check_no_double_occupation(self, state: GameState) -> list[str]:
        """Invariant 6: No square has two rocketmen from the same player."""
        violations = []
        for row in range(3):
            for col in range(3):
                square = state.get_square(row, col)
                for player in [Player.ONE, Player.TWO]:
                    count = sum(1 for rm in square.rocketmen if rm.player == player)
                    if count > 1:
                        violations.append(
                            f"Double occupation: P{player.value + 1} has "
                            f"{count} rocketmen at ({row},{col})"
                        )
        return violations

    def _check_contested_squares(self, state: GameState) -> list[str]:
        """Invariant 10: Contested squares have exactly 2 rocketmen (one per player)."""
        violations = []
        for row in range(3):
            for col in range(3):
                square = state.get_square(row, col)
                if square.is_contested:
                    if len(square.rocketmen) != 2:
                        violations.append(
                            f"Contested square ({row},{col}) has "
                            f"{len(square.rocketmen)} rocketmen, expected 2"
                        )
                    elif square.rocketmen[0].player == square.rocketmen[1].player:
                        violations.append(
                            f"Contested square ({row},{col}) has two rocketmen "
                            f"from same player"
                        )
        return violations

    def _check_face_down_validity(self, state: GameState) -> list[str]:
        """Invariant 8: Face-down cards have powers in the config's face-down set."""
        violations = []
        for row in range(3):
            for col in range(3):
                for rm in state.get_square(row, col).rocketmen:
                    if rm.face_down and rm.power not in self.config.face_down_powers:
                        violations.append(
                            f"Face-down card at ({row},{col}) has power "
                            f"{rm.power}, not in face_down_powers "
                            f"{self.config.face_down_powers}"
                        )
        return violations

    def _check_phase_transition(
        self, prev: GameState, new: GameState
    ) -> list[str]:
        """Invariant 5b: Phase transitions only go forward."""
        valid_transitions = {
            Phase.PLACEMENT: {Phase.PLACEMENT, Phase.DOGFIGHTS, Phase.ENDED},
            Phase.DOGFIGHTS: {Phase.DOGFIGHTS, Phase.ENDED},
            Phase.ENDED: {Phase.ENDED},
        }
        allowed = valid_transitions.get(prev.phase, set())
        if new.phase not in allowed:
            return [
                f"Invalid phase transition: {prev.phase.value} -> {new.phase.value}"
            ]
        return []

    def _check_turn_alternation(
        self, prev: GameState, new: GameState
    ) -> list[str]:
        """Invariant 9: During placement, current_player alternates each turn."""
        if prev.phase == Phase.PLACEMENT and new.phase == Phase.PLACEMENT:
            if new.current_player == prev.current_player:
                return [
                    f"Turn alternation violated: player stayed "
                    f"P{new.current_player.value + 1} after placement turn"
                ]
        return []

    def _check_game_over_consistency(self, state: GameState) -> list[str]:
        """Invariant 7: game_over correlates with phase==ENDED."""
        violations = []
        if state.game_over and state.phase != Phase.ENDED:
            violations.append(
                f"game_over=True but phase={state.phase.value}"
            )
        if state.phase == Phase.ENDED and not state.game_over:
            violations.append(
                "phase=ENDED but game_over=False"
            )
        return violations

    def _check_three_in_row_winner(self, state: GameState) -> list[str]:
        """Invariant 12: If a player has 3-in-a-row at game end, they must be the winner."""
        if not state.game_over:
            return []

        violations = []
        p1_has = state.check_three_in_row(Player.ONE)
        p2_has = state.check_three_in_row(Player.TWO)

        if p1_has and not p2_has and state.winner != Player.ONE:
            violations.append(
                f"P1 has 3-in-a-row but winner is "
                f"{'P2' if state.winner == Player.TWO else 'Draw'}"
            )
        if p2_has and not p1_has and state.winner != Player.TWO:
            violations.append(
                f"P2 has 3-in-a-row but winner is "
                f"{'P1' if state.winner == Player.ONE else 'Draw'}"
            )
        return violations
