"""
Game engine for utala: kaos 9.

Architecture principles:
- All randomness lives in the engine, never in agents
- Agents only propose actions, never apply them
- The engine validates and applies all actions
- State is deterministic given a seed and action sequence
"""

import random
from copy import deepcopy
from dataclasses import dataclass

from .actions import Action, ActionType, get_action_space
from .state import DogfightContext, GameState, Phase, Player, Rocketman


@dataclass
class DogfightCommit:
    """Cards committed by both players during a dogfight."""
    player_one_action: Action | None = None
    player_two_action: Action | None = None


@dataclass
class DogfightResult:
    """Result of a dogfight resolution."""
    winner: Player | None  # None if both eliminated
    eliminated: list[Player]  # Players whose rocketmen were eliminated
    kaos_draws: dict[Player, list[int]]  # Kaos cards drawn during dogfight


@dataclass
class DogfightTurnState:
    """Tracks the current state of a turn-based dogfight."""
    position: tuple[int, int]  # Grid position of dogfight
    underdog: Player  # Player with lower power (acts first)
    other: Player  # Other player
    underdog_action: Action | None = None  # Underdog's first action
    other_action: Action | None = None  # Other player's action
    underdog_second_action: Action | None = None  # Underdog's counter-response if needed
    current_actor: Player | None = None  # Whose turn it is now
    rocket_in_play: Player | None = None  # Which player played a rocket (if any)
    complete: bool = False  # Whether all actions collected


class GameEngine:
    """
    Core game engine for utala: kaos 9.

    Manages game state, applies actions, handles all randomness.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize a new game.

        Args:
            seed: RNG seed for deterministic replay. If None, uses system random.
        """
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.rng = random.Random(self.seed)
        self.state = self._initialize_state()
        self.action_space = get_action_space()
        self.action_history: list[tuple[Player, int]] = []  # (player, action_index)
        self.current_dogfight: DogfightTurnState | None = None  # Turn-based dogfight state

    def _initialize_state(self) -> GameState:
        """Initialize game state."""
        state = GameState()
        state.rng_seed = self.seed

        # Initialize Kaos decks (shuffled)
        for player in [Player.ONE, Player.TWO]:
            kaos_deck = list(range(1, 14))  # 1-13 (A-K, Ace low)
            self.rng.shuffle(kaos_deck)
            state.player_resources[player].kaos_deck = kaos_deck

        return state

    def get_state_copy(self) -> GameState:
        """
        Return a deep copy of the current state.

        Agents receive state copies for observation only.
        """
        return deepcopy(self.state)

    def get_legal_actions(self, player: Player | None = None) -> list[int]:
        """
        Get legal action indices for a player.

        Args:
            player: Player to get actions for. If None, uses current player.

        Returns:
            List of legal action indices.
        """
        if player is None:
            player = self.state.current_player
        return self.action_space.get_legal_actions(self.state, player)

    def get_legal_actions_mask(self, player: Player | None = None) -> list[bool]:
        """
        Get legal action mask for a player.

        Args:
            player: Player to get mask for. If None, uses current player.

        Returns:
            Boolean mask of legal actions.
        """
        if player is None:
            player = self.state.current_player
        return self.action_space.get_legal_actions_mask(self.state, player)

    def apply_action(self, action_index: int) -> bool:
        """
        Apply an action to the game state.

        Args:
            action_index: Index of the action to apply.

        Returns:
            True if action was legal and applied, False otherwise.
        """
        # Validate action is legal
        legal_actions = self.get_legal_actions()
        if action_index not in legal_actions:
            return False

        action = self.action_space.get_action(action_index)
        player = self.state.current_player

        # Record action in history
        self.action_history.append((player, action_index))

        # Apply action based on phase
        if self.state.phase == Phase.PLACEMENT:
            self._apply_placement_action(action, player)
        elif self.state.phase == Phase.DOGFIGHTS:
            # Dogfight actions are handled differently (need both players)
            # This shouldn't be called directly during dogfights
            raise RuntimeError("Use apply_dogfight_actions for dogfight phase")

        return True

    def _apply_placement_action(self, action: Action, player: Player):
        """Apply a placement action."""
        assert action.action_type == ActionType.PLACE_ROCKETMAN
        assert action.rocketman_power is not None
        assert action.row is not None
        assert action.col is not None

        # Remove rocketman from player's hand
        resources = self.state.get_resources(player)
        resources.rocketmen.remove(action.rocketman_power)

        # Place rocketman on grid (v1.3: cards 2, 3, 9, 10 placed face-down)
        face_down = action.rocketman_power in [2, 3, 9, 10]
        rocketman = Rocketman(player=player, power=action.rocketman_power, face_down=face_down)
        square = self.state.get_square(action.row, action.col)
        square.rocketmen.append(rocketman)

        # Advance turn
        self.state.turn_number += 1
        self.state.current_player = self.state.current_player.opponent()

        # Check if placement phase is complete
        if self._all_rocketmen_placed():
            self._transition_to_dogfights()

    def _all_rocketmen_placed(self) -> bool:
        """Check if all rocketmen have been placed."""
        for player in [Player.ONE, Player.TWO]:
            if len(self.state.get_resources(player).rocketmen) > 0:
                return False
        return True

    def _transition_to_dogfights(self):
        """Transition from placement to dogfights phase."""
        self.state.phase = Phase.DOGFIGHTS

        # Build dogfight order: center, edges, corners
        # Center: (1,1)
        # Edges: (0,1), (1,0), (1,2), (2,1)
        # Corners: (0,0), (0,2), (2,0), (2,2)
        order = [
            (1, 1),  # center
            (0, 1), (1, 0), (1, 2), (2, 1),  # edges
            (0, 0), (0, 2), (2, 0), (2, 2),  # corners
        ]

        # Only include contested squares
        self.state.dogfight_order = [
            pos for pos in order
            if self.state.get_square(pos[0], pos[1]).is_contested
        ]

        self.state.current_dogfight_index = 0

        # If no dogfights, go straight to game end
        if len(self.state.dogfight_order) == 0:
            self._check_game_end()

    def get_current_dogfight_square(self) -> tuple[int, int] | None:
        """Get the current dogfight square position, or None if no dogfights left."""
        if (self.state.phase == Phase.DOGFIGHTS and
            self.state.current_dogfight_index < len(self.state.dogfight_order)):
            return self.state.dogfight_order[self.state.current_dogfight_index]
        return None

    def _get_underdog_at_position(self, row: int, col: int) -> Player:
        """
        Determine which player is the underdog (has lower power) at a position.

        Args:
            row, col: Grid position

        Returns:
            Player with lower rocketman power (underdog acts first)

        Raises:
            RuntimeError: If square is not contested
        """
        square = self.state.get_square(row, col)
        if len(square.rocketmen) != 2:
            raise RuntimeError(f"Square [{row},{col}] is not contested")

        rm1, rm2 = square.rocketmen

        # v1.3: Verify cards are revealed (showdown must occur before underdog determination)
        assert not rm1.face_down and not rm2.face_down, \
            "Cards must be revealed before determining underdog"

        # Underdog is player with lower power
        if rm1.power < rm2.power:
            return rm1.player
        elif rm2.power < rm1.power:
            return rm2.player
        else:
            # v1.8: Equal power - joker holder acts first, then passes joker
            joker_holder = self.state.joker_holder
            # Swap joker to other player (used in this dogfight)
            self.state.joker_holder = joker_holder.opponent()
            return joker_holder

    def begin_current_dogfight(self):
        """
        Initialize turn-based dogfight for the current contested square.

        Sets up dogfight state with underdog determined and ready for first action.
        """
        if self.state.phase != Phase.DOGFIGHTS:
            raise RuntimeError("Not in dogfight phase")

        dogfight_pos = self.get_current_dogfight_square()
        if dogfight_pos is None:
            raise RuntimeError("No dogfights remaining")

        row, col = dogfight_pos

        # v1.3: Showdown - reveal any face-down cards before dogfight resolution
        square = self.state.get_square(row, col)
        if len(square.rocketmen) == 2:
            revealed_rocketmen = []
            for rm in square.rocketmen:
                if rm.face_down:
                    # Rocketman is frozen dataclass, must create new instance
                    revealed = Rocketman(player=rm.player, power=rm.power, face_down=False)
                    revealed_rocketmen.append(revealed)
                else:
                    revealed_rocketmen.append(rm)
            square.rocketmen = revealed_rocketmen

        underdog = self._get_underdog_at_position(row, col)
        other = Player.TWO if underdog == Player.ONE else Player.ONE

        self.current_dogfight = DogfightTurnState(
            position=dogfight_pos,
            underdog=underdog,
            other=other,
            current_actor=underdog,  # Underdog acts first
            complete=False
        )

        # v1.4+: Expose dogfight context to agents for strategic decisions
        self.state.dogfight_context = DogfightContext(
            position=dogfight_pos,
            underdog=underdog,
            other=other,
            rocket_in_play=None  # No rocket yet at start
        )

    def get_dogfight_current_actor(self) -> Player:
        """Get the player whose turn it is in the current dogfight."""
        if self.current_dogfight is None:
            raise RuntimeError("No active dogfight - call begin_current_dogfight() first")
        if self.current_dogfight.complete:
            raise RuntimeError("Dogfight is complete - call finish_current_dogfight()")
        assert self.current_dogfight.current_actor is not None
        return self.current_dogfight.current_actor

    def get_dogfight_legal_actions_for_player(self, player: Player) -> list[int]:
        """
        Get legal actions for a player in the current dogfight turn.

        Returns:
            List of legal action indices based on dogfight state
        """
        if self.current_dogfight is None:
            raise RuntimeError("No active dogfight")
        if player != self.current_dogfight.current_actor:
            raise RuntimeError(f"Not {player}'s turn")

        resources = self.state.get_resources(player)
        legal = []

        # v1.4: All weapons are always legal - context determines role
        # (offensive/defensive) when resolving
        for i, action in enumerate(self.action_space.actions):
            if action.action_type == ActionType.PLAY_WEAPON:
                assert action.card_index is not None
                if action.card_index < len(resources.weapons):
                    legal.append(i)

        # Pass always legal
        for i, action in enumerate(self.action_space.actions):
            if action.action_type == ActionType.PASS:
                legal.append(i)

        return legal

    def apply_dogfight_turn_action(self, player: Player, action_index: int):
        """
        Apply an action for the current player's turn in the dogfight.

        Advances the dogfight state and determines next actor.
        """
        if self.current_dogfight is None:
            raise RuntimeError("No active dogfight")
        if player != self.current_dogfight.current_actor:
            raise RuntimeError(f"Not {player}'s turn")

        action = self.action_space.get_action(action_index)

        # Record action
        self.action_history.append((player, action_index))

        df = self.current_dogfight

        # Apply based on whose turn it is
        if df.underdog_action is None:
            # Round 1: Underdog's first action
            df.underdog_action = action

            if action.action_type == ActionType.PLAY_WEAPON:
                # v1.4: Underdog played weapon (offensive role - Rocket)
                df.rocket_in_play = df.underdog
                df.current_actor = df.other
                # Update context for agents
                self.state.dogfight_context = DogfightContext(
                    position=df.position,
                    underdog=df.underdog,
                    other=df.other,
                    rocket_in_play=df.rocket_in_play
                )
            elif action.action_type == ActionType.PASS:
                # Underdog passed - other player can play weapon or pass
                df.current_actor = df.other
            else:
                raise RuntimeError(f"Invalid first action: {action.action_type}")

        elif df.other_action is None:
            # Round 2: Other player's action
            df.other_action = action

            if df.rocket_in_play == df.underdog:
                # Responding to underdog's weapon (defensive role - Flare)
                # Dogfight is complete
                df.complete = True
            elif action.action_type == ActionType.PLAY_WEAPON:
                # v1.4: Other player played weapon (offensive role - Rocket)
                df.rocket_in_play = df.other
                df.current_actor = df.underdog  # Underdog can now respond
                # Update context for agents
                self.state.dogfight_context = DogfightContext(
                    position=df.position,
                    underdog=df.underdog,
                    other=df.other,
                    rocket_in_play=df.rocket_in_play
                )
            else:
                # Other player passed too - dogfight is complete
                df.complete = True

        else:
            # Round 3: Underdog's counter-response to other player's weapon
            df.underdog_second_action = action
            df.complete = True

    def is_dogfight_complete(self) -> bool:
        """Check if the current dogfight has all actions collected."""
        if self.current_dogfight is None:
            return False
        return self.current_dogfight.complete

    def finish_current_dogfight(self) -> DogfightResult:
        """
        Resolve the current dogfight and advance to the next one.

        Must be called after dogfight is complete (all actions collected).

        Returns:
            DogfightResult with winner, eliminated players, and Kaos draws
        """
        if self.current_dogfight is None:
            raise RuntimeError("No active dogfight")
        if not self.current_dogfight.complete:
            raise RuntimeError("Dogfight not complete - collect all actions first")

        df = self.current_dogfight

        # Underdog always has an action (Round 1)
        assert df.underdog_action is not None

        # Determine final actions for each player
        if df.underdog == Player.ONE:
            p1_action = df.underdog_action
            p2_action = df.other_action if df.other_action is not None else Action(ActionType.PASS)
        else:
            p1_action = df.other_action if df.other_action is not None else Action(ActionType.PASS)
            p2_action = df.underdog_action

        # Handle underdog's second action (counter-flare)
        if df.underdog_second_action is not None:
            if df.underdog == Player.ONE:
                p1_action = df.underdog_second_action
            else:
                p2_action = df.underdog_second_action

        # Resolve dogfight using existing logic (pass turn state for context)
        result = self._resolve_dogfight(df.position, p1_action, p2_action, df)

        # Apply result: remove eliminated rocketmen
        square = self.state.get_square(df.position[0], df.position[1])
        if Player.ONE in result.eliminated:
            square.rocketmen = [rm for rm in square.rocketmen if rm.player != Player.ONE]
        if Player.TWO in result.eliminated:
            square.rocketmen = [rm for rm in square.rocketmen if rm.player != Player.TWO]

        # Clear dogfight state
        self.current_dogfight = None
        self.state.dogfight_context = None  # Clear context

        # Advance to next dogfight
        self.state.current_dogfight_index += 1

        # v1.8: Check for 3-in-a-row after EACH dogfight (first to achieve wins)
        if self.state.check_three_in_row(Player.ONE):
            self.state.winner = Player.ONE
            self.state.game_over = True
            self.state.phase = Phase.ENDED
            return result
        elif self.state.check_three_in_row(Player.TWO):
            self.state.winner = Player.TWO
            self.state.game_over = True
            self.state.phase = Phase.ENDED
            return result

        # Check if all dogfights are done
        if self.state.current_dogfight_index >= len(self.state.dogfight_order):
            self._check_game_end()

        return result

    def apply_dogfight_actions(self, p1_action_index: int, p2_action_index: int):
        """
        Apply dogfight actions for both players simultaneously.

        Args:
            p1_action_index: Action index for player 1
            p2_action_index: Action index for player 2
        """
        if self.state.phase != Phase.DOGFIGHTS:
            raise RuntimeError("Not in dogfight phase")

        dogfight_pos = self.get_current_dogfight_square()
        if dogfight_pos is None:
            raise RuntimeError("No dogfights remaining")

        # Get actions
        p1_action = self.action_space.get_action(p1_action_index)
        p2_action = self.action_space.get_action(p2_action_index)

        # Record actions
        self.action_history.append((Player.ONE, p1_action_index))
        self.action_history.append((Player.TWO, p2_action_index))

        # Resolve dogfight
        result = self._resolve_dogfight(dogfight_pos, p1_action, p2_action)

        # Apply result: remove eliminated rocketmen
        square = self.state.get_square(dogfight_pos[0], dogfight_pos[1])
        if Player.ONE in result.eliminated:
            square.rocketmen = [rm for rm in square.rocketmen if rm.player != Player.ONE]
        if Player.TWO in result.eliminated:
            square.rocketmen = [rm for rm in square.rocketmen if rm.player != Player.TWO]

        # Advance to next dogfight
        self.state.current_dogfight_index += 1

        # Check if all dogfights are done
        if self.state.current_dogfight_index >= len(self.state.dogfight_order):
            self._check_game_end()

    def _resolve_dogfight(
        self,
        pos: tuple[int, int],
        p1_action: Action,
        p2_action: Action,
        dogfight_state: 'DogfightTurnState | None' = None
    ) -> DogfightResult:
        """
        Resolve a dogfight at the given position.

        Implements the full dogfight resolution rules from utala-kaos-9.md.
        For v1.4, infers weapon roles (Rocket vs Flare) from turn sequence context.
        """
        result = DogfightResult(
            winner=None,
            eliminated=[],
            kaos_draws={Player.ONE: [], Player.TWO: []}
        )

        square = self.state.get_square(pos[0], pos[1])
        p1_rocketman = next(rm for rm in square.rocketmen if rm.player == Player.ONE)
        p2_rocketman = next(rm for rm in square.rocketmen if rm.player == Player.TWO)

        # v1.4: Infer weapon roles from turn sequence context
        p1_is_rocket = False
        p2_is_rocket = False
        p1_is_flare = False
        p2_is_flare = False

        if dogfight_state is not None:
            # Determine roles based on who acted first and what they did
            df = dogfight_state
            if df.underdog == Player.ONE:
                # P1 acted first (underdog)
                if p1_action.action_type == ActionType.PLAY_WEAPON:
                    p1_is_rocket = True  # First weapon = offensive (Rocket)
                if p2_action.action_type == ActionType.PLAY_WEAPON:
                    if p1_is_rocket:
                        p2_is_flare = True  # Response to rocket = defensive (Flare)
                    else:
                        p2_is_rocket = True  # No rocket to respond to = offensive
            else:
                # P2 acted first (underdog)
                if p2_action.action_type == ActionType.PLAY_WEAPON:
                    p2_is_rocket = True  # First weapon = offensive (Rocket)
                if p1_action.action_type == ActionType.PLAY_WEAPON:
                    if p2_is_rocket:
                        p1_is_flare = True  # Response to rocket = defensive (Flare)
                    else:
                        p1_is_rocket = True  # No rocket to respond to = offensive

        # Apply weapon cards
        self._apply_cards(p1_action, p2_action)

        # Step 2: Weapon Interaction
        rocket_ended_dogfight = False

        # Case 1: Both play rockets
        if p1_is_rocket and p2_is_rocket:
            k1 = self._draw_kaos(Player.ONE)
            k2 = self._draw_kaos(Player.TWO)
            result.kaos_draws[Player.ONE].append(k1)
            result.kaos_draws[Player.TWO].append(k2)

            if k1 > k2:
                result.eliminated.append(Player.TWO)
                result.winner = Player.ONE
            elif k2 > k1:
                result.eliminated.append(Player.ONE)
                result.winner = Player.TWO
            else:  # Tie: both eliminated
                result.eliminated.extend([Player.ONE, Player.TWO])
                result.winner = None

            rocket_ended_dogfight = True

        # Case 2: Rocket vs Flare - cancel out
        elif (p1_is_rocket and p2_is_flare) or (p2_is_rocket and p1_is_flare):
            # Rocket and flare cancel out, proceed to Kaos resolution
            pass

        # Case 3: P1 plays rocket, P2 passes
        elif p1_is_rocket and p2_action.action_type == ActionType.PASS:
            k1 = self._draw_kaos(Player.ONE)
            result.kaos_draws[Player.ONE].append(k1)

            if k1 >= 7:  # Hit (7 or higher)
                result.eliminated.append(Player.TWO)
                result.winner = Player.ONE
                rocket_ended_dogfight = True
            # else: Miss (6 or lower), proceed to Kaos resolution

        # Case 4: P2 plays rocket, P1 passes
        elif p2_is_rocket and p1_action.action_type == ActionType.PASS:
            k2 = self._draw_kaos(Player.TWO)
            result.kaos_draws[Player.TWO].append(k2)

            if k2 >= 7:  # Hit (7 or higher)
                result.eliminated.append(Player.ONE)
                result.winner = Player.TWO
                rocket_ended_dogfight = True
            # else: Miss (6 or lower), proceed to Kaos resolution

        # Step 3: Kaos Resolution (if dogfight continues)
        if not rocket_ended_dogfight:
            k1 = self._draw_kaos(Player.ONE)
            k2 = self._draw_kaos(Player.TWO)
            result.kaos_draws[Player.ONE].append(k1)
            result.kaos_draws[Player.TWO].append(k2)

            total1 = p1_rocketman.power + k1
            total2 = p2_rocketman.power + k2

            if total1 > total2:
                result.eliminated.append(Player.TWO)
                result.winner = Player.ONE
            elif total2 > total1:
                result.eliminated.append(Player.ONE)
                result.winner = Player.TWO
            else:  # Tie: both eliminated
                result.eliminated.extend([Player.ONE, Player.TWO])
                result.winner = None

        return result

    def _apply_cards(self, p1_action: Action, p2_action: Action):
        """Remove played cards from player resources."""
        p1_res = self.state.get_resources(Player.ONE)
        p2_res = self.state.get_resources(Player.TWO)

        # v1.4: All weapons from unified list
        if p1_action.action_type == ActionType.PLAY_WEAPON:
            assert p1_action.card_index is not None
            p1_res.weapons.pop(p1_action.card_index)

        if p2_action.action_type == ActionType.PLAY_WEAPON:
            assert p2_action.card_index is not None
            p2_res.weapons.pop(p2_action.card_index)

    def _draw_kaos(self, player: Player) -> int:
        """
        Draw a Kaos card for a player.

        Returns the value of the drawn card.
        """
        resources = self.state.get_resources(player)

        # If deck is empty, reshuffle discard pile
        if len(resources.kaos_deck) == 0:
            resources.kaos_deck = resources.kaos_discard.copy()
            resources.kaos_discard.clear()
            self.rng.shuffle(resources.kaos_deck)

        # Draw card
        card = resources.kaos_deck.pop(0)
        resources.kaos_discard.append(card)

        return card

    def _check_game_end(self):
        """Check if game is over and determine winner."""
        # v1.8: Check BOTH players for 3-in-a-row before determining winner
        p1_has_three = self.state.check_three_in_row(Player.ONE)
        p2_has_three = self.state.check_three_in_row(Player.TWO)

        if p1_has_three and p2_has_three:
            # v1.8: Both achieved 3-in-a-row simultaneously - joker holder wins
            self.state.winner = self.state.joker_holder
            self.state.game_over = True
            self.state.phase = Phase.ENDED
            return
        elif p1_has_three:
            self.state.winner = Player.ONE
            self.state.game_over = True
            self.state.phase = Phase.ENDED
            return
        elif p2_has_three:
            self.state.winner = Player.TWO
            self.state.game_over = True
            self.state.phase = Phase.ENDED
            return

        # Check for most squares controlled
        p1_squares = self.state.count_controlled_squares(Player.ONE)
        p2_squares = self.state.count_controlled_squares(Player.TWO)

        if p1_squares > p2_squares:
            self.state.winner = Player.ONE
        elif p2_squares > p1_squares:
            self.state.winner = Player.TWO
        else:
            self.state.winner = None  # Draw

        self.state.game_over = True
        self.state.phase = Phase.ENDED

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.state.game_over

    def get_winner(self) -> Player | None:
        """Get the winner, or None if game is a draw or not over."""
        return self.state.winner if self.state.game_over else None
