"""
Monte Carlo baseline agent for utala: kaos 9 v1.8.

Rollout evaluation for placement and optionally dogfight decisions.
Supports information set sampling for proper partial observability handling.

v1.8: Compatible with joker tie-breaking and early 3-in-a-row wins.
"""

import random
from copy import deepcopy

from ..state import GameState, Phase, Player
from .base import Agent


class MonteCarloAgent(Agent):
    """
    Monte Carlo baseline for v1.8 with optional information set sampling.

    Uses random rollouts to evaluate actions in both phases:
    - Placement: Always evaluates via rollouts
    - Dogfights: Optionally evaluates via rollouts (if evaluate_dogfights=True)

    Information handling:
    - use_information_sets=False (default): Perfect information rollouts
      Uses deepcopy which sees face-down cards and opponent Kaos deck.
      Matches v1.8 baseline behavior.

    - use_information_sets=True: Information set sampling
      Samples possible face-down cards (2, 3, 9, 10) and Kaos deck orders
      consistent with observations. More realistic but slower.

    v1.8: Automatically respects joker tie-breaking and early 3-in-a-row wins
    through engine delegation.
    """

    def __init__(
        self,
        name: str = "MonteCarlo",
        num_rollouts: int = 50,
        seed: int | None = None,
        use_information_sets: bool = False,
        is_samples_per_rollout: int = 1,
        evaluate_dogfights: bool = False
    ):
        """
        Initialize Monte Carlo agent.

        Args:
            name: Agent name
            num_rollouts: Number of rollouts per placement action
            seed: Optional seed for deterministic behavior
            use_information_sets: If True, sample hidden information instead of using perfect info
            is_samples_per_rollout: Number of information set samples per rollout (if enabled)
            evaluate_dogfights: If True, evaluate dogfight actions via rollouts instead of random play
        """
        super().__init__(name)
        self.num_rollouts = num_rollouts
        self.use_information_sets = use_information_sets
        self.is_samples_per_rollout = is_samples_per_rollout
        self.evaluate_dogfights = evaluate_dogfights
        self.rng = random.Random(seed)

    def select_action(
        self,
        state: GameState,
        legal_actions: list[int],
        player: Player
    ) -> int:
        """
        Select action.

        Placement: evaluate via rollouts
        Dogfights: evaluate via rollouts (if enabled) or play randomly
        """
        if len(legal_actions) == 1:
            return legal_actions[0]

        # For dogfights
        if state.phase == Phase.DOGFIGHTS:
            if self.evaluate_dogfights:
                # Check if we're at the start of a dogfight (can evaluate)
                # or mid-dogfight (fall back to random)
                context = state.dogfight_context
                if context is None or context.rocket_in_play is not None:
                    # Mid-dogfight or no context - fall back to random
                    # (Preserving mid-dogfight state is complex, so we only
                    # evaluate the first decision in each dogfight)
                    return self.rng.choice(legal_actions)

                # Check if it's actually our turn (we must be the underdog acting first)
                # If opponent is underdog, they act first and we can't evaluate our action yet
                if context.underdog != player:
                    # Not our turn yet - fall back to random
                    return self.rng.choice(legal_actions)

                # Start of dogfight AND our turn: Evaluate each action via rollouts
                best_score = -1.0
                best_actions = []

                for action_idx in legal_actions:
                    score = self._evaluate_dogfight_action(state, action_idx, player)

                    if score > best_score:
                        best_score = score
                        best_actions = [action_idx]
                    elif score == best_score:
                        best_actions.append(action_idx)

                return self.rng.choice(best_actions)
            else:
                # Original: play randomly
                return self.rng.choice(legal_actions)

        # For placement, evaluate each action via rollouts
        best_score = -1.0
        best_actions = []

        for action_idx in legal_actions:
            score = self._evaluate_placement_action(state, action_idx, player)

            if score > best_score:
                best_score = score
                best_actions = [action_idx]
            elif score == best_score:
                best_actions.append(action_idx)

        return self.rng.choice(best_actions)

    def _sample_hidden_information(
        self,
        state: GameState,
        player: Player,
        sample_seed: int
    ) -> GameState:
        """
        Sample a possible state consistent with player's observations.

        Hidden information to sample:
        1. Face-down cards (2, 3, 9, 10): player doesn't know opponent's values
        2. Opponent's Kaos deck: player only knows discard pile

        Args:
            state: Current game state (may have perfect information)
            player: Player making decisions (whose perspective to sample from)
            sample_seed: RNG seed for this sample

        Returns:
            Sampled state with hidden information randomized
        """
        sampled_state = deepcopy(state)
        opponent = player.opponent()
        sample_rng = random.Random(sample_seed)

        # 1. Sample face-down card values
        # First, figure out which opponent cards are already known (not face-down)
        opp_resources = sampled_state.get_resources(opponent)
        known_powers = set(opp_resources.rocketmen)  # Cards still in hand

        # Find face-up opponent cards on grid
        for row in range(3):
            for col in range(3):
                square = sampled_state.get_square(row, col)
                for rm in square.rocketmen:
                    if rm.player == opponent and not rm.face_down:
                        known_powers.add(rm.power)

        # Available powers for face-down cards: {2,3,9,10} minus known
        face_down_options = [p for p in [2, 3, 9, 10] if p not in known_powers]

        # If no options (all face-down powers already visible), fall back to all
        if not face_down_options:
            face_down_options = [2, 3, 9, 10]

        # Now sample face-down cards
        for row in range(3):
            for col in range(3):
                square = sampled_state.get_square(row, col)
                for idx, rm in enumerate(square.rocketmen):
                    if rm.player == opponent and rm.face_down:
                        # Sample from available powers, avoiding duplicates
                        if face_down_options:
                            sampled_power = sample_rng.choice(face_down_options)
                            # Remove from options to avoid duplicates
                            face_down_options = [p for p in face_down_options
                                               if p != sampled_power]
                        else:
                            # Shouldn't happen, but fallback
                            sampled_power = sample_rng.choice([2, 3, 9, 10])

                        # Replace with sampled value (frozen dataclass, create new)
                        from ..state import Rocketman
                        square.rocketmen[idx] = Rocketman(
                            player=rm.player,
                            power=sampled_power,
                            face_down=True
                        )

        # 2. Sample opponent's Kaos deck order
        opp_resources = sampled_state.get_resources(opponent)

        # Known: cards in discard pile
        known_cards = set(opp_resources.kaos_discard)

        # Unknown: cards still in deck (1-13 minus discarded)
        all_kaos = list(range(1, 14))
        unknown_cards = [c for c in all_kaos if c not in known_cards]

        # IMPORTANT: Preserve the original deck size
        # We can only shuffle cards that are actually still in the deck
        original_deck_size = len(opp_resources.kaos_deck)

        # Reshuffle unknown cards and take only the deck size we need
        sample_rng.shuffle(unknown_cards)
        opp_resources.kaos_deck = unknown_cards[:original_deck_size]

        return sampled_state

    def _evaluate_placement_action(
        self,
        state: GameState,
        action_idx: int,
        player: Player
    ) -> float:
        """
        Evaluate a placement action via random rollouts.

        Returns win rate (0.0 to 1.0, draws count as 0.5).

        If use_information_sets=True, samples hidden information for each rollout.
        Otherwise uses perfect information (original v1.8 behavior).
        """
        from ..engine import GameEngine

        wins = 0
        draws = 0

        # If using information sets, do multiple samples per rollout
        samples = self.is_samples_per_rollout if self.use_information_sets else 1
        total_rollouts = self.num_rollouts * samples

        for rollout_idx in range(self.num_rollouts):
            for sample_idx in range(samples):
                # Create fresh engine from current state
                assert state.rng_seed is not None
                engine = GameEngine(seed=state.rng_seed)

                # Sample hidden information if enabled
                if self.use_information_sets:
                    sample_seed = (state.rng_seed + state.turn_number +
                                  rollout_idx * 1000 + sample_idx +
                                  self.rng.randint(0, 1_000_000))
                    sampled_state = self._sample_hidden_information(state, player, sample_seed)
                    engine.state = sampled_state
                else:
                    # Original behavior: perfect information
                    engine.state = deepcopy(state)

                # Reseed for this rollout
                rollout_seed = (state.rng_seed + state.turn_number +
                              self.rng.randint(0, 1_000_000))
                engine.rng = random.Random(rollout_seed)

                # Apply the action we're evaluating
                engine.apply_action(action_idx)

                # Random rollout to end
                winner = self._random_rollout(engine, player)

                if winner == player:
                    wins += 1
                elif winner is None:
                    draws += 1

        return (wins + draws * 0.5) / total_rollouts

    def _evaluate_dogfight_action(
        self,
        state: GameState,
        action_idx: int,
        actor: Player
    ) -> float:
        """
        Evaluate a dogfight weapon action via random rollouts.

        This is more complex than placement:
        - Action doesn't end game immediately
        - Must continue playing through dogfight completion
        - Value function: P(win game from resulting state)

        Args:
            state: Current game state (mid-dogfight)
            action_idx: Action to evaluate (PLAY_WEAPON or PASS)
            actor: Player making the decision

        Returns:
            Expected win rate (0.0 to 1.0)
        """
        from ..engine import GameEngine

        wins = 0
        draws = 0
        errors = 0

        # If using information sets, do multiple samples per rollout
        samples = self.is_samples_per_rollout if self.use_information_sets else 1
        total_rollouts = self.num_rollouts * samples

        for rollout_idx in range(self.num_rollouts):
            for sample_idx in range(samples):
                try:
                    # Create fresh engine from current state
                    assert state.rng_seed is not None
                    engine = GameEngine(seed=state.rng_seed)

                    # Sample hidden information if enabled
                    if self.use_information_sets:
                        sample_seed = (state.rng_seed + state.turn_number +
                                      rollout_idx * 1000 + sample_idx +
                                      self.rng.randint(0, 1_000_000))
                        sampled_state = self._sample_hidden_information(state, actor, sample_seed)
                        engine.state = sampled_state
                    else:
                        # Original behavior: perfect information
                        engine.state = deepcopy(state)

                    # Reseed for this rollout
                    rollout_seed = (state.rng_seed + state.turn_number +
                                  self.rng.randint(0, 1_000_000))
                    engine.rng = random.Random(rollout_seed)

                    # Begin dogfight if not already started
                    if engine.current_dogfight is None:
                        engine.begin_current_dogfight()

                    # Get the actual current actor from engine (might differ from our player
                    # if opponent is underdog and acts first)
                    current_actor = engine.get_dogfight_current_actor()

                    # Apply the dogfight action we're evaluating
                    engine.apply_dogfight_turn_action(current_actor, action_idx)

                    # Continue rollout: finish current dogfight, then play to end
                    winner = self._continue_dogfight_rollout(engine, actor)

                    if winner == actor:
                        wins += 1
                    elif winner is None:
                        draws += 1
                except Exception:
                    # State copying issues can cause inconsistencies
                    # Count as neutral outcome (0.5)
                    draws += 1
                    errors += 1

        # If too many errors, this evaluation is unreliable - return neutral
        if errors > total_rollouts * 0.5:
            return 0.5

        return (wins + draws * 0.5) / total_rollouts

    def _continue_dogfight_rollout(self, engine, original_player: Player) -> Player | None:
        """
        Continue playing from mid-dogfight to game end.

        Similar to _random_rollout but starts in dogfight phase.

        Args:
            engine: Game engine with partially-completed dogfight
            original_player: Player to evaluate win from perspective of

        Returns:
            Winner (Player | None for draw)
        """
        while not engine.is_game_over():
            state = engine.get_state_copy()

            if state.phase == Phase.PLACEMENT:
                # Should not happen (dogfights follow placement)
                legal = engine.get_legal_actions()
                action = self.rng.choice(legal)
                engine.apply_action(action)

            elif state.phase == Phase.DOGFIGHTS:
                # Continue current dogfight
                if engine.current_dogfight is None:
                    engine.begin_current_dogfight()

                # Play out remaining dogfight turns
                while not engine.is_dogfight_complete():
                    actor = engine.get_dogfight_current_actor()
                    legal = engine.get_dogfight_legal_actions_for_player(actor)
                    action = self.rng.choice(legal)
                    engine.apply_dogfight_turn_action(actor, action)

                # Finish and move to next dogfight
                engine.finish_current_dogfight()

        winner: Player | None = engine.get_winner()
        return winner

    def _random_rollout(self, engine, original_player: Player) -> Player | None:
        """
        Play randomly to game end.

        Returns winner.
        """
        while not engine.is_game_over():
            state = engine.get_state_copy()

            if state.phase == Phase.PLACEMENT:
                legal = engine.get_legal_actions()
                action = self.rng.choice(legal)
                engine.apply_action(action)

            elif state.phase == Phase.DOGFIGHTS:
                # Handle turn-based dogfights
                if engine.current_dogfight is None:
                    engine.begin_current_dogfight()

                while not engine.is_dogfight_complete():
                    actor = engine.get_dogfight_current_actor()
                    legal = engine.get_dogfight_legal_actions_for_player(actor)
                    action = self.rng.choice(legal)
                    engine.apply_dogfight_turn_action(actor, action)

                engine.finish_current_dogfight()

        winner: Player | None = engine.get_winner()
        return winner


class FastMonteCarloAgent(MonteCarloAgent):
    """Fast MC with 10 rollouts. Strategic placement + dogfight evaluation."""

    def __init__(
        self,
        name: str = "MonteCarlo-Fast",
        seed: int | None = None,
        use_information_sets: bool = False,
        evaluate_dogfights: bool = True  # Strategic dogfights by default
    ):
        super().__init__(
            name=name,
            num_rollouts=10,
            seed=seed,
            use_information_sets=use_information_sets,
            evaluate_dogfights=evaluate_dogfights
        )


class StrongMonteCarloAgent(MonteCarloAgent):
    """Strong MC with 20 rollouts."""

    def __init__(
        self,
        name: str = "MonteCarlo-Strong",
        seed: int | None = None,
        use_information_sets: bool = False,
        evaluate_dogfights: bool = False
    ):
        super().__init__(
            name=name,
            num_rollouts=20,
            seed=seed,
            use_information_sets=use_information_sets,
            evaluate_dogfights=evaluate_dogfights
        )


class VeryStrongMonteCarloAgent(MonteCarloAgent):
    """Very strong MC with 30 rollouts for deeper search."""

    def __init__(
        self,
        name: str = "MonteCarlo-VeryStrong",
        seed: int | None = None,
        use_information_sets: bool = False,
        evaluate_dogfights: bool = False
    ):
        super().__init__(
            name=name,
            num_rollouts=30,
            seed=seed,
            use_information_sets=use_information_sets,
            evaluate_dogfights=evaluate_dogfights
        )


class UltraStrongMonteCarloAgent(MonteCarloAgent):
    """Ultra strong MC with 50 rollouts for maximum search depth."""

    def __init__(
        self,
        name: str = "MonteCarlo-Ultra",
        seed: int | None = None,
        use_information_sets: bool = False,
        evaluate_dogfights: bool = False
    ):
        super().__init__(
            name=name,
            num_rollouts=50,
            seed=seed,
            use_information_sets=use_information_sets,
            evaluate_dogfights=evaluate_dogfights
        )
