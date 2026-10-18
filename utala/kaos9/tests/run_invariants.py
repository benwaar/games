"""
Smoke runner: plays random games and checks all invariants after every state transition.

Usage:
    python -m tests.run_invariants --games 50
    python -m tests.run_invariants --games 50 --face-down 5
"""

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

# Add src to path (same as run_tests.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utala.agents.random_agent import RandomAgent
from utala.engine import GameEngine
from utala.state import GameConfig, Phase, Player

from tests.invariants import InvariantChecker


def run_invariant_game(seed: int, config: GameConfig, checker: InvariantChecker) -> list[str]:
    """Run a single game with invariant checking after every state transition."""
    engine = GameEngine(seed=seed, config=config)
    all_violations: list[str] = []

    agent_one = RandomAgent(seed=seed)
    agent_two = RandomAgent(seed=seed + 1)

    # Check initial state
    violations = checker.check_all(engine.state)
    if violations:
        all_violations.extend([f"[seed={seed} init] {v}" for v in violations])

    while not engine.is_game_over():
        prev_state = deepcopy(engine.state)

        if engine.state.phase == Phase.PLACEMENT:
            legal = engine.get_legal_actions()
            player = engine.state.current_player
            agent = agent_one if player == Player.ONE else agent_two
            state_copy = engine.get_state_copy()
            action = agent.select_action(state_copy, legal, player)
            engine.apply_action(action)

        elif engine.state.phase == Phase.DOGFIGHTS:
            if engine.state.awaiting_dogfight_choice:
                chooser = engine.state.dogfight_choice_player
                assert chooser is not None
                agent = agent_one if chooser == Player.ONE else agent_two
                legal = engine.get_legal_actions(chooser)
                state_copy = engine.get_state_copy()
                action = agent.select_action(state_copy, legal, chooser)
                engine.apply_dogfight_choice(action)
            else:
                engine.begin_current_dogfight()

                while not engine.is_dogfight_complete():
                    player = engine.get_dogfight_current_actor()
                    agent = agent_one if player == Player.ONE else agent_two
                    legal = engine.get_dogfight_legal_actions_for_player(player)
                    state_copy = engine.get_state_copy()
                    action = agent.select_action(state_copy, legal, player)
                    engine.apply_dogfight_turn_action(player, action)

                engine.finish_current_dogfight()

        # Check invariants after transition
        violations = checker.check_all(engine.state)
        if violations:
            all_violations.extend(
                [f"[seed={seed} turn={engine.state.turn_number}] {v}" for v in violations]
            )

        violations = checker.check_transition(prev_state, engine.state)
        if violations:
            all_violations.extend(
                [f"[seed={seed} transition] {v}" for v in violations]
            )

    # Check terminal invariants
    violations = checker.check_terminal(engine.state)
    if violations:
        all_violations.extend([f"[seed={seed} terminal] {v}" for v in violations])

    return all_violations


def main():
    parser = argparse.ArgumentParser(description="Run invariant checks on random games")
    parser.add_argument("--games", type=int, default=50, help="Number of games to run")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed")
    parser.add_argument("--face-down", type=int, default=None,
                        help="Number of face-down powers (4=default, 5, 6)")
    parser.add_argument("--choosable-order", action="store_true",
                        help="Enable Variant A: choosable dogfight order")
    args = parser.parse_args()

    # Build config
    kwargs: dict = {}

    if args.face_down is not None:
        face_down_tiers = {
            4: frozenset({2, 3, 9, 10}),
            5: frozenset({2, 3, 8, 9, 10}),
            6: frozenset({2, 3, 4, 8, 9, 10}),
        }
        if args.face_down not in face_down_tiers:
            print(f"Error: --face-down must be 4, 5, or 6 (got {args.face_down})")
            sys.exit(1)
        kwargs["face_down_powers"] = face_down_tiers[args.face_down]

    if args.choosable_order:
        kwargs["fixed_dogfight_order"] = False

    config = GameConfig(**kwargs)

    checker = InvariantChecker(config)

    print(f"Running {args.games} games with invariant checking...")
    print(f"  Config: face_down={sorted(config.face_down_powers)}, "
          f"kaos_deck={len(config.kaos_deck_values)} cards, "
          f"fixed_order={config.fixed_dogfight_order}")

    start = time.time()
    total_violations: list[str] = []

    for i in range(args.games):
        seed = args.seed + i
        violations = run_invariant_game(seed, config, checker)
        total_violations.extend(violations)

        if violations:
            print(f"  FAIL game {i + 1} (seed={seed}): {len(violations)} violations")

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Games: {args.games}  Time: {elapsed:.2f}s")

    if total_violations:
        print(f"VIOLATIONS: {len(total_violations)}")
        for v in total_violations[:20]:
            print(f"  - {v}")
        if len(total_violations) > 20:
            print(f"  ... and {len(total_violations) - 20} more")
        sys.exit(1)
    else:
        print("ALL INVARIANTS PASSED")


if __name__ == "__main__":
    main()
