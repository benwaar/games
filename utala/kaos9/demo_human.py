#!/usr/bin/env python3
"""
Demo: Play utala: kaos 9 as a human against an AI agent.

This demo lets you play as either Player 1 or Player 2 against
a computer opponent (Heuristic, Monte Carlo, or Random).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'src')

from utala.actions import get_action_space
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.human_agent import HumanAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.agents.random_agent import RandomAgent
from utala.engine import GameEngine
from utala.replays.format import ReplayMetadata, ReplayV1
from utala.state import Phase, Player

SAVE_FILE = Path("saved_game.json")


def save_game(engine: GameEngine, human_player: Player, ai_agent_type: str, agents: dict):
    """Save current game state to file."""
    # Convert action history to replay format
    actions = [(p.value, action_idx) for p, action_idx in engine.action_history]

    metadata = ReplayMetadata(
        format_version="v1",
        rules_version="1.8",
        game_variant="level1",
        player_one_name=agents[Player.ONE].name,
        player_two_name=agents[Player.TWO].name,
        timestamp=datetime.now().isoformat()
    )

    replay = ReplayV1(
        seed=engine.seed,
        actions=actions,
        metadata=metadata,
        winner=None  # Game in progress
    )

    # Add extra info for restoration
    save_data = replay.to_dict()
    save_data["human_player"] = human_player.value
    save_data["ai_agent_type"] = ai_agent_type

    SAVE_FILE.write_text(json.dumps(save_data, indent=2))
    print(f"\n✓ Game saved to {SAVE_FILE}")


def load_game():
    """Load saved game if it exists."""
    if not SAVE_FILE.exists():
        return None

    try:
        import json
        save_data = json.loads(SAVE_FILE.read_text())
        return save_data
    except Exception as e:
        print(f"Error loading saved game: {e}")
        return None


def select_action_with_save(
    human: HumanAgent,
    engine: GameEngine,
    legal_actions: list[int],
    current_player: Player,
    human_player: Player,
    ai_agent_type: str,
    agents: dict
) -> int | None:
    """
    Let human select action with option to save and exit.
    Returns action_idx or None if user chose to save and exit.
    """
    # Display state and actions manually, then prompt for input with save option
    state = engine.get_state_copy()
    # Pass dogfight info if available for better context
    dogfight_info = engine.current_dogfight if engine.state.phase == Phase.DOGFIGHTS else None
    human._display_state(state, current_player, dogfight_info)

    # Display legal actions (consolidate by type for dogfights)
    print("\nLegal actions:")
    print("-" * 60)

    seen_types = set()
    display_options = []

    for action_idx in legal_actions:
        action = human.action_space.get_action(action_idx)
        action_type = action.action_type.name

        # For dogfight actions, only show first of each type
        if action_type in ["PLAY_WEAPON", "PASS"]:
            if action_type not in seen_types:
                seen_types.add(action_type)
                display_options.append((action_idx, action_type))
        else:
            # For placement actions, show all
            display_options.append((action_idx, str(action)))

    # Display consolidated options
    for i, (_action_idx, display_text) in enumerate(display_options):
        if display_text == "PLAY_WEAPON":
            print(f"  {i}: Attack")
        elif display_text == "PASS":
            print(f"  {i}: Pass")
        else:
            print(f"  {i}: {display_text}")

    print(" 99: Save and exit")

    # Prompt for selection
    while True:
        try:
            choice = input(f"\n{human.name}, select action [0-{len(display_options)-1}, or 99]: ").strip()

            # Check for save request
            if choice == "99":
                save_game(engine, human_player, ai_agent_type, agents)
                print("Exiting game...")
                return None

            idx = int(choice)
            if 0 <= idx < len(display_options):
                return display_options[idx][0]  # Return the actual action_idx
            print(f"Invalid choice. Please enter a number between 0 and {len(display_options)-1}, or 99")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            raise


def restore_engine(save_data: dict) -> GameEngine:
    """Restore engine state by replaying actions."""
    seed = save_data["seed"]
    actions = save_data["actions"]

    engine = GameEngine(seed=seed)

    # Replay all actions to restore state
    for player_id, action_idx in actions:
        if engine.state.phase == Phase.DOGFIGHTS:
            # In dogfights, apply turn-based actions
            if not engine.current_dogfight:
                engine.begin_current_dogfight()

            current_actor = engine.get_dogfight_current_actor()
            engine.apply_dogfight_turn_action(current_actor, action_idx)

            if engine.is_dogfight_complete():
                engine.finish_current_dogfight()
        else:
            # Placement phase
            engine.apply_action(action_idx)

    return engine


def main():
    """Run human vs AI demo."""
    print("=" * 60)
    print("utala: kaos 9 v1.8 - Human vs AI")
    print("Face-down cards (2,3,9,10) shown as ??")
    print("Joker token for equal-power dogfights")
    print("=" * 60)
    print()

    # Check for saved game
    saved_game = load_game()
    ai_agent = None
    human_player = None
    ai_player = None
    engine = None
    seed = None
    ai_agent_type = None

    if saved_game:
        print("📁 Found saved game!")
        print(f"   {saved_game['metadata']['player_one_name']} vs {saved_game['metadata']['player_two_name']}")
        print(f"   Saved: {saved_game['metadata'].get('timestamp', 'unknown')}")
        print(f"   Actions played: {len(saved_game['actions'])}")
        print()
        print("  1: Continue saved game")
        print("  2: Start new game")

        while True:
            try:
                choice = input("\nChoice [1-2]: ").strip()
                if choice == "1":
                    # Restore game
                    print("\n📂 Loading saved game...")
                    engine = restore_engine(saved_game)
                    seed = saved_game["seed"]
                    human_player = Player(saved_game["human_player"])
                    ai_player = human_player.opponent()
                    ai_agent_type = saved_game["ai_agent_type"]

                    # Recreate AI agent
                    if ai_agent_type == "random":
                        ai_agent = RandomAgent("AI-Random")
                    elif ai_agent_type == "heuristic":
                        ai_agent = HeuristicAgent("AI-Heuristic")
                    elif ai_agent_type == "mc-fast":
                        ai_agent = FastMonteCarloAgent("AI-MC-Fast")
                    elif ai_agent_type == "mc-ultra":
                        from utala.agents.monte_carlo_agent import UltraStrongMonteCarloAgent
                        ai_agent = UltraStrongMonteCarloAgent("AI-MC-Ultra")

                    print(f"✓ Game restored! Phase: {engine.state.phase.name}, Turn: {engine.state.turn_number}")
                    break
                elif choice == "2":
                    # Delete saved game and start fresh
                    SAVE_FILE.unlink()
                    print("\n🗑️  Saved game deleted. Starting new game...")
                    break
                print("Invalid choice. Please enter 1 or 2")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                return

    # Select opponent (if not loaded from save)
    if ai_agent is None:
        print("Select your opponent:")
        print("  1: Random (easiest)")
        print("  2: Heuristic (medium - 65% vs Random)")
        print("  3: Monte Carlo Fast (hard - 79% vs Random, strategic)")
        print("  4: Monte Carlo Ultra (very hard - 50 rollouts)")

        while True:
            try:
                choice = input("\nOpponent [1-4]: ").strip()
                if choice == "1":
                    ai_agent = RandomAgent("AI-Random")
                    ai_agent_type = "random"
                    break
                elif choice == "2":
                    ai_agent = HeuristicAgent("AI-Heuristic")
                    ai_agent_type = "heuristic"
                    break
                elif choice == "3":
                    ai_agent = FastMonteCarloAgent("AI-MC-Fast")
                    ai_agent_type = "mc-fast"
                    break
                elif choice == "4":
                    from utala.agents.monte_carlo_agent import UltraStrongMonteCarloAgent
                    ai_agent = UltraStrongMonteCarloAgent("AI-MC-Ultra")
                    ai_agent_type = "mc-ultra"
                    break
                print("Invalid choice. Please enter 1, 2, 3, or 4")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                return

        # Select player side
        print("\nSelect your player:")
        print("  1: Player 1 (X) - moves first")
        print("  2: Player 2 (O) - moves second")

        while True:
            try:
                choice = input("\nPlayer [1-2]: ").strip()
                if choice == "1":
                    human_player = Player.ONE
                    ai_player = Player.TWO
                    break
                elif choice == "2":
                    human_player = Player.TWO
                    ai_player = Player.ONE
                    break
                print("Invalid choice. Please enter 1 or 2")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                return

    # Create agents
    human = HumanAgent("You")
    agents = {
        human_player: human,
        ai_player: ai_agent
    }

    # Initialize engine (if not loaded)
    if engine is None:
        seed = 42  # Fixed seed for reproducibility
        engine = GameEngine(seed=seed)

        # Notify agents
        agents[Player.ONE].game_start(Player.ONE, seed)
        agents[Player.TWO].game_start(Player.TWO, seed)
    else:
        # Loaded from save - notify agents
        agents[Player.ONE].game_start(Player.ONE, seed)
        agents[Player.TWO].game_start(Player.TWO, seed)

    print(f"\nStarting game: {agents[Player.ONE].name} (X) vs {agents[Player.TWO].name} (O)")
    print("=" * 60)

    # Game loop
    while not engine.is_game_over():
        # Handle dogfight phase with turn-based action/reaction
        if engine.state.phase == Phase.DOGFIGHTS:
            # Start the dogfight (determines underdog)
            engine.begin_current_dogfight()

            # Show dogfight setup
            df = engine.current_dogfight
            if df:
                pos = df.position
                square = engine.state.get_square(pos[0], pos[1])
                if len(square.rocketmen) == 2:
                    rm1, rm2 = square.rocketmen
                    underdog_rm = rm1 if rm1.player == df.underdog else rm2
                    other_rm = rm2 if rm1.player == df.underdog else rm1
                    print(f"\n🥊 Dogfight at [{pos[0]},{pos[1]}]:")
                    print(f"   {agents[df.underdog].name} (underdog, power {underdog_rm.power}) acts first")
                    print(f"   vs {agents[df.other].name} (power {other_rm.power})")

            # Collect actions turn by turn until dogfight is complete
            while not engine.is_dogfight_complete():
                current_player = engine.get_dogfight_current_actor()
                agent = agents[current_player]

                # Show what's happened so far
                df = engine.current_dogfight
                opponent_acted = False
                if df:
                    # Show underdog's action if taken and we're not the underdog
                    if df.underdog_action and current_player != df.underdog:
                        print(f"\n{'='*60}")
                        print(f"➤ {agents[df.underdog].name} (underdog) played: {df.underdog_action}")
                        print(f"{'='*60}")
                        opponent_acted = True

                    # Show other player's action if taken and we're not the other player
                    if df.other_action and current_player != df.other:
                        print(f"\n{'='*60}")
                        print(f"➤ {agents[df.other].name} played: {df.other_action}")
                        print(f"{'='*60}")
                        opponent_acted = True

                    # Emphasize if there's a rocket in play and we didn't fire it
                    if df.rocket_in_play is not None and current_player != df.rocket_in_play:
                        # Set flag for human agent to show "Defend" option
                        if agent == human:
                            human.rocket_in_play = True
                            print(f"\n⚠️  ROCKET ALERT!")
                            print("  0: Defend")
                            print("  1: Pass")
                        else:
                            print(f"⚠️  ROCKET ALERT! AI must defend or pass")

                # If human is about to act and opponent just acted, pause for acknowledgment
                # (unless responding to rocket, in which case options are already shown)
                if agent == human and opponent_acted and not human.rocket_in_play:
                    input("\nPress Enter to continue...")

                # Get legal actions for this turn
                legal_actions = engine.get_dogfight_legal_actions_for_player(current_player)

                # Agent selects action
                if agent == human:
                    # Auto-pass if PASS is the only legal action
                    if len(legal_actions) == 1:
                        action_space = get_action_space()
                        action = action_space.get_action(legal_actions[0])
                        if action.action_type.name == "PASS":
                            print("\n(Auto-passing - no weapons available)")
                            action_idx = legal_actions[0]
                        else:
                            # Single non-pass action, still prompt
                            action_idx = select_action_with_save(
                                human,
                                engine,
                                legal_actions,
                                current_player,
                                human_player,
                                ai_agent_type,
                                agents
                            )
                            if action_idx is None:
                                return
                    # If responding to rocket, just get input directly (options already shown)
                    elif human.rocket_in_play:
                        while True:
                            try:
                                choice = input(f"\n{human.name}, select action [0-1, or 99 to save & exit]: ").strip()
                                if choice == "99":
                                    save_game(engine, human_player, ai_agent_type, agents)
                                    print("Exiting game...")
                                    return
                                idx = int(choice)
                                if 0 <= idx <= 1:
                                    action_idx = legal_actions[idx]
                                    human.rocket_in_play = False
                                    break
                                print("Invalid choice. Please enter 0, 1, or 99")
                            except (ValueError, EOFError, KeyboardInterrupt):
                                print("\nInvalid input. Please enter 0, 1, or 99.")
                    else:
                        # Use modified select_action that supports save
                        action_idx = select_action_with_save(
                            human,
                            engine,
                            legal_actions,
                            current_player,
                            human_player,
                            ai_agent_type,
                            agents
                        )
                        if action_idx is None:  # User chose to save and exit
                            return
                else:
                    print(f"\n{agent.name} is thinking...")
                    action_idx = agent.select_action(
                        engine.get_state_copy(),
                        legal_actions,
                        current_player
                    )
                    action = get_action_space().get_action(action_idx)
                    print(f"{agent.name} plays: {action}")

                # Apply the action to the dogfight
                engine.apply_dogfight_turn_action(current_player, action_idx)

            # Dogfight complete - capture state before resolution
            df = engine.current_dogfight
            if df:
                # Store pre-resolution state
                pos = df.position
                pre_resolution_square = engine.state.get_square(pos[0], pos[1])
                pre_rocketmen = list(pre_resolution_square.rocketmen)
                rm1, rm2 = pre_rocketmen
                underdog_rm = rm1 if rm1.player == df.underdog else rm2
                other_rm = rm2 if rm1.player == df.underdog else rm1

                # Store actions for display
                underdog_action = df.underdog_action
                other_action = df.other_action
                underdog_second_action = df.underdog_second_action

            # Resolve the dogfight and get result
            result = engine.finish_current_dogfight()

            # Display resolution summary
            if df:
                print("\n" + "-" * 60)
                print("📋 DOGFIGHT RESOLUTION")
                print("-" * 60)

                # Show what each player did
                print(f"\n{agents[df.underdog].name} (underdog): {underdog_action}")
                if underdog_second_action:
                    print(f"  → Then: {underdog_second_action}")
                print(f"{agents[df.other].name}: {other_action}")

                # Display Kaos draws and resolution details
                underdog_kaos = result.kaos_draws[df.underdog]
                other_kaos = result.kaos_draws[df.other]

                # Check if this was undefended rocket (one player has Kaos, other doesn't initially)
                underdog_action_type = underdog_action.action_type if underdog_action else None
                other_action_type = other_action.action_type if other_action else None

                # v1.4: Check for weapon vs weapon cancellation
                weapons_cancel = False
                if underdog_action_type and other_action_type:
                    if (underdog_action_type.name == "PLAY_WEAPON" and other_action_type.name == "PLAY_WEAPON"):
                        weapons_cancel = True
                        print("\n⚔️  Weapons cancel out (attack vs defense)! Proceeding to Kaos resolution...")

                print("\n🎲 Kaos Resolution:")

                # v1.4: Weapon without response case
                if (underdog_action_type and underdog_action_type.name == "PLAY_WEAPON" and
                    other_action_type and other_action_type.name == "PASS"):
                    # Underdog played weapon (attack), other passed
                    rocket_kaos = underdog_kaos[0]
                    print(f"   {agents[df.underdog].name} drew Kaos: {rocket_kaos}")
                    if rocket_kaos >= 7:
                        print(f"   💥 HIT! ({rocket_kaos} ≥ 7)")
                    else:
                        print(f"   ❌ MISS ({rocket_kaos} < 7) - continuing to Kaos resolution...")
                        if len(underdog_kaos) > 1:
                            print(f"   {agents[df.underdog].name} drew: {underdog_kaos[1]} (total: {underdog_rm.power}+{underdog_kaos[1]}={underdog_rm.power + underdog_kaos[1]})")
                            print(f"   {agents[df.other].name} drew: {other_kaos[0]} (total: {other_rm.power}+{other_kaos[0]}={other_rm.power + other_kaos[0]})")

                elif (other_action_type and other_action_type.name == "PLAY_WEAPON" and
                      underdog_action_type and underdog_action_type.name == "PASS"):
                    # Other played weapon (attack), underdog passed
                    rocket_kaos = other_kaos[0]
                    print(f"   {agents[df.other].name} drew Kaos: {rocket_kaos}")
                    if rocket_kaos >= 7:
                        print(f"   💥 HIT! ({rocket_kaos} ≥ 7)")
                    else:
                        print(f"   ❌ MISS ({rocket_kaos} < 7) - continuing to Kaos resolution...")
                        if len(other_kaos) > 1:
                            print(f"   {agents[df.underdog].name} drew: {underdog_kaos[0]} (total: {underdog_rm.power}+{underdog_kaos[0]}={underdog_rm.power + underdog_kaos[0]})")
                            print(f"   {agents[df.other].name} drew: {other_kaos[1]} (total: {other_rm.power}+{other_kaos[1]}={other_rm.power + other_kaos[1]})")

                # Other cases: both passed, or rocket vs flare
                elif len(underdog_kaos) > 0 or len(other_kaos) > 0:
                    # Show all Kaos draws
                    if underdog_kaos:
                        kaos_list = ", ".join(str(k) for k in underdog_kaos)
                        if len(underdog_kaos) > 1:
                            print(f"   {agents[df.underdog].name} drew Kaos: {kaos_list} (final: {underdog_kaos[-1]})")
                        else:
                            print(f"   {agents[df.underdog].name} drew Kaos: {kaos_list}")
                        print(f"     → Total power: {underdog_rm.power} + {underdog_kaos[-1]} = {underdog_rm.power + underdog_kaos[-1]}")

                    if other_kaos:
                        kaos_list = ", ".join(str(k) for k in other_kaos)
                        if len(other_kaos) > 1:
                            print(f"   {agents[df.other].name} drew Kaos: {kaos_list} (final: {other_kaos[-1]})")
                        else:
                            print(f"   {agents[df.other].name} drew Kaos: {kaos_list}")
                        print(f"     → Total power: {other_rm.power} + {other_kaos[-1]} = {other_rm.power + other_kaos[-1]}")

                # Check post-resolution state
                post_resolution_square = engine.state.get_square(pos[0], pos[1])
                post_rocketmen = list(post_resolution_square.rocketmen)

                # Determine outcome
                if len(post_rocketmen) == 0:
                    print("\n💀 Both rocketmen eliminated!")
                elif len(post_rocketmen) == 1:
                    survivor = post_rocketmen[0]
                    eliminated_players = [rm.player for rm in pre_rocketmen if rm.player != survivor.player]
                    if eliminated_players:
                        print(f"\n🎯 {agents[survivor.player].name} wins! {agents[eliminated_players[0]].name}'s rocketman eliminated.")
                    print(f"   Square [{pos[0]},{pos[1]}] now controlled by {agents[survivor.player].name} (power {survivor.power})")

                # Display updated board
                print("\n📍 Updated board:")
                for row in range(3):
                    row_str = []
                    for col in range(3):
                        square = engine.state.get_square(row, col)
                        if not square.rocketmen:
                            row_str.append("   .   ")
                        elif len(square.rocketmen) == 1:
                            rm = square.rocketmen[0]
                            owner = "X" if rm.player == Player.ONE else "O"
                            row_str.append(f"   {owner}   ")
                        else:
                            # Contested - show both rocketmen with values
                            rm1, rm2 = square.rocketmen
                            p1_rm = rm1 if rm1.player == Player.ONE else rm2
                            p2_rm = rm2 if rm1.player == Player.ONE else rm1
                            p1_val = str(p1_rm.power) if not p1_rm.face_down else "??"
                            p2_val = str(p2_rm.power) if not p2_rm.face_down else "??"
                            row_str.append(f"X:{p1_val}/O:{p2_val}")
                    print(f"  {row}: {'|'.join(row_str)}")

                print("\n" + "-" * 60)

                # Show clear winner summary
                if len(post_rocketmen) == 1:
                    winner_name = agents[post_rocketmen[0].player].name
                    print(f"\n🏆 DOGFIGHT WINNER: {winner_name}")
                elif len(post_rocketmen) == 0:
                    print(f"\n⚔️  DOGFIGHT RESULT: Both eliminated")

                # Show current score
                current_state = engine.get_state_copy()
                p1_squares = current_state.count_controlled_squares(Player.ONE)
                p2_squares = current_state.count_controlled_squares(Player.TWO)
                print(f"Current score: {agents[Player.ONE].name} {p1_squares} - {p2_squares} {agents[Player.TWO].name}")

                # Pause before next dogfight (if not game over)
                if not engine.is_game_over() and engine.state.phase == Phase.DOGFIGHTS:
                    input("\nPress Enter to continue to next dogfight...")
                    print()

        else:
            # Placement phase (turn-based)
            current_player = engine.state.current_player
            agent = agents[current_player]

            # Get legal actions
            legal_actions = engine.get_legal_actions()

            # Agent selects action
            if agent == human:
                # Auto-pass if PASS is the only legal action (rare but possible)
                if len(legal_actions) == 1:
                    action_space = get_action_space()
                    action = action_space.get_action(legal_actions[0])
                    if action.action_type.name == "PASS":
                        print("\n(Auto-passing - no legal placements)")
                        action_idx = legal_actions[0]
                    else:
                        action_idx = select_action_with_save(
                            human,
                            engine,
                            legal_actions,
                            current_player,
                            human_player,
                            ai_agent_type,
                            agents
                        )
                        if action_idx is None:
                            return
                else:
                    action_idx = select_action_with_save(
                        human,
                        engine,
                        legal_actions,
                        current_player,
                        human_player,
                        ai_agent_type,
                        agents
                    )
                    if action_idx is None:  # User chose to save and exit
                        return
            else:
                # For AI, show what it's thinking
                print(f"\n{agent.name} is thinking...")
                action_idx = agent.select_action(
                    engine.get_state_copy(),
                    legal_actions,
                    current_player
                )
                action = get_action_space().get_action(action_idx)
                print(f"{agent.name} plays: {action}")

            # Apply action
            if not engine.apply_action(action_idx):
                print(f"ERROR: Invalid action from {agent.name}")
                break

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)

    winner = engine.get_winner()
    final_state = engine.get_state_copy()

    # Display final board
    print("\nFinal board:")
    for row in range(3):
        row_str = []
        for col in range(3):
            square = final_state.get_square(row, col)
            if not square.rocketmen:
                row_str.append("   .   ")
            elif len(square.rocketmen) == 1:
                rm = square.rocketmen[0]
                owner = "X" if rm.player == Player.ONE else "O"
                row_str.append(f"   {owner}   ")
            else:
                # Contested - show both rocketmen with values
                rm1, rm2 = square.rocketmen
                p1_rm = rm1 if rm1.player == Player.ONE else rm2
                p2_rm = rm2 if rm1.player == Player.ONE else rm1
                p1_val = str(p1_rm.power) if not p1_rm.face_down else "??"
                p2_val = str(p2_rm.power) if not p2_rm.face_down else "??"
                row_str.append(f"X:{p1_val}/O:{p2_val}")
        print(f"  {row}: {'|'.join(row_str)}")

    # Announce winner
    print()
    if winner is None:
        print("Result: DRAW")
    elif winner == human_player:
        print("🎉 YOU WIN! Congratulations!")
    else:
        print("AI wins. Better luck next time!")

    # Display final score
    p1_score = final_state.count_controlled_squares(Player.ONE)
    p2_score = final_state.count_controlled_squares(Player.TWO)
    print(f"\nFinal score: Player 1 (X): {p1_score}, Player 2 (O): {p2_score}")

    # Notify agents
    agents[Player.ONE].game_end(final_state, winner)
    agents[Player.TWO].game_end(final_state, winner)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n\nGame interrupted. Exiting...")
        sys.exit(0)
