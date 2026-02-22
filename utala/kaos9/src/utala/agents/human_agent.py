"""
Human agent for utala: kaos 9.

Allows a human player to select actions via text input.
"""

from ..actions import get_action_space
from ..state import GameState, Phase, Player
from .base import Agent


class HumanAgent(Agent):
    """Agent that prompts human for actions via text input."""

    def __init__(self, name: str = "Human"):
        """Initialize human agent."""
        super().__init__(name)
        self.action_space = get_action_space()
        self.player: Player | None = None
        self.rocket_in_play: bool = False  # Set by external context

    def game_start(self, player: Player, seed: int | None = None):
        """Store which player we are."""
        self.player = player

    def select_action(
        self,
        state: GameState,
        legal_actions: list[int],
        player: Player
    ) -> int:
        """Prompt human to select an action."""
        # Display current state
        self._display_state(state, player)

        # Display legal actions (consolidate by type for dogfights)
        print("\nLegal actions:")
        print("-" * 60)

        # For dogfight actions, consolidate by type
        seen_types = set()
        display_options = []

        for action_idx in legal_actions:
            action = self.action_space.get_action(action_idx)
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
                if self.rocket_in_play and state.phase == Phase.DOGFIGHTS:
                    print(f"  {i}: Defend")
                else:
                    print(f"  {i}: Attack")
            elif display_text == "PASS":
                print(f"  {i}: Pass")
            else:
                print(f"  {i}: {display_text}")

        # Prompt for selection
        while True:
            try:
                choice = input(f"\n{self.name}, select action [0-{len(display_options)-1}]: ").strip()
                idx = int(choice)
                if 0 <= idx < len(display_options):
                    selected_action = display_options[idx][0]
                    # Reset rocket flag after selection
                    self.rocket_in_play = False
                    return selected_action
                print(f"Invalid choice. Please enter a number between 0 and {len(display_options)-1}")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\nInvalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")

    def _display_state(self, state: GameState, player: Player, dogfight_info=None):
        """Display current game state with optional dogfight turn information."""
        print("\n" + "=" * 60)
        print(f"Turn {state.turn_number} - Phase: {state.phase.name}")
        print(f"You are Player {player.value + 1} {'(X)' if player == Player.ONE else '(O)'}")

        # Display joker token holder
        if hasattr(state, 'joker_holder') and state.joker_holder is not None:
            if state.joker_holder == player:
                print("🃏 Joker: YOU (tie-breaking advantage)")
            else:
                print("🃏 Joker: OPPONENT (tie-breaking advantage)")

        print("=" * 60)

        # Display board
        print("\nBoard (row,col):")
        for row in range(3):
            row_str = []
            for col in range(3):
                square = state.get_square(row, col)
                if not square.rocketmen:
                    row_str.append("   .   ")
                elif len(square.rocketmen) == 1:
                    # Won square - just show owner
                    rm = square.rocketmen[0]
                    owner = "X" if rm.player == Player.ONE else "O"
                    row_str.append(f"   {owner}   ")
                else:
                    # Contested - show both with values
                    rm1, rm2 = square.rocketmen
                    p1_rm = rm1 if rm1.player == Player.ONE else rm2
                    p2_rm = rm2 if rm1.player == Player.ONE else rm1
                    p1_val = str(p1_rm.power) if not p1_rm.face_down else "??"
                    p2_val = str(p2_rm.power) if not p2_rm.face_down else "??"
                    row_str.append(f"X:{p1_val}/O:{p2_val}")
            print(f"  {row}: {'|'.join(row_str)}")

        # Display resources
        resources = state.get_resources(player)
        opp_resources = state.get_resources(Player.ONE if player == Player.TWO else Player.TWO)

        print("\nYour resources:")
        print(f"  Rocketmen: {sorted(resources.rocketmen)}")
        # v1.4: Unified weapons list (all dual-purpose)
        print(f"  Weapons: {sorted(resources.weapons)}")
        print(f"  Kaos deck: {len(resources.kaos_deck)} cards, discard: {sorted(resources.kaos_discard)}")

        print("\nOpponent resources:")
        opp_weapon_count = len(opp_resources.weapons)
        print(f"  Weapons: {opp_weapon_count} cards")
        print(f"  Kaos deck: {len(opp_resources.kaos_deck)} cards, discard: {sorted(opp_resources.kaos_discard)}")

        # Additional context for dogfight phase
        if state.phase == Phase.DOGFIGHTS and state.current_dogfight_index < len(state.dogfight_order):
            row, col = state.dogfight_order[state.current_dogfight_index]
            square = state.get_square(row, col)
            print(f"\nCurrent dogfight at [{row},{col}]:")
            if len(square.rocketmen) == 2:
                rm1, rm2 = square.rocketmen

                # Find which is yours and which is opponent's based on player ID
                if rm1.player == player:
                    your_rm = rm1
                    opp_rm = rm2
                else:
                    your_rm = rm2
                    opp_rm = rm1

                # Determine who is underdog based on actual power values
                if your_rm.power < opp_rm.power:
                    underdog_player = player
                elif opp_rm.power < your_rm.power:
                    underdog_player = player.opponent()
                else:
                    # Equal power - check joker holder
                    underdog_player = state.joker_holder if state.joker_holder else Player.ONE

                your_power_display = str(your_rm.power) if not your_rm.face_down else "??"
                opp_power_display = str(opp_rm.power) if not opp_rm.face_down else "??"

                print(f"  Your rocketman: {your_power_display}")
                print(f"  Opponent rocketman: {opp_power_display}")
                print(f"  Underdog acts first: {'You' if underdog_player == player else 'Opponent'}")

                # Show opponent's action if they've already acted (from dogfight_info)
                if dogfight_info:
                    opponent = player.opponent()
                    # Check if opponent was underdog and has acted
                    if dogfight_info.underdog == opponent and dogfight_info.underdog_action:
                        print(f"  → Opponent played: {dogfight_info.underdog_action}")
                    # Check if opponent was other player and has acted
                    elif dogfight_info.other == opponent and dogfight_info.other_action:
                        print(f"  → Opponent played: {dogfight_info.other_action}")
