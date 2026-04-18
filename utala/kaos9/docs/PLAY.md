# How to Play utala: kaos 9 v1.8

## Quick Start

Run the interactive demo:
```bash
python demo_human.py
```

## Game Overview

**utala: kaos 9** is a tactical 3×3 grid combat card game for 2 players.

- **Placement Phase**: Take turns placing your 9 rocketmen (cards 2-10) on the 3×3 board
- **Dogfight Phase**: Resolve contested squares through turn-based combat
- **Win Condition**: Get 3 in a row, or control the most squares

## v1.8 Features

### Face-Down Cards (v1.3)
**Cards 2, 3, 9, and 10 are placed face-down** and shown as `??` during placement:
- Your own face-down cards: You know what they are (shown in your hand)
- Opponent's face-down cards: Hidden until dogfight showdown
- Face-up cards (4-8): Visible to both players

**Showdown**: When a dogfight begins, all cards at that square are revealed.

### Joker Token (v1.8)
**Equal-power dogfights** use the joker token:
- Physical token starts with one player (typically P2)
- When two Rocketmen have **equal power** in a dogfight, the **joker holder acts first**
- After use, joker passes to opponent (alternating advantage)
- Makes tie-breaking fair and trackable

### Immediate 3-in-a-Row Victory (v1.8)
**First player to achieve 3-in-a-row wins immediately**:
- Checked after EACH dogfight resolves (not just at game end)
- Standard tournament rule (like chess checkmate)
- Eliminates simultaneous wins

## Opponent Difficulty

1. **Random** (easiest) - Makes random legal moves
2. **Heuristic** (medium) - Uses strategic placement + dogfight tactics (65% vs Random)
3. **MC-Fast** (hard) - Monte Carlo with 10 rollouts, strategic dogfights (79% vs Random, 72% vs Heuristic)
4. **MC-Ultra** (very hard) - Monte Carlo with 50 rollouts (slower but stronger)

## How to Play

### Placement Phase
- The board shows: `X:power` (Player 1) and `O:power` (Player 2)
- Face-down cards show as `X:??` or `O:??`
- Choose which rocketman to place and where
- Squares can be contested (both players present)

Example board display:
```
  0:    .   | O:??  |   .
  1:  X:7   | X:??  | O:6
  2:    .   |  .    | X:5
```

### Dogfight Phase
Turn-based combat at each contested square:

1. **Underdog acts first** (weaker power, or joker holder if equal power)
   - Play a Weapon (any of your 4 weapons) as Rocket to attack, OR
   - Pass

2. **Opponent responds**
   - If Rocket played: Play Flare (Queen or Jack) to defend, OR Pass
   - If no Rocket: Play Rocket to attack, OR Pass

3. **Resolution**
   - **Undefended Rocket**: Draw Kaos card. ≥7 = hit (eliminate opponent)
   - **Rocket vs Flare**: Cancel out, go to Kaos resolution
   - **Both Pass**: Go to Kaos resolution
   - **Kaos Resolution**: Each draws card, add to power, highest total wins

### Strategy Tips

**Placement:**
- Center square (1,1) is most valuable (forms 4 potential lines)
- Edge squares (middle of sides) form 2-3 potential lines
- Corner squares form 2 potential lines
- Use face-down cards (2,3,9,10) for deception
- Contest opponent squares when beneficial

**Dogfights:**
- Save Rockets/Flares for critical fights
- As underdog: Rocket when very behind (power ≤-2)
- As favorite: Pass and win via Kaos (don't waste weapons)
- Track opponent's discarded Kaos cards for probability

**Resource Management:**
- You have 2 Rockets (Ace, King) and 2 Flares (Queen, Jack)
- Each is single-use - choose wisely!
- **Note**: Ace and King work identically (both Rockets), Queen and Jack work identically (both Flares). The strategic decision is WHEN to use them, not WHICH card to pick.
- Kaos deck (1-13) is finite but reshuffles when empty
- Discard pile is visible - use for probability tracking

## Board Notation

- Rows and columns are 0-indexed (0-2)
- Position [1,1] is the center
- Player 1 is "X", Player 2 is "O"

Example placements:
- `PLACE(5 @ [0,1])` - Place rocketman 5 at row 0, column 1
- `PLACE(10 @ [1,1])` - Place rocketman 10 at center (face-down!)

## Controls

During your turn:
- **Placement**: Enter rocketman power and grid position
- **Dogfight**: Choose Rocket, Flare, or Pass
- The game validates all inputs and shows legal options

Press Ctrl+C to quit at any time.

## Game End

Game ends when all dogfights are resolved:
- **3 in a row**: Instant win (horizontal, vertical, or diagonal)
- **Most squares controlled**: Winner
- **Tie**: Draw

Final score shows controlled squares for each player.

---

**Good luck and have fun!** 🚀
