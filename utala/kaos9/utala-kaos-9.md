# utala: kaos 9
## version 1.9

*a tactical grid combat card game of strategy, risk and aerial duels*

---

## v1.9 Update: Choosable Dogfight Order

**New in v1.9:**

1. **Winner Chooses Next Dogfight**
   - Center square (if contested) is always resolved first
   - After each dogfight, the **winner chooses which contested square to fight next**
   - If both Rocketmen are eliminated (double knockout), the **joker holder** chooses
   - Adds a new strategic layer: fight where you're strong, or block where opponent threatens a line

**Previous versions:**
- v1.8: Joker token for equal-power tie-breaking, immediate 3-in-a-row victory
- v1.4: Dual-purpose weapons (sacrifice mechanic)
- v1.3: Face-down placement for cards 2, 3, 9, 10 (hidden information)
- v1.2: Turn-based dogfight mechanics
- v1.1 and earlier: Development versions

---

## Overview

utala: kaos 9 is a 2-player tactical grid game played on a **3×3 board**.
Players deploy Rocketmen (number cards) to control the grid, then resolve contested squares through probabilistic dogfights using limited dual-purpose weapons and personal Kaos decks.

- **Placement is deterministic and skill-based**
- **Combat is probabilistic but trackable**
- **Better players win by managing risk, timing, and odds**

---

## Uniqueness Notes (Design Intent)

utala: kaos 9 sits in the family of “3-in-a-row with cards” games, but differs in several key ways:

- **Two-phase structure**: the board is completely filled before any combat resolves (deploy → dogfight), unlike most variants that resolve conflicts immediately.
- **Contested squares persist**: both players occupy a square until it is resolved, rather than overwriting by rank.
- **Personal finite odds decks**: each player uses their own **Kaos deck** with visible discard, turning randomness into a skill-based probability game.
- **Interactive interrupts**: rockets do not automatically succeed; their outcomes are themselves resolved via Kaos draws.
- **Incomplete information**: The *highest and lowest* units values are hidden, creating bluffing without increasing component count.

Together, these elements make Dogfight less about card rank dominance and more about **commitment timing, risk management, and information control**.

---

## Components

- One standard 52-card deck  
- A Joker or the card box as a token to say who acts first in draws
- A 3×3 grid (drawn, printed, or imagined)

---

## terminology

- **rocketman**: a numbered unit (2–10) placed on the board  
- **weapon**:  A,K,Q & J placed on the board in the square during a dogfight 
- **kaos deck**: a personal 1–13 resolution deck used to determine combat outcomes

---

## Card Roles

### 🚀 Rocketmen (Placement Cards)
- Cards: **2–10**
- Each card represents one Rocketman
- The number is the Rocketman’s **power**
- Rocketmen fill the grid

### 💥 Weapons (Dual-Purpose)
- Cards: **Ace, King, Queen, Jack**
- **v1.4 Rules**: ALL 4 weapon cards are dual-purpose
- Each weapon can be used as EITHER:
  - **Rocket** (offensive - attack your opponent)
  - **Flare** (defensive - defend against incoming rocket)
- Single-use, discarded after play
- Which card you choose (A/K/Q/J) has no strategic value—they're interchangeable
- **The strategic decision**: Use now as defense OR save for future attack

**The Sacrifice Mechanic:**
Like sacrificing a pawn in chess, using a weapon defensively means spending a resource that could have been used offensively later. This creates meaningful trade-offs:
- **Defend now** = Block incoming attack, but lose an offensive option
- **Take the hit** = Accept damage/elimination risk, save weapon for your own attack
- No "always defend" strategy—every defense costs you future offense

### 🎲 Kaos Decks
- Each player has **their own Kaos deck**
- Cards: **A–K** (13 cards, a full suit, Aces are low A=1)
- Shuffled face-down at setup
- Used to resolve combat outcomes
- Cards are revealed, applied, then discarded
- Discard piles remain **visible at all times**

Kaos decks are finite and trackable, allowing skilled players to manage odds.

---

## Setup

1. Separate cards by role:
   - Rocketmen: 2–10
   - Weapons (dual-purpose): A, K, Q, J
2. Each player:
   - Takes all Rocketmen of their color (e.g. DIAMONDS)
   - Takes all 4 weapon cards (A, K, Q, J) of their color (e.g. DIAMONDS)
   - Creates a **Kaos deck (A–K)** of their color (e.g. HEARTS)
3. Shuffle Kaos decks separately.
4. Decide first player.

---

## Rules

### Phase 1: Placement
- Players alternate turns.
- Each player must place **their highest Rocketman (9 & 10)** and **their lowest Rocketman (2 & 3)** **face-down** when they are played.

- On your turn:
  - Place **one Rocketman (2–10)** onto any empty square.
- A square may contain:
  - One Rocketman (uncontested, yet placement continues)
  - Or one Rocketman from each player (contested)

Continue until **all 9 squares are filled**.

No combat occurs during this phase.

---

### Phase 2: Dogfights
A **dogfight** occurs in any square containing **two Rocketmen** (one from each player).

Resolve dogfights **one square at a time**.

**v1.9 Choosable Order:**
- The **center square** (if contested) is always resolved first
- After each dogfight, the **winner chooses** which contested square to fight next
- If both Rocketmen are eliminated (no winner), the **joker holder** chooses
- This continues until all contested squares are resolved

(Order matters and is part of the strategy — choosing where to fight next is a key tactical decision.)

---

### Dogfight Resolution

- Facedown cards are flipped over so their value is revelaed (showdown)

#### Step 1: Turn-Based Action/Reaction

**Turn Order (v1.8):**
- The **underdog** (player with lower Rocketman power) acts first
- **If powers are equal**: The **joker holder** acts first, then passes joker to opponent

**Weapon System:**
- You have 4 dual-purpose weapon cards: **Ace, King, Queen, Jack**
- Each weapon can be used as **Rocket** (offensive) OR **Flare** (defensive)
- **Context determines role**: First weapon played = Rocket, weapon played in response = Flare
- Every defense costs you a future attack (the sacrifice mechanic)

**Round 1: Underdog Acts**

The underdog may:
- Play a **weapon** (used as Rocket to attack, provided they have some left), OR
- Play **nothing** (pass)

**Round 2: Response**

*If underdog played weapon (Rocket):*
- Opponent may play **weapon** (used as Flare to defend), OR
- Opponent may play **nothing** (take the rocket attack risk)

*If underdog passed:*
- Opponent may play **weapon** (used as Rocket to attack, if they still have some), OR
- Opponent may play **nothing** (pass)

**Round 3: Counter-Response (if needed)**

*If opponent played weapon as Rocket in Round 2:*
- Underdog may now play **weapon** (used as Flare to defend), OR
- Underdog may play **nothing** (take the rocket attack risk)

**Key Rules:**
- First weapon played in a turn = **offensive** (Rocket)
- Weapon played in response to rocket = **defensive** (Flare)
- If both players pass with no weapons played, skip to Step 3 (Kaos Resolution)

**Strategic Consideration:**
Using a weapon to defend means you can't use it to attack later. Sometimes letting your Rocketman take a hit (or risk it) is better than spending your limited weapons on defense.

#### Step 2: Weapon Interaction

**🚀 Undefended Attack (Rocket with No Flare)**
- If a **weapon is used offensively** (Rocket) and **opponent passes**:
  - Attacking player draws **one Kaos card**
  - **7 or higher → HIT**
    - Target Rocketman is instantly removed
    - Dogfight ends
  - **6 or lower → MISS**
    - Weapon is discarded
    - Proceed to Kaos Resolution

**⚔️ Attack vs Defense (Rocket vs Flare)**
- If one player attacks (uses weapon as Rocket) and opponent defends (uses weapon as Flare):
  - Both weapons cancel each other out, both are discarded
  - Proceed to Kaos Resolution


Outcomes:
- Undefended attack wins → target Rocketman removed, dogfight ends
- Attack vs Defense → weapons cancel, proceed to Kaos

#### Step 3: Kaos Resolution (If Dogfight Continues)
If the dogfight was not ended by a rocket:

Each Rocketman draws **one Kaos card** from their own Kaos deck.

Total Power =  
Rocketman Power + Kaos Value

- Highest total wins the square
- Losing Rocketman is removed
- **Tie → both Rocketmen are removed**

---

### Winning the Game

**v1.8: Immediate Victory Rule**

After EACH dogfight is resolved, check if a player has achieved **3-in-a-row**:
- **Three Rocketmen in a row** (horizontal, vertical, or diagonal)
- **First player to achieve this wins immediately** (game ends)
- Standard tournament rule (like checkmate in chess)

If all dogfights complete without anyone achieving 3-in-a-row:
- Player controlling **the most squares** wins
- Tie → draw or rematch

**Why this matters**: Makes 3-in-a-row the primary win condition. Strategic play focuses on building lines during placement and fighting for line-completing squares during dogfights.

---

## Design Notes

- **v1.9 Choosable dogfight order** adds strategic depth — fight where you're strong or block opponent lines
- **v1.8 Joker token** creates fair tie-breaking with alternating advantage
- **v1.8 Immediate 3-in-a-row** makes spatial tactics primary win condition
- **v1.4 Dual-purpose weapons** create meaningful sacrifice decisions (defend now vs attack later)
- **Attacks are powerful but uncertain** (Kaos draw ≥7 to hit)
- **Defense costs offense** - every blocked attack means one fewer future attack
- **Kaos decks reward memory and probability tracking** (finite, visible discard)
- **Face-down cards (2, 3, 9, 10)** add bluffing and incomplete information
- **Resource tension throughout** - only 4 weapons total, must choose when to spend

The sacrifice mechanic (like chess) means skilled players must evaluate: "Is this Rocketman worth a weapon?" This prevents "always defend" strategies and rewards risk management.

**v1.9 Strategic Implication**: Choosable dogfight order rewards players who create multiple threats during placement. After winning a fight, choosing the next square wisely — completing your own line or blocking the opponent's — is a critical skill. Combined with v1.8's immediate victory rule, the order you fight in can determine the game.

**v1.8 Strategic Implication**: The immediate 3-in-a-row victory rule makes line completion/blocking the dominant strategy. Good placement creates 3-in-a-row threats; good dogfight play defends critical line-completing squares.

---

## Copyright

© 2026 David Benoy.

The text of this document, the name *utala: kaos 9*, and the presentation of the rules are protected works. Game mechanics themselves are not claimed as exclusive.
