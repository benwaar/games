# Void Duel

A 2-player space shooter that runs on real ZX Spectrum hardware and through a WASM emulator in the browser with P2P multiplayer.

Primary learning aim: AI (classic ML, multi-agent RL, extreme-edge distillation).
Secondary learning aim: WASM (Rust emulator compiled to WebAssembly, input injection, P2P sync).

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Layer 1: Python AI Lab                 │
│  Train → evaluate → distill             │
│  Classic ML, multi-agent RL, PyTorch    │
│  Output: lookup tables / decision trees │
│  that fit in Z80 memory (48K total)     │
├─────────────────────────────────────────┤
│  Layer 2: Z80 Game                      │
│  Actual game binary (z88dk C or asm)    │
│  Baked-in AI data from Layer 1          │
│  Runs on real Spectrum hardware         │
│  2P local = shared keyboard             │
├─────────────────────────────────────────┤
│  Layer 3: Rust WASM Emulator Host       │
│  Fork of rustzx (MIT), WASM frontend    │
│  Injects remote player inputs via P2P   │
│  Duello arch: WebRTC + Nostr signaling  │
│  2P remote = each runs own emulator     │
└─────────────────────────────────────────┘
```

The Z80 game doesn't know whether the second player is local or remote. In WASM mode, Layer 3 intercepts remote inputs over WebRTC and feeds them into the emulator's keyboard port. Same ROM, both modes.

---

## Phases

```
Phase 0: WASM PoC (coin flip)
   │
Phase 1a: Z80 Game ──────── Phase 1b: Python Simulator
   │                              │
   ├── WASM 2P (from Ph 0)       │
   │                              │
   │         ┌────────────────────┘
   │         │
Phase 2: Classic ML AI (trains on simulator, deploys to Z80)
   │
Phase 3: Multi-Agent RL (extends simulator + training)
   │
Phase 4: Polish + IP exploration
```

Game code (1a) and Python simulator (1b) are parallel tracks — both
must exist before AI training (Ph 2) can start. Art, audio, and
polish (Ph 4) are independent and can happen any time.

### Phase 0 — PoC: Coin Flip on Emulator

Prove the full stack with the simplest possible game. Reuse Duello's coin flip.

1. Fork rustzx, build WASM frontend (replaces SDL2 with canvas/WebAudio)
2. Write trivial Z80 program — coin flip with sprite animation, loads via .tap
3. Wire Duello P2P transport — remote player's "flip" injected as keyboard input
4. Validate: same .tap runs on FUSE (real emulator) and in browser via WASM

**Checkpoint 0 — Does the emulator + P2P pipeline work?**
Pass: two browsers play coin flip over WebRTC, same ROM runs on FUSE. Fail: WASM emulator can't run a .tap, or input injection breaks timing.

### Phase 1 — Minimal Shooter + Python Simulator

Build the game and its Python twin in parallel. The Z80 game is the artifact; the Python simulator is the AI training environment. Both must implement identical rules.

**1a. Z80 game** (z88dk C)
1. Two sprites, 8-directional movement, projectiles, collision detection
2. Same-keyboard 2P on real hardware (or FUSE)
3. WASM 2P via Duello P2P (proven in Phase 0)
4. Frame-rate and input latency validation

**1b. Python simulator** (parallel with 1a)
1. Mirror the Z80 game rules exactly — same physics, same collision, same timing
2. Headless, fast rollouts (thousands of games per second for training)
3. Deterministic replay with explicit RNG seeds (same pattern as Utala)
4. Validate against Z80 game — same inputs must produce same outcomes

**Checkpoint 1 — Is the game playable and the simulator faithful?**
Pass: responsive controls on real hardware, no desync in WASM P2P, and Python simulator produces identical outcomes to Z80 game for the same input sequence. Fail: input lag, desync, or simulator/game divergence.

### Phase 2 — Classic ML AI Opponent

Train AI pilots in Python using the full classic ML toolkit. This is the primary learning phase — it fills the entire "Classic ML" row in STUDY.md, which currently only has a 🔶 from 2004. Depends on the Python simulator from Phase 1b.

**2a. Feature engineering + baselines**
1. Feature engineering — position, velocity, angle, distance, ammo, health as tabular features
2. PCA on game state — reduce the feature space, visualise which dimensions matter most
3. k-means clustering on recorded play sessions — discover player behaviour archetypes (aggressive, defensive, evasive)
4. Use clusters to label training data and seed difficulty tiers

**2b. Classic ML agents (one per algorithm family)**
5. Linear / logistic regression — simplest agent, baseline for everything else
6. Decision tree / random forest — interpretable agent, the one most likely to ship to Z80
7. Gradient boosting (XGBoost or LightGBM) — strongest classic ML agent, benchmark ceiling
8. SVM — non-linear decision boundaries on the PCA-reduced feature space

**2c. Evaluation + interpretability**
9. Tournament harness — round-robin: each agent vs every other + random baseline, win rates, Elo-style ranking
10. SHAP values on the tree/forest agent — why did the AI dodge left? Feature importance per decision
11. LIME on individual predictions — local explanations for specific game moments
12. Compare: which algorithm family produces the best pilot? Document the tradeoffs (accuracy vs interpretability vs Z80 size)

**2d. Distillation to Z80**
13. Distill best agent to Z80-compatible format (lookup table or compact decision tree)
14. Bake into ROM, test on real hardware
15. Measure: does the Z80 version play recognisably like the Python original?

**Checkpoint 2 — Can classic ML produce a competent AI pilot?**
Pass: at least 3 algorithm families trained and compared; best agent beats random baseline >70%; SHAP/LIME explanations documented; distilled Z80 version plays recognisably similar. Fail: classic ML can't capture the decision space, or distillation loses too much.

### Phase 3 — Multi-Agent RL

Train two AI agents against each other. Fill the multi-agent RL gap.

1. Self-play training loop — two agents learn simultaneously
2. Explore co-adaptation, Nash equilibria, strategy cycling
3. Compare multi-agent RL agents vs Phase 2 classic ML agents
4. Distill winning strategies to Z80

**Checkpoint 3 — Does multi-agent training produce emergent behaviour?**
Pass: agents develop recognisable strategies (flanking, retreating, baiting) that beat single-agent-trained opponents. Fail: training collapses or agents don't surpass classic ML.

### Phase 4 — Polish and IP Exploration

1. Game polish — title screen, sound effects, score tracking, attract mode
2. WASM security study — can someone extract the ROM from the WASM binary? What's actually protected?
3. Countermeasures — encrypted ROM decrypted at runtime, integrity self-checks
4. Cryptographic match ledger via Duello (Ed25519 signed results)

---

## Skills Coverage (what this project adds)

| Skill | Phase | How |
|---|---|---|
| Linear / logistic regression | Ph 2b | Baseline agent on tabular game state |
| Decision trees, random forests | Ph 2b | Interpretable agent, primary distillation candidate |
| Gradient boosting (XGBoost/LightGBM) | Ph 2b | Benchmark ceiling — strongest classic ML agent |
| SVM | Ph 2b | Non-linear boundaries on PCA-reduced features |
| k-means clustering | Ph 2a | Discover player behaviour archetypes from replays |
| PCA | Ph 2a | Dimensionality reduction on game state features |
| Interpretability (SHAP, LIME) | Ph 2c | Explain AI pilot decisions per algorithm |
| Tabular data | Ph 2 | Game state = position/velocity/ammo/health |
| Multi-agent / offline RL | Ph 3 | Self-play, co-adaptation, Nash equilibria |
| Edge inference / extreme distillation | Ph 2–3 | Train in Python, compress to Z80 (48K, 3.5MHz) |
| WASM | Ph 0–1 | Rust emulator → WebAssembly, audio, input, canvas |
| WASM security / IP protection | Ph 4 | ROM inside WASM sandbox, extraction study |

---

## Tech Stack

| Layer | Tech |
|---|---|
| AI training | Python, PyTorch, scikit-learn, XGBoost/LightGBM, SHAP |
| Z80 game | z88dk (C cross-compiler) or Z80 assembly |
| Emulator | rustzx fork (Rust, MIT license) |
| WASM build | wasm-pack / wasm-bindgen |
| Browser frontend | Canvas API, WebAudio API |
| P2P multiplayer | Duello architecture (WebRTC, Nostr relay signaling) |
| Crypto ledger | Ed25519 signed match results (from Duello) |

---

## Dependencies

- [Duello](../../duello/) — P2P transport, signaling, crypto ledger
- [rustzx](https://github.com/rustzx/rustzx) — ZX Spectrum emulator (MIT, `no_std` core)
- [Utala: KAOS 9](../utala/kaos9/) — evaluation harness patterns, distillation pipeline reference
