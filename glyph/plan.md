# glyph — Plan

LLM-first presentation layer: game specs → platform code (WASM + Z80).

---

## M1 — HAL definition + RAG knowledge base

Define the minimal API surface the LLM will write against. Build the RAG knowledge base for both targets.

- [ ] Define HAL: `screen_clear`, `sprite_draw`, `sprite_move`, `input_read`, `collision`, `screen_width`, `screen_height`
- [ ] Document each call: signature, semantics, constraints, example (WASM and Z80 side by side)
- [ ] Add macroquad sprite API docs to knowledge base
- [ ] Add ULA memory map + sprite write pattern docs to knowledge base
- [ ] Embed into RAG (`--collection glyph`) using existing incant RAG pipeline
- [ ] Test: RAG query `"draw a sprite at position x,y"` returns relevant chunks for both targets

**Gate:** `python -m incant rag query "draw sprite" --collection glyph` returns macroquad and ULA docs.

---

## M2 — WASM target (macroquad)

Implement the WASM-side of the HAL. The LLM generates Rust game logic; macroquad provides the rendering.

- [ ] Rust crate scaffold: `glyph-wasm/` with macroquad dependency
- [ ] Implement HAL trait in Rust backed by macroquad calls
- [ ] `wasm-pack` build pipeline → `.wasm` output, loads in browser with no JS game logic
- [ ] Example: single sprite, moves on input, wraps at screen edge
- [ ] Gate: sprite renders at correct position, input moves it, no JS required

**Gate:** `bash build_wasm.sh` → `glyph.wasm` loads in browser, sprite draws and moves.

---

## M3 — Z80 target (ULA)

Implement the Z80-side of the HAL. LLM generates asm that calls ULA write routines.

- [ ] HAL asm stubs: `screen_clear`, `sprite_draw(b=x, c=y, d=sprite_id)`, `input_read → a`
- [ ] ULA address calculation routine (pixel + attribute addresses)
- [ ] 8×8 sprite data format: 8 bytes per frame, packed into data block
- [ ] Example: same single-sprite demo as M2 (same spec, different target)
- [ ] Gate: assembled binary runs in `zxw`, sprite visible at correct position

**Gate:** Z80 binary from same spec as M2 loads in zxw emulator, sprite draws and moves.

---

## M4 — incant integration (game sigil type)

Wire glyph into the incant pipeline as a new sigil target.

- [ ] `game` sigil type: extends existing sigil format with `sprites`, `screen_size`, `loop` fields
- [ ] New `--target wasm-glyph` and `--target z80-glyph` flags in `incant cast`
- [ ] LLM prompt template for game sigils (includes HAL API docs from RAG)
- [ ] Gate: `python -m incant cast sigils/examples/sprite_move.sigil.yaml --target wasm-glyph`

**Gate:** `incant cast` with a game sigil produces a working binary for both targets from one spec.

---

## M5 — Sprite demo (ML hook)

Build a minimal sprite demo designed for ML experiment hookup — the first void-duel precursor.

- [ ] Two sprites, 8-directional movement, boundary collision
- [ ] Input: player 1 from keyboard, player 2 from a function pointer (AI hook)
- [ ] Python harness: headless WASM run for fast ML rollouts
- [ ] Spec drives both WASM and Z80 output from one file

**Gate:** Python harness can run 100 headless games/second. Z80 binary runs identically in zxw.

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| LLM-first API design | new |
| WASM sprite rendering (no JS) | new |
| Z80 ULA sprite writes | new |
| Cross-platform HAL (one spec, two targets) | new |
| RAG for visual/game APIs | extends incant |
| Game sigil type | extends incant |

---

## Feeds into

- [void-duel](../void-duel/) — Phase 0/1 game engine, ML environment hookup
