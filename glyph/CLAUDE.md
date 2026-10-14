# CLAUDE.md — glyph

## What this is

An LLM-first presentation layer. Extends incant with a `game` sigil type that generates sprite-based programs targeting WASM (macroquad) and Z80 (ULA) from one spec.

The library is designed for LLMs to write against, not humans. Minimal, consistent API — the LLM learns it via RAG.

## Architecture

```
game sigil → rag.py (--collection glyph) → gen.py → gate
                                                       ↓
                                          wasm-glyph: Rust + macroquad → .wasm
                                          z80-glyph:  Z80 asm + ULA writes → .bin
```

## HAL surface (what the LLM writes against)

```
screen_clear()
sprite_draw(id, x, y)
sprite_move(id, dx, dy)
input_read() → u8          // UP=0x01 DOWN=0x02 LEFT=0x04 RIGHT=0x08 FIRE=0x10
collision(id_a, id_b) → bool
screen_width() → u16
screen_height() → u16
```

Both targets implement this surface. Game logic only calls HAL functions — no platform code in sigils.

## Conventions

- One sigil = one game loop or one game function. Same rule as incant.
- Sprites are 8×8, 1-bit (black + 1 colour). Same constraint on both targets.
- Z80 screen is 256×192. WASM screen matches (or scales).
- No audio in M1–M4. Separate HAL extension later.

## Tech stack

- **WASM**: Rust, macroquad, wasm-pack
- **Z80**: z80 pip package (assembler + emulator), ULA memory map
- **RAG**: nomic-embed-text via Ollama, JSONL store (shared with incant)
- **LLM**: qwen3-coder via Ollama (shared with incant)

## Running

```bash
# (setup coming in M2)
python -m incant cast sigils/examples/sprite_move.sigil.yaml --target wasm-glyph
python -m incant cast sigils/examples/sprite_move.sigil.yaml --target z80-glyph
python -m incant rag query "draw sprite" --collection glyph
```

## Key files (planned)

```
glyph/
  knowledge/         — HAL docs embedded into RAG
  glyph-wasm/        — Rust crate: macroquad HAL implementation
  sigils/examples/   — example game sigils
  plan.md            — milestones
  explainers/        — glyph-specific concepts
```
