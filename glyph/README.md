# glyph

An LLM-first presentation layer — sprite-based rendering from specs, targeting WASM and Z80 from one codebase.

Part of the [incant](../incant/) pipeline. Adds a `game` sigil type and a `glyph` target, so incant can generate both visual programs (via glyph) and non-visual computation (existing pipeline) from the same spec-driven approach.

---

## What it is

A library + RAG knowledge base designed for LLMs to write against, not humans.

The API surface is minimal and consistent — one way to do each thing. The LLM knows the surface via RAG (same approach as Z80/WAT knowledge in incant). Specs describe intent, LLM generates code against the lib, gate verifies the output renders correctly.

```
spec (sprites, movement, input)
  → RAG retrieves glyph API docs
  → LLM generates platform code
      → WASM target: Rust + macroquad (canvas, no JS game logic)
      → Z80 target:  asm + ULA memory writes
```

---

## HAL principle

**Same intent, native implementation per platform.** Each target renders natively — it doesn't emulate another target.

- **Z80** renders like a Spectrum: 256×192, 2 colours per 8×8 cell, ULA memory. SCR is the native format.
- **WASM** renders like a modern canvas: full colour, smooth scaling, no cell-colour constraint.
- **Same spec drives both.** The spec says what to do; each target decides how.

See [explainers/hal-principle.md](explainers/hal-principle.md) for the full rationale.

---

## POC: Image → Both Targets (complete)

Proves the HAL principle with static image rendering. Three design iterations showed that prompt-engineering the source art for Spectrum constraints produces dramatically better Z80 output while keeping WASM quality high.

**Pipeline:**
```
source image (any size, full colour)
  ├── WASM: resize + JPEG compress → render full colour to canvas
  └── Z80:  resize to 256×192 → k-means quantise per 8×8 cell → SCR → load into ULA
```

**What was built:**
- `scr/` — Python image-to-SCR converter (k-means clustering, Spectrum palette mapping, interleaved bitmap packing). 39 tests.
- `z80/` — Z80 SCR loader (LDIR into ULA memory, runs in zxw emulator). 6 tests.
- `glyph-wasm/` — Rust + macroquad image renderer (full colour, desktop + WASM builds).
- `prompt-skeleton.md` — prompt template for generating LLM art that works on both targets.

**Key finding:** designing art with Spectrum constraints in the prompt (bold shapes, palette colours, high contrast, no fine detail under 8px) is the most effective optimisation. The k-means quantiser can only work with what it's given.

---

## Targets

| Target | Runtime | Rendering |
|--------|---------|-----------|
| WASM | macroquad (Rust → WASM) | Canvas, no JS game logic |
| Z80 | Z80 asm | ULA memory (`0x4000–0x57FF`) |
| Desktop | macroquad native | Same Rust codebase as WASM |
| Mobile | macroquad (future) | iOS/Android from same source |

---

## API surface (HAL)

The LLM writes to this abstraction. Platform implementations are provided by the lib.

```
screen_clear()
sprite_draw(id, x, y)
sprite_move(id, dx, dy)
input_read() → u8          // bitmask: UP DOWN LEFT RIGHT FIRE
collision(id_a, id_b) → bool
screen_width() → u16
screen_height() → u16
```

---

## Project status

| Milestone | Status |
|-----------|--------|
| POC: Image → Both Targets | ✅ Complete |
| M1: HAL definition + RAG knowledge base | Planned |
| M2: WASM target (macroquad) | Planned |
| M3: Z80 target (ULA) | Planned |
| M4: incant integration (game sigil type) | Planned |
| M5: Sprite demo (ML hook) | Planned |

See [plan.md](plan.md) for milestone details. See [NEXT.md](../NEXT.md) for the POC plan.

---

## Connections

| Project | Relationship |
|---------|-------------|
| [incant](../incant/) | Core codegen pipeline — glyph extends it with game sigils |
| [void-duel](../void-duel/) | First real consumer of glyph |
| [zxw](../zxw/) | Spectrum emulator — runs Z80 glyph output |
