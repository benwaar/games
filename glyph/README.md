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

See [plan.md](plan.md) for milestones.

---

## Connections

| Project | Relationship |
|---------|-------------|
| [incant](../incant/) | Core codegen pipeline — glyph extends it with game sigils |
| [void-duel](../void-duel/) | First real consumer of glyph |
