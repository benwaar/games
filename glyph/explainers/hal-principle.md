# HAL Principle: Same Intent, Native Implementation

The HAL separates **what** (assets, AI, game logic) from **how** (platform rendering).

Each target implements the HAL natively — it doesn't emulate another target.

## What this means

- **Z80 target** renders like a Spectrum. 256×192, 2 colours per 8×8 cell, colour clash, interleaved ULA memory. That's the hardware. An SCR *is* the native format.
- **WASM target** renders like a modern canvas. Full colour, smooth scaling, no cell-colour constraint. It can look *better* — that's the point of having the hardware.
- **Same spec drives both.** The spec says "draw sprite at (x, y)" — Z80 writes to ULA memory, WASM draws a texture to canvas. Different implementations, same intent.

## Why not just emulate Z80 in WASM?

We already have that — it's [zxw](../../zxw/). The emulator faithfully reproduces Spectrum limitations in the browser.

Glyph is the opposite: the spec is the source of truth, and each target uses its platform's strengths. The Z80 target has colour clash because the hardware does. The WASM target doesn't because it doesn't have to.

## For SCR specifically

- **Z80**: load 6912 bytes into `0x4000`. Done — the ULA renders it. SCR *is* the native screen.
- **WASM**: the spec describes what to display (an image, sprites, a scene). The WASM renderer uses the source image directly — full colour, no quantisation. It doesn't need an SCR.

SCR is a Z80 artefact, not a cross-platform format. The HAL takes the *intent* (display this image) and each target does its own thing.
