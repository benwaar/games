# zxw — ZX Spectrum 48K Emulator (Rust → WASM)

A minimal ZX Spectrum 48K emulator written in Rust, compiled to WebAssembly, with Chrome DevTools Protocol (CDP) debug integration.

Built from scratch — small enough to understand completely. The primary target is running Jetpac in the browser with full memory inspection via CDP.

## Why

- **Binary protection research** — dog-food the [binary-protection explainer](../incant/explainers/binary-protection.md) by using CDP to inspect/patch WASM linear memory live
- **Void Duel foundation** — becomes the Layer 3 emulator host for the ZX Spectrum shooter project
- **Rust + WASM practice** — build pipeline patterns transfer to Acoustic Odyssey's mobile runtime

## What it does

- Z80 CPU emulation (incremental — opcodes added as games need them)
- 48K RAM + 16K ROM (standard Spectrum ROM)
- ULA screen rendering to HTML canvas
- TAP file loading (instant-load, no tape hardware emulation)
- Keyboard input via browser events
- CDP debug interface: inspect memory, patch values, set breakpoints — the POK file workflow in a modern browser

## Prerequisites

- Rust toolchain (rustup)
- `wasm-pack` (installed by setup.sh)
- Python 3 (for dev server)

## Quick start

```bash
bash setup.sh        # install Rust + wasm-pack if needed
make build           # compile to WASM
make serve           # start dev server at localhost:8080
```

## Status

See [PLAN.md](PLAN.md) for milestones. Currently at M1.

## Connections

| Project | Relationship |
|---------|-------------|
| [incant](../incant/) | TAP tools, Z80 knowledge, binary-protection research |
| [void-duel](../void-duel/) | Layer 3 emulator host, P2P input via debug API |
| [acoustic-odyssey](../acoustic-odyssey/) | Rust + WASM build patterns |
