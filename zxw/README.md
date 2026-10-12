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

See [PLAN.md](PLAN.md) for milestones. M1–M7 complete.

## Debug API (M7)

The emulator exposes a `window.zxw` object in the browser. All functions are
available from the Chrome DevTools console or via the CDP client.

```javascript
zxw.regs()                   // Z80 registers as a JS object
zxw.mem(addr, len)           // Uint8Array — default len 16
zxw.poke(addr, b1, b2, ...)  // write bytes at addr
zxw.bp(addr)                 // set breakpoint
zxw.clearBp(addr)            // clear one breakpoint
zxw.clearAllBp()             // clear all breakpoints
zxw.pause()                  // pause emulation
zxw.resume()                 // resume emulation
zxw.step()                   // single-step one instruction
zxw.screen()                 // read screen memory (0x4000, 6912 bytes)
zxw.watch(addr)              // add write watchpoint
zxw.unwatch(addr)            // remove write watchpoint
zxw.hits()                   // drain watchpoint hits → [addr, old, new, ...]
```

### CDP command-line client

```bash
pip install websockets
# Start Chrome with remote debugging:
#   Google Chrome --remote-debugging-port=9222
python3 tools/cdp_debug.py

# Commands at the zxw> prompt:
# regs              — print all Z80 registers
# mem 0x4000 32     — hex dump 32 bytes
# poke 0x5C78 9     — write one byte
# bp 0x61E5         — set breakpoint
# step              — single-step
# watch 0x4000 64   — poll memory every 0.5 s (Ctrl-C to stop)
```

## Connections

| Project | Relationship |
|---------|-------------|
| [incant](../incant/) | TAP tools, Z80 knowledge, binary-protection research |
| [void-duel](../void-duel/) | Layer 3 emulator host, P2P input via debug API |
| [acoustic-odyssey](../acoustic-odyssey/) | Rust + WASM build patterns |
