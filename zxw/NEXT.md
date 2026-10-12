# Next: M7 — CDP debug integration

Wire up a WASM debug API so Python CDP scripts can inspect and control the running emulator.

## What to do

1. **WASM debug API** — expose via `window.zxw.*`:
   - `read_memory(addr, len)` → Uint8Array
   - `write_memory(addr, data)` — poke bytes
   - `read_registers()` → JS object with all Z80 registers
   - `set_breakpoint(addr)` / `clear_breakpoint(addr)`
   - `step()` — single-step one instruction
   - `pause()` / `resume()` — halt/resume the frame loop

2. **`tools/cdp_debug.py`** — WebSocket CDP client with commands:
   - `regs` — print Z80 register state
   - `mem <addr> <len>` — hex dump
   - `poke <addr> <val>` — write single byte
   - `bp <addr>` — set breakpoint, print when hit
   - `screen` — dump screen memory (0x4000, 6912 bytes) to file
   - `watch <addr>` — print value each frame

3. **Memory-write watchpoint** (optional) — callback when address range is written.

4. **Test** — while Jetpac runs:
   - `python3 tools/cdp_debug.py regs` prints Z80 state
   - `mem 0x4000 32` dumps screen bytes
   - `poke <lives_addr> 9` changes displayed lives

## Gate

Jetpac running → `python3 tools/cdp_debug.py` connects → `regs` prints Z80 state → `mem 0x4000 32` dumps screen bytes → `poke <lives_addr> 9` changes displayed lives.

## Reference

- CDP WebSocket endpoint: `ws://localhost:9222/json` (Chrome with `--remote-debugging-port=9222`)
- wasm-bindgen exports: annotate functions with `#[wasm_bindgen]` and expose on `window.zxw`
- Breakpoint hook: check `cpu.pc` against a set at start of each `step()` call
