# Next: zxw M6 — Jetpac bring-up

zxw M4 (ROM + TAP loading) and M5 (keyboard input) complete. Both tracks converge on M6: get Jetpac running.

---

## zxw M6 — Jetpac bring-up (incremental opcode completion)

- [ ] Load `jetpac_fixed.tap` (rebuilt 3-block version) → screen at 0x4000, code at 0x6000, entry 0x61E5
- [ ] Unimplemented-opcode trap: log PC, opcode, context, halt — feedback loop
- [ ] Implement IX/IY prefix opcodes (DD/FD): indexed addressing `(IX+d)` / `(IY+d)`
- [ ] Fill remaining ED-prefix gaps as Jetpac hits them
- [ ] IM 1 interrupt handling: RST 0x38 every ~69888 T-states
- [ ] Verify: title screen → key press → gameplay → movement + fire works

**Gate:** Jetpac loads, title screen renders, game starts on keypress, player moves and collects rockets. ~50fps, no visible slowdown.

**Dir:** `zxw/`

---

## Completed

- [x] zxw M1 — scaffold + WASM canvas hello world
- [x] zxw M2 — Z80 CPU core (11 tests, all unprefixed + CB + ED opcodes)
- [x] zxw M3 — ULA screen rendering (17 tests, Spectrum palette + interleaved layout)
- [x] zxw M4 — ROM + TAP loading (TAP parser, instant-load trap, 11 integration tests)
- [x] zxw M5 — Keyboard input (port 0xFE matrix, Kempston joystick, browser mapping, 11 tests)
- [x] Incant M7 — unified harness library (Z80Harness + WasmHarness)
- [x] Incant M8 — TAP read/write + disassembly (25 tests, --tap flag, disasm CLI)

---

## What's next after M6

| Milestone | Depends on | Parallel? |
|-----------|-----------|-----------|
| zxw M7 (CDP debug) | M6 | Sequential |
| Incant M9 (binary protection PoC) | zxw M7 + incant M8 | Both projects converge here |
| Incant M10 (ONNX → Z80) | incant M8 | Independent |
