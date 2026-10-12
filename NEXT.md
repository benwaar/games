# Next: Incant M7 + zxw M1–M3 (parallel tracks)

Two workstreams running in parallel. Incant M7 is a Python refactor with no blockers. zxw M1–M3 is a new Rust/WASM project built via `/research`.

---

## Track 1: Incant M7 — Unified test harness

Refactor scattered test logic into `incant/harness.py` with a shared interface for both targets.

- [ ] Create `incant/harness.py` — `Z80Harness` and `WasmHarness` with shared API
- [ ] Refactor `z80.py` `run_harness_io_test()` to use `Z80Harness`
- [ ] Refactor `wat.py` `run_wasi_test()` to use `WasmHarness`
- [ ] Simplify run scripts to use the harness module
- [ ] CLI: `python -m incant run output/greet.bin "Ben"`
- [ ] Tests for both harnesses
- [ ] Docs: explainer, update README + CLAUDE.md

**Gate:** `python -m incant run output/greet.bin "Ben"` and `python -m incant run output/greet_user.wasm "Ben"` both print `hello Ben`.

**Branch:** `feature/no-ticket-incant-m7-harness`

---

## Track 2: zxw M1–M3 — Spectrum emulator scaffold + CPU + screen

New Rust project. Use `/research` skill — plan with checkboxes, one step at a time, `setup.sh`, reproducible from cold clone.

- [ ] **M1** — Scaffold: `Cargo.toml`, `setup.sh`, `wasm-pack`, checkerboard on canvas
- [ ] **M2** — Z80 CPU core: registers, decode/execute loop, foundational opcodes, `cargo test`
- [ ] **M3** — ULA screen: Spectrum bitmap layout → RGBA → canvas, Jetpac loading screen renders

**Gate (M3):** Load Jetpac screen dump into memory at 0x4000 → loading screen renders with correct colours.

**Branch:** `feature/no-ticket-zxw-m1-scaffold` (then new branch per milestone)

---

## After these complete

| Next | Depends on | What |
|------|-----------|------|
| Incant M8 (TAP read/write) | M7 done | TAP writer wraps `.bin` → `.tap`, reader + disassembler |
| zxw M4–M5 (ROM + keyboard) | M2–M3 done | Load Spectrum ROM, parse TAPs in Rust, keyboard input |
| zxw M6 (Jetpac bring-up) | M4–M5 done | Run Jetpac, fill opcode gaps incrementally |
| zxw M7 (CDP debug) | M6 done | `cdp_debug.py` — inspect memory, poke values, set breakpoints |
| Incant M9 (binary protection PoC) | zxw M7 done | Self-verifying code, demoed live via CDP |
| Incant M10 (ONNX → Z80) | M7–M8 done | TinyNN inference engine on Z80 |
