# zxw — Binary Protection Demos

**Goal:** Show incant's self-verifying binaries in action — running clean, then caught tampering.

Skipping Demo 2 (POK file parser) for now — Demo 1 and 3 first.

---

## Prerequisites

- zxw built and serving (`make dev` from `zxw/`)
- incant venv active (`source incant/.venv/bin/activate`)
- `wat2wasm` available (`brew install wabt`)

---

## Demo 1 — Z80 self-verifying binary on zxw

Show a protected Z80 binary running clean on the emulator, then tampered via `zxw.poke()` — the verifier catches it and crashes to HALT.

### Steps

- [ ] Write `incant/scripts/make_protected_tap.py`:
  - Simple Z80 payload: draws a message to screen memory and loops
  - `wrap_z80_checksum(payload)` → protected binary
  - `write_tap(binary, name, load_addr)` → `zxw/www/tap/protected.tap`
  - Print the tamper address (first byte of payload = `load_addr + VERIFY_PREFIX_LEN`)
- [ ] Run the script, verify `protected.tap` appears in `zxw/www/tap/`
- [ ] Run `make assets` in `zxw/` so `assets.json` picks it up
- [ ] Write `zxw/www/demo-protection.html`:
  - Loads the emulator + `protected.tap` automatically
  - Shows instructions: "open DevTools console and paste the tamper command"
  - Displays the tamper `poke` command to copy
- [ ] Gate: load `protected.tap` in zxw → emulator runs → paste `zxw.poke(<tamper_addr>, <tampered_byte>)` in DevTools → emulator halts (verifier triggered)

**Commit:** `feat(zxw): Demo 1 — Z80 self-verifying binary + tamper demo`

---

## Demo 3 — WASM self-verifying module in browser

Show the WASM module running clean, then tampered via JS memory write — the `unreachable` trap fires.

### Steps

- [ ] Write `incant/scripts/make_protected_wasm.py`:
  - `generate_self_verify_wat(payload)` + `assemble_wat()` → `zxw/www/protect.wasm`
  - Print the payload bytes and expected checksum for reference
- [ ] Run the script, verify `protect.wasm` lands in `zxw/www/`
- [ ] Write `zxw/www/demo-wasm-protect.html`:
  - Fetch + instantiate `protect.wasm`
  - Button: **Run (intact)** → calls `run()` → shows result (42)
  - Button: **Tamper + run** → flips one byte in the data segment via `Uint8Array` on the fetched bytes, re-instantiates, calls `run()` → catches the `unreachable` trap and shows "DETECTED"
  - Shows the raw bytes before/after tamper
- [ ] Gate: "Run (intact)" shows 42; "Tamper + run" shows DETECTED

**Commit:** `feat(zxw): Demo 3 — WASM self-verifying module tamper demo`

---

## After both demos

- [ ] Update `zxw/LOG.md` with demo milestones
- [ ] Update `NEXT.md` — clear Demo 1 and 3, leave Demo 2 (POK) for later
- [ ] Commit docs + PR

