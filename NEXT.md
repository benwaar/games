# Next

Remaining work before starting Acoustic Odyssey or Artefact.

---

## 1. zxw — POK file support (Demo 2)

Study the real-hardware pattern for educational purposes: load a protected TAP, apply pokes
from a `.pok` file to bypass the startup check — the same technique players used with
Multiface hardware or POKE listings from magazines. The goal is understanding how binary
protection works, not breaking anything current.

Test subject: `j83.tap`. Three sysvar pokes bypass the startup protection check — without
them the game bails immediately. The `.pok` file is the bypass.

### Prerequisites

- zxw builds cleanly: `wasm-pack build --target web` in `zxw/`
- `make serve` runs the dev server (`python3 -m http.server 8080 --directory www`)
- `cargo test` passes from `zxw/`

---

### M8 — Create `j83.pok`

Create the POK file for j83 with the three sysvar pokes from `j83-analysis.md`.

POK format (standard ZX Spectrum):
```
Y <description>
M <bank> <address> <value> <oldvalue|N>
Z
```
Bank 8 = 48K RAM. Addresses in decimal. Old value `N` = unknown/runtime.

- [x] Create `zxw/www/pok/j83.pok` with the three pokes:
  - `M 8 23672 90 N` (0x5C78 — ERR_SP low byte)
  - `M 8 23673 131 N` (0x5C79 — ERR_SP high byte)
  - `M 8 23728 233 N` (0x5CB0 — JP (HL) instruction)
- [x] Manually verify addresses match `incant/data/raw/tap/j83-analysis.md`

**Gate:** file exists, addresses match the analysis  
**Commit:** `feat(zxw): add j83.pok — sysvar pokes for startup bypass`

---

### M9 — POK parser in Rust (`src/pok.rs`)

New module: parse a `.pok` file into a list of `(address, value)` pairs.

- [x] Create `zxw/src/pok.rs`:
  - `pub struct Poke { pub addr: u16, pub value: u8 }`
  - `pub fn parse_pok(data: &str) -> Result<Vec<Poke>, String>` — skip `Y`/`Z` lines, parse `M` lines
  - Reject bank != 8 (only 48K RAM supported), addresses > 65535, values > 255
  - Ignore malformed lines gracefully (skip with warning)
- [x] Add `mod pok;` to `src/lib.rs`
- [x] Write `zxw/tests/test_pok.rs`:
  - valid 3-poke file → 3 results
  - `Y` and `Z` lines ignored
  - bank != 8 → error or skip
  - malformed line → skipped
  - empty file → empty vec

**Gate:** `cargo test` passes  
**Commit:** `feat(zxw): POK file parser — parse_pok() + tests`

---

### M10 — Apply pokes on TAP load

Expose `apply_pok()` on `Emulator` so JS can pass POK data after loading a TAP. Wire into
the demo so pokes are applied before the game starts.

- [ ] Add to `src/lib.rs`:
  ```rust
  pub fn apply_pok(&mut self, pok_data: &str) -> Result<(), JsValue>
  ```
  Calls `parse_pok()`, then `write_memory()` for each poke. Returns error string on parse failure.
- [ ] Rebuild WASM: `wasm-pack build --target web`
- [ ] Update `zxw/www/index.html` (or create a demo page) — after `load_tap()`, fetch
  `<tapname>.pok` if it exists, call `apply_pok()`, then start the emulator
- [ ] Run `make assets` to pick up `j83.pok` in `assets.json`

**Gate:** `j83.tap` loads in the browser, pokes applied, loading screen appears  
**Commit:** `feat(zxw): apply_pok() on Emulator — wire POK into TAP load`

---

### M11 — CDP path *(next session)*

Apply pokes via the existing CDP debug API as an alternative source.

---

### M12 — Demo page *(next session)*

Browser page showing: TAP name, POK file applied, addresses + values listed. Suitable for
a non-technical show-and-tell.
