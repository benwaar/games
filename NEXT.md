# Next

Remaining work before starting Acoustic Odyssey or Artefact.

---

## 1. zxw — Binary protection demos

Showcase incant's self-verifying binaries running on the zxw emulator, with live tampering via the CDP debug API.

### Steps (Sonnet — use `/research` then implement)

**Demo 1 — Z80 self-verifying binary on zxw**
1. `/research` — review `incant/incant/protect.py` (XOR checksum self-verification) and `zxw/` CDP debug API
2. Load incant's self-verifying `.tap` onto zxw
3. Show it running clean, then tamper a byte via CDP `poke` command
4. Capture the crash/halt — screenshot or automated test
5. Write up in a demo script or doc

**Demo 2 — POK file support**
1. `/research` — ZX Spectrum POK file format (classic cheat/poke format)
2. Implement POK parser in zxw — read POK files and apply pokes on load
3. Tie into the CDP debug API as an alternative poke source
4. Test with a known game POK

**Demo 3 — WASM self-verifying module**
1. `/research` — review `incant/incant/protect.py` WASM data-segment mirror protection
2. Load the WASM self-verifying module in browser
3. Tamper the data segment, show detection
4. Document the bypass (patch constant, intercept load)

---

## 2. Log entries

Update LOG.md for both projects with what was completed this session:

- [ ] `incant/LOG.md` — WASM Phase 1–2 already logged; add any missing detail
- [ ] `zxw/LOG.md` — note the binary protection demo work once complete
- [ ] Delete stale `utala/kaos9/demo/screenshot.png` (old WASM screenshot, renamed)
