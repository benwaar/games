# Next

## WASM player — a11y (feature/no-ticket-wasm-responsive)

Phased a11y for web game content. Reusable lib: pipeline-side WAT encoding + player-side mirror DOM.

Research: `wasm/A11Y.md` | `wasm/SCALING.md` (done)

### Phase 1: Mirror DOM POC

Slim mirror DOM — scene description JSON → WAT data segment → player reads and builds hidden DOM overlay.

**1a — WASM import contract (preferences bridge)**
- [x] Define `env` imports: `prefers_dark`, `prefers_reduced_motion`, `prefers_high_contrast`
- [x] Player provides imports at `WebAssembly.instantiate()` — reads `matchMedia` by default
- [x] WAT generators accept optional `imports` list — games opt in to what they need
- [x] Tests: WASM instantiation with imports, default values match browser
- [x] Doc: update `glyph/CLAUDE.md` WASM export contract section with imports

**1b — a11y data segment in pipeline**
- [x] `_a11y_data_segment(nodes_json, offset)` helper in `pipeline.py`
- [x] `type()` bit 3 (value 8) = has a11y metadata
- [x] New exports: `a11y_offset() → i32`, `a11y_size() → i32`
- [x] Optional `a11y` param on `rgba_to_wat`, `audio_to_wat`, `combined_to_wat`
- [x] CLI: `--a11y scene.json` flag
- [x] Sample: `glyph/assets/scene.json` for `i1.png`
- [x] Tests: encoding, offset/size, bitmask, backward compat (no a11y = same output)
- [x] Doc: update `glyph/CLAUDE.md`

**1c — Mirror DOM in player**
- [x] `buildMirrorDom(canvas, nodes)` in `wasm/index.html`
- [x] Positioned container overlaying canvas, percentage-based coords (scales with responsive CSS)
- [x] Per node: `<div>` with `role`, `aria-label`, sr-only (clip-rect)
- [x] `aria-live="polite"` region for announcements
- [x] Status div gets `role="status"` + `aria-live="polite"`
- [x] Fallback when no bit 3: `aria-label="Game content"` + `role="img"` on canvas
- [x] Doc: update `wasm/A11Y.md` implementation section

**1d — Explainer**
- [x] `wasm/explainers/mirror-dom.md` — what, why, how, connection to toolkit/flare pattern

**1e — Verify + commit**
- [x] Build with `--a11y`, serve `:8999`, inspect DOM overlay in DevTools
- [x] VoiceOver test — tab through nodes, hear labels (manual — a11y tree verified via DevTools snapshot)
- [x] Old WASM (no bit 3) — fallback works, no crash
- [x] Mobile emulation — mirror DOM scales with responsive canvas
- [x] `python -m pytest glyph/ -v` — all pass
- [x] Commit: `feat(wasm): mirror DOM a11y + preference imports`

### Phase 2: Settings panel + palette override

Browser preferences flow through to WASM via imports. Settings panel lets user override.

- [x] Settings gear icon in player UI
- [x] Toggle overrides: dark mode, reduced motion, high contrast
- [x] Show detected browser value vs override
- [x] `localStorage` persistence
- [x] CSS custom properties for player chrome (`--bg`, `--border`, `--text`)
- [x] Palette presets: default, high contrast, deuteranopia-friendly
- [x] Re-instantiate WASM when override changes (imports return new values)
- [x] Tests + docs
- [x] Commit: `feat(wasm): a11y settings panel + palette override`

### Phase 3: Sound customisation

- [ ] Separate volume controls (SFX / narration / music) when multiple audio channels exist
- [ ] Web Speech API self-voicing toggle — player speaks scene descriptions
- [ ] Spatial audio via `PannerNode` when game provides positional data
- [ ] `localStorage` persistence
- [ ] Tests + docs
- [ ] Commit: `feat(wasm): a11y sound customisation`

---

## Glyph — deferred from audio POC

- HAL audio surface (`tone_play`, `sfx_play`) — needed for game sigils
- SFX: sweep, noise burst — melody proved the pipeline
- ADSR envelopes on WASM side
- WASM player as wrapper (bundle player + content for deploy)
- Keyboard input mechanism (store key state in WASM memory)

## Cross-project

- incant to follow same debug pipeline as glyph, with wrappers for testing
- Better build pipeline later that combines stuff
