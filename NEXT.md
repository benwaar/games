# Next

## wex — a11y Phase 3: Sound customisation (feature/no-ticket-wasm-responsive)

Research: `wex/A11Y.md` | `wex/SCALING.md`

- [x] Separate volume controls (SFX / narration / music) when multiple audio channels exist
- [x] Web Speech API self-voicing toggle — player speaks scene descriptions
- [x] Spatial audio via `PannerNode` when game provides positional data
- [x] `localStorage` persistence
- [x] Tests + docs
- [ ] Commit: `feat(wex): a11y sound customisation`

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
