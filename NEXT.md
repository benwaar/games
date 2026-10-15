# Next

wrap wasm with the cross or something

pretty much done the poc,
wrap both together,
nail down the pipeline in docs
check for any unused code and add tests

incant to follow same debug pipeline as glyph, with wrappers for testing

do a better build pipeline later that combines stuff

---

# Glyph Audio POC — status

**Goal:** Play audio on both targets from one spec. Same intent, native implementation per platform.

```
audio spec (glyph-audio JSON)
  ├── WASM: Web Audio oscillators — pure WAT, data in memory
  └── Z80:  Spectrum beeper — 1-bit audio, port 0xFE, cycle-counted loops
```

## Done

- [x] **A0** — zxw beeper support: T-state accurate sampling + Web Audio playback
- [x] **A1** — explainer: `glyph/explainers/beeper-audio.md`
- [x] **A2** — Z80 beeper engine: `glyphz80/beeper.py`, 34 tests, TAP via `incant.tap`
- [x] **A3** — Pure-WAT audio WASM: freq/duration f64 arrays as data segments, `wat2wasm`
- [x] **WASM player** — generic player (`wasm/index.html`): image + audio + keyboard via `type()` bitmask
- [x] **zxw debug mode** — `zxw/www/debug.html`: auto-load ROM + TAP + boot + LOAD ""
- [x] **Pipeline** — `pipeline.py` handles image and audio → both targets

## Remaining

- [x] Verify audio playback in browser (Play button on WASM player)
- [ ] Verify TAP audio in zxw debug mode (needs ROM file)
- [x] Add pipeline tests (audio WAT generation, type bitmask) — 34 tests in `glyph/tests/test_pipeline.py`
- [ ] WASM player: keyboard input mechanism (store key state in WASM memory)
- [x] Combined image+audio WASM (type=3: image with soundtrack)
- [x] Remove glyphwasm Rust crate (replaced by pure WAT)

## Deferred

- HAL audio surface (`tone_play`, `sfx_play`) — needed for game sigils, not this POC
- SFX: sweep, noise burst — melody is enough to prove the pipeline
- ADSR envelopes on WASM side — future enhancement
- WASM player as wrapper (bundle player + content for deploy)

## WASM export contract

```
type()         → i32    # bitmask: bit 0=image, bit 1=audio, bit 2=keyboard
width()        → i32    # canvas dimensions (image only)
height()       → i32
render()                # populate framebuffer (image only)
audio_count()  → i32    # number of notes (audio only)
audio_offset() → i32    # byte offset of note data (audio only)
memory                  # linear memory
```

Audio data layout at `audio_offset()`: `f64[count]` frequencies, then `f64[count]` durations.

---

## Skills covered

| Skill | Status |
|-------|--------|
| 1-bit audio / beeper engine | Done |
| Web Audio API | Done |
| Cycle-counted Z80 timing | Done |
| Pure WAT generation | Done |
| WASM export contract design | Done |

## Feeds into

- [Artefact](../artefact/) — beeper engine is the playback target for generative chip-tunes
- [Void Duel](../void-duel/) — game SFX on both platforms
- glyph game sigils — HAL audio calls (future)
