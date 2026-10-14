# Next 

Audio WASM: the wasm-pack/Rust approach produces a separate JS glue file — wrong pattern.
Should follow the same pure WAT approach as images: embed freq/duration arrays as WAT data
segments, compile with wat2wasm, HTML reads from WASM memory and plays via Web Audio.
No Rust, no wasm-pack, no extra files. Need to decide the export contract:
  - WASM exports `type()` (0=image, 1=audio) so the HTML knows which payload it's hosting
  - Audio WASM exports `audio_count()` + `audio_offset()` for the note data in memory
  - Image WASM exports `render()`, `width()`, `height()`, `memory` as now
  - HTML stays static — just branches on `type()` to blit pixels or play oscillators

wrap wasm with the cross or something 

pretty much done the poc, 
wrap both together,
nail down the pipeline in docs
check for any unused code and add tests

incant to follow same debug pipeline as glph, with wrappers for testing

do a better build pipeline later that combines stuff

---

# Next — Glyph Audio POC: Sound → Both Targets



**Goal:** Play audio on both targets from one spec. Extends the HAL with sound — same principle as the image POC: same intent, native implementation per platform.

```
audio spec (tone, SFX, melody)
  ├── WASM: Web Audio API — oscillators, envelopes, full fidelity
  └── Z80:  Spectrum beeper — 1-bit audio, port 0xFE, timed loops
```

---

## Why this POC

1. **Audio is required.** Void Duel, Artefact, and any game sigil needs sound. Can't ship games without it.
2. **1-bit audio is a real constraint.** The Spectrum has no sound chip (48K) — just a speaker toggled via port 0xFE. Generating tones means cycle-counted loops. This is the audio equivalent of the colour-clash problem.
3. **HAL audio surface.** Extends the HAL API with `tone_play`, `sfx_play`, `music_play` — same calls, radically different implementations.
4. **Feeds into Artefact.** The beeper engine built here becomes the playback target for Artefact's generative chip-tunes.

---

## Milestones

### A0 — zxw beeper support (prerequisite)

The emulator currently ignores beeper output. Add 1-bit audio so A2 can be tested.

- [ ] Rust: capture bit 4 from `port_write` on port 0xFE in `spectrum.rs`
- [ ] Accumulate 1-bit samples per T-state into a ring buffer
- [ ] Expose sample buffer to JS via wasm-bindgen
- [ ] JS: Web Audio API playback — drain buffer into AudioWorklet/ScriptProcessor
- [ ] Test: a known beeper program (simple tone) produces audible output in the browser

**Gate:** A simple `OUT (0xFE)` toggle loop in zxw produces audible sound in the browser.

---

### A1 — Explainer: 1-bit audio + beeper engine

Write `glyph/explainers/beeper-audio.md`.

- [ ] How the Spectrum beeper works: OUT (0xFE), bit 4, timed toggles
- [ ] Frequency from cycle counting: `freq = 3,500,000 / (2 × loop_cycles)`
- [ ] Pulse wave basics: duty cycle, volume is on/off only
- [ ] Multi-channel illusion: interleaving (rapid switching between frequencies)
- [ ] Connection to Web Audio: oscillator nodes, gain envelopes — same concepts, better hardware
- [ ] "Coming from JS/TS": Web Audio API mapping to beeper equivalents
- [ ] "In practice": embedded audio, resource-constrained playback, codec design

**Gate:** Explainer reviewed. Someone could read it and understand both targets.

---

### A2 — Z80 beeper engine

Build the Z80 tone generator. This is the hard target — WASM is easy once this works.

- [ ] Single tone: given frequency + duration, generate beeper output
- [ ] Frequency table: map note names (C4, D4, ...) to cycle counts
- [ ] Duration from frame count (50 Hz interrupt)
- [ ] Simple SFX: ascending/descending sweep, noise burst
- [ ] Simple melody: sequence of (note, duration) pairs, played in order
- [ ] Test in zxw emulator — audible output
- [ ] `glyph/z80/beeper.asm` — reusable beeper routines
- [ ] `glyph/z80/build_audio.py` — assembler + test harness

**Gate:** `python -m glyph.z80.build_audio melody.json --output melody.bin` produces a binary that plays a recognisable tune in zxw.

---

### A3 — WASM audio engine

The WASM target uses Web Audio API — full fidelity, no constraints.

- [ ] Rust crate or JS module: tone playback via Web Audio oscillators
- [ ] Same frequency table as A2 (same notes, same API)
- [ ] Envelope: attack/decay/sustain/release (ADSR) — luxury the beeper can't do
- [ ] SFX: same sweep/noise as A2 but richer
- [ ] Melody: same sequence format as A2
- [ ] Runs in browser alongside glyph-wasm renderer

**Gate:** Same melody JSON plays in browser with Web Audio. Richer than Z80 but same tune.

---

### A4 — HAL audio surface

Define and document the audio API that game sigils will call.

- [ ] `tone_play(note, duration_ms)` — play a single tone
- [ ] `sfx_play(id)` — trigger a named sound effect
- [ ] `music_play(melody_id)` — start a melody sequence
- [ ] `music_stop()` — stop current melody
- [ ] Document per-target behaviour: WASM gets envelopes, Z80 gets beeper approximation
- [ ] Add to `glyph/knowledge/` for RAG embedding
- [ ] Update `glyph/CLAUDE.md` HAL surface with audio calls

**Gate:** HAL audio calls documented. Both targets implement the same interface.

---

### A5 — One spec, both targets (audio)

Wire it together — same pattern as the image POC.

- [ ] Spec format: `type: play_melody`, `notes: [{note: "C4", ms: 500}, ...]`
- [ ] CLI: `python -m glyph cast spec.yaml --target z80` → beeper binary
- [ ] CLI: `python -m glyph cast spec.yaml --target wasm` → Web Audio build
- [ ] Side-by-side: Z80 beeper vs WASM Web Audio, same melody
- [ ] Document the contrast: 1-bit vs full-fidelity from one spec

**Gate:** One spec, two outputs, both audibly play the same tune.

---

## Skills covered

| Skill | Study guide ref | Status before |
|-------|----------------|---------------|
| 1-bit audio / beeper engine | Systems / Retro audio | New |
| Web Audio API | Web platform | New |
| Cycle-counted Z80 timing | Systems / Emulation | Extends zxw |
| Audio HAL design | New (glyph) | New |
| Cross-platform audio | New (glyph) | New |

---

## Feeds into

- [Artefact](../artefact/) — beeper engine is the playback target for generative chip-tunes
- [Void Duel](../void-duel/) — game SFX on both platforms
- glyph M1–M5 — HAL audio calls used by game sigils

---

## Multi-agent build approach

A0 (zxw beeper) must come first. Then A1, A2, A3 can run in parallel:

- **Agent A** — A1: beeper explainer
- **Agent B** — A2: Z80 beeper engine (depends on A0 for testing)
- **Agent C** — A3: WASM Web Audio engine

A4 (HAL surface) and A5 (integration) run after all three converge.
