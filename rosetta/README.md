# Rosetta

Round-trip translation pipeline: Z80 binary → glyph intermediate → Z80 + WASM output, verified against the known original.

## What this is

Given a Z80 binary (game ROM, TAP code block), Rosetta:

1. **Extracts** routines from the binary (sound effects, screen loaders, game logic)
2. **Translates** them into glyph's intermediate representation (sigils)
3. **Emits** back to both targets: Z80 assembly (TAP) and WASM (WAT → wasm)
4. **Verifies** output matches the original — byte-identical Z80, behaviourally equivalent WASM

The core claim: if you can go Z80 → glyph → Z80 and get the same thing back, the intermediate representation is faithful. If you can also go → WASM and get equivalent behaviour, the translation is correct.

## Why

- **Proves the glyph pipeline** end-to-end with real-world input, not hand-authored sigils
- **Feeds artefact** — ROM scanner finds sound routines, Rosetta translates them, artefact plays them
- **Tests incant** — the local LLM + RAG pipeline generates the output, Rosetta verifies it
- **Builds the knowledge corpus** — binary analysis patterns feed back into the MCP for future extraction

## Pipeline

```
┌─────────────┐     ┌───────────┐     ┌──────────────┐
│  Z80 binary  │────▶│  extract   │────▶│  glyph sigil │
│  (TAP/ROM)   │     │  (analyse) │     │  (IR)        │
└─────────────┘     └───────────┘     └──────┬───────┘
                                             │
                                    ┌────────┴────────┐
                                    ▼                  ▼
                              ┌──────────┐      ┌──────────┐
                              │  Z80 asm  │      │  WAT/WASM │
                              │  (TAP)    │      │  (wex)    │
                              └────┬─────┘      └────┬─────┘
                                   │                  │
                                   ▼                  ▼
                              ┌──────────┐      ┌──────────┐
                              │  verify   │      │  verify   │
                              │  vs orig  │      │  behaviour│
                              └──────────┘      └──────────┘
```

## Milestones

### M1 — Extract and replay (Z80 → Z80)

Extract a known sound routine from j83, emit it as standalone Z80 assembly, build a test TAP, compare output against the original bytes.

- [ ] Sound routine extractor — identify OUT (FE) sites, trace call boundaries
- [ ] Z80 emitter — extracted routine → relocatable assembly
- [ ] TAP builder — wrap in BASIC loader + test harness
- [ ] Byte comparison — original bytes vs emitted bytes (must match)

### M2 — Glyph intermediate representation

Define the sigil format for extracted routines. A sigil captures what the routine does (square wave at frequency F for duration D) not how (DJNZ loop count).

- [ ] Sigil schema for beeper effects (tone, sweep, noise)
- [ ] Z80 → sigil translator (pattern-match known beeper idioms)
- [ ] Sigil → Z80 emitter (generate equivalent assembly from sigil)
- [ ] Round-trip test: Z80 → sigil → Z80, verify byte-identical or behaviourally equivalent

### M3 — WASM target

Emit the same sigil as WASM audio via glyph's existing WAT pipeline.

- [ ] Sigil → WAT emitter (AudioWorklet-compatible output)
- [ ] Behavioural test: Z80 beeper output vs WASM audio output (waveform comparison)
- [ ] Both outputs from same sigil — proves the IR is target-independent

### M4 — Corpus-assisted extraction

Use the knowledge corpus MCP (z80-binary-analysis category) to assist extraction. Local LLM identifies routines, the corpus provides pattern knowledge.

- [ ] Binary analysis corpus category (see NEXT.md)
- [ ] Incant integration — local LLM + RAG for routine identification
- [ ] Automated extraction pipeline — binary in, sigils out, no manual disassembly

### M5 — Multi-routine and game-scale

Scale beyond single sound effects to full sound engines, screen routines, and game logic.

- [ ] Sound engine extraction (channel drivers, note tables, frame ticks)
- [ ] Screen routine extraction (attribute painting, scroll, sprite blitting)
- [ ] Composition — multiple sigils assembled into a complete program

## Dependencies

| Project | What Rosetta uses |
|---------|-------------------|
| glyph | Sigil format, WAT pipeline, Z80 assembly pipeline |
| incant | Local LLM inference, RAG over knowledge corpus |
| zxw | Emulator for Z80 verification, AudioWorklet for playback |
| wex | WASM player for WASM verification |
| knowledge corpus | z80-binary-analysis category (to be built) |

## Status

Planned. POC work done: j83 sound analysis in `_data/j83-analysis-sound.md`, manual extraction proved the approach.
