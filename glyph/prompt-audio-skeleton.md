# Glyph Audio Prompt Skeleton

Template for generating audio via LLMs (Claude, GPT, Gemini, etc.)
that plays on both WASM (Web Audio API) and Z80 (1-bit beeper).

The LLM outputs a JSON spec. Both engines consume the same format.

---

## Output format

The LLM produces a `glyph-audio` JSON object. Every field maps directly
to the HAL audio surface — no post-processing needed.

```json
{
  "type": "glyph-audio",
  "version": 1,
  "pieces": [
    {
      "id": "title_theme",
      "kind": "melody",
      "bpm": 120,
      "notes": [
        { "note": "C4", "ms": 500 },
        { "note": "E4", "ms": 250 },
        { "note": "G4", "ms": 250 },
        { "note": "C5", "ms": 1000 }
      ]
    },
    {
      "id": "jump",
      "kind": "sfx",
      "sfx": "sweep_up"
    },
    {
      "id": "hit",
      "kind": "sfx",
      "sfx": "noise_burst"
    },
    {
      "id": "coin_pickup",
      "kind": "sfx",
      "sfx": "coin"
    },
    {
      "id": "alert",
      "kind": "tone",
      "note": "A4",
      "ms": 200
    }
  ]
}
```

---

## Piece kinds

### `melody` — a sequence of notes

```json
{
  "id": "level_music",
  "kind": "melody",
  "bpm": 140,
  "loop": true,
  "notes": [
    { "note": "C4", "ms": 214 },
    { "note": "REST", "ms": 107 },
    { "note": "E4", "ms": 214 }
  ]
}
```

- `note` — pitch name: `C3`–`C6`, sharps as `C#4`, flats as `Db4`
- `ms` — duration in milliseconds (both targets convert to their native timing)
- `REST` — silence for the given duration
- `bpm` — optional, advisory for the LLM (durations are absolute, not derived from bpm)
- `loop` — optional, default false. WASM loops seamlessly; Z80 restarts from the top

### `sfx` — a named sound effect

```json
{
  "id": "explosion",
  "kind": "sfx",
  "sfx": "explosion"
}
```

Built-in SFX types (both targets implement these):

| SFX ID | WASM behaviour | Z80 behaviour |
|--------|---------------|---------------|
| `sweep_up` | Frequency sweep 200→800 Hz, 150ms | Descending delay loop (rising pitch) |
| `sweep_down` | Frequency sweep 800→200 Hz, 150ms | Ascending delay loop (falling pitch) |
| `noise_burst` | White noise, 100ms decay | Random OUT toggling, 80ms |
| `coin` | Two-tone ping (E5→B5), 100ms | Two quick beeps |
| `explosion` | Noise + low sweep, 300ms | Long random toggle burst |

### `tone` — a single note

```json
{
  "id": "beep",
  "kind": "tone",
  "note": "A4",
  "ms": 200
}
```

Simplest form — one note, one duration.

---

## Note range

The shared range is **C3 to C6** (3 octaves). Both targets can produce these.

| Note | Frequency | Z80 half-period (T-states) |
|------|-----------|---------------------------|
| C3 | 130.8 Hz | 13,379 |
| C4 | 261.6 Hz | 6,689 |
| A4 | 440.0 Hz | 3,977 |
| C5 | 523.3 Hz | 3,344 |
| C6 | 1046.5 Hz | 1,672 |

Below C3 the Z80 can't maintain pitch during interleaving.
Above C6 the beeper loop overhead distorts the waveform.

---

## Base prompt structure

```
Generate a glyph-audio JSON spec for [game/scene description].

Requirements:
- Output valid JSON matching the glyph-audio format (version 1)
- Use note names from C3 to C6 (e.g. C4, F#4, Bb3)
- Durations in milliseconds — keep above 50ms (Z80 can't resolve shorter)
- Melodies: simple, catchy, 4–16 bars. Think chiptune — not orchestral
- SFX: use built-in types (sweep_up, sweep_down, noise_burst, coin, explosion)
- This will play on a 1-bit beeper AND full Web Audio — keep melodies
  monophonic (one note at a time). No chords.

Style guidance:
- [upbeat / dark / tense / playful / triumphant]
- [tempo: slow ~80 bpm / medium ~120 bpm / fast ~160 bpm]
- [reference: "like Manic Miner loading music" / "like Jetpac SFX"]
```

---

## Example prompts

### Game title theme

```
Generate a glyph-audio JSON spec for a retro space shooter title screen.

Requirements:
- One melody: "title_theme" — 8 bars, heroic feel, 140 bpm
- Three SFX: "laser" (sweep_up), "explosion" (explosion), "powerup" (coin)
- Keep the melody in the C4–C5 range for clean beeper output
- Output valid JSON, glyph-audio version 1

Style: triumphant, energetic, think Jetpac or Manic Miner
```

### Puzzle game audio set

```
Generate a glyph-audio JSON spec for a puzzle game.

Requirements:
- One melody: "background" — 16 bars, calm and repetitive, 100 bpm, loop: true
- SFX: "place_piece" (coin), "clear_row" (sweep_up), "game_over" (sweep_down)
- One tone: "tick" — short A5 beep, 80ms
- Keep it minimal — this plays on repeat

Style: calm, methodical, like Tetris but simpler
```

### Card game (KAOS 9 style)

```
Generate a glyph-audio JSON spec for a card battle game.

Requirements:
- One melody: "battle_theme" — 8 bars, tense, 130 bpm
- SFX: "card_play" (coin), "damage" (noise_burst), "victory" (sweep_up)
- One tone: "turn_start" — C5, 150ms
- Melodies should feel aggressive but stay in C3–C5

Style: dark, intense, think combat music on limited hardware
```

---

## Post-generation checklist

After the LLM generates the JSON:

1. **Validate JSON** — parse it. LLMs sometimes produce trailing commas or comments.
2. **Check note range** — all notes within C3–C6? Flag anything outside.
3. **Check durations** — anything under 50ms? The Z80 can't play it. Merge or extend.
4. **Check melody length** — over 64 notes gets unwieldy on Z80 (memory). Split if needed.
5. **Check SFX types** — only the 5 built-in types are supported. Map anything else.
6. **Test both targets:**
   - WASM: `npx ts-node glyph/audio-wasm/demo-cli.ts spec.json`
   - Z80: `python -m glyph.z80.build_audio spec.json --output audio.bin` then play in zxw
