# Jetpac TAP Analysis — Multi-Stage Loader and System Variable Patching

Analysis of the Jetpac TAP file structure. The file is structurally valid (all checksums pass) but uses a tricky multi-stage loader that some emulators struggle with.

## Block inventory

| Block | Name | Type | Address | Size | What it is |
|-------|------|------|---------|------|-----------|
| 1–2 | Jetpac1 | Program | — | 94 | BASIC autostart loader |
| 3–4 | Jetpac2 | Code | 0x4000 | 6912 | Loading screen (fills screen memory) |
| 5–6 | Jetpac3 | Code | 0x6000 | 8192 | Main game code |
| 7–8 | Jetpac4 | Code | 0x5B80 | 15 | Relocation stub — copies code then jumps to entry |
| 9–10 | Jetpac5 | Code | 0x5CB0 | 1 | Single `JP (HL)` instruction (0xE9) |
| 11–12 | Jetpac6 | Code | 0x5C78 | 2 | ERR_SP system variable patch (0x5A83) |

## Memory map

```
0x4000–0x5AFF  Screen memory (6912 bytes)        ← Jetpac2
0x5B00–0x5CB5  System variables                   ← Jetpac4, 5, 6 load HERE
  0x5B80       Relocation stub (15 bytes)         ← Jetpac4
  0x5C78       ERR_SP system variable             ← Jetpac6 patches this
  0x5CB0       Inside system area                 ← Jetpac5
0x5CB6+        BASIC program / workspace
0x6000–0x7FFF  Main game code (8192 bytes)        ← Jetpac3
```

## BASIC loader (line 1, autostart)

```basic
1 CLEAR VAL "24575": ... : FOR q=2 TO 6: LOAD "Jetpac"+STR$(q) CODE: NEXT q: PRINT USR 24576
```

Loads 5 blocks in a loop, then jumps to 0x6000.

## The relocation stub (Jetpac4 at 0x5B80)

```z80
5B80  LD HL, 6004h    ; source = game code + 4 byte offset
5B83  LD DE, 6000h    ; dest = start of game code
5B86  LD BC, 2000h    ; length = 8192 bytes
5B89  LDIR            ; block copy (overlapping, forward)
5B8B  JP 61E5h        ; jump to game entry point
```

This is a self-modifying trick. The LDIR copies from 0x6004 to 0x6000 — a 4-byte overlap. On real hardware this works because LDIR copies byte-by-byte forward, so each destination byte is written before it's needed as a source. The net effect is the first 4 bytes of the game code get overwritten with bytes from offset +4, then execution jumps to the real entry point.

## System variable patching

The loader writes directly into the Spectrum's system variable area using `LOAD CODE`:

- **0x5C78 (ERR_SP)** — the error stack pointer. Patching this redirects where the Spectrum goes on error, part of the copy protection / anti-break scheme.
- **0x5CB0** — `JP (HL)` instruction. If something tries to break into the program, this redirects execution.

This is a common Ultimate Play The Game technique from 1983. By patching ERR_SP and planting jump instructions in the system area, pressing BREAK doesn't return to BASIC — it either crashes or jumps back into the game.

## Why it fails in some emulators

The TAP format is valid. The problem is the loader's runtime behaviour:

1. **Multi-block LOAD loop** — the BASIC `FOR q=2 TO 6: LOAD ... CODE: NEXT q` does 5 sequential tape loads. Some emulators with "fast load" or "instant load" don't handle multiple LOADs from a single BASIC line correctly.
2. **System variable writes** — loading code into 0x5B80/0x5C78/0x5CB0 modifies the Spectrum's own housekeeping. Emulators that protect or shadow the system area may reject these loads silently.
3. **Block naming** — the loader expects blocks named "Jetpac2" through "Jetpac6" in order. If the emulator's tape simulation doesn't match names correctly, it may skip blocks or load them out of order.

## Fix approaches

1. **Use FUSE** — most accurate Spectrum emulator, handles multi-load TAPs correctly
2. **Disable fast load** — some emulators' speed hacks break multi-stage loaders
3. **Rebuild the TAP** — combine all code into a single block with a simple loader, bypassing the multi-stage tricks entirely

## Relevance to incant / WASM security research

This is a real-world example of binary-level protection techniques:

- **Code loaded into system areas** — equivalent to patching runtime metadata in WASM (table entries, memory layout)
- **Self-relocating code** — the LDIR stub modifies code in place before jumping to it. In WASM terms, this is like rewriting function bodies via linear memory
- **Error handler hijacking** — patching ERR_SP to prevent break-out. Analogous to trap handler manipulation in WASM runtimes
- **Multi-stage loading** — the program isn't usable until all stages complete. Similar to WASM module linking and lazy instantiation patterns

The Z80 Spectrum is a useful testbed because the protection techniques are simple enough to fully understand, but the concepts map directly to modern binary security.

## Rebuild: jetpac_fixed.tap

The original TAP is structurally valid but the multi-stage loader fails in some emulators. We rebuilt it with `tap_rebuild_jetpac.py`.

### What the rebuild does

1. **Extracts** screen data (6912 bytes at 0x4000) and game code (8192 bytes at 0x6000) from the original
2. **Applies the relocation offline** — the original stub at 0x5B80 does an LDIR that shifts code left by 4 bytes at runtime. We do that shift in Python instead, producing the post-relocation binary
3. **Strips the copy protection** — the system variable patches (ERR_SP hijack at 0x5C78, JP(HL) at 0x5CB0) are anti-BREAK tricks. We skip them entirely
4. **Wraps in a simple 3-block TAP:**

```
Block 1–2: BASIC loader (autostart line 10)
  10 CLEAR VAL "24575"
  20 LOAD ""SCREEN$
  30 LOAD ""CODE
  40 RANDOMIZE USR VAL "25061"

Block 3–4: Screen data → 0x4000 (6912 bytes)
Block 5–6: Game code → 0x6000 (8192 bytes, pre-relocated)
```

### Entry point

The original stub ends with `JP 61E5h`. After applying the 4-byte left-shift relocation, 0x61E5 in the final memory layout should be the game's real entry point. Our loader calls `USR 25061` (= 0x61E5).

### If it doesn't work

The relocation is the tricky part. The stub does:

```z80
LD HL, 6004h    ; source
LD DE, 6000h    ; dest
LD BC, 2000h    ; 8192 bytes
LDIR            ; byte-by-byte forward copy with 4-byte overlap
JP 61E5h
```

LDIR with overlapping source/dest copies byte-by-byte forward, so each byte is read before it's overwritten. Our Python replicates this. But if the game relies on timing, interrupt state, or other side effects of the original loader sequence (e.g. the system variable patches being present when code runs), the pre-relocated version may behave differently.

Things to try if it fails:
- **Different entry point** — try `USR 24576` (0x6000) instead, in case the first instruction is the real entry
- **Include the stub** — load the stub at 0x5B80 and the original (un-relocated) code at 0x6000, then `USR 23424` to let the stub do its own LDIR
- **Include the sysvar patches** — the game might check for the ERR_SP value as a self-integrity test
