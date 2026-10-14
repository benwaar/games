# glyph/z80 — SCR loader

Loads a `.scr` file into ZX Spectrum screen memory and halts.

## What an SCR file is

A `.scr` file is exactly 6912 bytes — the raw contents of the Spectrum's video RAM:

- `0x4000–0x57FF` — pixel bitmap (6144 bytes, 192 rows x 32 columns)
- `0x5800–0x5AFF` — colour attributes (768 bytes, 24 rows x 32 columns)

The ULA hardware reads these addresses directly to drive the display. Copying a valid
SCR into this region is all that's needed to show an image.

## Files

| File | Purpose |
|------|---------|
| `scr_loader.asm` | Z80 assembly source — 12-byte ldir loop |
| `build_scr.py` | Assemble + concatenate loader + SCR data into a single `.bin` |
| `tests/test_scr_loader.py` | Assembly, concat, and emulation tests |

## How it works

The loader is five instructions:

```
ld hl, scr_data   ; source: SCR bytes appended after this code
ld de, 0x4000     ; dest: screen memory start
ld bc, 6912       ; length: full screen
ldir              ; block copy (HL → DE, BC bytes)
halt
```

`ldir` is the Z80 block-copy instruction. It copies one byte per iteration,
incrementing HL and DE, decrementing BC, repeating until BC == 0.

The `build_scr.py` script appends 6912 bytes of SCR data immediately after the
assembled code so the `scr_data` label resolves to the right address at link time.

## Build

Requires the `z80` pip package (used by incant — activate the incant venv):

```bash
source ../incant/.venv/bin/activate
python build_scr.py path/to/image.scr --output scr_loader.bin
```

The output `scr_loader.bin` loads at `0x8000`. Execute from that address in the
zxw emulator to display the image.

## Tests

```bash
source ../incant/.venv/bin/activate
python -m pytest glyph/z80/tests/ -v
```

All tests run from the repo root (`games/`).
