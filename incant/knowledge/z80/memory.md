# Z80 Memory Map (ZX Spectrum 48K)

## Address space

```
0x0000 - 0x3FFF  ROM (16K) — Spectrum BASIC, not writable
0x4000 - 0x57FF  Screen bitmap (6144 bytes)
0x5800 - 0x5AFF  Screen attributes / colour (768 bytes)
0x5B00 - 0x5BFF  System variables
0x5C00 - 0xFFFF  Free RAM (~41K available for programs)
```

## For incant code generation

When generating standalone Z80 code (not a full Spectrum program), use:

- **Code starts at 0x0000** — the Z80 always starts execution at address 0
- **Stack at 0xFFFE** — set SP to top of memory, grows downward
- **Working memory at 0x8000** — safe area for data storage
- **End with HALT** — stops execution cleanly for the test harness

## Addressing modes

- **Immediate:** `LD A, 42` — value is in the instruction
- **Register:** `LD A, B` — value is in a register
- **Register indirect:** `LD A, (HL)` — HL holds the memory address
- **Absolute:** `LD A, (0x8000)` — address is in the instruction
- **Indexed:** `LD A, (IX+5)` — IX plus signed offset
- **Relative:** `JR offset` — PC plus signed 8-bit offset

## Important constraints

- All addresses are 16-bit (0x0000 to 0xFFFF = 64K total)
- No memory protection — any address is readable and writable (except ROM on real hardware)
- Stack grows downward: PUSH decrements SP by 2, POP increments by 2
- Z80 is little-endian: low byte stored first
