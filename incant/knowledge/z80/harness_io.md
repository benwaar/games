# Harness I/O — Memory-Mapped String I/O for Testing

Z80 has no stdin/stdout. The harness I/O convention provides a simple equivalent for testing string-based programs.

## Memory layout

```
0x8000–0x80FF  INPUT_BUF   (256 bytes, null-terminated string)
0x9000–0x90FF  OUTPUT_BUF  (256 bytes, null-terminated string)
```

The test harness pre-loads the input string at `INPUT_BUF` before execution. After the program halts, the harness reads from `OUTPUT_BUF` until the null terminator.

## Constants

Always define these at the top of your program:

```asm
INPUT_BUF  equ 0x8000
OUTPUT_BUF equ 0x9000
```

## Pattern: copy input to output

```asm
  ld hl, INPUT_BUF
  ld de, OUTPUT_BUF
copy:
  ld a, (hl)
  or a            ; check for null terminator
  jr z, done
  ld (de), a
  inc hl
  inc de
  jr copy
done:
  xor a
  ld (de), a      ; null-terminate output
  halt
```

## Pattern: prefix + input

Write a fixed prefix, then copy the input:

```asm
  ld de, OUTPUT_BUF
  ; write prefix bytes one at a time
  ld a, 'h'
  ld (de), a
  inc de
  ; ... more prefix bytes ...
  ; then copy input
  ld hl, INPUT_BUF
  ; ... copy loop as above ...
```

## On real hardware

On a ZX Spectrum, you'd replace the memory buffers with ROM calls:
- `RST 16` — print character in A to screen
- Spectrum INPUT routine at ROM address for keyboard input

The harness I/O convention lets us test the string logic without needing a full Spectrum emulator.
