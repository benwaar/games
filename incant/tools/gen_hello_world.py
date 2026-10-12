#!/usr/bin/env python3
"""Generate a Spectrum hello-world TAP using incant's TAP writer.

The Z80 program:
  - Opens screen channel (CALL 0x1601 with A=2)
  - Loops: print AT 10,8 + ink colour + "HELLO WORLD", cycle colour, delay
  - Checks BREAK (Space + Caps Shift) each loop — returns to BASIC on break

Uses ROM routines: RST 0x10 (print char), 0x1601 (open channel).
Loaded at 32768 (0x8000), auto-run via BASIC: 10 RANDOMIZE USR 32768.
"""

import sys
from pathlib import Path

# Add parent to path so we can import incant
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incant.tap import write_spectrum_tap


def assemble_hello_world() -> bytes:
    """Hand-assemble the hello world Z80 program.

    ORG 0x8000 (32768)

    start:
        ld a, 2             ; channel 2 = upper screen
        call 0x1601         ; ROM: OPEN_CHANNEL
        ld a, 1             ; blue border
        out (0xFE), a

    loop:
        ; Print AT 10,8 (position cursor mid-screen)
        ld a, 22            ; AT control code
        rst 0x10
        ld a, 10            ; row
        rst 0x10
        ld a, 8             ; column
        rst 0x10

        ; Set INK colour (cycles 1–7)
        ld a, 16            ; INK control code
        rst 0x10
        ld a, (colour)
        rst 0x10

        ; Print "HELLO WORLD"
        ld hl, message
        ld b, msg_len
    print:
        ld a, (hl)
        rst 0x10
        inc hl
        djnz print

        ; Cycle colour 1→2→3→4→5→6→7→1
        ld a, (colour)
        inc a
        cp 8
        jr nz, no_wrap
        ld a, 1
    no_wrap:
        ld (colour), a

        ; Delay loop (~0.3s at 3.5MHz)
        ld bc, 0x4000
    delay:
        dec bc
        ld a, b
        or c
        jr nz, delay

        ; Check BREAK: Space (row 7 bit 0) + Caps Shift (row 0 bit 0)
        ld a, 0x7F          ; select half-row 7
        in a, (0xFE)        ; read port 0x7FFE
        bit 0, a            ; Space?
        jr nz, loop         ; not pressed → keep going

        ld a, 0xFE          ; select half-row 0
        in a, (0xFE)        ; read port 0xFEFE
        bit 0, a            ; Caps Shift?
        jr nz, loop         ; not pressed → keep going

        ; Both pressed — return to BASIC
        ret

    colour:
        defb 2              ; start with red ink

    message:
        defm "HELLO WORLD"
    """
    code = bytearray()

    # start:
    code += bytes([0x3E, 0x02])             # ld a, 2
    code += bytes([0xCD, 0x01, 0x16])       # call 0x1601
    code += bytes([0x3E, 0x01])             # ld a, 1
    code += bytes([0xD3, 0xFE])             # out (0xFE), a

    # loop: (offset 9)
    loop_addr = len(code)

    code += bytes([0x3E, 22])               # ld a, 22 (AT)
    code += bytes([0xD7])                   # rst 0x10
    code += bytes([0x3E, 10])               # ld a, 10 (row)
    code += bytes([0xD7])                   # rst 0x10
    code += bytes([0x3E, 8])                # ld a, 8 (col)
    code += bytes([0xD7])                   # rst 0x10

    code += bytes([0x3E, 16])               # ld a, 16 (INK)
    code += bytes([0xD7])                   # rst 0x10

    # ld a, (colour) — address will be patched below
    colour_ref1 = len(code) + 1
    code += bytes([0x3A, 0x00, 0x00])       # ld a, (colour) — placeholder
    code += bytes([0xD7])                   # rst 0x10

    # ld hl, message — address will be patched below
    msg_ref = len(code) + 1
    code += bytes([0x21, 0x00, 0x00])       # ld hl, message — placeholder
    msg_text = b"HELLO WORLD"
    code += bytes([0x06, len(msg_text)])     # ld b, msg_len

    # print:
    print_addr = len(code)
    code += bytes([0x7E])                   # ld a, (hl)
    code += bytes([0xD7])                   # rst 0x10
    code += bytes([0x23])                   # inc hl
    back = print_addr - (len(code) + 2)
    code += bytes([0x10, back & 0xFF])      # djnz print

    # Cycle colour
    colour_ref2 = len(code) + 1
    code += bytes([0x3A, 0x00, 0x00])       # ld a, (colour) — placeholder
    code += bytes([0x3C])                   # inc a
    code += bytes([0xFE, 0x08])             # cp 8
    code += bytes([0x20, 0x02])             # jr nz, no_wrap (+2)
    code += bytes([0x3E, 0x01])             # ld a, 1
    # no_wrap:
    colour_ref3 = len(code) + 1
    code += bytes([0x32, 0x00, 0x00])       # ld (colour), a — placeholder

    # Delay loop
    code += bytes([0x01, 0x00, 0x40])       # ld bc, 0x4000
    delay_addr = len(code)
    code += bytes([0x0B])                   # dec bc
    code += bytes([0x78])                   # ld a, b
    code += bytes([0xB1])                   # or c
    back = delay_addr - (len(code) + 2)
    code += bytes([0x20, back & 0xFF])      # jr nz, delay

    # Check BREAK
    code += bytes([0x3E, 0x7F])             # ld a, 0x7F
    code += bytes([0xDB, 0xFE])             # in a, (0xFE)
    code += bytes([0xCB, 0x47])             # bit 0, a
    back_to_loop = loop_addr - (len(code) + 2)
    code += bytes([0x20, back_to_loop & 0xFF])  # jr nz, loop

    code += bytes([0x3E, 0xFE])             # ld a, 0xFE
    code += bytes([0xDB, 0xFE])             # in a, (0xFE)
    code += bytes([0xCB, 0x47])             # bit 0, a
    back_to_loop2 = loop_addr - (len(code) + 2)
    code += bytes([0x20, back_to_loop2 & 0xFF]) # jr nz, loop

    # ret
    code += bytes([0xC9])                   # ret

    # colour variable
    colour_addr = len(code)
    code += bytes([0x02])                   # initial colour = 2 (red)

    # message data
    message_addr = len(code)
    code += msg_text

    # Patch addresses (absolute: base + offset)
    base = 0x8000
    abs_colour = base + colour_addr
    abs_message = base + message_addr

    code[colour_ref1] = abs_colour & 0xFF
    code[colour_ref1 + 1] = (abs_colour >> 8) & 0xFF
    code[colour_ref2] = abs_colour & 0xFF
    code[colour_ref2 + 1] = (abs_colour >> 8) & 0xFF
    code[colour_ref3] = abs_colour & 0xFF
    code[colour_ref3 + 1] = (abs_colour >> 8) & 0xFF
    code[msg_ref] = abs_message & 0xFF
    code[msg_ref + 1] = (abs_message >> 8) & 0xFF

    return bytes(code)


def main():
    output = Path(__file__).resolve().parent.parent / "output" / "hello_world.tap"
    output.parent.mkdir(parents=True, exist_ok=True)

    binary = assemble_hello_world()
    write_spectrum_tap(binary, "HelloWorld", 0x8000, str(output))

    print(f"Generated {output} ({output.stat().st_size} bytes)")
    print(f"  Code: {len(binary)} bytes at 0x8000")
    print(f"  4 blocks: BASIC loader + CODE")
    print(f"  Auto-runs: 10 RANDOMIZE USR 32768")

    # Also disassemble for inspection
    from incant.tap import disassemble
    print(f"\nDisassembly:\n{disassemble(binary, 0x8000)}")


if __name__ == "__main__":
    main()
