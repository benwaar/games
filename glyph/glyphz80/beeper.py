"""Glyph-audio JSON to Z80 beeper binary converter.

Converts glyph-audio specs into raw Z80 machine code that plays melodies
on the ZX Spectrum beeper via port 0xFE bit 4.

CPU clock: 3,500,000 Hz (Spectrum 48K).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CPU_CLOCK = 3_500_000

# Delay loop: DEC HL (6T) + LD A,H (4T) + OR L (4T) + JR NZ (12T) = 26T per iteration
DELAY_LOOP_T = 26

# Overhead outside the loop body: PUSH HL (11T) + final DEC HL (6T) + LD A,H (4T)
# + OR L (4T) + JR NZ-fall-through (7T) + POP HL (10T) + RET (10T) = 52T
DELAY_OVERHEAD_T = 52

# play_note loop overhead per half-cycle (everything except the delay_hl body):
# LD A,E (4) + XOR (7) + LD E,A (4) + OUT (11) + CALL (17) + RET-from-delay (10)
# + DEC BC (6) + LD A,B (4) + OR C (4) + JR NZ (12) = 79T
PLAY_LOOP_OVERHEAD_T = 79

# Total overhead per half-cycle = delay_hl fixed cost + play_note loop cost
HALF_CYCLE_OVERHEAD_T = DELAY_OVERHEAD_T + PLAY_LOOP_OVERHEAD_T  # 131T

# Note name -> semitone offset from C in that octave
_NOTE_SEMITONES: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def note_to_midi(name: str) -> int:
    """Convert note name (e.g. 'C3', 'A#4') to MIDI number. C3=48, A4=69."""
    if name.upper() == "REST":
        return -1

    # Parse: letter + optional accidental + octave digit
    i = 1
    if len(name) > 1 and name[1] in "#b":
        i = 2
    pitch = name[:i]
    octave = int(name[i:])

    semitone = _NOTE_SEMITONES[pitch]
    # MIDI: C-1 = 0, so C{octave} = (octave + 1) * 12 + semitone
    return (octave + 1) * 12 + semitone


def midi_to_freq(midi: int) -> float:
    """MIDI number to frequency (Hz). A4 (69) = 440 Hz, equal temperament."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def note_to_freq(name: str) -> float:
    """Note name to frequency. Returns 0.0 for REST."""
    if name.upper() == "REST":
        return 0.0
    return midi_to_freq(note_to_midi(name))


def freq_to_half_period(freq: float) -> int:
    """Frequency to half-period in T-states."""
    if freq <= 0:
        return 0
    return round(CPU_CLOCK / (2.0 * freq))


def duration_to_cycles(freq: float, ms: int) -> int:
    """How many full wave cycles fit in the given duration."""
    if freq <= 0 or ms <= 0:
        return 0
    return max(1, round(freq * (ms / 1000.0)))


def half_period_to_hl(half_period_tstates: int) -> int:
    """Solve for HL delay loop counter from desired half-period T-states.

    actual_half_period = HL * 26 + 131  =>  HL = (half_period - 131) / 26
    """
    hl = (half_period_tstates - HALF_CYCLE_OVERHEAD_T) / DELAY_LOOP_T
    return max(1, round(hl))


def duration_to_rest_hl(ms: int) -> int:
    """For a REST, compute HL for the outer delay loop that gives roughly ms duration.

    We'll use a nested delay: outer loop BC iterations of inner delay HL.
    But simpler: total T-states = ms * CPU_CLOCK / 1000.
    We use a big delay loop. For long rests, we use BC * inner_delay approach.

    Actually, let's use the same delay_hl but called in a loop of BC iterations.
    Total T = BC * (HL * 26 + 52 + call/ret overhead).
    We pick HL=0xFFFF for max inner delay, then compute BC.
    """
    total_t = int(CPU_CLOCK * ms / 1000.0)
    return total_t


# ----- Z80 code generation -----

def _le16(val: int) -> bytes:
    """Little-endian 16-bit value."""
    return (val & 0xFFFF).to_bytes(2, "little")


def _emit_play_note(hl: int, bc: int) -> bytes:
    """Emit: LD HL, half_period; LD BC, cycles; CALL play_note"""
    code = bytearray()
    code += b"\x21" + _le16(hl)    # LD HL, nn
    code += b"\x01" + _le16(bc)    # LD BC, nn
    code += b"\xCD"                 # CALL play_note (address filled later)
    code += b"\x00\x00"             # placeholder for address
    return bytes(code)


def _emit_rest(ms: int) -> bytes:
    """Emit delay-only code for a REST note.

    Uses a nested loop: outer counter in BC, inner delay_hl with HL=max.
    For short rests, use smaller HL.
    """
    total_t = int(CPU_CLOCK * ms / 1000.0)

    # Inner loop with HL=0x0100 gives roughly 0x100 * 26 + 52 = 6708 T per call
    # Plus CALL (17T) + RET overhead already in delay_overhead
    # Let's use a simpler approach: call delay_hl in a loop.
    # Each call: HL * 26 + 52 + CALL(17) = HL * 26 + 69
    # We want BC calls where each has HL iterations.
    # Pick HL such that BC fits in 16 bits.

    # With HL = 1000: each call = 1000*26 + 69 = 26069 T
    # For 1 second = 3,500,000 T: BC = 3500000/26069 = ~134

    inner_hl = 1000
    t_per_call = inner_hl * DELAY_LOOP_T + DELAY_OVERHEAD_T + 17  # +17 for CALL itself
    bc = max(1, round(total_t / t_per_call))

    if bc > 0xFFFF:
        # Scale up inner HL
        inner_hl = 10000
        t_per_call = inner_hl * DELAY_LOOP_T + DELAY_OVERHEAD_T + 17
        bc = max(1, round(total_t / t_per_call))

    code = bytearray()
    code += b"\x21" + _le16(inner_hl)  # LD HL, inner_hl
    code += b"\x01" + _le16(bc)         # LD BC, bc
    # CALL rest_loop (address filled later)
    code += b"\xCD"
    code += b"\x00\x00"  # placeholder
    return bytes(code)


# Size constants for the subroutines
_PLAY_NOTE_SIZE = 0   # computed dynamically
_REST_LOOP_SIZE = 0


def build_play_note_routine() -> bytes:
    """The play_note subroutine. HL = half-period counter, BC = half-cycle count.

    Speaker state lives in E so A is free for the BC zero-check.

    play_note:
        LD E, 0x10
    .loop:
        LD A, E
        XOR 0x10
        LD E, A
        OUT (0xFE), A
        CALL delay_hl
        DEC BC
        LD A, B
        OR C
        JR NZ, .loop
        RET
    """
    code = bytearray()
    code += bytes([0x1E, 0x10])        # LD E, 0x10
    # .loop: (offset 2)
    code += bytes([0x7B])              # LD A, E
    code += bytes([0xEE, 0x10])        # XOR 0x10
    code += bytes([0x5F])              # LD E, A
    code += bytes([0xD3, 0xFE])        # OUT (0xFE), A
    code += bytes([0xCD, 0x00, 0x00])  # CALL delay_hl (patched)
    code += bytes([0x0B])              # DEC BC
    code += bytes([0x78])              # LD A, B
    code += bytes([0xB1])              # OR C
    code += bytes([0x20])              # JR NZ, .loop
    loop_start = 2
    jr_addr = len(code)
    code += bytes([(loop_start - (jr_addr + 1)) & 0xFF])
    code += bytes([0xC9])              # RET
    return bytes(code)


def build_delay_hl_routine() -> bytes:
    """The delay_hl subroutine. Delays HL * 26 + 52 T-states.

    delay_hl:
        PUSH HL           ; E5
    .loop:
        DEC HL            ; 2B
        LD A, H           ; 7C
        OR L              ; B5
        JR NZ, .loop      ; 20 FB
        POP HL            ; E1
        RET               ; C9
    """
    code = bytearray()
    code += bytes([0xE5])              # PUSH HL
    # .loop: (offset 1)
    code += bytes([0x2B])              # DEC HL
    code += bytes([0x7C])              # LD A, H
    code += bytes([0xB5])              # OR L
    code += bytes([0x20, 0xFB])        # JR NZ, .loop (-5 -> offset 1)
    code += bytes([0xE1])              # POP HL
    code += bytes([0xC9])              # RET
    return bytes(code)


def build_rest_loop_routine() -> bytes:
    """rest_loop: HL = inner delay count, BC = outer iterations.

    rest_loop:
        CALL delay_hl     ; CD xx xx (patched)
        DEC BC            ; 0B
        LD A, B           ; 78
        OR C              ; B1
        JR NZ, rest_loop  ; 20 F8
        RET               ; C9
    """
    code = bytearray()
    # rest_loop: (offset 0)
    code += bytes([0xCD, 0x00, 0x00])  # CALL delay_hl (patched)
    code += bytes([0x0B])              # DEC BC
    code += bytes([0x78])              # LD A, B
    code += bytes([0xB1])              # OR C
    code += bytes([0x20, 0xF8])        # JR NZ, rest_loop (-8 -> offset 0)
    code += bytes([0xC9])              # RET
    return bytes(code)


def compile_glyph_audio(spec: dict[str, Any], base_addr: int = 0x8000) -> bytes:
    """Compile a glyph-audio spec into a raw Z80 binary.

    Layout at base_addr:
      1. Note sequence (LD HL/BC + CALL for each note)
      2. HALT / RET
      3. play_note subroutine
      4. delay_hl subroutine
      5. rest_loop subroutine
    """
    if spec.get("type") != "glyph-audio":
        raise ValueError(f"Expected type 'glyph-audio', got '{spec.get('type')}'")

    pieces = spec.get("pieces", [])

    # Build note call stubs (addresses to be patched)
    # A short silence gap after each tone gives articulation between notes.
    gap_ms = 30
    note_stubs: list[tuple[str, bytes]] = []  # (kind, stub_bytes)
    for piece in pieces:
        if piece.get("kind") != "melody":
            continue
        for entry in piece.get("notes", []):
            note_name = entry["note"]
            ms = entry["ms"]
            if note_name.upper() == "REST":
                stub = _emit_rest(ms)
                note_stubs.append(("rest", stub))
            else:
                tone_ms = max(1, ms - gap_ms)
                freq = note_to_freq(note_name)
                hp = freq_to_half_period(freq)
                hl = half_period_to_hl(hp)
                cycles = duration_to_cycles(freq, tone_ms)
                stub = _emit_play_note(hl, cycles * 2)
                note_stubs.append(("note", stub))
                note_stubs.append(("rest", _emit_rest(gap_ms)))

    if not note_stubs:
        raise ValueError("No melody notes found in spec")

    # Calculate sizes
    # Each stub is 8 bytes: LD HL (3) + LD BC (3) + CALL (3) = 9 bytes... wait
    # LD HL,nn = 3 bytes, LD BC,nn = 3 bytes, CALL nn = 3 bytes = 9 bytes each
    # Actually: 21 xx xx = 3, 01 xx xx = 3, CD xx xx = 3 = 9 bytes
    stub_total = sum(len(s) for _, s in note_stubs)
    halt_size = 2  # DI + HALT

    play_note_code = build_play_note_routine()
    delay_hl_code = build_delay_hl_routine()
    rest_loop_code = build_rest_loop_routine()

    # Addresses
    play_note_addr = base_addr + stub_total + halt_size
    delay_hl_addr = play_note_addr + len(play_note_code)
    rest_loop_addr = delay_hl_addr + len(delay_hl_code)

    # Patch play_note: CALL delay_hl at offset 8 (bytes 9-10)
    play_note_patched = bytearray(play_note_code)
    play_note_patched[9] = delay_hl_addr & 0xFF
    play_note_patched[10] = (delay_hl_addr >> 8) & 0xFF

    # Patch rest_loop: CALL delay_hl at offset 0 (bytes 1-2)
    rest_loop_patched = bytearray(rest_loop_code)
    rest_loop_patched[1] = delay_hl_addr & 0xFF
    rest_loop_patched[2] = (delay_hl_addr >> 8) & 0xFF

    # Patch note stubs: CALL addresses
    output = bytearray()
    for kind, stub in note_stubs:
        patched = bytearray(stub)
        if kind == "note":
            # CALL play_note at offset 6 (bytes 7-8)
            call_offset = 6  # after LD HL (3) + LD BC (3)
            patched[call_offset + 1] = play_note_addr & 0xFF
            patched[call_offset + 2] = (play_note_addr >> 8) & 0xFF
        elif kind == "rest":
            # CALL rest_loop at offset 6 (bytes 7-8)
            call_offset = 6
            patched[call_offset + 1] = rest_loop_addr & 0xFF
            patched[call_offset + 2] = (rest_loop_addr >> 8) & 0xFF
        output += patched

    output += bytes([0xF3, 0x76])  # DI + HALT (disable interrupts so HALT is permanent)

    output += play_note_patched
    output += delay_hl_code
    output += rest_loop_patched

    return bytes(output)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate a glyph-audio JSON spec."""
    with open(path) as f:
        spec = json.load(f)
    if spec.get("type") != "glyph-audio":
        raise ValueError(f"Not a glyph-audio spec: type={spec.get('type')}")
    return spec


