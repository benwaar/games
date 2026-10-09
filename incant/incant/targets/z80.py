"""Z80 target — prompt template, assembler gate, test runner."""

import z80 as z80lib

from ..sigil import Sigil, SigilTest

INPUT_BUF = 0x8000
OUTPUT_BUF = 0x9000

SYSTEM_PROMPT = """\
You are a Z80 assembly programmer. You write clean, correct Z80 assembly code.

Rules:
- Output ONLY valid Z80 assembly. No prose, no markdown fences, no explanation.
- End every program with HALT.
- Registers are pre-loaded by the test harness — do not set input registers yourself.
- Use lowercase mnemonics (ld, add, sub, jp, jr, halt).
- Use labels for jumps (not raw addresses).
- Keep code minimal — do exactly what the spec asks, nothing more.
"""

HARNESS_IO_PROMPT = """\
You are a Z80 assembly programmer. You write clean, correct Z80 assembly code.

Memory-mapped I/O conventions:
- Input buffer:  0x8000 (null-terminated string, pre-loaded by test harness)
- Output buffer: 0x9000 (write your null-terminated output here)

Constants you MUST define at the top:
  INPUT_BUF  equ 0x8000
  OUTPUT_BUF equ 0x9000

Rules:
- Output ONLY valid Z80 assembly. No prose, no markdown fences, no explanation.
- End every program with HALT.
- Use lowercase mnemonics (ld, add, sub, jp, jr, halt).
- Use labels for jumps (not raw addresses).
- NEVER use character literals like 'h' or "h" — the assembler does not support them.
  Use hex values instead: 0x68 for 'h', 0x65 for 'e', 0x6c for 'l', 0x6f for 'o', 0x20 for ' '.
- Read input from INPUT_BUF using HL as a pointer.
- Write output to OUTPUT_BUF using DE as a pointer.
- Null-terminate the output string (write a 0x00 byte at the end).
- Keep code minimal — do exactly what the spec asks, nothing more.

ASCII hex reference: ' '=0x20 'A'=0x41 'B'=0x42 'a'=0x61 'b'=0x62 'e'=0x65 'h'=0x68 'l'=0x6c 'o'=0x6f

Working example — copy "hi " then input to output:

INPUT_BUF  equ 0x8000
OUTPUT_BUF equ 0x9000

  ld de, OUTPUT_BUF
  ; write "hi " using hex values
  ld a, 0x68
  ld (de), a
  inc de
  ld a, 0x69
  ld (de), a
  inc de
  ld a, 0x20
  ld (de), a
  inc de
  ; copy input to output
  ld hl, INPUT_BUF
copy:
  ld a, (hl)
  or a
  jr z, done
  ld (de), a
  inc hl
  inc de
  jr copy
done:
  xor a
  ld (de), a
  halt
"""


def build_prompt(sigil: Sigil, rag_context: str) -> str:
    if sigil.harness_io:
        return f"""{rag_context}

Write Z80 assembly for the following spec:

Name: {sigil.name}
Description: {sigil.description}
Input: null-terminated string at INPUT_BUF (0x8000)
Output: null-terminated string at OUTPUT_BUF (0x9000)

Example test: stdin "{sigil.tests[0].inputs.get('stdin', '')}" → expect "{sigil.tests[0].expect}"

Output ONLY the assembly code. No explanation."""

    inputs_desc = ", ".join(
        f"{p.name} in register {p.register} ({p.type})"
        for p in sigil.signature.inputs
    )
    output_desc = f"register {sigil.signature.output_register} ({sigil.signature.output_type})"

    return f"""{rag_context}

Write Z80 assembly for the following spec:

Name: {sigil.name}
Description: {sigil.description}
Inputs: {inputs_desc}
Output: {output_desc}

Example test: inputs {sigil.tests[0].inputs} → expect {sigil.tests[0].expect}

Output ONLY the assembly code. No explanation."""


def assemble(source: str) -> tuple[bool, bytes | None, str | None]:
    """Assemble Z80 source. Returns (success, bytes, error_message)."""
    try:
        asm = z80lib.Asm()
        sf = z80lib.SourceFile("generated.asm", source.lower())
        code = asm.assemble(sf)
        code.resolve()
        chunks = code.encode()
        if not chunks:
            return False, None, "Assembly produced no output"
        _, data = chunks[0]
        return True, data, None
    except Exception as e:
        return False, None, str(e)


REGISTER_MAP = {
    "a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "h": "h", "l": "l",
    "bc": "bc", "de": "de", "hl": "hl", "sp": "sp", "pc": "pc",
    "ix": "ix", "iy": "iy",
}


def run_test(data: bytes, test: SigilTest, sigil: Sigil) -> tuple[bool, str]:
    """Load code into Z80 machine, set input registers, run, check output."""
    machine = z80lib.Z80Machine()
    machine.set_memory_block(0, data)
    machine.ticks_to_stop = 10000

    for param in sigil.signature.inputs:
        reg_name = param.register.lower()
        value = test.inputs[param.name]
        if reg_name in REGISTER_MAP:
            setattr(machine, REGISTER_MAP[reg_name], value)

    machine.run()

    if not machine.halted:
        return False, "Program did not halt within tick limit"

    if isinstance(test.expect, dict):
        for reg, expected in test.expect.items():
            reg_name = reg.lower()
            actual = getattr(machine, REGISTER_MAP[reg_name])
            if actual != expected:
                return False, f"Register {reg.upper()}: expected {expected}, got {actual}"
    else:
        out_reg = sigil.signature.output_register.lower()
        actual = getattr(machine, REGISTER_MAP[out_reg])
        if actual != test.expect:
            return False, f"Register {out_reg.upper()}: expected {test.expect}, got {actual}"

    return True, "pass"


def run_harness_io_test(data: bytes, test: SigilTest) -> tuple[bool, str]:
    """Run a harness_io test: pre-load stdin at INPUT_BUF, check OUTPUT_BUF after."""
    machine = z80lib.Z80Machine()
    machine.set_memory_block(0, data)
    machine.ticks_to_stop = 100000

    stdin_val = test.inputs.get("stdin", "")
    stdin_bytes = stdin_val.encode("ascii") + b"\x00"
    machine.set_memory_block(INPUT_BUF, stdin_bytes)

    machine.run()

    if not machine.halted:
        return False, "Program did not halt within tick limit"

    output_bytes = []
    for i in range(256):
        b = machine.memory[OUTPUT_BUF + i]
        if b == 0:
            break
        output_bytes.append(b)

    actual = bytes(output_bytes).decode("ascii", errors="replace")
    expected = test.expect if isinstance(test.expect, str) else str(test.expect)

    if actual != expected:
        return False, f"Expected output '{expected}', got '{actual}'"

    return True, "pass"


def extract_code(llm_output: str) -> str:
    """Strip markdown fences and prose from LLM output."""
    lines = llm_output.strip().splitlines()

    in_fence = False
    code_lines = []
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    return llm_output.strip()
