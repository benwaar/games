"""Z80 target — prompt template, assembler gate, test runner."""

import z80 as z80lib

from ..sigil import Sigil, SigilTest

SYSTEM_PROMPT = """\
You are a Z80 assembly programmer. You write clean, correct Z80 assembly code.

Rules:
- Output ONLY valid Z80 assembly. No prose, no markdown fences, no explanation.
- End every program with HALT.
- Registers are pre-loaded by the test harness — do not set input registers yourself.
- Use standard Zilog mnemonics (LD, ADD, SUB, etc.).
- Use labels for jumps (not raw addresses).
- Keep code minimal — do exactly what the spec asks, nothing more.
"""


def build_prompt(sigil: Sigil, rag_context: str) -> str:
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
        sf = z80lib.SourceFile("generated.asm", source)
        code = asm.assemble(sf)
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
