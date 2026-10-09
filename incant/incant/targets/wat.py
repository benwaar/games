"""WAT target — prompt template, wat2wasm gate, wasm-interp test runner."""

import subprocess
import tempfile
from pathlib import Path

from ..sigil import Sigil, SigilTest

SYSTEM_PROMPT = """\
You are a WebAssembly Text (WAT) programmer. You write clean, correct WAT code.

Rules:
- Output ONLY a valid WAT module. No prose, no markdown fences, no explanation.
- Always wrap code in (module ... ).
- Export the function with the exact name from the spec.
- Use flat (stack) style, not folded S-expressions, for clarity.
- If the spec includes memory, declare (memory N) and export it.
- Keep code minimal — do exactly what the spec asks, nothing more.
"""


def build_prompt(sigil: Sigil, rag_context: str) -> str:
    params = ", ".join(
        f"{p.name}: {p.type}" for p in sigil.signature.inputs
    )
    test = sigil.tests[0]
    test_args = ", ".join(str(test.inputs[p.name]) for p in sigil.signature.inputs)

    memory_note = ""
    if sigil.memory:
        memory_note = f"\nMemory: {sigil.memory} page(s) — include (memory {sigil.memory}) and export it."

    return f"""{rag_context}

Write a WAT module for the following spec:

Name: {sigil.name}
Description: {sigil.description}
Export name: {sigil.export}
Signature: ({params}) -> {sigil.signature.output_type}{memory_note}

Example: {sigil.export}({test_args}) → {test.expect}

Output ONLY the WAT module. No explanation."""


def assemble(source: str) -> tuple[bool, Path | None, str | None]:
    """Assemble WAT source to .wasm using wat2wasm. Returns (success, wasm_path, error)."""
    with tempfile.NamedTemporaryFile(suffix=".wat", mode="w", delete=False) as f:
        f.write(source)
        wat_path = Path(f.name)

    wasm_path = wat_path.with_suffix(".wasm")

    result = subprocess.run(
        ["wat2wasm", str(wat_path), "-o", str(wasm_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False, None, result.stderr.strip()

    return True, wasm_path, None


def run_test(wasm_path: Path, test: SigilTest, sigil: Sigil) -> tuple[bool, str]:
    """Run a test case using wasm-interp."""
    args = [str(test.inputs[p.name]) for p in sigil.signature.inputs]

    result = subprocess.run(
        ["wasm-interp", str(wasm_path), f"--run-export={sigil.export}", "--"] + args,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False, f"wasm-interp failed: {result.stderr.strip()}"

    output = result.stdout.strip()
    # wasm-interp prints: export_name(args) => value
    # parse the result value
    if "=>" in output:
        result_str = output.split("=>")[-1].strip()
        try:
            # handle i32:N format
            if ":" in result_str:
                result_str = result_str.split(":")[-1]
            actual = int(result_str)
        except ValueError:
            try:
                actual = float(result_str)
            except ValueError:
                return False, f"Could not parse wasm-interp output: {output}"

        expected = test.expect
        if actual != expected:
            return False, f"Expected {expected}, got {actual} (output: {output})"
        return True, "pass"

    return False, f"Unexpected wasm-interp output: {output}"


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

    # no fences — find the (module ...) block
    text = llm_output.strip()
    start = text.find("(module")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text
