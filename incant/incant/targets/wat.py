"""WAT target — prompt template, wat2wasm gate, wasm-interp test runner."""

import math
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
    if not sigil.tests:
        raise ValueError(f"Sigil '{sigil.name}' has no tests — at least one required for prompt example")

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
    tmp_dir = tempfile.mkdtemp(prefix="incant_")
    wat_path = Path(tmp_dir) / "generated.wat"
    wat_path.write_text(source)
    wasm_path = wat_path.with_suffix(".wasm")

    result = subprocess.run(
        ["wat2wasm", str(wat_path), "-o", str(wasm_path)],
        capture_output=True,
        text=True,
    )

    wat_path.unlink()

    if result.returncode != 0:
        return False, None, result.stderr.strip()

    return True, wasm_path, None


def _bits_for_type(wasm_type: str) -> int:
    return 64 if wasm_type in ("i64", "f64") else 32


def _to_unsigned(value: int, bits: int = 32) -> int:
    """Convert a signed int to unsigned representation for wasm-interp."""
    if value < 0:
        return value + (1 << bits)
    return value


def _to_signed(value: int, bits: int = 32) -> int:
    """Convert unsigned wasm-interp output back to signed if needed."""
    if value >= (1 << (bits - 1)):
        return value - (1 << bits)
    return value


def run_test(wasm_path: Path, test: SigilTest, sigil: Sigil) -> tuple[bool, str]:
    """Run a test case using wasm-interp."""
    cmd = ["wasm-interp", str(wasm_path), "-r", sigil.export]
    for param in sigil.signature.inputs:
        value = test.inputs[param.name]
        bits = _bits_for_type(param.type)
        cmd.extend(["-a", f"{param.type}:{_to_unsigned(value, bits)}"])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return False, f"wasm-interp failed: {result.stderr.strip()}"

    output = result.stdout.strip()
    # wasm-interp prints: func_name(type:val, ...) => type:val
    if "=>" not in output:
        return False, f"Unexpected wasm-interp output: {output}"

    result_str = output.split("=>")[-1].strip()
    # strip type prefix (e.g. "i32:5" → "5")
    if ":" in result_str:
        result_str = result_str.split(":")[-1]

    try:
        actual = int(result_str)
    except ValueError:
        try:
            actual = float(result_str)
        except ValueError:
            return False, f"Could not parse wasm-interp output: {output}"

    expected = test.expect
    out_bits = _bits_for_type(sigil.signature.output_type)
    if isinstance(expected, int) and expected < 0:
        actual_signed = _to_signed(actual, out_bits)
        if actual_signed != expected:
            return False, f"Expected {expected}, got {actual_signed} (output: {output})"
    elif isinstance(expected, float) or isinstance(actual, float):
        if not math.isclose(actual, expected, rel_tol=1e-6):
            return False, f"Expected {expected}, got {actual} (output: {output})"
    elif actual != expected:
        return False, f"Expected {expected}, got {actual} (output: {output})"
    return True, "pass"


WASI_SYSTEM_PROMPT = """\
You are a WebAssembly Text (WAT) programmer writing WASI programs.

Rules:
- Output ONLY a valid WAT module. No prose, no markdown fences, no explanation.
- Always wrap code in (module ... ).
- Import fd_read and fd_write from "wasi_snapshot_preview1".
- Export "_start" as the entry point and "memory" as the memory.
- Keep code minimal — do exactly what the spec asks, nothing more.

WASI function signatures:
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_read" (func $fd_read (param i32 i32 i32 i32) (result i32)))
  fd 0 = stdin, fd 1 = stdout

CRITICAL WAT syntax rules:
- i32.store takes TWO stack args: (i32.store <addr> <value>). Example: (i32.store (i32.const 0) (i32.const 100))
- i32.load takes ONE stack arg: (i32.load <addr>). Example: (i32.load (i32.const 8))
- NEVER put a local.set or anything else as an arg to i32.load. The result goes on the stack; use local.set AFTER.
  WRONG: (i32.load (i32.const 8) (local.set $x))
  RIGHT: (local.set $x (i32.load (i32.const 8)))
- Declare locals: (local $name i32) inside the function, before any instructions.
- drop the i32 result of fd_write/fd_read calls: (drop (call $fd_write ...))

Working WASI example — reads a name from stdin, writes "hello {name}\\n" to stdout:
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_read"
    (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 100) "hello ")
  ;; Memory layout:
  ;;   0-7: iovec struct (4b pointer + 4b length)
  ;;   8-11: nread/nwritten
  ;;   100-105: "hello " prefix (6 bytes — NOTE the trailing space)
  ;;   200-299: stdin read buffer
  ;;   300-399: output assembly buffer
  (func (export "_start")
    (local $nread i32)
    ;; Read stdin into buffer at 200, max 100 bytes
    (i32.store (i32.const 0) (i32.const 200))  ;; iov.buf = 200
    (i32.store (i32.const 4) (i32.const 100))  ;; iov.len = 100
    (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
    (local.set $nread (i32.load (i32.const 8)))
    ;; Copy "hello " (6 bytes from offset 100) to output buffer at 300
    (memory.copy (i32.const 300) (i32.const 100) (i32.const 6))
    ;; Copy name from 200 to 306 (right after "hello ")
    (memory.copy (i32.const 306) (i32.const 200) (local.get $nread))
    ;; Write newline after name
    (i32.store8 (i32.add (i32.const 306) (local.get $nread)) (i32.const 10))
    ;; Write output: iov points to 300, length = 6 + nread + 1
    (i32.store (i32.const 0) (i32.const 300))
    (i32.store (i32.const 4) (i32.add (i32.const 7) (local.get $nread)))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))
  )
)
"""


def build_wasi_prompt(sigil: Sigil, rag_context: str) -> str:
    """Build a prompt for WASI program generation."""
    scenario_text = ""
    if sigil.tests:
        examples = []
        for t in sigil.tests:
            stdin_val = t.inputs.get("stdin", "")
            stdout_val = t.expect if isinstance(t.expect, str) else str(t.expect)
            examples.append(f'  stdin: "{stdin_val}" → stdout: "{stdout_val}"')
        scenario_text = "\n".join(examples)

    return f"""{rag_context}

Write a WASI WAT module for the following spec:

Name: {sigil.name}
Description: {sigil.description}

The program reads from stdin (fd 0) and writes to stdout (fd 1).
Import fd_read and fd_write from "wasi_snapshot_preview1".
Export "_start" as the entry point and "memory".

Examples:
{scenario_text}

Memory layout:
- Offset 0-7: iovec struct for current I/O operation (4b pointer + 4b length)
- Offset 8-11: nread/nwritten result
- Offset 100-199: string constants ("hello " etc via data segments)
- Offset 200-299: read buffer (for stdin input)
- Offset 300-399: output assembly buffer

Use i32.store with two explicit args: (i32.store (i32.const addr) (i32.const val))
Use (local $name i32) to declare local variables inside the function.
Use (drop ...) around fd_read/fd_write calls to discard the result.

Output ONLY the WAT module. No explanation."""


def run_wasi_test(
    wasm_path: Path, stdin_input: str, expected_stdout: str,
) -> tuple[bool, str]:
    """Run a WASI program with stdin input and check stdout."""
    result = subprocess.run(
        ["wasm-interp", "--wasi", str(wasm_path)],
        capture_output=True,
        text=True,
        input=stdin_input,
    )

    if result.returncode != 0:
        return False, f"wasm-interp --wasi failed: {result.stderr.strip()}"

    actual = result.stdout
    if actual == expected_stdout:
        return True, "pass"

    return False, f"Expected stdout {expected_stdout!r}, got {actual!r}"


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
