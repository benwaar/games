# Z80 Backend — Assembly, Emulation, and Gate-Driven Iteration

How incant generates, assembles, and tests Z80 machine code using a local LLM.

## The pipeline

```
sigil → RAG context → prompt → LLM → extract code → assemble → run tests → binary
                                        ↑                          |
                                        └── retry with error ──────┘  (max 3)
```

## Prompt construction

The Z80 target builds a prompt from three parts:

1. **RAG context** — top-5 relevant chunks from the Z80 knowledge base (instruction patterns, register usage, flag behaviour)
2. **System prompt** — rules for the LLM: output-only assembly, lowercase mnemonics, labels for jumps, end with `halt`
3. **Sigil spec** — function name, inputs (with register assignments), output register, one example test case

```python
# From incant/targets/z80.py
def build_prompt(sigil, rag_context):
    inputs_desc = ", ".join(
        f"{p.name} in register {p.register} ({p.type})"
        for p in sigil.signature.inputs
    )
    # ...
```

The prompt tells the LLM that registers are pre-loaded by the test harness — so the generated code should not set input registers itself. This is a key design choice: it keeps the generated code pure (just the algorithm) and lets the test harness control inputs.

## Assembly

The `z80` pip package provides `z80.Asm()` — a pure-Python Z80 assembler.

### Case sensitivity

The assembler requires **lowercase mnemonics**. Uppercase `ADD A, B` is parsed as a label definition (`ADD:`) and fails. The LLM often outputs uppercase Zilog convention, so we lowercase the entire output before assembling:

```python
sf = z80lib.SourceFile("generated.asm", source.lower())
```

This works because Z80 assembly has no case-sensitive identifiers — register names, mnemonics, and labels are all case-insensitive in the specification.

### Label resolution

The assembler's `assemble()` method parses source into a `Code` object but does not resolve label addresses. You must call `code.resolve()` explicitly before `code.encode()`:

```python
code = asm.assemble(sf)
code.resolve()          # bind labels to addresses
chunks = code.encode()  # emit bytes
```

Without `resolve()`, any code using labels (loops, jumps, subroutines) fails with "Undefined symbol".

> **Coming from C:** This is the linker step. The assembler creates an object with unresolved symbols; `resolve()` is the link pass that patches addresses.

> **Coming from JS/TS:** Think of it like a two-pass compiler — first pass collects symbol definitions, second pass fills in references. The API just makes the second pass explicit.

## Emulation

Tests run on `z80.Z80Machine()` — a cycle-accurate Z80 emulator.

```python
machine = z80lib.Z80Machine()
machine.set_memory_block(0, data)   # load binary at address 0
machine.ticks_to_stop = 10000       # safety limit (prevents infinite loops)

# set input registers from test case
for param in sigil.signature.inputs:
    setattr(machine, param.register.lower(), test.inputs[param.name])

machine.run()

# check output registers
actual = getattr(machine, output_register)
```

The tick limit (10,000 cycles) catches infinite loops. A simple add takes ~8 ticks; a 255-iteration loop takes ~2,000.

## Retry loop

If assembly or tests fail, the error message is appended to the prompt and the LLM tries again (up to 3 attempts total). This is gate-driven iteration — the gate's error output becomes the LLM's correction input.

```python
retry_prompt = (
    f"{prompt}\n\n"
    f"Previous attempt failed with error:\n{last_error}\n\n"
    f"Fix the error and output ONLY the corrected code."
)
```

> **In practice:** This is the same pattern production LLM systems use — structured feedback loops where validation errors are fed back to the model. The difference from chat is that the feedback is machine-generated (assembler errors, test failures) rather than human.

## Example: what the LLM generates

### add (A + B → A)
```asm
add a, b
halt
```
2 bytes. Simplest possible Z80 program.

### counter (loop B times, count in A)
```asm
ld a, 0
loop:
    inc a
    djnz loop
    halt
```
6 bytes. Uses `DJNZ` — "decrement B, jump if not zero". A Z80-specific loop instruction.

### store_load (memory round-trip)
```asm
ld (0x8000), a
ld a, (0x8000)
halt
```
7 bytes. Direct memory addressing — store A to 0x8000, load it back.

## Code extraction

The LLM sometimes wraps output in markdown fences despite being told not to. `extract_code()` strips fences and falls back to raw output:

```python
def extract_code(llm_output):
    # If markdown fences found, extract content between them
    # Otherwise, return the raw output
```
