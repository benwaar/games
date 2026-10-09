# incant — Libraries

Libraries and specific functions used in this project.

## ollama (`ollama>=0.4.0`)

Python client for the Ollama local LLM server.

### `ollama.embed(model, input)`

Generates embeddings (dense float vectors) for text. Returns `{"embeddings": [[float, ...]]}`.

```python
response = ollama.embed(model="nomic-embed-text", input="add two numbers")
vector = response["embeddings"][0]  # 768-dimensional float vector
```

Used in: `incant/rag.py` — embedding knowledge chunks and queries.

### `ollama.chat(model, messages)`

Chat completion with a local model. Takes a list of `{"role": ..., "content": ...}` messages.

```python
response = ollama.chat(
    model="qwen3-coder:latest",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)
text = response["message"]["content"]
```

Used in: `incant/gen.py` — generating code from sigil specs.

## z80 (`z80>=1.2.0`)

Z80 assembler and CPU emulator in pure Python.

### `z80.Asm()`

Assembles Z80 assembly source into a bytes object.

```python
asm = z80.Asm()
binary = asm.assemble("add a, b\nhalt")  # returns bytes
```

Used in: `incant/targets/z80.py` — the assembly gate.

### `z80.Z80Machine()`

Z80 CPU emulator. Load binary, set registers, step/run, inspect state.

```python
machine = z80.Z80Machine()
machine.load(binary, addr=0x0000)
machine.regs[z80.A] = 2
machine.regs[z80.B] = 3
machine.run()
result = machine.regs[z80.A]  # 5
```

Used in: `incant/targets/z80.py` — running test cases against generated code.

## pyyaml (`pyyaml>=6.0`)

YAML parser.

### `yaml.safe_load(stream)`

Parses YAML into Python dicts/lists. `safe_load` is preferred over `load` — it refuses to execute arbitrary Python objects embedded in YAML.

```python
with open(path) as f:
    raw = yaml.safe_load(f)  # returns dict
```

Used in: `incant/sigil.py` — parsing sigil spec files.

## wabt (system package, `brew install wabt`)

WebAssembly Binary Toolkit — command-line tools, not a Python package.

### `wat2wasm`

Compiles WAT (WebAssembly Text Format) to .wasm binary.

```bash
wat2wasm input.wat -o output.wasm
```

Used in: `incant/targets/wat.py` — the WAT assembly gate.

### `wasm-interp`

Interprets a .wasm file. `--run-all-exports` calls every exported function.

```bash
wasm-interp output.wasm --run-all-exports
```

Used in: `incant/targets/wat.py` — running WAT test cases.
