# Python Concepts for ML

Patterns you'll see throughout this codebase. If you're coming from another language, these are the Python-specific bits that might look unfamiliar.

Each section has "Coming from..." callouts that map Python concepts to what you already know.

---

## Tuples — returning multiple values

A tuple is an immutable ordered collection. No named fields — just positions. Python uses them heavily for returning multiple values from a function.

```python
def load_audio(path) -> tuple[np.ndarray, int]:
    ...
    return y, sr   # returns a tuple of (array, sample_rate)

# Unpack into separate variables
y, sr = load_audio("file.wav")
```

> **Coming from C:** A tuple is like returning a struct, but without defining one. The closest C pattern is passing output pointers (`load_audio(path, &y, &sr)`), which Python replaces with tuple unpacking. A `NamedTuple` (below) is the real struct equivalent — named fields, typed, but immutable.

> **Coming from JS/TS:** A tuple is like returning an array and destructuring it: `const [y, sr] = loadAudio(file)`. A Python dict `{"signal": y, "sr": sr}` is closer to a JS object `{ signal, sr }`, but Python devs reach for tuples more often because unpacking is so clean. If you want named access like `result.signal`, use a `NamedTuple` — that's the JS `{ signal, sr }` equivalent.

**When to use which:**
- **Tuple** — 2-3 return values where the meaning is obvious from context (e.g. `y, sr`)
- **NamedTuple** — more fields, or when the meaning isn't obvious at the call site
- **Dataclass** — when you need mutability or methods

```python
from typing import NamedTuple

class AudioResult(NamedTuple):
    signal: np.ndarray
    sample_rate: int

result = AudioResult(audio_array, 22050)
result.signal      # by name
result[0]          # by position — still works
```

---

## Type hints

Python is dynamically typed, but type hints document what a function expects. They don't enforce anything at runtime — they're for readability and tooling (IDE autocomplete, type checkers).

```python
def pad_or_truncate(signal: np.ndarray, target_length: int) -> np.ndarray:
```

Read as: "takes a NumPy array and an int, returns a NumPy array."

`str | Path` means "accepts either a string or a Path object" (union type, Python 3.10+).

> **Coming from C:** Like function prototypes, but optional and not enforced by the compiler. Python won't stop you passing a string where an int is expected — the hint is documentation, not a contract. Use `mypy` for static checking if you want enforcement.

> **Coming from TS:** Very similar to TypeScript's type annotations. The key difference: TS types are erased at compile time but checked before then. Python type hints are ignored entirely at runtime unless you run a separate tool like `mypy`. Think of them as TS types with `// @ts-ignore` on every line — helpful for editors, invisible to the interpreter.

---

## Fixtures in pytest

A fixture is a function that sets up test data. The `@pytest.fixture` decorator tells pytest to call it automatically and pass the result to any test that asks for it by name.

```python
@pytest.fixture
def sine_signal() -> np.ndarray:
    t = np.linspace(0, 1.0, 22050, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

class TestAddNoise:
    def test_shape(self, sine_signal):   # pytest injects the fixture here
        result = add_noise(sine_signal)
        assert result.shape == sine_signal.shape
```

**Why not just create the array in each test?** DRY — define the test signal once, reuse it everywhere. Fixtures can also handle setup/teardown (creating temp files, database connections, etc.).

> **Coming from C:** Like a test helper function, but with automatic injection. In C you'd call `setup()` at the top of each test; pytest calls it for you based on the parameter name.

> **Coming from JS/TS:** Similar to `beforeEach` in Jest/Mocha, but more targeted. Instead of running before *every* test, a fixture only runs for tests that ask for it by name in their argument list. Like dependency injection for tests.

---

## `if __name__ == "__main__"`

```python
if __name__ == "__main__":
    main()
```

This guard means "only run this code when the file is executed directly, not when it's imported." Without it, `import pipeline.ingest` would execute the script's main logic as a side effect.

> **Coming from C:** Same idea as having a `main()` function — but in C, only one file's `main()` runs. In Python, every file *could* run top-level code on import, so you need this guard to prevent it.

> **Coming from JS/TS:** There's no direct equivalent because JS modules don't execute on import (ES modules) or have a clear entry point. The closest is checking `process.argv[1]` or `import.meta.url` to detect if a module is the entry point. In Python, `__name__` is set to `"__main__"` for the entry file and to the module name for imports.

---

## f-strings

```python
print(f"Duration: {duration:.2f}s")
```

String interpolation. The `{expression}` is evaluated and inserted. `:.2f` formats as a float with 2 decimal places.

> **Coming from C:** Like `printf("Duration: %.2fs", duration)` but inline — the variable goes inside the string, not after it.

> **Coming from JS/TS:** Same as template literals: `` `Duration: ${duration.toFixed(2)}s` ``. The `f` prefix is Python's backtick equivalent. The `:.2f` format spec is more powerful than JS — handles padding, alignment, and number formatting in one mini-language.

---

## Dicts as lightweight config

```python
def default_augmentations() -> list[dict]:
    return [
        {"name": "clean"},
        {"name": "noise_low", "fn": "add_noise", "kwargs": {"snr_db": 20.0}},
    ]
```

Python dicts are the go-to for configuration objects — flexible, no class definition needed. The batch pipeline uses a list of dicts to define which augmentations to apply. Each dict is a self-contained config: a name, an optional function reference, and keyword arguments.

The `**kwargs` unpacking pattern passes dict entries as named arguments: `fn(**{"snr_db": 20.0})` becomes `fn(snr_db=20.0)`.

> **Coming from C:** This is what you'd do with a struct array — but without defining the struct. Flexible, but no compile-time checking. In C you'd define `struct AugConfig { char* name; void (*fn)(float*, int, float); float snr_db; }` and get type safety. In Python you trade that for speed of iteration.

> **Coming from JS/TS:** Identical to a plain object `{ name: "clean" }` or an array of options objects. The `**kwargs` spread is like `fn(...config)` or `fn({...config})`. If you're used to TypeScript interfaces for config shapes, know that Python dicts have no schema enforcement unless you add runtime validation (e.g. Pydantic, TypedDict).

---

## argparse — CLI argument parsing

```python
import argparse

parser = argparse.ArgumentParser(description="Batch audio → tensor pipeline")
parser.add_argument("input_dir", type=Path, help="Folder of .wav files")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
```

Standard library module for building CLI tools. Positional args are required, `--flag` args are optional. Handles type conversion, defaults, and `--help` generation automatically.

> **Coming from C:** Like `getopt()` but higher-level. No manual `argc`/`argv` parsing — argparse handles the loop, validation, and error messages.

> **Coming from JS/TS:** Similar to `commander` or `yargs` in Node. Python's version is built in — no npm install needed. `args.input_dir` is like `program.opts().inputDir`.

---

## Path.glob() and sorted() — file discovery

```python
audio_files = sorted(input_dir.glob("*.wav"))
```

`Path.glob()` returns all files matching a pattern — like shell globbing. Returns a generator (lazy), so `sorted()` collects and orders them. Sorting ensures deterministic processing order across platforms (filesystem order isn't guaranteed).

> **Coming from C:** Like `opendir()` + `readdir()` with a filter, then `qsort()`. Python's `pathlib` wraps all of that in one line.

> **Coming from JS/TS:** Like `fs.readdirSync()` filtered with `.filter(f => f.endsWith('.wav'))`. The `glob` npm package adds pattern matching. Python's `Path.glob()` is built in and returns `Path` objects (not strings), so you can chain `.stem`, `.name`, `.parent` etc. without string manipulation.

---

## json.dumps / Path.write_text — serialisation

```python
import json
from pathlib import Path

manifest_path = output_dir / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
```

`json.dumps()` converts a Python object (list, dict) to a JSON string. `Path.write_text()` writes a string to a file atomically. Combined: one-line file serialisation.

> **Coming from C:** Like `fprintf()` with manual JSON formatting — except `json.dumps` handles escaping, nesting, and pretty-printing for you. No need to build the string yourself.

> **Coming from JS/TS:** `JSON.stringify(manifest, null, 2)` + `fs.writeFileSync()`. Nearly identical — Python's `json` module has the same API shape. The `indent=2` parameter matches JS's third argument to `stringify`.

---

## sys.argv — lightweight CLI arguments

```python
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "data/raw/hum_440hz.wav"
```

`sys.argv` is a list of command-line arguments. `sys.argv[0]` is the script name, `sys.argv[1]` onward are user arguments. For scripts with one or two optional args, this is simpler than `argparse`.

> **Coming from C:** Exactly `argv[1]` from `int main(int argc, char *argv[])`. Same indexing, same concept. Python just wraps it in a list instead of a pointer array.

> **Coming from JS/TS:** Like `process.argv[2]` in Node (Node's `argv[0]` is the node binary, `argv[1]` is the script — so user args start at index 2). Python's `sys.argv[0]` is the script, user args start at index 1.

**When to use which:**
- **`sys.argv`** — 0–2 optional args, simple scripts, demos
- **`argparse`** — named flags, help text, type validation, anything a user will run regularly

---

## np.pad — padding arrays

```python
padded = np.pad(signal, (0, max(0, target_len - len(signal))))[:target_len]
```

`np.pad` adds values to the edges of an array. The tuple `(before, after)` controls how many elements to add at each end. Default padding value is 0. Combined with slicing `[:target_len]`, it handles both too-short and too-long inputs in one line.

> **Coming from C:** Like `memset` after a `realloc` — extend the buffer and zero-fill the new space. Python does it without manual memory management.

> **Coming from JS/TS:** No built-in equivalent. You'd spread into a new array: `[...signal, ...new Array(padding).fill(0)]`. NumPy's `np.pad` is more flexible — supports constant, edge, reflect, and wrap modes for different padding strategies.

---

## Dunder methods — implementing protocols

Dunder methods (double-underscore, e.g. `__len__`, `__getitem__`) are Python's way of making your class behave like a built-in type. Implement the right set of dunders and your object works with `len()`, indexing `obj[i]`, iteration, comparison operators, and more.

```python
class SignalDataset:
    def __len__(self):
        return len(self.records)      # enables: len(dataset)

    def __getitem__(self, idx):
        return self.records[idx]      # enables: dataset[0], dataset[-1]
```

Once those two are implemented, Python's `for item in dataset` works automatically — Python calls `__getitem__` with indices 0, 1, 2, … until `IndexError`. PyTorch's `DataLoader` uses the same mechanism.

Common dunders you'll see in this codebase:

| Dunder | Enables | Example |
|--------|---------|---------|
| `__len__` | `len(obj)` | `len(dataset)` |
| `__getitem__` | `obj[i]` | `dataset[0]` |
| `__repr__` | `repr(obj)` in debugger | `"SignalDataset(231 items)"` |
| `__eq__` | `obj == other` | comparing configs |

> **Coming from C:** This is a vtable. `__len__` and `__getitem__` are function pointers in a struct. Any struct (class) that populates those slots satisfies the "array-like" interface that Python's `len()` and `[]` operator check for. The double underscores mark them as protocol-level — the interpreter looks for them, not user code.

> **Coming from TS:** This is like implementing a TypeScript interface, but implicit. If your class has `length` and `[Symbol.iterator]`, it's iterable — no explicit `implements Iterable` needed. Python's dunder protocol is the same: implement the right methods, get the behaviour, no declaration required. `__len__` is `.length`; `__getitem__` is indexed access.

---

## Dict comprehensions

A compact way to build a dict from a sequence:

```python
# {"clap": 0, "hum": 1, "whistle": 2}
label_map = {label: i for i, label in enumerate(sorted(labels))}
```

`enumerate(iterable)` yields `(index, value)` pairs. `sorted()` sorts the list first — so the mapping is alphabetical and deterministic regardless of insertion order.

> **Coming from C:** No direct equivalent. You'd build a hash map manually with a loop. Python's comprehension syntax makes the declaration look like its mathematical definition: `{label: i for each (i, label) in enumerate(sorted(labels))}`.

> **Coming from JS/TS:** Like `Object.fromEntries(sorted.map((label, i) => [label, i]))`. Python's comprehension is more readable because the key and value are written in the natural `key: value` order: `{label: i for ...}` reads as "label maps to i".

---

## dataclass with `field(default_factory=...)`

A `dataclass` is like a struct with auto-generated `__init__`, `__repr__`, and `__eq__`.
Used in `model/config.py` for hyperparameters:

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
```

**Why `field(default_factory=...)`?** Mutable defaults (lists, dicts, objects) can't be
plain default values in a dataclass — Python would share one instance across all objects.
`default_factory` creates a fresh one for each instance. `Path("data/processed")` is
technically immutable but still requires `field` to avoid a dataclass restriction on
mutable types that could cause this bug.

> **Coming from C:** Like a struct with a default initialiser per field. The `@dataclass`
> decorator writes the `__init__` for you — equivalent to hand-writing a constructor that
> assigns each field.

> **Coming from TS:** Like an interface with default values: `interface TrainConfig { batchSize: number = 32 }`.
> Python's `@dataclass` generates the constructor automatically; TypeScript requires you to write it.

---

## `model.train()` / `model.eval()` — PyTorch mode switching

PyTorch models have two modes that affect behaviour at runtime:

```python
model.train()   # dropout active, BatchNorm uses batch statistics
model.eval()    # dropout off, BatchNorm uses running statistics
```

Always switch before the appropriate loop:

```python
# Training loop
model.train()
for x, y in train_loader:
    ...

# Validation / test loop
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        ...
```

Forgetting `model.eval()` before validation means dropout randomly zeroes activations —
you'd get different loss values every call on the same data. Forgetting `model.train()`
before the next training epoch means dropout is off — the model trains without regularisation.

> **Coming from C/JS/TS:** This is a global mode flag that changes the behaviour of
> specific operations. Like a `debugMode` flag that switches between debug and production
> code paths — except it's built into every layer that has train-vs-inference differences.

---

## `torch.no_grad()` — disabling gradient tracking

```python
with torch.no_grad():
    logits = model(x)
```

During training, PyTorch records every operation on tensors so it can compute gradients
during `loss.backward()`. During evaluation, you don't need gradients — and computing
them wastes memory and time.

`torch.no_grad()` turns off gradient tracking for the block. No computation graph is built,
no `.grad` fields are populated, and memory usage drops.

> **Coming from C:** Like disabling an instrumentation mode that records every operation
> for later replay. `no_grad` says "just compute the output, don't record how you got there."

> **Coming from JS/TS:** Like running without source maps — the output is the same but
> you're not tracking the path that produced it.

---

## `scope="module"` in pytest fixtures

```python
@pytest.fixture(scope="module")
def trained_output(tmp_path_factory):
    # Runs once for all tests in this file, not once per test
    train(config)
    return output_dir
```

By default, pytest fixtures run before every test. `scope="module"` runs the fixture once
for the entire test file and shares the result. Used in `test_evaluate.py` because training
takes seconds — running it 11 times (once per test) would be slow.

Note: `scope="module"` fixtures must use `tmp_path_factory` (not `tmp_path`) to create
temporary directories, because `tmp_path` is function-scoped.

> **Coming from C/JS/TS:** Like `beforeAll` in Jest vs `beforeEach`. `scope="module"` =
> `beforeAll`; default scope = `beforeEach`.
