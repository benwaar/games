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
