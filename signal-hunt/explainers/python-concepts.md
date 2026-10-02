# Python Concepts for ML

Patterns you'll see throughout this codebase. If you're coming from another language, these are the Python-specific bits that might look unfamiliar.

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

**How is this different from a struct?** A struct (C, Go, Rust) has named fields. A tuple only has positions — `result[0]`, `result[1]`. If you want named fields in Python, use a `NamedTuple` or dataclass:

```python
from typing import NamedTuple

class AudioResult(NamedTuple):
    signal: np.ndarray
    sample_rate: int

result = AudioResult(audio_array, 22050)
result.signal      # by name
result[0]          # by position — still works
```

**When to use which:**
- **Tuple** — 2-3 return values where the meaning is obvious from context (e.g. `y, sr`)
- **NamedTuple** — more fields, or when the meaning isn't obvious at the call site
- **Dataclass** — when you need mutability or methods

---

## Type hints

Python is dynamically typed, but type hints document what a function expects. They don't enforce anything at runtime — they're for readability and tooling (IDE autocomplete, type checkers).

```python
def pad_or_truncate(signal: np.ndarray, target_length: int) -> np.ndarray:
```

Read as: "takes a NumPy array and an int, returns a NumPy array."

`str | Path` means "accepts either a string or a Path object" (union type, Python 3.10+).

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

---

## `if __name__ == "__main__"`

```python
if __name__ == "__main__":
    main()
```

This guard means "only run this code when the file is executed directly, not when it's imported." Without it, `import pipeline.ingest` would execute the script's main logic as a side effect.

---

## f-strings

```python
print(f"Duration: {duration:.2f}s")
```

String interpolation. The `{expression}` is evaluated and inserted. `:.2f` formats as a float with 2 decimal places. Equivalent to `"Duration: %.2fs" % duration` but more readable.
