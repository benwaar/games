# incant — Python Concepts

Python patterns and idioms used in this project, with C and JS/TS equivalents.

## dataclass

Typed data containers with auto-generated `__init__`, `__repr__`, and `__eq__`.

```python
@dataclass
class Chunk:
    text: str
    source: str
    embedding: list[float]
```

> **Coming from C:** Like a `struct`, but with an auto-generated constructor and equality comparison.
> **Coming from JS/TS:** Like a TypeScript `interface` but as a runtime class. Closest equivalent: a class with a constructor that assigns every parameter.

## Generator expressions in sum/all/any

```python
dot = sum(x * y for x, y in zip(a, b))
```

The `(expr for x in iterable)` inside `sum()` is a generator — it yields values one at a time without building a list. Memory-efficient for large sequences.

> **Coming from JS/TS:** Similar to `Array.reduce()` but lazy — no intermediate array. JS equivalent: `a.reduce((sum, x, i) => sum + x * b[i], 0)`.

## zip

Iterates two (or more) sequences in lockstep:

```python
for x, y in zip(a, b):  # pairs (a[0],b[0]), (a[1],b[1]), ...
```

Stops at the shorter sequence. Used heavily in the cosine similarity calculation.

> **Coming from C:** Like iterating with a shared index `for (int i = 0; i < min(len_a, len_b); i++)` but without the index.

## Pathlib

Object-oriented filesystem paths. Used throughout instead of string manipulation.

```python
path = Path("knowledge/z80")
for md_file in path.glob("*.md"):   # find all .md files
    text = md_file.read_text()       # read file content
    name = md_file.stem              # filename without extension
```

> **Coming from JS/TS:** Like `path.join()` + `fs.readFileSync()` combined into one object. `Path("a") / "b"` is `path.join("a", "b")`.

## JSONL (JSON Lines)

One JSON object per line — no array wrapper, no commas between records.

```python
# Write
with open(path, "w") as f:
    for chunk in chunks:
        json.dump(chunk_dict, f)
        f.write("\n")

# Read
with open(path) as f:
    for line in f:
        obj = json.loads(line)
```

> **Coming from JS/TS:** `JSON.stringify(obj) + '\n'` per record. Read with `readline()` + `JSON.parse()`. Streaming-friendly — you don't need to parse the whole file to read one record.

## re.match vs re.search

`re.match` checks only at the **start** of the string. `re.search` scans the whole string.

```python
re.match(r"^#{1,3}\s+", line)   # heading detection
re.sub(r"^#+\s+", "", line)     # strip heading markers
```

> **Coming from JS/TS:** `re.match` is like `/^pattern/.test(str)` — anchored to start. `re.search` is like `/pattern/.test(str)` — scans anywhere. Python splits these into separate functions; JS uses the `^` anchor.

## setattr / getattr — dynamic property access

Access object attributes by name at runtime. Used in the Z80 gate to read/write registers dynamically:

```python
# Set register B to 5 on the Z80 machine
setattr(machine, "b", 5)

# Read register A after execution
result = getattr(machine, "a")
```

This lets us map sigil register names to machine properties without a giant switch statement.

> **Coming from C:** No direct equivalent — you'd use a function pointer table or a switch. The closest pattern is accessing struct members via `offsetof` + pointer arithmetic, but that's unsafe and manual.
> **Coming from JS/TS:** Equivalent to `machine["b"] = 5` and `machine["a"]`. JavaScript's bracket notation does the same thing — access properties by computed string key.

## subprocess.run — calling external tools

Runs a command-line tool and captures its output. Used by the WAT backend to call `wat2wasm` and `wasm-interp`:

```python
result = subprocess.run(
    ["wat2wasm", str(wat_path), "-o", str(wasm_path)],
    capture_output=True,  # capture stdout + stderr
    text=True,            # decode output as UTF-8 strings
)
if result.returncode != 0:
    error = result.stderr.strip()
```

Key parameters:
- `capture_output=True` — equivalent to `stdout=PIPE, stderr=PIPE`
- `text=True` — return strings, not bytes (Python default is bytes)
- Returns a `CompletedProcess` with `.returncode`, `.stdout`, `.stderr`

> **Coming from C:** Like `popen()` but safer — no shell interpolation. The command is a list of strings, not a single string, so arguments with spaces are handled correctly.
> **Coming from JS/TS:** Like `child_process.execFileSync()` with `encoding: 'utf8'`. The list form (`["cmd", "arg1"]`) is like `execFile`, not `exec` — no shell, no injection risk.

## Topological sort — dependency ordering

A DFS-based sort that outputs nodes after all their dependencies. Used to order sigils so dependencies are generated before dependents.

```python
def topological_sort(sigils):
    visited, in_progress, result = set(), set(), []

    def visit(name):
        if name in in_progress:
            raise ValueError(f"Circular dependency: '{name}'")
        if name in visited:
            return
        in_progress.add(name)
        for dep in by_name[name].dependencies:
            visit(dep)
        in_progress.remove(name)
        visited.add(name)
        result.append(by_name[name])

    for s in sigils:
        visit(s.name)
    return result
```

The `in_progress` set detects cycles — if we encounter a node we're currently visiting, we've found a loop.

> **Coming from C:** Same algorithm as `make` or `tsort` — the classic Cormen/Leiserson/Rivest DFS topological sort. The recursion is the DFS, and appending after the recursive calls gives a valid ordering.
> **Coming from JS/TS:** npm uses topological sort for `node_modules` installation order. The algorithm is the same — DFS with cycle detection via a "visiting" set.
