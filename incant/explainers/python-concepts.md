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
