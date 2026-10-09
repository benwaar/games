# RAG Pipeline — Retrieval-Augmented Generation for Code Gen

RAG grounds an LLM's output in factual reference material. Instead of relying on what the model memorised during training, we retrieve relevant documents at query time and inject them into the prompt.

## Why RAG for incant

The LLM (qwen3-coder) knows some Z80 and WAT, but not reliably — it may hallucinate opcodes, invent syntax, or mix instruction sets. By feeding it curated, correct reference docs alongside every prompt, we dramatically reduce hallucination and give the gate (assembler + test runner) a much better chance of passing on the first attempt.

## How it works

```
query → embed → cosine search → top-k chunks → inject into prompt
```

### 1. Chunking

Markdown knowledge docs are split by heading. Each chunk is a (heading, body) pair — small enough to embed meaningfully, large enough to be useful context.

```python
# From incant/rag.py
def chunk_markdown(path: Path) -> list[tuple[str, str]]:
    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+", line):
            # new heading = new chunk
```

> **Coming from JS/TS:** This is like splitting a markdown AST by heading node. Python's `re.match` is anchored to the start of the string (unlike JS `match` which scans) — so `^` is redundant but explicit.

### 2. Embedding

Each chunk is converted to a dense vector (768 floats) using `nomic-embed-text` via Ollama. The embedding captures semantic meaning — "add two registers" and "ADD A, B" will be close in vector space even though they share few words.

```python
def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]
```

We prefix the heading to the body before embedding: `f"{heading}\n{text}"`. This gives the embedding model context about what section the chunk belongs to.

> **Coming from C:** Think of embeddings as a hash function, but instead of collisions being bad, *proximity* is the goal. Similar inputs produce similar outputs, and you measure "similar" with cosine distance rather than equality.

### 3. Storage

Chunks + embeddings are stored as JSONL (one JSON object per line). No database, no external service — just a flat file.

```json
{"text": "ADD A, B adds register B to A", "source": "z80/instructions.md", "collection": "z80", "heading": "Arithmetic", "embedding": [0.12, -0.34, ...]}
```

> **Coming from JS/TS:** JSONL is `JSON.stringify(obj) + '\n'` per record — you can `readline()` and `JSON.parse()` each line. No array wrapper, no commas between records.

### 4. Querying

At query time, the search string is embedded with the same model, then compared to every chunk using cosine similarity. Top-k results are returned.

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    return dot / (mag_a * mag_b)
```

Cosine similarity ranges from -1 (opposite) to 1 (identical). For normalised embeddings, it simplifies to the dot product — but we compute the full formula since `nomic-embed-text` doesn't guarantee unit vectors.

> **Coming from C:** This is the same formula you'd use for the angle between two vectors in 3D graphics, just in 768 dimensions.

### 5. Injection

The top-k chunks are formatted as markdown sections and prepended to the LLM prompt:

```python
def get_rag_context(store, sigil, top_k=5) -> str:
    results = store.query(query, collection=sigil.target, top_k=top_k)
    chunks = [f"### {chunk.heading}\n{chunk.text}" for chunk, score in results]
    return "## Reference documentation\n\n" + "\n\n".join(chunks)
```

## In practice

RAG is the standard pattern for grounding LLM output in domain knowledge — customer support bots retrieving help docs, code assistants retrieving API specs, legal tools retrieving case law. The core loop (chunk → embed → store → retrieve → inject) is the same everywhere. What changes is the chunking strategy, the embedding model, and the retrieval threshold.

## Trade-offs in this implementation

- **No vector database.** Linear scan over all chunks. Fine for ~100 chunks, won't scale to millions. Production systems use FAISS, pgvector, or Pinecone.
- **No reranking.** We trust cosine similarity alone. Production systems often rerank the top-k with a cross-encoder for better precision.
- **Collection filtering.** We separate Z80 and WAT docs into collections so a Z80 query never returns WAT results (and vice versa). Simple but effective.

## Quality in, quality out — where the knowledge came from

RAG only works if the reference material is correct. This sounds obvious, but it's easy to get wrong.

### What happened here

The knowledge docs in `knowledge/z80/` and `knowledge/wat/` were originally written by Claude from training data — not copied from authoritative sources. They were plausible, well-structured, and completely unverified. The assumption was that the model "knows" Z80 and WAT well enough to write accurate reference material.

That assumption was wrong. When we cross-checked against the real specs, we found:

- **ZX Spectrum memory map had system variables at the wrong address** (0x5B00 instead of 0x5C00 — that's actually the printer buffer)
- **Free RAM was overstated** (~41K vs the real figure after system vars and BASIC workspace)
- **`wasm-interp` argument syntax was fabricated** (`-- args` instead of the real `-a i32:N` flag) — this would have broken the WAT test runner entirely
- **WASM type system was described as MVP-only** without noting that multi-value returns have been standard since 2.0
- **PUSH/POP was missing IX and IY** register pairs
- **Undocumented Z80 flag bits** had simplified behaviour that's wrong for CP, SCF, CCF instructions

Six errors across ten files. Some subtle (flag bit edge cases), some showstoppers (wrong CLI syntax for the test runner).

### The verification process

We verified against:

- **Z80:** Zilog Z80 CPU User Manual (the original datasheet), ClrHome opcode table, WoS/NVG Spectrum FAQ, speccy-bootcamp system variables reference
- **WAT:** WebAssembly spec (webassembly.github.io/spec), WABT documentation, MDN WebAssembly reference

Every claim in every file was checked. Corrections were committed, knowledge was re-embedded, and all tests were re-run.

### The lesson

This is the central risk of RAG: **the retrieval mechanism doesn't know whether the documents are correct.** Cosine similarity measures relevance, not truth. If you embed a wrong opcode table, the retrieval will faithfully return wrong opcodes with high confidence scores.

In production RAG systems, this shows up as:
- **Stale docs** — API reference that describes the v2 endpoint when v3 shipped six months ago
- **Contradictory sources** — two wiki pages that disagree on how a feature works
- **Authoritative-looking garbage** — LLM-generated docs that read well but contain hallucinations (which is exactly what we had here)

The fix isn't technical — no amount of reranking or chunk-size tuning compensates for bad source material. The fix is editorial: verify your knowledge base against primary sources, version it alongside your code, and treat it as a first-class artefact that needs review.

> **In practice:** The same principle applies to any retrieval system. A customer support bot grounded in an outdated help centre will confidently give wrong answers. A legal RAG system citing superseded case law will mislead. The embedding model and the retrieval pipeline are plumbing — the quality of the output is bounded by the quality of what you put in.
