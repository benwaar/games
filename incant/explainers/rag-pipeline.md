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
