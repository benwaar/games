# Embeddings and RAG — From Text to Lookup

How the knowledge corpus turns plain-text chunks into something a model can search at generation time.

---

## What's an embedding?

A function that takes text and returns a fixed-length vector of floats. "LD A, (HL)" might become `[0.12, -0.44, 0.87, ...]` — 384 numbers that encode the *meaning* of that text, not its spelling.

Texts with similar meaning land near each other in vector space. "load accumulator from memory" and "LD A, (HL)" are far apart as strings but close as vectors.

> **Coming from JS/TS:** Think of it like a hash function, except similar inputs produce similar outputs instead of completely different ones.

> **Coming from C:** It's a feature extractor. Raw text goes in, a dense float array comes out — like how an FFT turns time-domain audio into frequency-domain, this turns natural language into a geometric space where distance = semantic similarity.

---

## How embedding models work

A pre-trained transformer (same architecture family as GPT/Claude, but much smaller) reads your text and produces a vector. Common models:

| Model | Dimensions | Size | Runner |
|-------|-----------|------|--------|
| `all-MiniLM-L6-v2` | 384 | ~90 MB | sentence-transformers |
| `nomic-embed-text` | 768 | ~274 MB | ollama |
| `mxbai-embed-large` | 1024 | ~670 MB | ollama |

Smaller models are faster and use less memory. For a corpus of a few hundred chunks, the smallest model is more than enough — the bottleneck is retrieval quality from your chunk boundaries, not embedding precision.

---

## Two ways to run embeddings locally

### sentence-transformers (Python, in-process)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(["LD A, (HL)", "LDIR copies BC bytes"])
```

Runs in the Python process. Fast for batches (no HTTP overhead). Downloads the model on first run.

### ollama embeddings (HTTP, via ollama server)

```python
import ollama

result = ollama.embeddings(model="nomic-embed-text", prompt="LD A, (HL)")
vector = result["embedding"]
```

Runs through the ollama server. Same idea, different runner. Slightly more latency per call from the HTTP round-trip, but negligible for small queries.

**This project uses sentence-transformers** for the batch embed step at setup time. ollama is used separately by incant for LLM code generation.

> **In practice:** Most production systems use an API (OpenAI, Cohere, Voyage) rather than running locally. The tradeoff is latency + cost vs. no-dependency simplicity. For a local R&D corpus, local models are the obvious choice.

---

## FAISS — the vector index

Once every chunk is embedded, you need fast nearest-neighbour search. [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is a C++ library with Python bindings that does exactly this.

```python
import faiss
import numpy as np

# Build: one-time, at setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))  # shape: (n_chunks, 384)
faiss.write_index(index, "corpus.faiss")

# Query: at generation time
index = faiss.read_index("corpus.faiss")
query_vec = model.encode(["how do I play a note on the Spectrum?"])
distances, indices = index.search(np.array(query_vec), k=5)
# indices[0] = [42, 17, 88, 3, 61] → the 5 nearest chunk IDs
```

`IndexFlatL2` is brute-force L2 distance — compares the query against every vector. For a few hundred chunks this is instant. For millions of vectors you'd switch to approximate indices (`IndexIVFFlat`, `IndexHNSW`) that trade a tiny accuracy loss for massive speedup.

> **Coming from databases:** FAISS is like a specialised index — instead of B-trees on sorted keys, it's spatial partitioning on high-dimensional vectors. The query isn't "WHERE id = 42", it's "find the 5 points closest to this point in 384-dimensional space."

---

## RAG — Retrieval-Augmented Generation

RAG is the pattern that connects embeddings and FAISS to an LLM. Three steps:

1. **Embed the query.** User asks "how does the Spectrum BEEP routine work?" — embed that into a vector.
2. **Retrieve.** FAISS finds the k nearest chunks — e.g. the ROM BEEP disassembly, the beeper hardware docs, the Z80 OUT instruction.
3. **Generate.** Feed those chunks as context to the LLM alongside the original query. The model now has grounding material instead of guessing from training data.

```
User query
    ↓
[Embedding model] → query vector
    ↓
[FAISS index] → top-k chunk IDs
    ↓
[Read chunks from disk]
    ↓
[LLM prompt] = system + retrieved chunks + user query
    ↓
Generated answer (grounded in retrieved facts)
```

### Why not just put everything in the prompt?

Context windows are finite and expensive. The knowledge corpus has hundreds of chunks — you can't paste them all in. RAG selects only the relevant ones, keeping the prompt focused and the generation accurate.

> **In practice:** RAG is how most "chat with your docs" products work. The quality depends more on chunking strategy (how you split the source material) and retrieval precision (do the right chunks come back?) than on the LLM itself. A mediocre model with great retrieval beats a great model with no retrieval.

---

## How it fits in this project

```
knowledge/
  scripts/
    fetch_all.sh     ← download raw docs (ROM disassembly, Z80 tables, WASM specs)
    normalise.py      ← split into markdown chunks with frontmatter
    embed.py          ← sentence-transformers → FAISS index
  chunks/             ← the normalised markdown files (what the MCP server reads)
  corpus.faiss        ← the vector index (what incant queries)

knowledge-mcp/        ← Rust MCP server: keyword search + read (for Claude Code)
incant/               ← code gen pipeline: FAISS lookup → ollama LLM → Z80/WASM output
```

Two access paths to the same corpus:

- **MCP server** (keyword search) — Claude Code uses this to look things up during conversation. Text matching, no vectors.
- **FAISS index** (semantic search) — incant uses this during code generation. Vector similarity, finds conceptually related chunks even when the words don't match.

---

## Why two embedding libraries?

This project deliberately uses **sentence-transformers** for the knowledge corpus and **ollama embeddings** for incant. Same job, different runners — and that's the point.

| | Knowledge corpus (MCP) | Incant (RAG code gen) |
|---|---|---|
| **Embedding library** | sentence-transformers | ollama (`nomic-embed-text`) |
| **Consumer** | Claude Code (all projects) | Local LLM (incant only) |
| **Dependency** | Python + pip | Python + pip + ollama server |
| **When it runs** | Setup time (batch embed) | Query time (per generation) |

**The knowledge corpus serves every project** — zxw, glyph, artefact, void-duel. If someone is working on the Spectrum emulator and needs to look up a ROM routine, they shouldn't need ollama installed. Sentence-transformers is a pip install with no background server.

**Incant already needs ollama running** for code generation, so using ollama's embedding API there costs nothing extra. It keeps incant's stack self-contained — one server for both generation and retrieval.

The result: the shared knowledge layer has minimal dependencies (works everywhere), while the code gen pipeline uses the heavier tool it already depends on. Clean boundary between infrastructure and application.

---

## Key terms

| Term | What it means |
|------|--------------|
| **Embedding** | Text → fixed-length float vector encoding meaning |
| **Vector space** | The high-dimensional space where embeddings live. Nearby = semantically similar |
| **FAISS** | Library for fast nearest-neighbour search over vectors |
| **RAG** | Retrieve relevant chunks, then generate with them as context |
| **Chunk** | A single unit of knowledge (one file in `knowledge/chunks/`) — small enough to embed and retrieve individually |
| **k-NN** | k-nearest neighbours — "find the k vectors closest to my query" |
| **Cosine similarity** | Alternative distance metric to L2 — measures angle between vectors rather than distance. Often used interchangeably for normalised embeddings |
