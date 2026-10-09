# incant — Explainers

How the incant pipeline works, concept by concept.

## Pipeline overview

```
sigil.yaml → RAG retrieve → LLM prompt → code → assemble → test → binary
manifest.yaml → toposort → generate each → stitch → assemble → verify → combined binary
spec.md (BDD) → parse → LLM decompose → sigil YAMLs + manifest → cast_multi → WASI test → binary
```

1. **Sigil** — a YAML spec defining one function: name, signature, target (z80/wat), test cases
2. **RAG** — retrieves relevant instruction set docs to ground the LLM's output
3. **LLM** — generates Z80 asm or WAT from the spec + reference docs
4. **Gate** — assembles the code and runs test cases; retries on failure (max 3, or 5 for WASI)
5. **Manifest** — lists multiple sigils to compose into one program
6. **Stitch** — merges outputs (WAT: module merge + dedup, Z80: concatenation)
7. **Smelt** — BDD spec → LLM decomposition → sigil YAML → cast_multi → WASI scenario tests

## Explainers

| Doc | Covers |
|-----|--------|
| [Sigil Design](sigil-design.md) | YAML spec format, Z80 vs WAT differences, dependencies, design principles |
| [RAG Pipeline](rag-pipeline.md) | Chunking, embedding, vector search, prompt injection |
| [Z80 Backend](z80-backend.md) | Assembly, emulation, gate-driven retry, code extraction |
| [WAT Backend](wat-backend.md) | wat2wasm, wasm-interp, signed/unsigned conversion, structured control flow |
| [Multi-Sigil Orchestration](multi-sigil.md) | Manifests, stitching, dependencies, topological sort |
| [Smelt Pipeline](smelt-pipeline.md) | BDD specs, WASI I/O, library catalogue, LLM decomposition |
| [Python Concepts](python-concepts.md) | dataclass, zip, generators, pathlib, JSONL, regex, setattr/getattr, subprocess, topological sort, YAML frontmatter |
| [Libraries](libraries.md) | ollama, z80, pyyaml, wabt |

For shared concepts (CNNs, training loops, RL), see [../../explainers/](../../explainers/README.md).
