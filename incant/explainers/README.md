# incant — Explainers

How the incant pipeline works, concept by concept.

## Pipeline overview

```
sigil.yaml → RAG retrieve → LLM prompt → code → assemble → test → binary
```

1. **Sigil** — a YAML spec defining one function: name, signature, target (z80/wat), test cases
2. **RAG** — retrieves relevant instruction set docs to ground the LLM's output
3. **LLM** — generates Z80 asm or WAT from the spec + reference docs
4. **Gate** — assembles the code and runs test cases; retries on failure (max 3)

## Explainers

| Doc | Covers |
|-----|--------|
| [RAG Pipeline](rag-pipeline.md) | Chunking, embedding, vector search, prompt injection |
| [Z80 Backend](z80-backend.md) | Assembly, emulation, gate-driven retry, code extraction |
| [Python Concepts](python-concepts.md) | dataclass, zip, generators, pathlib, JSONL, regex, setattr/getattr |
| [Libraries](libraries.md) | ollama, z80, pyyaml, wabt |

For shared concepts (CNNs, training loops, RL), see [../../explainers/](../../explainers/README.md).
