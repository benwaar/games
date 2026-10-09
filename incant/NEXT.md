# Next: M1 — Project scaffold + knowledge base

Curate Z80 and WAT instruction set docs, embed them, write the sigil parser and example sigils.

**What's done:** project structure, README, PLAN, NEXT, CLAUDE.md, requirements, setup.sh

**What's next:**
1. Z80 knowledge docs (opcodes, registers, flags, memory map)
2. WAT knowledge docs (S-expressions, types, instructions, memory model)
3. Embed into vector store
4. Sigil format parser + 6 example sigils
5. Verify RAG retrieval

**Gate:** `python -m incant rag query "add two numbers" --collection z80` returns relevant chunks.
