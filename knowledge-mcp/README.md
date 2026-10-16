# knowledge-mcp

Rust MCP server for the knowledge corpus. Exposes search and read tools so Claude can query Z80/Spectrum/WASM chunks mid-session.

One of three access layers to the same knowledge base: this MCP server is for Claude, the `/asm` skill is for the user, and the FAISS embeddings are for the local LLM (incant RAG).

All three read from the same `knowledge/chunks/*.md` files — same source of truth, three access patterns for three consumers:

| Layer | Consumer | Access pattern |
|-------|----------|----------------|
| **knowledge-mcp** | Claude | In-memory keyword index, MCP stdio |
| **/asm skill** | User | Shell script, browsable from the CLI |
| **FAISS embeddings** | Local LLM (incant) | Semantic vector search, RAG retrieval |

## Build

```bash
cargo build --release
```

Binary: `target/release/knowledge-mcp`

## How it works

On startup, walks `../knowledge/chunks/` and indexes every `.md` file into memory. Frontmatter `topic:` fields, filenames, and body text are all searchable.

Scoring: +10 topic match, +5 filename match, +1 body match. Returns top 5.

## Tools

| Tool | Description |
|------|-------------|
| `search(query, category?)` | Keyword search across all chunks. Optional category filter (e.g. `z80-instruction`, `wasm-spec`). |
| `read(chunk_id)` | Read full content of a chunk. ID format: `category/stem` (e.g. `z80-instruction/LD`). |

## Resources

| URI | Description |
|-----|-------------|
| `corpus://categories` | List of chunk categories with file counts |
| `corpus://stats` | Total chunk count and last rebuild timestamp |

## Configuration

Registered in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "knowledge-corpus": {
      "command": "./knowledge-mcp/target/release/knowledge-mcp",
      "cwd": "/path/to/games"
    }
  }
}
```

The server expects to be launched with `cwd` set to the repo root (it looks for `knowledge/chunks/` relative to the working directory).

## Dependencies

- `rust-mcp-sdk` v2 (MCP protocol, stdio transport)
- `serde` / `serde_json` (frontmatter parsing)
- `tokio` (async runtime)
