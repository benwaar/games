"""
MCP server for the knowledge corpus.
Exposes search and read tools so Claude can query Z80/Spectrum/WASM chunks mid-session.
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, TextContent

CHUNKS_DIR = Path(__file__).parent.parent / "chunks"

app = Server("knowledge-corpus")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body) from a markdown string with YAML front matter."""
    fm: dict = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip()
            body = text[end + 4:].strip()
    return fm, body


def _score_chunk(path: Path, query: str, category_filter: str | None) -> int:
    """Return relevance score for a chunk file; 0 = skip."""
    ql = query.lower()

    # Category filter: only search within that subdirectory
    if category_filter and path.parent.name != category_filter:
        return 0

    text = path.read_text(encoding="utf-8", errors="ignore")
    fm, body = _parse_frontmatter(text)
    topic = fm.get("topic", "").lower()

    score = 0
    if ql in topic:
        score += 10
    if ql in path.stem.lower():
        score += 5
    if ql in body.lower():
        score += 1
    return score


def _chunk_id(path: Path) -> str:
    """category/stem — used as a stable chunk_id."""
    return f"{path.parent.name}/{path.stem}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description="Search the knowledge corpus for Z80, Spectrum, or WASM chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (instruction mnemonic, topic name, keyword)",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional category filter. One of: z80-instruction, z80-pattern, "
                            "spectrum-rom, spectrum-hardware, wasm-spec, wasm-pattern, wasi"
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read",
            description="Read the full content of a specific knowledge chunk by its chunk_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "chunk_id in the form category/filename (e.g. z80-instruction/ld)",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search":
        return await _search(arguments.get("query", ""), arguments.get("category"))
    if name == "read":
        return await _read(arguments.get("chunk_id", ""))
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _search(query: str, category: str | None) -> list[TextContent]:
    if not query.strip():
        return [TextContent(type="text", text="Query cannot be empty.")]

    if not CHUNKS_DIR.exists():
        return [TextContent(type="text", text="Chunks directory not found. Run the normaliser first.")]

    scored: list[tuple[int, Path]] = []
    for md in CHUNKS_DIR.rglob("*.md"):
        try:
            score = _score_chunk(md, query, category)
        except Exception:
            continue
        if score > 0:
            scored.append((score, md))

    scored.sort(key=lambda t: t[0], reverse=True)
    top5 = scored[:5]

    if not top5:
        return [TextContent(type="text", text=f"No chunks found for query: {query!r}")]

    lines: list[str] = [f"Top {len(top5)} results for '{query}':\n"]
    for score, path in top5:
        text = path.read_text(encoding="utf-8", errors="ignore")
        fm, body = _parse_frontmatter(text)
        topic = fm.get("topic", path.stem)
        cid = _chunk_id(path)
        preview = body[:200].replace("\n", " ").strip()
        lines.append(f"**{cid}** (score {score})\n  topic: {topic}\n  {preview}\n")

    return [TextContent(type="text", text="\n".join(lines))]


async def _read(chunk_id: str) -> list[TextContent]:
    if not chunk_id.strip():
        return [TextContent(type="text", text="chunk_id cannot be empty.")]

    # chunk_id = category/stem
    parts = chunk_id.strip("/").split("/", 1)
    if len(parts) != 2:
        return [TextContent(type="text", text="chunk_id must be category/filename (e.g. z80-instruction/ld)")]

    category, stem = parts
    target = (CHUNKS_DIR / category / f"{stem}.md").resolve()

    if not target.is_relative_to(CHUNKS_DIR.resolve()):
        return [TextContent(type="text", text=f"Invalid chunk_id: {chunk_id}")]

    if not target.exists():
        return [TextContent(type="text", text=f"Chunk not found: {chunk_id}")]

    return [TextContent(type="text", text=target.read_text(encoding="utf-8", errors="ignore"))]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@app.list_resources()
async def list_resources():
    return [
        Resource(
            uri="corpus://categories",
            name="Corpus categories",
            description="List of chunk categories with file counts",
            mimeType="text/plain",
        ),
        Resource(
            uri="corpus://stats",
            name="Corpus stats",
            description="Total chunk count and last rebuild timestamp",
            mimeType="text/plain",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "corpus://categories":
        return _categories_text()
    if uri == "corpus://stats":
        return _stats_text()
    return f"Unknown resource: {uri}"


def _categories_text() -> str:
    if not CHUNKS_DIR.exists():
        return "Chunks directory not found."
    lines = ["Categories:\n"]
    for cat_dir in sorted(CHUNKS_DIR.iterdir()):
        if cat_dir.is_dir():
            count = len(list(cat_dir.glob("*.md")))
            lines.append(f"  {cat_dir.name}: {count} chunks")
    return "\n".join(lines)


def _stats_text() -> str:
    if not CHUNKS_DIR.exists():
        return "Chunks directory not found."
    total = sum(1 for _ in CHUNKS_DIR.rglob("*.md"))
    # Use the most recent mtime across all chunks as "last rebuild"
    mtimes = [p.stat().st_mtime for p in CHUNKS_DIR.rglob("*.md")]
    if mtimes:
        last = datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()
    else:
        last = "unknown"
    return f"Total chunks: {total}\nLast rebuild: {last}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
