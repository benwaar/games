"""
MCP server for the knowledge corpus.
Exposes search and read tools so Claude can query Z80/Spectrum/WASM chunks mid-session.

Requires mcp >= 2.0 (MCPServer API).
"""

from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer

CHUNKS_DIR = Path(__file__).parent.parent / "chunks"

app = MCPServer("knowledge-corpus")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
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


def _chunk_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _score_chunk(path: Path, query: str, category_filter: str | None) -> int:
    ql = query.lower()
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


@app.tool()
def search(query: str, category: str | None = None) -> str:
    """Search the knowledge corpus for Z80, Spectrum, or WASM chunks."""
    if not query.strip():
        return "Query cannot be empty."
    if not CHUNKS_DIR.exists():
        return "Chunks directory not found. Run the normaliser first."

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
        return f"No chunks found for query: {query!r}"

    lines: list[str] = [f"Top {len(top5)} results for '{query}':\n"]
    for score, path in top5:
        text = path.read_text(encoding="utf-8", errors="ignore")
        fm, body = _parse_frontmatter(text)
        topic = fm.get("topic", path.stem)
        cid = _chunk_id(path)
        preview = body[:200].replace("\n", " ").strip()
        lines.append(f"**{cid}** (score {score})\n  topic: {topic}\n  {preview}\n")

    return "\n".join(lines)


@app.tool()
def read(chunk_id: str) -> str:
    """Read the full content of a specific knowledge chunk by its chunk_id (e.g. z80-instruction/LD)."""
    if not chunk_id.strip():
        return "chunk_id cannot be empty."

    parts = chunk_id.strip("/").split("/", 1)
    if len(parts) != 2:
        return "chunk_id must be category/filename (e.g. z80-instruction/LD)"

    category, stem = parts
    target = (CHUNKS_DIR / category / f"{stem}.md").resolve()

    if not target.is_relative_to(CHUNKS_DIR.resolve()):
        return f"Invalid chunk_id: {chunk_id}"

    if not target.exists():
        return f"Chunk not found: {chunk_id}"

    return target.read_text(encoding="utf-8", errors="ignore")


@app.resource("corpus://categories")
def categories() -> str:
    """List of chunk categories with file counts."""
    if not CHUNKS_DIR.exists():
        return "Chunks directory not found."
    lines = ["Categories:\n"]
    for cat_dir in sorted(CHUNKS_DIR.iterdir()):
        if cat_dir.is_dir():
            count = len(list(cat_dir.glob("*.md")))
            lines.append(f"  {cat_dir.name}: {count} chunks")
    return "\n".join(lines)


@app.resource("corpus://stats")
def stats() -> str:
    """Total chunk count and last rebuild timestamp."""
    if not CHUNKS_DIR.exists():
        return "Chunks directory not found."
    total = sum(1 for _ in CHUNKS_DIR.rglob("*.md"))
    mtimes = [p.stat().st_mtime for p in CHUNKS_DIR.rglob("*.md")]
    if mtimes:
        last = datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()
    else:
        last = "unknown"
    return f"Total chunks: {total}\nLast rebuild: {last}"


if __name__ == "__main__":
    app.run(transport="stdio")
