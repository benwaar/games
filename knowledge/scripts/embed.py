"""
Generate a FAISS vector index from knowledge chunks.

Reads all .md files under knowledge/chunks/, embeds them with
sentence-transformers (all-MiniLM-L6-v2), and writes:
  - knowledge/index/faiss.index  — the FAISS index
  - knowledge/index/metadata.json — maps index positions to chunk metadata

Run from repo root:
    python knowledge/scripts/embed.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = REPO_ROOT / "knowledge" / "chunks"
INDEX_DIR = REPO_ROOT / "knowledge" / "index"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Frontmatter parser (no pyyaml — mirrors normalise.py)
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into (frontmatter_dict, body)."""
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    fm_block = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")

    meta: dict = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw_val = line.partition(":")
        key = key.strip()
        val: object = raw_val.strip()

        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            val = [v.strip().strip('"').strip("'") for v in inner.split(",") if v.strip()]
        elif isinstance(val, str):
            val = val.strip('"').strip("'")

        meta[key] = val

    return meta, body


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def load_chunks(chunks_dir: Path) -> list[dict]:
    """
    Recursively load all .md files from chunks_dir.

    Returns a list of dicts with keys:
        path, topic, category, source_url, licence, tags, body, embed_text
    """
    chunks = []
    skipped = 0
    for md_file in sorted(chunks_dir.rglob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"  [SKIP] {md_file}: {exc}")
            skipped += 1
            continue
        meta, body = _parse_frontmatter(text)

        topic = meta.get("topic", md_file.stem)
        category = meta.get("category", md_file.parent.name)

        # Text to embed: topic header + body content
        embed_text = f"{topic}\n\n{body}".strip()

        chunks.append(
            {
                "path": str(md_file.relative_to(REPO_ROOT)),
                "topic": topic,
                "category": category,
                "source_url": meta.get("source_url", ""),
                "licence": meta.get("licence", ""),
                "tags": meta.get("tags", []),
                "embed_text": embed_text,
            }
        )

    if skipped:
        print(f"  ({skipped} files skipped)")

    return chunks


# ---------------------------------------------------------------------------
# Embedding + indexing
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatL2:
    """Embed all chunks and build a FAISS L2 index."""
    texts = [c["embed_text"] for c in chunks]

    print(f"Encoding {len(texts)} chunks with {MODEL_NAME}…")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalise to unit vectors so L2 distance ≈ cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CHUNKS_DIR.exists():
        print(f"ERROR: chunks directory not found: {CHUNKS_DIR}", file=sys.stderr)
        sys.exit(1)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {CHUNKS_DIR} …")
    chunks = load_chunks(CHUNKS_DIR)
    if not chunks:
        print("No .md chunks found — nothing to embed.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(chunks)} chunks across categories:")
    categories: dict[str, int] = {}
    for c in chunks:
        categories[c["category"]] = categories.get(c["category"], 0) + 1
    for cat, n in sorted(categories.items()):
        print(f"  {cat}: {n}")

    print(f"\nLoading model {MODEL_NAME!r} …")
    model = SentenceTransformer(MODEL_NAME)

    index = build_index(chunks, model)

    # Save FAISS index and metadata atomically (temp → rename)
    tmp_index = FAISS_INDEX_PATH.with_suffix(".tmp")
    tmp_meta = METADATA_PATH.with_suffix(".tmp")

    metadata = [
        {k: v for k, v in c.items() if k != "embed_text"}
        for c in chunks
    ]

    faiss.write_index(index, str(tmp_index))
    tmp_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    tmp_index.rename(FAISS_INDEX_PATH)
    tmp_meta.rename(METADATA_PATH)

    print(f"\nFAISS index saved → {FAISS_INDEX_PATH}")
    print(f"Metadata saved   → {METADATA_PATH}")

    dim = index.d
    print(f"\nDone. {index.ntotal} vectors, dim={dim}.")


if __name__ == "__main__":
    main()
