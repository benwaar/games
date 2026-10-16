"""
Query the FAISS knowledge index.

Returns the top-K nearest chunks for a given query string.

Usage:
    python knowledge/scripts/query_index.py "how to output to screen on ZX Spectrum"
    python knowledge/scripts/query_index.py "WASM memory model" --top 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "knowledge" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed query_text and return the top_k nearest chunks with scores.

    Each result dict has: rank, score, path, topic, category, source_url, tags
    """
    if not FAISS_INDEX_PATH.exists():
        print(
            f"ERROR: index not found at {FAISS_INDEX_PATH}\n"
            "Run 'python knowledge/scripts/embed.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata: list[dict] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    model = SentenceTransformer(MODEL_NAME)
    embedding = model.encode([query_text], convert_to_numpy=True)

    # Normalise to match the unit-norm vectors stored in the index
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    distances, indices = index.search(embedding.astype(np.float32), top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx == -1:
            continue
        chunk = metadata[idx]
        results.append(
            {
                "rank": rank,
                "score": float(dist),
                "path": chunk["path"],
                "topic": chunk["topic"],
                "category": chunk["category"],
                "source_url": chunk.get("source_url", ""),
                "tags": chunk.get("tags", []),
            }
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the FAISS knowledge index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python knowledge/scripts/query_index.py "ZX Spectrum screen output"',
    )
    parser.add_argument("query", nargs="+", help="Query string (quoted or space-separated words)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_K, metavar="N", help=f"Number of results (default {DEFAULT_TOP_K})")
    args = parser.parse_args()

    query_text = " ".join(args.query)
    print(f'Query: "{query_text}"')
    print(f"Top {args.top} results:\n")

    results = query(query_text, top_k=args.top)

    if not results:
        print("No results found.")
        return

    for r in results:
        tags = ", ".join(r["tags"]) if isinstance(r["tags"], list) else r["tags"]
        print(f"  [{r['rank']}] score={r['score']:.4f}  {r['topic']}  ({r['category']})")
        print(f"       path: {r['path']}")
        if r["source_url"]:
            print(f"       url:  {r['source_url']}")
        if tags:
            print(f"       tags: {tags}")
        print()


if __name__ == "__main__":
    main()
