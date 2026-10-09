"""RAG module — embed and query knowledge docs via Ollama."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import ollama

EMBED_MODEL = "nomic-embed-text"


@dataclass
class Chunk:
    text: str
    source: str
    collection: str
    heading: str
    embedding: list[float]


def chunk_markdown(path: Path) -> list[tuple[str, str]]:
    """Split a markdown file into chunks by heading. Returns (heading, text) pairs."""
    text = path.read_text()
    chunks = []
    current_heading = path.stem
    current_lines: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+", line):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    chunks.append((current_heading, body))
            current_heading = re.sub(r"^#+\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append((current_heading, body))

    return chunks


def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class VectorStore:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.chunks: list[Chunk] = []
        if store_path.exists():
            self._load()

    def _load(self):
        with open(self.store_path) as f:
            for line in f:
                raw = json.loads(line)
                self.chunks.append(Chunk(**raw))

    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w") as f:
            for chunk in self.chunks:
                json.dump(
                    {
                        "text": chunk.text,
                        "source": chunk.source,
                        "collection": chunk.collection,
                        "heading": chunk.heading,
                        "embedding": chunk.embedding,
                    },
                    f,
                )
                f.write("\n")

    def embed_directory(self, dir_path: Path, collection: str):
        """Embed all markdown files in a directory into a collection."""
        self.chunks = [c for c in self.chunks if c.collection != collection]

        for md_file in sorted(dir_path.glob("*.md")):
            for heading, text in chunk_markdown(md_file):
                embedding = embed_text(f"{heading}\n{text}")
                self.chunks.append(
                    Chunk(
                        text=text,
                        source=str(md_file.relative_to(dir_path.parent.parent)),
                        collection=collection,
                        heading=heading,
                        embedding=embedding,
                    )
                )

        self._save()

    def query(
        self, query_text: str, collection: str | None = None, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Query the store, return top-k chunks with similarity scores."""
        query_embedding = embed_text(query_text)

        candidates = self.chunks
        if collection:
            candidates = [c for c in candidates if c.collection == collection]

        scored = [
            (chunk, cosine_similarity(query_embedding, chunk.embedding))
            for chunk in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
