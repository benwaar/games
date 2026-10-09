"""Tests for RAG module — chunking, vector store, and query."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from incant.rag import Chunk, VectorStore, chunk_markdown, cosine_similarity


@pytest.fixture
def tmp_md(tmp_path):
    """Create a temp markdown file with headings."""
    md = tmp_path / "test.md"
    md.write_text(
        "# Arithmetic\n\n"
        "ADD A, B adds two registers.\n\n"
        "## Carry flag\n\n"
        "Set when result > 255.\n\n"
        "# Jumps\n\n"
        "JP addr jumps unconditionally.\n"
    )
    return md


@pytest.fixture
def tmp_store(tmp_path):
    return VectorStore(tmp_path / "store.jsonl")


def fake_embedding(dims: int = 4) -> list[float]:
    """Deterministic fake embedding for testing."""
    return [0.1] * dims


class TestChunkMarkdown:
    def test_splits_by_heading(self, tmp_md):
        chunks = chunk_markdown(tmp_md)
        assert len(chunks) == 3
        assert chunks[0][0] == "Arithmetic"
        assert chunks[1][0] == "Carry flag"
        assert chunks[2][0] == "Jumps"

    def test_body_content(self, tmp_md):
        chunks = chunk_markdown(tmp_md)
        assert "ADD A, B" in chunks[0][1]
        assert "result > 255" in chunks[1][1]
        assert "JP addr" in chunks[2][1]

    def test_empty_file(self, tmp_path):
        md = tmp_path / "empty.md"
        md.write_text("")
        assert chunk_markdown(md) == []

    def test_no_headings(self, tmp_path):
        md = tmp_path / "flat.md"
        md.write_text("Just some text\nwith no headings.\n")
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0][0] == "flat"  # uses filename stem

    def test_heading_only_no_body(self, tmp_path):
        md = tmp_path / "heading_only.md"
        md.write_text("# Title\n\n# Another\n\nSome text.\n")
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0][0] == "Another"


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 1.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestVectorStore:
    @patch("incant.rag.embed_text", return_value=[0.5, 0.5, 0.0, 0.0])
    def test_embed_directory(self, mock_embed, tmp_path, tmp_store):
        docs = tmp_path / "knowledge" / "z80"
        docs.mkdir(parents=True)
        (docs / "ops.md").write_text("# Add\n\nADD A, B\n")

        tmp_store.embed_directory(docs, "z80")

        assert len(tmp_store.chunks) == 1
        assert tmp_store.chunks[0].collection == "z80"
        assert tmp_store.chunks[0].heading == "Add"
        assert mock_embed.called

    @patch("incant.rag.embed_text", return_value=[0.5, 0.5, 0.0, 0.0])
    def test_re_embed_replaces_collection(self, mock_embed, tmp_path, tmp_store):
        docs = tmp_path / "knowledge" / "z80"
        docs.mkdir(parents=True)
        (docs / "ops.md").write_text("# Add\n\nADD A, B\n")

        tmp_store.embed_directory(docs, "z80")
        tmp_store.embed_directory(docs, "z80")

        assert len(tmp_store.chunks) == 1  # not 2

    @patch("incant.rag.embed_text")
    def test_query_filters_by_collection(self, mock_embed, tmp_path, tmp_store):
        mock_embed.return_value = [1.0, 0.0, 0.0, 0.0]

        docs_z80 = tmp_path / "knowledge" / "z80"
        docs_z80.mkdir(parents=True)
        (docs_z80 / "ops.md").write_text("# Z80 Add\n\nADD A, B\n")

        docs_wat = tmp_path / "knowledge" / "wat"
        docs_wat.mkdir(parents=True)
        (docs_wat / "ops.md").write_text("# WAT Add\n\ni32.add\n")

        tmp_store.embed_directory(docs_z80, "z80")
        tmp_store.embed_directory(docs_wat, "wat")

        results = tmp_store.query("add", collection="z80")
        assert all(c.collection == "z80" for c, _ in results)

    @patch("incant.rag.embed_text", return_value=[0.5, 0.5, 0.0, 0.0])
    def test_persist_and_reload(self, mock_embed, tmp_path):
        store_path = tmp_path / "store.jsonl"
        store1 = VectorStore(store_path)

        docs = tmp_path / "knowledge" / "z80"
        docs.mkdir(parents=True)
        (docs / "ops.md").write_text("# Add\n\nADD A, B\n")

        store1.embed_directory(docs, "z80")

        store2 = VectorStore(store_path)
        assert len(store2.chunks) == 1
        assert store2.chunks[0].heading == "Add"

    @patch("incant.rag.embed_text")
    def test_query_returns_top_k(self, mock_embed, tmp_path, tmp_store):
        call_count = 0
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # query embedding
        ]

        def side_effect(text):
            nonlocal call_count
            idx = min(call_count, len(embeddings) - 1)
            call_count += 1
            return embeddings[idx]

        mock_embed.side_effect = side_effect

        docs = tmp_path / "knowledge" / "z80"
        docs.mkdir(parents=True)
        (docs / "ops.md").write_text(
            "# Add\n\nADD instruction\n\n"
            "# Jump\n\nJP instruction\n\n"
            "# Mixed\n\nADD and JP\n"
        )
        tmp_store.embed_directory(docs, "z80")

        results = tmp_store.query("add numbers", collection="z80", top_k=2)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]  # sorted by score
