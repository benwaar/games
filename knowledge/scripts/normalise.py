"""
Normalise raw knowledge chunks.

Reads every .md file under knowledge/raw/, validates YAML frontmatter,
splits files that exceed the token budget, and writes uniform chunks to
knowledge/chunks/{category}/.

Run from repo root:
    python knowledge/scripts/normalise.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "knowledge" / "raw"
CHUNKS_DIR = REPO_ROOT / "knowledge" / "chunks"

VALID_CATEGORIES = {
    "z80-instruction",
    "z80-pattern",
    "spectrum-rom",
    "spectrum-hardware",
    "wasm-spec",
    "wasm-pattern",
    "wasi",
}

REQUIRED_FIELDS = {"topic", "category", "source_url", "licence", "tags"}

# ~750 words ≈ 1000 tokens
WORD_BUDGET = 750


# ---------------------------------------------------------------------------
# YAML frontmatter parser (no pyyaml dep)
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Split a markdown file into (frontmatter_dict, body).
    Returns ({}, full_text) when no frontmatter block is found.
    Raises ValueError on a malformed block.
    """
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

        # Inline lists: [a, b, c]
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            val = [v.strip().strip('"').strip("'") for v in inner.split(",") if v.strip()]
        elif isinstance(val, str):
            # Strip surrounding quotes
            val = val.strip('"').strip("'")

        meta[key] = val

    return meta, body


def _render_frontmatter(meta: dict) -> str:
    lines = ["---"]
    for key, val in meta.items():
        if isinstance(val, list):
            rendered = "[" + ", ".join(val) + "]"
            lines.append(f"{key}: {rendered}")
        else:
            # Quote values that contain colons or special chars
            s = str(val)
            if any(c in s for c in (':', '#', '{', '}')):
                lines.append(f'{key}: "{s}"')
            else:
                lines.append(f"{key}: {s}")
    lines.append("---")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _split_on_headings(body: str) -> list[str]:
    """Split body into sections at top-level headings (## or #)."""
    # Keep the heading with its section
    parts = re.split(r"(?=^#{1,2} )", body, flags=re.MULTILINE)
    return [p for p in parts if p.strip()]


def _split_body(body: str, budget: int) -> list[str]:
    """
    Split body into chunks each within budget words.
    Prefers heading boundaries; falls back to paragraph boundaries.
    """
    if _word_count(body) <= budget:
        return [body]

    sections = _split_on_headings(body)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for section in sections:
        section_words = _word_count(section)
        if current and current_words + section_words > budget:
            chunks.append("\n".join(current))
            current = []
            current_words = 0
        current.append(section)
        current_words += section_words

    if current:
        chunks.append("\n".join(current))

    # If a single section is still too large, split on paragraphs
    result: list[str] = []
    for chunk in chunks:
        if _word_count(chunk) <= budget:
            result.append(chunk)
        else:
            paras = chunk.split("\n\n")
            sub: list[str] = []
            sub_words = 0
            for para in paras:
                pw = _word_count(para)
                if sub and sub_words + pw > budget:
                    result.append("\n\n".join(sub))
                    sub = []
                    sub_words = 0
                sub.append(para)
                sub_words += pw
            if sub:
                result.append("\n\n".join(sub))

    return result if result else [body]


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _output_paths(stem: str, category: str, n_chunks: int) -> list[Path]:
    cat_dir = CHUNKS_DIR / category
    if n_chunks == 1:
        return [cat_dir / f"{stem}.md"]
    return [cat_dir / f"{stem}_{i + 1}.md" for i in range(n_chunks)]


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(src: Path) -> tuple[int, str | None]:
    """
    Process one raw markdown file.
    Returns (chunks_written, error_message_or_None).
    """
    text = src.read_text(encoding="utf-8")

    try:
        meta, body = _parse_frontmatter(text)
    except ValueError as exc:
        return 0, f"frontmatter parse error: {exc}"

    # Validate required fields
    missing = REQUIRED_FIELDS - set(meta.keys())
    if missing:
        return 0, f"missing fields: {', '.join(sorted(missing))}"

    category = meta.get("category", "")
    if category not in VALID_CATEGORIES:
        return 0, f"invalid category: {category!r}"

    # Split if needed
    body_parts = _split_body(body, WORD_BUDGET)
    stem = src.stem

    cat_dir = CHUNKS_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    out_paths = _output_paths(stem, category, len(body_parts))

    for idx, (part, out_path) in enumerate(zip(body_parts, out_paths)):
        chunk_meta = dict(meta)
        if len(body_parts) > 1:
            # Annotate multi-part chunks
            chunk_meta["topic"] = f"{meta['topic']} (part {idx + 1}/{len(body_parts)})"
        content = _render_frontmatter(chunk_meta) + "\n" + part.strip() + "\n"
        out_path.write_text(content, encoding="utf-8")

    return len(body_parts), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not RAW_DIR.exists():
        print(f"ERROR: raw dir not found: {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    raw_files = sorted(RAW_DIR.rglob("*.md"))

    total_input = len(raw_files)
    total_chunks = 0
    skipped: list[tuple[Path, str]] = []

    for src in raw_files:
        n, err = process_file(src)
        if err:
            skipped.append((src, err))
            rel = src.relative_to(RAW_DIR)
            print(f"  SKIP {rel}: {err}")
        else:
            total_chunks += n

    # Summary
    print()
    print("=" * 50)
    print(f"Input files  : {total_input}")
    print(f"Output chunks: {total_chunks}")
    print(f"Skipped      : {len(skipped)}")
    if skipped:
        print()
        print("Skipped files:")
        for path, reason in skipped:
            print(f"  {path.relative_to(RAW_DIR)}: {reason}")
    print("=" * 50)


if __name__ == "__main__":
    main()
