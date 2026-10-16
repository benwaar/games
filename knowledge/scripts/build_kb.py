#!/usr/bin/env python3
"""Build human-readable kb/ from knowledge/chunks/ and knowledge/curated/."""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = ROOT / "knowledge" / "chunks"
CURATED_DIR = ROOT / "knowledge" / "curated"
KB_DIR = ROOT / "_data" / "kb"

# category → (output_subdir, output_filename)
CATEGORY_MAP = {
    "z80-instruction": ("z80", "instructions.md"),
    "z80-pattern": ("z80", "patterns.md"),
    "z80-undocumented": ("z80", "undocumented.md"),
    "z80-beeper": ("z80", "beeper-engines.md"),
    "z80-library": ("z80", "library.md"),
    "spectrum-rom": ("spectrum", "rom-routines.md"),
    "spectrum-hardware": ("spectrum", "hardware.md"),
    "wasm-spec": ("wasm", "spec.md"),
    "wasm-binary": ("wasm", "binary-format.md"),
    "wasm-pattern": ("wasm", "patterns.md"),
    "wasi": ("wasm", "wasi.md"),
}

# curated files that map directly to kb output
CURATED_MAP = {
    ("wat", "types.md"): ("wasm", "types.md"),
    ("wat", "syntax.md"): ("wasm", "syntax.md"),
    ("wat", "memory.md"): ("wasm", "memory.md"),
    ("wat", "instructions.md"): ("wasm", "wat-instructions.md"),
    ("wat", "patterns.md"): ("wasm", "wat-patterns.md"),
    ("z80", "registers.md"): ("z80", "registers.md"),
    ("z80", "flags.md"): ("z80", "flags.md"),
    ("z80", "memory.md"): ("z80", "memory.md"),
    ("z80", "harness_io.md"): ("z80", "harness-io.md"),
}

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
PART_RE = re.compile(r"^(.+?)_(\d+)$")


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text).strip()


def extract_frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    result = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", text.lower()).strip("-")


def group_chunks(category_dir: Path) -> list[tuple[str, str]]:
    """Read chunks, reassemble multi-part files, return (topic, body) pairs sorted by topic."""
    files = sorted(category_dir.glob("*.md"))
    parts: dict[str, list[tuple[int, str, str]]] = {}

    for f in files:
        text = f.read_text(encoding="utf-8")
        fm = extract_frontmatter(text)
        body = strip_frontmatter(text)
        topic = fm.get("topic", f.stem)
        stem = f.stem

        m = PART_RE.match(stem)
        if m:
            base, idx = m.group(1), int(m.group(2))
            parts.setdefault(base, []).append((idx, topic, body))
        else:
            parts.setdefault(stem, []).append((0, topic, body))

    result = []
    for base in sorted(parts.keys()):
        entries = sorted(parts[base], key=lambda x: x[0])
        topic = entries[0][1]
        # strip " (part N/M)" from topic
        topic = re.sub(r"\s*\(part \d+/\d+\)", "", topic)
        combined = "\n\n".join(e[2] for e in entries)
        result.append((topic, combined))

    return result


def build_toc(sections: list[tuple[str, str]]) -> str:
    lines = []
    for topic, _ in sections:
        anchor = slugify(topic)
        lines.append(f"- [{topic}](#{anchor})")
    return "\n".join(lines)


def build_file(title: str, sections: list[tuple[str, str]]) -> str:
    parts = [f"# {title}\n"]
    if len(sections) > 5:
        parts.append(build_toc(sections))
        parts.append("")

    for topic, body in sections:
        # if body already starts with the topic as an h1, don't duplicate
        if body.startswith(f"# {topic}"):
            parts.append(body)
        else:
            parts.append(f"## {topic}\n\n{body}")
        parts.append("---\n")

    return "\n\n".join(parts).rstrip() + "\n"


TITLES = {
    ("z80", "instructions.md"): "Z80 Instruction Set",
    ("z80", "patterns.md"): "Z80 Common Patterns",
    ("z80", "undocumented.md"): "Z80 Undocumented Opcodes",
    ("z80", "beeper-engines.md"): "ZX Spectrum 1-Bit Beeper Engines",
    ("z80", "library.md"): "Z80 Library Routines",
    ("spectrum", "rom-routines.md"): "ZX Spectrum ROM Routines",
    ("spectrum", "hardware.md"): "ZX Spectrum Hardware",
    ("wasm", "spec.md"): "WebAssembly Specification Reference",
    ("wasm", "binary-format.md"): "WebAssembly Binary Format",
    ("wasm", "patterns.md"): "WebAssembly Common Patterns",
    ("wasm", "wasi.md"): "WASI Preview 1",
}


def main():
    if KB_DIR.exists():
        shutil.rmtree(KB_DIR)

    for category, (subdir, filename) in CATEGORY_MAP.items():
        cat_dir = CHUNKS_DIR / category
        if not cat_dir.exists():
            print(f"  skip {category} (no chunks)")
            continue

        sections = group_chunks(cat_dir)
        title = TITLES.get((subdir, filename), category)
        content = build_file(title, sections)

        out_path = KB_DIR / subdir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"  {out_path.relative_to(ROOT)}: {len(sections)} sections")

    # curated files — copy with frontmatter stripped
    for (cur_subdir, cur_file), (kb_subdir, kb_file) in CURATED_MAP.items():
        src = CURATED_DIR / cur_subdir / cur_file
        if not src.exists():
            print(f"  skip curated {cur_subdir}/{cur_file} (not found)")
            continue
        text = src.read_text(encoding="utf-8")
        body = strip_frontmatter(text)
        out_path = KB_DIR / kb_subdir / kb_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body + "\n", encoding="utf-8")
        print(f"  {out_path.relative_to(ROOT)}: curated")

    print(f"\nDone. Output: {KB_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
