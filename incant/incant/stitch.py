"""Stitch multiple generated outputs into a single program."""

import re
from pathlib import Path


def stitch_wat(modules: list[str]) -> str:
    """Merge multiple WAT modules into one.

    Extracts inner declarations from each (module ...) and combines them.
    Deduplicates memory declarations (keeps the largest) and functions
    (first definition wins — later modules that redefine $add are skipped).
    """
    seen_funcs: set[str] = set()
    seen_exports: set[str] = set()
    inner_blocks: list[str] = []
    max_memory_pages = 0
    has_memory = False
    memory_exports: list[str] = []

    for module in modules:
        inner = _extract_wat_inner(module)
        declarations = _split_top_level_declarations(inner)

        kept: list[str] = []
        for decl in declarations:
            stripped = decl.strip()

            mem_pages = _parse_memory_decl(stripped)
            if mem_pages is not None:
                has_memory = True
                max_memory_pages = max(max_memory_pages, mem_pages)
                continue

            if _is_memory_export(stripped):
                if not memory_exports:
                    memory_exports.append(decl)
                continue

            func_name = _extract_func_name(stripped)
            if func_name:
                if func_name in seen_funcs:
                    continue
                seen_funcs.add(func_name)

            export_name = _extract_export_name(stripped)
            if export_name:
                if export_name in seen_exports:
                    continue
                seen_exports.add(export_name)

            kept.append(decl)

        if kept:
            inner_blocks.append("\n".join(kept))

    parts = []
    if has_memory:
        parts.append(f"  (memory {max_memory_pages})")
        if memory_exports:
            parts.append(memory_exports[0])

    for block in inner_blocks:
        block = block.strip()
        if block:
            parts.append(block)

    return "(module\n" + "\n".join(parts) + "\n)\n"


def _split_top_level_declarations(inner: str) -> list[str]:
    """Split module inner content into top-level S-expression declarations."""
    declarations = []
    i = 0
    text = inner.strip()

    while i < len(text):
        if text[i] == "(":
            depth = 1
            start = i
            i += 1
            while i < len(text) and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            declarations.append(text[start:i])
        else:
            i += 1

    return declarations


def _extract_func_name(decl: str) -> str | None:
    """Extract function name from (func $name ...)."""
    match = re.match(r"\(func\s+(\$\w+)", decl)
    return match.group(1) if match else None


def _extract_export_name(decl: str) -> str | None:
    """Extract export string from (export "name" ...)."""
    match = re.match(r'\(export\s+"([^"]+)"', decl)
    return match.group(1) if match else None


def _extract_wat_inner(module: str) -> str:
    """Extract content between outer (module and closing )."""
    text = module.strip()
    start = text.find("(module")
    if start < 0:
        return text

    # skip past "(module" and optional whitespace/newline
    inner_start = start + len("(module")
    # find matching close paren
    depth = 1
    i = inner_start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1

    return text[inner_start : i - 1]


def _parse_memory_decl(line: str) -> int | None:
    """Parse (memory N) and return N, or None if not a memory declaration."""
    match = re.match(r"^\(memory\s+(\d+)\)$", line)
    if match:
        return int(match.group(1))
    return None


def _is_memory_export(line: str) -> bool:
    return '(export' in line and '(memory' in line


def stitch_z80(sources: list[str], names: list[str]) -> str:
    """Concatenate Z80 asm sources into one program.

    Each source becomes a labelled section. Removes trailing halt from
    all but the last source so execution falls through.
    """
    parts: list[str] = []

    for i, (source, name) in enumerate(zip(sources, names)):
        is_last = i == len(sources) - 1
        cleaned = _strip_trailing_halt(source) if not is_last else source

        parts.append(f"; --- {name} ---")
        parts.append(cleaned.strip())
        parts.append("")

    return "\n".join(parts)


def _strip_trailing_halt(source: str) -> str:
    """Remove trailing halt instruction from Z80 source."""
    lines = source.rstrip().splitlines()
    while lines and lines[-1].strip().lower() == "halt":
        lines.pop()
    return "\n".join(lines)
