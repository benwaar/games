"""
Fetch additional Z80 reference data: undocumented opcodes, beeper engines, and
ZX Spectrum library routines.

deps: (none beyond stdlib for pdftotext; requests for GitHub)

Sources:
  - Sean Young's "The Undocumented Z80 Documented" v0.91 (GFDL)
    PDF expected at /tmp/z80-undocumented.pdf (or pass --pdf <path>)
  - utz82/ZX-Spectrum-1-Bit-Routines (MIT)
    https://github.com/utz82/ZX-Spectrum-1-Bit-Routines
  - breakintoprogram/lib-spectrum (MIT)
    https://github.com/breakintoprogram/lib-spectrum
"""

import argparse
import re
import subprocess
import sys
import time
import logging
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent.parent / "raw" / "z80"

UNDOCUMENTED_DIR = RAW_DIR / "undocumented"
BEEPER_DIR = RAW_DIR / "beeper_engines"
LIB_DIR = RAW_DIR / "lib_routines"

DEFAULT_PDF_PATH = Path("/tmp/z80-undocumented.pdf")
PDF_SOURCE_URL = "http://datasheets.chipdb.org/Zilog/Z80/z80-documented-v0.91.pdf"

LIB_SPECTRUM_RAW = "https://raw.githubusercontent.com/breakintoprogram/lib-spectrum/master/lib/"
LIB_SPECTRUM_README = "https://raw.githubusercontent.com/breakintoprogram/lib-spectrum/master/README.md"
LIB_SPECTRUM_LICENSE = "https://raw.githubusercontent.com/breakintoprogram/lib-spectrum/master/LICENSE"
LIB_SPECTRUM_REPO = "https://github.com/breakintoprogram/lib-spectrum"

LIB_SPECTRUM_FILES = [
    "gui.z80",
    "keyboard.z80",
    "macros.z80",
    "math.z80",
    "output.z80",
    "score.z80",
    "screen_buffer.z80",
    "scroll.z80",
    "scroll_attr.z80",
    "sound.z80",
    "sprite.z80",
    "sprite_masked.z80",
    "vector.z80",
    "vector_filled.z80",
]

# Map filename stem → routine-type tag
_LIB_TAGS: dict[str, str] = {
    "gui": "gui",
    "keyboard": "input",
    "macros": "macros",
    "math": "math",
    "output": "output",
    "score": "score",
    "screen_buffer": "display",
    "scroll": "display",
    "scroll_attr": "display",
    "sound": "sound",
    "sprite": "sprites",
    "sprite_masked": "sprites",
    "vector": "graphics",
    "vector_filled": "graphics",
}
GITHUB_API = "https://api.github.com/repos/utz82/ZX-Spectrum-1-Bit-Routines/contents/"
GITHUB_RAW = "https://raw.githubusercontent.com/utz82/ZX-Spectrum-1-Bit-Routines/master/"
GITHUB_README = "https://raw.githubusercontent.com/utz82/ZX-Spectrum-1-Bit-Routines/master/README.md"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; knowledge-corpus-fetcher/1.0)"
}

TIMEOUT = 15  # seconds per request

# Priority engines to fetch first if rate-limited
PRIORITY_ENGINES = [
    "octode2k16",
    "nanobeep",
    "tftone",
    "pulsatilla",
    "zbmod",
    "CLICK_DRUMS",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (mirrored from fetch_z80.py)
# ---------------------------------------------------------------------------

def get(url: str) -> str | None:
    """Fetch URL and return response text, or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.HTTPError as exc:
        log.error("HTTP %s fetching %s — skipping", exc.response.status_code, url)
    except requests.RequestException as exc:
        log.error("Request failed for %s: %s — skipping", url, exc)
    return None


def get_json(url: str) -> list | dict | None:
    """Fetch URL and return parsed JSON, or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        log.error("HTTP %s fetching %s — skipping", exc.response.status_code, url)
    except requests.RequestException as exc:
        log.error("Request failed for %s: %s — skipping", url, exc)
    except ValueError as exc:
        log.error("Bad JSON from %s: %s — skipping", url, exc)
    return None


_UNSAFE_RE = re.compile(r"[^\w\-]")


def slug(text: str) -> str:
    """Convert free text to a safe filename stem."""
    name = re.sub(r"[\s/\\]+", "_", text.strip())
    name = _UNSAFE_RE.sub("", name)
    return name.strip("_") or "unknown"


def frontmatter(topic: str, category: str, source_url: str, licence: str, tags: list[str]) -> str:
    """Return a YAML frontmatter block as a string."""
    tag_str = ", ".join(tags)
    return (
        "---\n"
        f'topic: "{topic}"\n'
        f"category: {category}\n"
        f'source_url: "{source_url}"\n'
        f'licence: "{licence}"\n'
        f"tags: [{tag_str}]\n"
        "---\n"
    )


def write_raw(path: Path, content: str) -> None:
    """Write content to path, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Source 1: Undocumented Z80 PDF
# ---------------------------------------------------------------------------

# Chapter/section headings we want to split on.
# Each entry: (section_key, heading_pattern, topic_label, tags)
UNDOC_SECTIONS = [
    ("ch3_cb",       r"3\.1\s+CB Prefix",               "CB Prefix — Undocumented Z80 Opcodes",      ["z80", "undocumented", "opcodes", "CB"]),
    ("ch3_dd",       r"3\.2\s+DD Prefix",               "DD Prefix — Undocumented Z80 Opcodes",      ["z80", "undocumented", "opcodes", "DD", "IX"]),
    ("ch3_fd",       r"3\.3\s+FD Prefix",               "FD Prefix — Undocumented Z80 Opcodes",      ["z80", "undocumented", "opcodes", "FD", "IY"]),
    ("ch3_ed",       r"3\.4\s+ED Prefix",               "ED Prefix — Undocumented Z80 Opcodes",      ["z80", "undocumented", "opcodes", "ED"]),
    ("ch3_ddcb",     r"3\.5\s+DDCB Prefix",             "DDCB Prefix — Undocumented Z80 Opcodes",    ["z80", "undocumented", "opcodes", "DDCB", "IX"]),
    ("ch3_fdcb",     r"3\.6\s+FDCB Prefix",             "FDCB Prefix — Undocumented Z80 Opcodes",    ["z80", "undocumented", "opcodes", "FDCB", "IY"]),
    ("ch3_combo",    r"3\.7\s+Comb",                    "Combinations — Undocumented Z80 Opcodes",   ["z80", "undocumented", "opcodes"]),
    ("ch4_bit",      r"4\.1\s+BIT",                     "BIT Instructions — Undocumented Flag Effects", ["z80", "undocumented", "flags", "BIT"]),
    ("ch4_memblock", r"4\.2\s+[Mm]emory [Bb]lock",      "Memory Block — Undocumented Flag Effects",  ["z80", "undocumented", "flags", "block"]),
    ("ch4_ioblock",  r"4\.3\s+I/O [Bb]lock",            "I/O Block — Undocumented Flag Effects",     ["z80", "undocumented", "flags", "IO"]),
    ("ch4_16io",     r"4\.4\s+16-bit I/O",              "16-Bit I/O — Undocumented Flag Effects",    ["z80", "undocumented", "flags", "IO"]),
    ("ch4_blkinstr", r"4\.5\s+[Bb]lock [Ii]nstruction", "Block Instructions — Undocumented Flags",   ["z80", "undocumented", "flags", "block"]),
    ("ch4_16add",    r"4\.6\s+16-bit [Aa]dd",           "16-Bit Additions — Undocumented Flags",     ["z80", "undocumented", "flags", "arithmetic"]),
    ("ch4_daa",      r"4\.7\s+DAA",                     "DAA — Undocumented Flag Effects",           ["z80", "undocumented", "flags", "DAA"]),
    ("ch5_int",      r"5\.",                             "Interrupts",                                ["z80", "undocumented", "interrupts"]),
    ("ch6_timing",   r"6\.",                             "Timing and R Register",                     ["z80", "undocumented", "timing", "R"]),
    ("ch8_tables",   r"8\.",                             "Instruction Tables — Opcodes by Group",     ["z80", "undocumented", "tables", "opcodes"]),
    ("ch9_by_opcode",r"9\.",                             "Instructions Sorted by Opcode",             ["z80", "undocumented", "tables", "opcodes"]),
    ("ch10_by_mnem", r"10\.",                            "Instructions Sorted by Mnemonic",           ["z80", "undocumented", "tables", "mnemonics"]),
]


def extract_pdf_text(pdf_path: Path) -> str | None:
    """Run pdftotext -layout and return the extracted text, or None if unavailable."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.error("pdftotext exited %d: %s", result.returncode, result.stderr.strip())
            return None
        return result.stdout
    except FileNotFoundError:
        log.warning("pdftotext not found — falling back to stub content")
        return None
    except subprocess.TimeoutExpired:
        log.error("pdftotext timed out on %s", pdf_path)
        return None


def split_pdf_into_sections(text: str) -> list[tuple[str, str, str, list[str]]]:
    """
    Walk UNDOC_SECTIONS in order; for each pattern, find the first match in text
    and capture everything up to the next section boundary.

    Returns list of (section_key, topic, body_text, tags).
    """
    # Build list of (match_pos, section_index) for all sections found
    hits: list[tuple[int, int]] = []
    patterns = [(i, re.compile(pat, re.IGNORECASE)) for i, (_, pat, _, _) in enumerate(UNDOC_SECTIONS)]

    for idx, compiled in patterns:
        m = compiled.search(text)
        if m:
            hits.append((m.start(), idx))

    # Sort by position in document
    hits.sort(key=lambda x: x[0])

    sections: list[tuple[str, str, str, list[str]]] = []
    for i, (pos, sec_idx) in enumerate(hits):
        key, _, topic, tags = UNDOC_SECTIONS[sec_idx]
        end_pos = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[pos:end_pos].strip()
        sections.append((key, topic, body, tags))

    return sections


def write_undocumented_sections(pdf_path: Path, out_dir: Path) -> int:
    """Extract the PDF and write one markdown file per section. Returns count written."""
    if not pdf_path.exists():
        log.warning("PDF not found at %s — skipping undocumented Z80 section", pdf_path)
        return 0

    log.info("Extracting text from %s", pdf_path)
    text = extract_pdf_text(pdf_path)

    if text is None:
        # Fallback: write a single stub file so the category still exists in the corpus
        log.info("Writing stub undocumented file (pdftotext unavailable)")
        stub_body = (
            "pdftotext was not available when this file was generated.\n\n"
            "Re-run `fetch_z80_extras.py --pdf /tmp/z80-undocumented.pdf` after installing\n"
            "poppler-utils (`brew install poppler` on macOS) to populate this section.\n\n"
            f"Source: {PDF_SOURCE_URL}\n"
        )
        fm = frontmatter(
            topic="Undocumented Z80 Opcodes (stub)",
            category="z80-undocumented",
            source_url=PDF_SOURCE_URL,
            licence="GFDL-1.1",
            tags=["z80", "undocumented", "opcodes"],
        )
        write_raw(out_dir / "undocumented_stub.md", fm + "\n" + stub_body)
        return 1

    sections = split_pdf_into_sections(text)
    if not sections:
        log.warning("No sections found in PDF text — writing full text as single file")
        fm = frontmatter(
            topic="The Undocumented Z80 Documented — Full Text",
            category="z80-undocumented",
            source_url=PDF_SOURCE_URL,
            licence="GFDL-1.1",
            tags=["z80", "undocumented", "opcodes"],
        )
        write_raw(out_dir / "undocumented_full.md", fm + "\n" + text[:50000])
        return 1

    written = 0
    for key, topic, body, tags in sections:
        fm = frontmatter(
            topic=topic,
            category="z80-undocumented",
            source_url=PDF_SOURCE_URL,
            licence="GFDL-1.1",
            tags=tags,
        )
        filename = slug(key) + ".md"
        write_raw(out_dir / filename, fm + "\n" + body)
        written += 1

    log.info("Undocumented Z80: wrote %d section files to %s", written, out_dir)
    return written


# ---------------------------------------------------------------------------
# Source 2: utz82/ZX-Spectrum-1-Bit-Routines (beeper engines)
# ---------------------------------------------------------------------------

def fetch_repo_readme() -> str:
    """Fetch the root README for engine context."""
    log.info("Fetching repo README from %s", GITHUB_README)
    text = get(GITHUB_README)
    return text or "_README not available._\n"


def list_repo_engines() -> list[str]:
    """
    Call GitHub API to list repo root contents.
    Returns engine directory names (non-hidden dirs, excluding docs etc.).
    """
    log.info("Listing repo contents: %s", GITHUB_API)
    data = get_json(GITHUB_API)
    if not data or not isinstance(data, list):
        log.error("Could not list repo contents — will use priority list only")
        return []

    dirs = [
        item["name"]
        for item in data
        if item.get("type") == "dir" and not item["name"].startswith(".")
    ]
    log.info("Found %d engine directories", len(dirs))
    return dirs


def _primary_asm_candidates(engine: str) -> list[str]:
    """Return ordered list of raw URLs to try for the primary ASM file."""
    candidates = [
        f"{GITHUB_RAW}{engine}/main.asm",
        f"{GITHUB_RAW}{engine}/{engine}.asm",
        f"{GITHUB_RAW}{engine}/player.asm",
    ]
    return candidates


def fetch_engine_asm(engine: str) -> tuple[str, str] | None:
    """
    Fetch the primary ASM file for an engine.

    Returns (filename, source_text) or None if nothing found.
    """
    for url in _primary_asm_candidates(engine):
        text = get(url)
        if text is not None:
            filename = url.split("/")[-1]
            log.info("  %s: fetched %s (%d bytes)", engine, filename, len(text))
            return filename, text
        time.sleep(0.5)

    log.warning("  %s: no ASM file found (tried %d candidates)", engine, len(_primary_asm_candidates(engine)))
    return None


def extract_register_usage(asm_text: str) -> str:
    """
    Pull the comment block at the top of an ASM file that describes register usage.

    Most utz82 engines have a header like:
        ; REGISTERS: DE=period, HL=duration, ...
    or a block of '; ---' bordered comments.
    """
    lines = asm_text.splitlines()
    comment_lines: list[str] = []
    for line in lines[:60]:  # only look at the top
        stripped = line.strip()
        if stripped.startswith(";"):
            comment_lines.append(stripped)
        elif stripped == "" and comment_lines:
            # Allow one blank line inside the header block
            continue
        elif comment_lines:
            # Non-comment, non-blank after we've collected some — stop
            break

    return "\n".join(comment_lines) if comment_lines else "_No register usage comments found._"


def write_engine_file(engine: str, asm_filename: str, asm_text: str,
                      readme_snippet: str, out_dir: Path) -> Path:
    """Write a single beeper engine markdown file."""
    fm = frontmatter(
        topic=f"{engine} — ZX Spectrum 1-Bit Beeper Engine",
        category="z80-beeper",
        source_url=f"https://github.com/utz82/ZX-Spectrum-1-Bit-Routines/tree/master/{engine}",
        licence="MIT",
        tags=["z80", "beeper", "1-bit", "sound", slug(engine.lower())],
    )

    reg_usage = extract_register_usage(asm_text)

    body = (
        f"# {engine}\n\n"
        f"{readme_snippet}\n\n"
        "## Register Usage\n\n"
        f"{reg_usage}\n\n"
        f"## Source: `{asm_filename}`\n\n"
        "```asm\n"
        + asm_text
        + "\n```\n"
    )

    out_path = out_dir / (slug(engine.lower()) + ".md")
    write_raw(out_path, fm + "\n" + body)
    return out_path


def fetch_beeper_engines(out_dir: Path) -> int:
    """Fetch all beeper engines and write markdown files. Returns count written."""
    readme = fetch_repo_readme()
    time.sleep(1)

    # Merge priority engines + discovered dirs, priority first, no duplicates
    discovered = list_repo_engines()
    time.sleep(1)

    engine_order: list[str] = []
    seen: set[str] = set()
    for e in PRIORITY_ENGINES + discovered:
        if e not in seen:
            engine_order.append(e)
            seen.add(e)

    if not engine_order:
        log.error("No engines to fetch")
        return 0

    # Extract a brief per-engine description from the README (best effort)
    # README typically has a table or bullet list: "| engine | description |"
    engine_descriptions: dict[str, str] = {}
    for line in readme.splitlines():
        for engine in engine_order:
            if engine.lower() in line.lower():
                # Strip markdown table pipes, clean up
                desc = re.sub(r"\|", "", line).strip()
                desc = re.sub(r"\s+", " ", desc)
                if len(desc) > 5 and engine not in engine_descriptions:
                    engine_descriptions[engine] = desc

    written = 0
    for engine in engine_order:
        log.info("Fetching engine: %s", engine)
        result = fetch_engine_asm(engine)
        time.sleep(1)

        if result is None:
            log.warning("Skipping %s — no ASM found", engine)
            continue

        asm_filename, asm_text = result
        readme_snippet = engine_descriptions.get(engine, f"Beeper engine: {engine}")

        try:
            write_engine_file(engine, asm_filename, asm_text, readme_snippet, out_dir)
            written += 1
        except Exception as exc:
            log.error("Failed to write %s: %s", engine, exc)

    log.info("Beeper engines: wrote %d files to %s", written, out_dir)
    return written


# ---------------------------------------------------------------------------
# Source 3: breakintoprogram/lib-spectrum (ZX Spectrum library routines)
# ---------------------------------------------------------------------------

def _detect_lib_licence() -> str:
    """Fetch the LICENSE file and return a short SPDX-ish label."""
    log.info("Checking lib-spectrum licence: %s", LIB_SPECTRUM_LICENSE)
    text = get(LIB_SPECTRUM_LICENSE)
    if text is None:
        return "MIT"  # most likely
    text_lower = text.lower()
    if "mit license" in text_lower or "mit licence" in text_lower:
        return "MIT"
    if "apache" in text_lower:
        return "Apache-2.0"
    if "gpl" in text_lower:
        return "GPL"
    # Return the first non-blank line as a fallback label
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:60]
    return "unknown"


def _extract_topic_from_z80(asm_text: str, filename: str) -> str:
    """
    Pull a topic string from the leading comment block.

    Most Dean Belfield files have:
        ; Title:       Some Routine
    or similar. Fall back to the filename stem.
    """
    for line in asm_text.splitlines()[:30]:
        m = re.match(r";\s*[Tt]itle\s*:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    # Fallback: prettify the filename stem
    stem = Path(filename).stem.replace("_", " ").title()
    return f"{stem} — ZX Spectrum Library"


def fetch_lib_spectrum(out_dir: Path) -> int:
    """Fetch lib-spectrum Z80 files and write markdown. Returns count written."""
    licence = _detect_lib_licence()
    time.sleep(1)

    readme_text = get(LIB_SPECTRUM_README) or "_README not available._\n"
    time.sleep(1)

    written = 0
    for filename in LIB_SPECTRUM_FILES:
        url = LIB_SPECTRUM_RAW + filename
        log.info("Fetching lib-spectrum: %s", filename)
        asm_text = get(url)
        time.sleep(1)

        if asm_text is None:
            log.warning("Could not fetch %s — skipping", filename)
            continue

        stem = Path(filename).stem
        routine_tag = _LIB_TAGS.get(stem, slug(stem))
        topic = _extract_topic_from_z80(asm_text, filename)

        fm = frontmatter(
            topic=topic,
            category="z80-library",
            source_url=f"{LIB_SPECTRUM_REPO}/blob/master/lib/{filename}",
            licence=licence,
            tags=["z80", "spectrum", "library", routine_tag],
        )

        # Brief README context: find any line mentioning this file
        readme_line = ""
        for line in readme_text.splitlines():
            if stem.lower() in line.lower():
                readme_line = line.strip()
                break

        body = (
            f"# {topic}\n\n"
            + (f"{readme_line}\n\n" if readme_line else "")
            + f"## Source: `{filename}`\n\n"
            "```asm\n"
            + asm_text
            + "\n```\n"
        )

        out_path = out_dir / (slug(stem) + ".md")
        try:
            write_raw(out_path, fm + "\n" + body)
            written += 1
        except Exception as exc:
            log.error("Failed to write %s: %s", filename, exc)

    log.info("lib-spectrum: wrote %d files to %s", written, out_dir)
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch undocumented Z80 data and ZX Spectrum beeper engines."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help=f"Path to the undocumented Z80 PDF (default: {DEFAULT_PDF_PATH})",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip the PDF extraction step",
    )
    parser.add_argument(
        "--skip-beeper",
        action="store_true",
        help="Skip the beeper engines step",
    )
    parser.add_argument(
        "--skip-lib",
        action="store_true",
        help="Skip the lib-spectrum step",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    total = 0

    if not args.skip_pdf:
        UNDOCUMENTED_DIR.mkdir(parents=True, exist_ok=True)
        count = write_undocumented_sections(args.pdf, UNDOCUMENTED_DIR)
        log.info("Undocumented Z80: %d file(s) written", count)
        total += count

    if not args.skip_beeper:
        BEEPER_DIR.mkdir(parents=True, exist_ok=True)
        count = fetch_beeper_engines(BEEPER_DIR)
        log.info("Beeper engines: %d file(s) written", count)
        total += count

    if not args.skip_lib:
        LIB_DIR.mkdir(parents=True, exist_ok=True)
        count = fetch_lib_spectrum(LIB_DIR)
        log.info("lib-spectrum: %d file(s) written", count)
        total += count

    if total == 0:
        log.error("Nothing was written — check errors above")
        sys.exit(1)

    log.info("Done. Total files written: %d", total)
