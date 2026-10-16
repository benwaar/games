"""
Fetch ZX Spectrum 48K hardware and ROM reference data from public sources.

deps: requests beautifulsoup4

Sources:
  - https://skoolkid.github.io/rom/  (GPL — Skoolkit ROM disassembly)
  - https://worldofspectrum.org/faq/reference/z80reference.htm  (community archive)

Output:
  knowledge/raw/spectrum/rom/       — ROM routines (one .md per entry point)
  knowledge/raw/spectrum/hardware/  — system variables, ULA timing, I/O ports
"""

import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_ROM = REPO_ROOT / "knowledge" / "raw" / "spectrum" / "rom"
OUT_HW = REPO_ROOT / "knowledge" / "raw" / "spectrum" / "hardware"

SKOOLKID_BASE = "https://skoolkid.github.io/rom"
SKOOLKID_ASM_INDEX = f"{SKOOLKID_BASE}/asm/0000.html"  # first page of ROM listing
SKOOLKID_MAP = f"{SKOOLKID_BASE}/maps/all.html"         # entry-point map

WOS_REF = "https://worldofspectrum.org/faq/reference/z80reference.htm"

HEADERS = {"User-Agent": "Mozilla/5.0 (knowledge-corpus-fetcher/1.0; educational use)"}
DELAY = 0.4  # seconds between requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get(url: str) -> BeautifulSoup | None:
    """GET a URL and return a BeautifulSoup, or None on error."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [skip] {url} — {e}", file=sys.stderr)
        return None


def slug(name: str) -> str:
    """Turn a routine name into a safe filename slug."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name).strip("-")


def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def frontmatter(topic: str, category: str, source_url: str, tags: list[str], licence: str = "gpl-3.0") -> str:
    tag_str = ", ".join(tags)
    return (
        "---\n"
        f'topic: "{topic}"\n'
        f"category: {category}\n"
        f'source_url: "{source_url}"\n'
        f"licence: {licence}\n"
        f"tags: [{tag_str}]\n"
        "---\n\n"
    )


# ---------------------------------------------------------------------------
# Skoolkid ROM — entry-point map
# ---------------------------------------------------------------------------


def fetch_rom_entry_map() -> list[dict]:
    """
    Parse the Skoolkid 'all routines' map page.
    Returns list of {name, addr, url, description}.
    """
    print(f"Fetching ROM map: {SKOOLKID_MAP}")
    soup = get(SKOOLKID_MAP)
    if soup is None:
        return []

    entries = []
    # The map page has a table with columns: address, label, description
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        addr_cell = cells[0].get_text(strip=True)
        label_cell = cells[1]
        desc_cell = cells[2]
        name = label_cell.get_text(strip=True)
        desc = desc_cell.get_text(strip=True)
        link = label_cell.find("a")
        href = link["href"] if link and link.get("href") else None
        if not name or not addr_cell:
            continue
        url = f"{SKOOLKID_BASE}/asm/{href}" if href else None
        entries.append({"name": name, "addr": addr_cell, "url": url, "desc": desc})

    print(f"  Found {len(entries)} map entries")
    return entries


def fetch_rom_routine(entry: dict) -> str | None:
    """
    Fetch the detail page for one ROM routine and return markdown body, or None.
    """
    if not entry.get("url"):
        return None
    soup = get(entry["url"])
    if soup is None:
        return None

    # Extract the main comment block and instruction table
    comments: list[str] = []
    instructions: list[str] = []

    # Skoolkid pages: <div class="comment"> paragraphs + <table class="disassembly">
    for p in soup.select("div.comment p, div.description p"):
        txt = p.get_text(" ", strip=True)
        if txt:
            comments.append(txt)

    for row in soup.select("table.disassembly tr"):
        cells = row.find_all("td")
        if len(cells) >= 3:
            addr = cells[0].get_text(strip=True)
            mnemonic = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            comment = cells[-1].get_text(strip=True) if len(cells) > 2 else ""
            if addr:
                line = f"  {addr}  {mnemonic}"
                if comment:
                    line += f"  ; {comment}"
                instructions.append(line)

    # Collect "Related" links
    related: list[str] = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if href.endswith(".html") and a.get_text(strip=True) not in (entry["name"], ""):
            related.append(a.get_text(strip=True))
    related = list(dict.fromkeys(related))[:10]  # dedupe, cap at 10

    body = "\n".join(comments) if comments else entry.get("desc", "")
    asm_block = ""
    if instructions:
        asm_block = "\n## Assembly\n\n```asm\n" + "\n".join(instructions[:40]) + "\n```\n"
        if len(instructions) > 40:
            asm_block += f"\n_(truncated — {len(instructions)} lines total)_\n"

    related_block = ""
    if related:
        related_block = "\n## Related routines\n\n" + "".join(f"- {r}\n" for r in related)

    return body + asm_block + related_block


def build_rom_routine_md(entry: dict, body: str) -> str:
    name = entry["name"]
    addr = entry["addr"]
    desc = entry.get("desc", "")
    url = entry.get("url") or SKOOLKID_MAP

    fm = frontmatter(name, "spectrum-rom", url, ["rom", "routine", slug(name).lower()])
    header = f"# {name}\n\n**Address:** `{addr}`\n\n"
    if desc:
        header += f"**Description:** {desc}\n\n"
    return fm + header + body


def fetch_all_rom_routines(entries: list[dict]) -> None:
    print(f"\nFetching {len(entries)} ROM routines → {OUT_ROM}/")
    written = 0
    for i, entry in enumerate(entries):
        name = entry["name"]
        if not name:
            continue
        filename = OUT_ROM / f"{slug(name)}.md"
        # Skip if already fetched in this run
        if filename.exists():
            continue
        print(f"  [{i+1}/{len(entries)}] {name} ({entry['addr']})")
        body = fetch_rom_routine(entry)
        if body is None:
            # Fallback: just the map description
            body = entry.get("desc", "")
        md = build_rom_routine_md(entry, body)
        write_md(filename, md)
        written += 1
        time.sleep(DELAY)
    print(f"  Done — {written} files written")


# ---------------------------------------------------------------------------
# World of Spectrum — system variables + hardware
# ---------------------------------------------------------------------------

# Known section anchors / headings on the WoS reference page
HW_SECTIONS = {
    "system-variables": ("System Variables", "spectrum-hardware", ["sysvar", "memory-map"]),
    "io-ports": ("I/O Ports", "spectrum-hardware", ["io", "ports", "ula"]),
    "ula": ("ULA", "spectrum-hardware", ["ula", "display", "timing"]),
}


def fetch_wos_reference() -> None:
    """
    Fetch WoS reference page and split into per-category markdown files.
    """
    print(f"\nFetching WoS reference: {WOS_REF}")
    soup = get(WOS_REF)
    if soup is None:
        print("  [skip] WoS reference unavailable — skipping hardware section")
        return

    body_text = soup.get_text("\n", strip=False)
    # Save the full page as a single hardware reference
    full_path = OUT_HW / "wos-reference-full.md"
    fm = frontmatter(
        "World of Spectrum — Z80/Spectrum Reference",
        "spectrum-hardware",
        WOS_REF,
        ["system-variables", "io-ports", "ula", "hardware"],
        licence="community-archive",
    )
    write_md(full_path, fm + body_text)
    print(f"  Wrote {full_path} ({len(body_text)} chars)")

    # Attempt to split by heading into separate files
    _split_wos_by_heading(soup)


def _split_wos_by_heading(soup: BeautifulSoup) -> None:
    """
    Walk the WoS page headings and emit one file per major section.
    """
    sections: list[tuple[str, str]] = []  # (heading_text, content_html)
    current_heading = "Introduction"
    current_parts: list[str] = []

    for el in soup.body.children if soup.body else []:
        tag = getattr(el, "name", None)
        if tag in ("h1", "h2", "h3"):
            if current_parts:
                sections.append((current_heading, "\n".join(current_parts)))
            current_heading = el.get_text(strip=True)
            current_parts = []
        else:
            txt = el.get_text("\n", strip=True) if hasattr(el, "get_text") else str(el).strip()
            if txt:
                current_parts.append(txt)

    if current_parts:
        sections.append((current_heading, "\n".join(current_parts)))

    print(f"  Splitting into {len(sections)} sections")
    for heading, content in sections:
        if not content.strip():
            continue
        fname = slug(heading) or "section"
        path = OUT_HW / f"{fname}.md"

        # Guess tags from heading text
        heading_lower = heading.lower()
        tags = ["hardware"]
        if "variable" in heading_lower:
            tags += ["sysvar", "memory-map"]
        if "port" in heading_lower or "i/o" in heading_lower:
            tags += ["io", "ports"]
        if "ula" in heading_lower or "display" in heading_lower:
            tags += ["ula", "display"]
        if "timing" in heading_lower:
            tags += ["timing"]

        fm = frontmatter(heading, "spectrum-hardware", WOS_REF, tags, licence="community-archive")
        write_md(path, fm + f"# {heading}\n\n" + content)
        print(f"    → {path.name}")


# ---------------------------------------------------------------------------
# Skoolkid HTML — supplementary hardware pages
# ---------------------------------------------------------------------------

SKOOLKID_HW_PAGES = [
    ("display-file", f"{SKOOLKID_BASE}/reference/display.html", "ULA display file layout"),
    ("system-variables", f"{SKOOLKID_BASE}/reference/sysvars.html", "System variables"),
    ("io-ports", f"{SKOOLKID_BASE}/reference/io.html", "I/O ports"),
]


def fetch_skoolkid_hw() -> None:
    print(f"\nFetching Skoolkid hardware reference pages → {OUT_HW}/")
    for name, url, desc in SKOOLKID_HW_PAGES:
        print(f"  {name}: {url}")
        soup = get(url)
        if soup is None:
            continue
        content = soup.get_text("\n", strip=False)
        tags = ["hardware", name]
        fm = frontmatter(desc, "spectrum-hardware", url, tags)
        path = OUT_HW / f"skoolkid-{name}.md"
        write_md(path, fm + f"# {desc}\n\nSource: {url}\n\n" + content)
        print(f"    → {path.name} ({len(content)} chars)")
        time.sleep(DELAY)


# ---------------------------------------------------------------------------
# Skoolkid — entry points only (fast fallback index)
# ---------------------------------------------------------------------------

KNOWN_ENTRY_POINTS = [
    ("0000", "MAIN-EXEC", "Main execution loop entry"),
    ("0008", "ERROR-1", "Error restart"),
    ("0010", "PRINT-A", "Print character in A"),
    ("0018", "GET-CHAR", "Get next BASIC character"),
    ("0020", "NEXT-CHAR", "Get next character from BASIC line"),
    ("0028", "FP-CALC", "Floating-point calculator"),
    ("0030", "BC-SPACES", "Make BC spaces in workspace"),
    ("0038", "MASK-INT", "Maskable interrupt routine"),
    ("003B", "KEYBOARD", "Keyboard scanning"),
    ("0048", "TEMPS", "Set temporary colours"),
    ("0053", "ATTR-BITS", "Collect attribute bits"),
    ("0077", "BORDER", "Set border colour"),
    ("007D", "BEEPER", "Produce sound effect"),
    ("00E2", "CL-CHAN", "Open a channel"),
    ("1601", "CLS", "Clear screen"),
    ("1605", "CLS-LOWER", "Clear lower screen"),
    ("1615", "CHAN-OPEN", "Open channel A in register"),
    ("1617", "CHAN-OPEN2", "Open channel (main)"),
    ("229B", "PRINT-OUT", "Output character to current stream"),
    ("2AB1", "SA-BYTES", "Save bytes to tape"),
    ("0556", "LD-BYTES", "Load bytes from tape"),
    ("028E", "MAKE-ROOM", "Make room in BASIC line"),
    ("1795", "LOOK-VARS", "Look up variable in VARS area"),
    ("2BF1", "COPY", "Copy screen to printer"),
    ("0D6B", "DISPLAY-6", "Display routine (ULA)"),
    ("1B17", "PLOT", "Plot pixel at (X,Y)"),
    ("1B5B", "DRAW", "Draw line"),
    ("1EAC", "CIRCLE", "Draw circle"),
    ("2211", "PRINT-FP", "Print floating-point number"),
]


def write_known_entry_points() -> None:
    """
    Write stub markdown for well-known ROM entry points as a fast fallback.
    Only writes files that don't already exist (live fetch takes precedence).
    """
    print(f"\nWriting {len(KNOWN_ENTRY_POINTS)} known ROM entry-point stubs → {OUT_ROM}/")
    for addr, name, desc in KNOWN_ENTRY_POINTS:
        path = OUT_ROM / f"{slug(name)}.md"
        if path.exists():
            continue  # live-fetched version already written
        url = f"{SKOOLKID_BASE}/asm/{addr.lower()}.html"
        fm = frontmatter(name, "spectrum-rom", url, ["rom", "entry-point", name.lower()])
        content = (
            fm
            + f"# {name}\n\n"
            + f"**Address:** `{addr}H`\n\n"
            + f"**Description:** {desc}\n\n"
            + "## Calling convention\n\n"
            + "_See source URL for full disassembly and register usage._\n\n"
            + "## Related routines\n\n"
            + "_See Skoolkid ROM disassembly for cross-references._\n"
        )
        write_md(path, content)
    print(f"  Done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== fetch_spectrum.py ===")
    print(f"Output: {OUT_ROM} | {OUT_HW}")
    OUT_ROM.mkdir(parents=True, exist_ok=True)
    OUT_HW.mkdir(parents=True, exist_ok=True)

    # 1. Known entry-point stubs (instant, no network) — baseline always written
    write_known_entry_points()

    # 2. Live ROM routine map + detail pages from Skoolkid
    entries = fetch_rom_entry_map()
    if entries:
        # Limit to first 100 to avoid hammering the server
        fetch_all_rom_routines(entries[:100])
    else:
        print("  [warn] ROM map unavailable — stubs only")

    # 3. Hardware reference from World of Spectrum
    fetch_wos_reference()

    # 4. Supplementary hardware pages from Skoolkid
    fetch_skoolkid_hw()

    print("\n=== Done ===")
    rom_files = list(OUT_ROM.glob("*.md"))
    hw_files = list(OUT_HW.glob("*.md"))
    print(f"  ROM files:      {len(rom_files)}")
    print(f"  Hardware files: {len(hw_files)}")


if __name__ == "__main__":
    main()
