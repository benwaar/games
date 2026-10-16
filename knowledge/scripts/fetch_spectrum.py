"""
Fetch ZX Spectrum 48K hardware and ROM reference data from public sources.

deps: requests beautifulsoup4

Sources:
  - https://skoolkid.github.io/rom/  (GPL — Skoolkit ROM disassembly)

Output:
  knowledge/raw/spectrum/rom/       — ROM routines (one .md per entry point)
  knowledge/raw/spectrum/hardware/  — system variables, ULA timing, I/O ports

Changes from v1:
  - Fixed map selector: `td.map-c/map-s/map-u` cells, `a.map-entry-title` for labels
  - Fixed URL resolution: urllib.parse.urljoin instead of string concat
  - Fixed detail page parsing: `div.paragraph`, correct assembly cell classes
  - Fixed hardware pages: sysvars at buffers/sysvars.html, I/O ports curated
  - Fixed KNOWN_ENTRY_POINTS: corrected wrong addresses (BEEPER at 03B5 not 007D)
"""

import re
import sys
import time
import urllib.parse
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
SKOOLKID_ROUTINES_MAP = f"{SKOOLKID_BASE}/maps/routines.html"
SKOOLKID_SYSVARS = f"{SKOOLKID_BASE}/buffers/sysvars.html"

HEADERS = {"User-Agent": "Mozilla/5.0 (knowledge-corpus-fetcher/1.0; educational use)"}
DELAY = 0.5  # seconds between requests

# Routines to fetch in detail (curated — covers all projects: beeper, tape, display, memory)
PRIORITY_ROUTINES = {
    "03B5",  # BEEPER subroutine
    "03F8",  # BEEP command
    "028E",  # KEYBOARD SCANNING
    "02BF",  # KEYBOARD
    "031E",  # K-TEST
    "0333",  # KEYBOARD DECODING
    "04C2",  # SA-BYTES (save to tape)
    "053F",  # SA/LD-RET
    "0556",  # LD-BYTES (load from tape)
    "05E3",  # LD-EDGE-2/LD-EDGE-1
    "0605",  # SAVE/LOAD/VERIFY/MERGE
    "0D6B",  # DISPLAY routines
    "1601",  # CLS
    "1605",  # CLS-LOWER
    "1B17",  # PLOT
    "1B5B",  # DRAW
    "1EAC",  # CIRCLE
    "229B",  # PRINT-OUT
    "0010",  # PRINT-A CHARACTER restart
    "0000",  # START
    "0038",  # MASKABLE INTERRUPT
    "0030",  # MAKE-BC-SPACES
    "0556",  # LD-BYTES
    "2AB1",  # SA-BYTES alias
    "1795",  # LOOK-VARS
    "11B7",  # NEW
    "12AC",  # MAIN-EXEC
}


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


def _clean_text(el) -> str:
    """Extract text from a BS4 element, collapsing whitespace."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


# ---------------------------------------------------------------------------
# Skoolkid ROM — routines map
# ---------------------------------------------------------------------------


def fetch_routines_map() -> list[dict]:
    """
    Parse the Skoolkid routines map page.
    Returns list of {name, addr, url, desc}.

    HTML structure:
      <table class="map">
        <tr>
          <td class="map-c"><a href="../asm/0000.html">0000</a></td>
          <td class="map-c-desc">
            <div class="map-entry-title-10">
              <a class="map-entry-title" href="../asm/0000.html">THE 'START'</a>
            </div>
          </td>
        </tr>
      """
    print(f"Fetching routines map: {SKOOLKID_ROUTINES_MAP}")
    soup = get(SKOOLKID_ROUTINES_MAP)
    if soup is None:
        return []

    entries = []
    for row in soup.select("table.map tr"):
        # Address cell can be map-c, map-s, map-u, map-g etc.
        addr_cell = row.select_one("td[class]")
        if addr_cell is None:
            continue
        addr_link = addr_cell.select_one("a[href]")
        if addr_link is None:
            continue

        # Label is in the sibling desc cell
        desc_cell = row.select_one("a.map-entry-title")
        if desc_cell is None:
            continue

        addr = addr_link.get_text(strip=True)
        name = desc_cell.get_text(strip=True)
        href = desc_cell.get("href", "")
        # Resolve relative URL against the map page URL
        url = urllib.parse.urljoin(SKOOLKID_ROUTINES_MAP, href) if href else None

        if addr and name:
            entries.append({"name": name, "addr": addr, "url": url})

    print(f"  Found {len(entries)} routines")
    return entries


# ---------------------------------------------------------------------------
# Skoolkid ROM — routine detail pages
# ---------------------------------------------------------------------------


def fetch_routine_detail(entry: dict) -> str | None:
    """
    Fetch a ROM routine detail page and return markdown body, or None.

    HTML structure:
      <div class="description">03B5: THE 'BEEPER' SUBROUTINE</div>
      <table class="disassembly">
        <td class="routine-comment"><div class="details"><div class="paragraph">...</div></div></td>
        <tr>
          <td class="asm-label">BEEPER</td>
          <td class="address-2">03B5</td>
          <td class="instruction">DI</td>
          <td class="comment-1">Disable the interrupt...</td>
        </tr>
      </table>
      <table class="input"><tr><td class="register">DE</td><td>...</td></tr></table>
      <table class="output">...</table>
    """
    url = entry.get("url")
    if not url:
        return None
    soup = get(url)
    if soup is None:
        return None

    # Description heading
    desc_div = soup.select_one("div.description")
    page_title = _clean_text(desc_div) if desc_div else entry.get("name", "")

    # Comment paragraphs
    paragraphs: list[str] = []
    for p in soup.select("td.routine-comment div.details div.paragraph"):
        txt = _clean_text(p)
        if txt:
            paragraphs.append(txt)

    # Input/output register tables
    reg_sections: list[str] = []
    for table_cls, heading in [("input", "Input"), ("output", "Output")]:
        table = soup.select_one(f"table.{table_cls}")
        if table is None:
            continue
        rows = []
        for row in table.select("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                reg = _clean_text(cells[0])
                desc = _clean_text(cells[1])
                if reg and desc:
                    rows.append(f"  {reg}: {desc}")
        if rows:
            reg_sections.append(f"\n## {heading}\n\n" + "\n".join(rows))

    # Assembly listing
    instructions: list[str] = []
    for row in soup.select("table.disassembly tr"):
        label_cell = row.select_one("td.asm-label")
        addr_cell = row.select_one("td[class^='address-']")
        instr_cell = row.select_one("td.instruction")
        comment_cell = row.select_one("td[class^='comment-']")

        if addr_cell is None and instr_cell is None:
            continue

        label = _clean_text(label_cell) if label_cell else ""
        addr = _clean_text(addr_cell) if addr_cell else ""
        instr = _clean_text(instr_cell) if instr_cell else ""
        comment = _clean_text(comment_cell) if comment_cell else ""

        if not addr and not instr:
            continue

        line = f"{addr:<6}  {instr:<20}"
        if label:
            line = f"{label:<12} " + line
        if comment:
            line += f"  ; {comment}"
        instructions.append(line.rstrip())

    # Build body
    body = "\n\n".join(paragraphs) if paragraphs else ""
    if reg_sections:
        body += "\n" + "\n".join(reg_sections)
    if instructions:
        shown = instructions[:60]
        asm = "\n## Assembly\n\n```asm\n" + "\n".join(shown) + "\n```\n"
        if len(instructions) > 60:
            asm += f"\n_(truncated — {len(instructions)} instructions total)_\n"
        body += asm

    return body.strip() if body.strip() else None


def build_rom_routine_md(entry: dict, body: str) -> str:
    name = entry["name"]
    addr = entry["addr"]
    url = entry.get("url") or SKOOLKID_ROUTINES_MAP

    # Derive a short label from the full name (strip "THE '...' SUBROUTINE" boilerplate)
    short = re.sub(r"^THE '(.+?)'\s*(SUBROUTINE|ROUTINE|RESTART|COMMAND.*)?$", r"\1", name, flags=re.IGNORECASE).strip()

    fm = frontmatter(short or name, "spectrum-rom", url, ["rom", "routine", slug(short or name).lower()])
    header = f"# {short or name}\n\n**Address:** `{addr}H`\n\n**Full title:** {name}\n\n"
    return fm + header + body


def fetch_selected_routines(entries: list[dict]) -> None:
    """Fetch detail pages for priority routines (skip already-fetched)."""
    priority = {e["addr"] for e in entries if e["addr"] in PRIORITY_ROUTINES}
    to_fetch = [e for e in entries if e["addr"] in PRIORITY_ROUTINES]

    print(f"\nFetching {len(to_fetch)} priority ROM routines → {OUT_ROM}/")
    written = 0
    for i, entry in enumerate(to_fetch):
        # Derive a safe label for the file
        short = re.sub(r"^THE '(.+?)'\s*(SUBROUTINE|ROUTINE|RESTART|COMMAND.*)?$", r"\1", entry["name"], flags=re.IGNORECASE).strip()
        label = short or entry["name"]
        filename = OUT_ROM / f"{slug(label)}.md"

        print(f"  [{i+1}/{len(to_fetch)}] {label} ({entry['addr']})")
        body = fetch_routine_detail(entry)
        if not body:
            body = f"_See {entry.get('url', SKOOLKID_ROUTINES_MAP)} for full disassembly._\n"
        md = build_rom_routine_md(entry, body)
        write_md(filename, md)
        written += 1
        time.sleep(DELAY)

    print(f"  Done — {written} files written")


# ---------------------------------------------------------------------------
# Fallback stubs for non-priority routines
# ---------------------------------------------------------------------------

def write_stub(entry: dict) -> None:
    """Write a stub .md for a routine, only if no file exists yet."""
    name = entry["name"]
    short = re.sub(r"^THE '(.+?)'\s*(SUBROUTINE|ROUTINE|RESTART|COMMAND.*)?$", r"\1", name, flags=re.IGNORECASE).strip()
    label = short or name
    filename = OUT_ROM / f"{slug(label)}.md"
    if filename.exists():
        return
    url = entry.get("url") or SKOOLKID_ROUTINES_MAP
    fm = frontmatter(label, "spectrum-rom", url, ["rom", "entry-point", slug(label).lower()])
    content = (
        fm
        + f"# {label}\n\n"
        + f"**Address:** `{entry['addr']}H`\n\n"
        + f"**Full title:** {name}\n\n"
        + "_See source URL for full disassembly and register usage._\n"
    )
    write_md(filename, content)


# ---------------------------------------------------------------------------
# System variables from buffers/sysvars.html
# ---------------------------------------------------------------------------


def fetch_system_variables() -> None:
    """
    Fetch the system variables index and write a single comprehensive chunk.

    HTML structure (same as routines map but with 3 columns):
      <td class="map-g"><a href="../asm/5C00.html">5C00</a></td>
      <td class="map-length">8</td>
      <td class="map-g-desc">
        <div class="map-entry-title-10">
          <a class="map-entry-title" href="../asm/5C00.html">KSTATE - Used in reading the keyboard</a>
        </div>
      </td>
    """
    print(f"\nFetching system variables: {SKOOLKID_SYSVARS}")
    soup = get(SKOOLKID_SYSVARS)
    if soup is None:
        print("  [skip] sysvars unavailable")
        return

    rows_md: list[str] = []
    for row in soup.select("table.map tr"):
        addr_cell = row.select_one("td[class]")
        if addr_cell is None:
            continue
        addr_link = addr_cell.select_one("a[href]")
        if addr_link is None:
            continue

        length_cell = row.select_one("td.map-length")
        label_link = row.select_one("a.map-entry-title")
        if label_link is None:
            continue

        addr = _clean_text(addr_link)
        length = _clean_text(length_cell) if length_cell else "?"
        full_label = _clean_text(label_link)

        # Split "KSTATE - Used in reading the keyboard" into name + desc
        if " - " in full_label:
            var_name, _, var_desc = full_label.partition(" - ")
        elif ": " in full_label:
            var_name, _, var_desc = full_label.partition(": ")
        else:
            var_name = full_label
            var_desc = ""

        url = urllib.parse.urljoin(SKOOLKID_SYSVARS, label_link.get("href", ""))

        rows_md.append(f"| `{addr}H` | {length} | `{var_name.strip()}` | {var_desc.strip()} |")

    if not rows_md:
        print("  [warn] no system variables parsed")
        return

    print(f"  Parsed {len(rows_md)} system variables")

    table_header = (
        "| Address | Bytes | Name | Description |\n"
        "|---------|-------|------|-------------|\n"
    )
    table = table_header + "\n".join(rows_md)

    intro = (
        "System variables are stored in RAM at 5C00H–5CB5H. "
        "They are referenced by the ROM via the IY register (IY=5C3AH). "
        "Each variable is accessible by name or by offset from IY.\n\n"
        "The ROM initialises system variables on `NEW` and `RESET`. "
        "User programs can read and write them directly to control "
        "border colour, cursor position, streams, and more.\n\n"
    )

    fm = frontmatter(
        "ZX Spectrum System Variables",
        "spectrum-hardware",
        SKOOLKID_SYSVARS,
        ["sysvar", "memory-map", "hardware", "spectrum"],
    )

    content = fm + "# ZX Spectrum System Variables\n\n" + intro + table + "\n"
    path = OUT_HW / "system-variables.md"
    write_md(path, content)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Curated I/O ports and ULA hardware reference
# ---------------------------------------------------------------------------

_IO_PORTS_CONTENT = """\
# ZX Spectrum I/O Ports and ULA

The ULA (Uncommitted Logic Array) handles the Spectrum's display,
keyboard, tape, and speaker via a single I/O port.

## Port FEH — ULA port

The primary hardware interface. Used by `IN` and `OUT` instructions.

### OUT to port FEH

Write bits control border colour and speaker/tape output:

| Bits | Name | Description |
|------|------|-------------|
| 2–0  | BORDER | Border colour (0=black … 7=white, matches INK/PAPER codes) |
| 3    | MIC   | Tape output (MIC socket) |
| 4    | EAR   | Loudspeaker drive. Toggle to generate sound. |

Example — toggle speaker bit for beeper:
```asm
  LD A, (speaker_state)  ; load current OUT value
  XOR %00010000          ; toggle bit 4 (EAR/speaker)
  OUT ($FE), A           ; write to ULA
  LD (speaker_state), A  ; save for next toggle
```

### IN from port FEH (keyboard)

Read bits report keyboard half-row and EAR input.
The address lines select which half-row to read.

| Bit | Name | Description |
|-----|------|-------------|
| 4–0 | KEY0–KEY4 | Key state (0=pressed, 1=released) |
| 6   | EAR input | Signal from EAR socket (tape input) |

Address lines A8–A14 select the keyboard row (one bit per row, active low):

| A8–A14 | Row | Keys (bit 0–4) |
|--------|-----|----------------|
| FEFEH  | 3   | SHIFT, Z, X, C, V |
| FDFEH  | 4   | A, S, D, F, G |
| FBFEH  | 2   | Q, W, E, R, T |
| F7FEH  | 5   | 1, 2, 3, 4, 5 |
| EFFEH  | 6   | 0, 9, 8, 7, 6 |
| DFFEH  | 1   | P, O, I, U, Y |
| BFFEH  | 0   | ENTER, L, K, J, H |
| 7FFEH  | 7   | SPACE, SYM, M, N, B |

Example — read SPACE key (row 7, bit 0):
```asm
  LD A, $7F          ; row 7 address high byte
  IN A, ($FE)        ; read keyboard
  AND %00000001      ; bit 0 = SPACE
  JR Z, space_pressed ; Z set if key pressed (bit=0)
```

## ULA display timing

The ULA generates a PAL TV signal at 50Hz. One frame = 69888 T-states.

| Phase | T-states | Description |
|-------|----------|-------------|
| Active display | 224 × 192 | Pixel output (192 scan lines) |
| Border (vertical) | 224 × 56 | Top + bottom border |
| HSYNC/VSYNC | — | Horizontal and vertical sync pulses |
| Interrupt | — | Maskable interrupt fired at frame start |

The interrupt fires at the start of the top border (not the active display).
The ROM interrupt handler (`0038H`) increments the FRAMES counter (5C78H)
and handles keyboard repeat timing.

**Contended memory:** CPU memory accesses during the active display are
delayed by the ULA. Addresses 4000H–7FFFH are contended — the ULA
inserts wait states to prevent bus conflicts when reading display data.

## Display file layout

The display file occupies 6912 bytes from 4000H:

- **Pixel data:** 4000H–57FFH (6144 bytes, 256×192 pixels)
- **Attribute data:** 5800H–5AFFH (768 bytes, 32×24 cells)

### Pixel addressing

The pixel address encodes row within cell, cell row, and cell column:

```
Bit pattern: 0 1 0 R R R Y Y Y X X X X X
             ^       ^   ^
             |       |   └── cell column (0–31)
             |       └──── Y within 8×8 cell (0–7)
             └────── R = third of screen (0, 1, 2)
```

This means the pixel data is stored in a non-linear order — consecutive
bytes in memory are not consecutive rows on screen.

### Attribute cell format

Each attribute byte covers an 8×8 pixel cell (32×24 cells total):

| Bits | Description |
|------|-------------|
| 7    | FLASH (blink between INK and PAPER) |
| 6    | BRIGHT (enhanced luminance) |
| 5–3  | PAPER colour (0–7) |
| 2–0  | INK colour (0–7) |

Colour codes: 0=black, 1=blue, 2=red, 3=magenta, 4=green, 5=cyan, 6=yellow, 7=white.
"""


def write_io_ports() -> None:
    """Write the curated I/O ports and ULA hardware reference."""
    fm = frontmatter(
        "ZX Spectrum I/O Ports and ULA",
        "spectrum-hardware",
        "https://skoolkid.github.io/rom/",
        ["io", "ports", "ula", "display", "hardware", "timing"],
    )
    path = OUT_HW / "io-ports-ula.md"
    write_md(path, fm + _IO_PORTS_CONTENT)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== fetch_spectrum.py (v2 — fixed selectors) ===")
    print(f"Output: {OUT_ROM} | {OUT_HW}")
    OUT_ROM.mkdir(parents=True, exist_ok=True)
    OUT_HW.mkdir(parents=True, exist_ok=True)

    # 1. Curated hardware reference (no network needed)
    print("\nWriting curated hardware reference…")
    write_io_ports()

    # 2. System variables from Skoolkid
    fetch_system_variables()

    # 3. Parse routines map
    entries = fetch_routines_map()

    if entries:
        # Fetch detail pages for priority routines
        fetch_selected_routines(entries)

        # Write stubs for the rest (if file doesn't exist)
        print(f"\nWriting stubs for remaining {len(entries)} routines…")
        stubs = 0
        for entry in entries:
            short = re.sub(r"^THE '(.+?)'\s*(SUBROUTINE|ROUTINE|RESTART|COMMAND.*)?$", r"\1", entry["name"], flags=re.IGNORECASE).strip()
            label = short or entry["name"]
            filename = OUT_ROM / f"{slug(label)}.md"
            if not filename.exists():
                write_stub(entry)
                stubs += 1
        print(f"  {stubs} stubs written")
    else:
        print("  [warn] routines map unavailable — no ROM routines fetched")

    print("\n=== Done ===")
    rom_files = list(OUT_ROM.glob("*.md"))
    hw_files = list(OUT_HW.glob("*.md"))
    print(f"  ROM files:      {len(rom_files)}")
    print(f"  Hardware files: {len(hw_files)}")


if __name__ == "__main__":
    main()
