"""
Fetch Z80 instruction reference data from public sources and write raw markdown files.

Dependencies (requirements comment):
    requests>=2.28
    beautifulsoup4>=4.11

Output: knowledge/raw/z80/{mnemonic}.md — one file per instruction

Sources:
    - https://clrhome.org/table/        Z80 opcode table (opcodes, T-states)
    - http://z80-heaven.wikidot.com/instructions-set  instruction descriptions + flags
"""

import re
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent.parent / "raw" / "z80"

SOURCES = {
    "clrhome": "https://clrhome.org/table/",
    "z80heaven": "http://z80-heaven.wikidot.com/instructions-set",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; knowledge-corpus-fetcher/1.0)"
}

TIMEOUT = 15  # seconds per request

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def get_html(url: str) -> BeautifulSoup | None:
    """Fetch URL and return parsed BeautifulSoup, or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.HTTPError as exc:
        log.error("HTTP %s fetching %s — skipping", exc.response.status_code, url)
    except requests.RequestException as exc:
        log.error("Request failed for %s: %s — skipping", url, exc)
    return None


# ---------------------------------------------------------------------------
# clrhome.org/table/ — opcode table
# ---------------------------------------------------------------------------

def parse_clrhome(soup: BeautifulSoup) -> dict[str, dict]:
    """
    Parse the clrhome opcode table.

    Returns dict keyed by normalised mnemonic:
        {
            "LD A,n": {
                "mnemonic": "LD A,n",
                "opcodes": ["3E", "n"],
                "t_states": "7",
                "description": "",   # filled later from z80-heaven
            },
            ...
        }
    """
    instructions: dict[str, dict] = {}

    # The table is rendered as <td> cells, each containing the mnemonic,
    # opcode bytes, and timing inside nested spans/divs.  The site uses
    # a grid where each <td> may carry data-* attributes.
    for td in soup.find_all("td"):
        mnemonic_tag = td.find("span", class_="mnem") or td.find("span", class_="mnemonic")
        if not mnemonic_tag:
            # Fallback: look for bold text that looks like a mnemonic
            strong = td.find("b")
            if strong:
                mnemonic_tag = strong

        if not mnemonic_tag:
            continue

        raw_mnem = mnemonic_tag.get_text(" ", strip=True)
        if not raw_mnem or len(raw_mnem) > 40:
            continue

        # Normalise whitespace
        mnemonic = re.sub(r"\s+", " ", raw_mnem).strip()

        # Extract opcode bytes from data-op or sibling spans
        opcodes: list[str] = []
        op_tag = td.get("data-op", "")
        if op_tag:
            opcodes = op_tag.upper().split()
        else:
            for span in td.find_all("span", class_=re.compile(r"opcode|byte|op")):
                opcodes.append(span.get_text(strip=True).upper())

        # T-states
        t_states = ""
        for span in td.find_all("span", class_=re.compile(r"time|tstates|cycles")):
            t_states = span.get_text(strip=True)
            break
        if not t_states:
            # look for data-t attribute on the td
            t_states = td.get("data-t", "")

        if mnemonic not in instructions:
            instructions[mnemonic] = {
                "mnemonic": mnemonic,
                "opcodes": opcodes,
                "t_states": t_states,
                "description": "",
                "flags": {},
                "source_clrhome": SOURCES["clrhome"],
            }

    return instructions


# ---------------------------------------------------------------------------
# z80-heaven.wikidot.com — descriptions + flags
# ---------------------------------------------------------------------------

_FLAG_NAMES = ("S", "Z", "H", "P/V", "N", "C")

def _parse_flag_row(row_text: str) -> dict[str, str]:
    """Best-effort parse of a flags row like 'S Z * 0 P 0'."""
    parts = row_text.split()
    flags: dict[str, str] = {}
    for i, name in enumerate(_FLAG_NAMES):
        flags[name] = parts[i] if i < len(parts) else "?"
    return flags


def parse_z80heaven_index(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """
    Return list of (mnemonic_text, href) pairs from the instruction index page.
    """
    links: list[tuple[str, str]] = []
    content = soup.find("div", id="page-content") or soup.find("div", class_="page-content") or soup
    for a in content.find_all("a", href=True):
        href: str = a["href"]
        if "/instructions/" in href or "instructions-set:" in href:
            text = a.get_text(strip=True)
            if text:
                links.append((text, href))
    return links


def fetch_z80heaven_instruction(url: str) -> dict:
    """
    Fetch one z80-heaven instruction page and extract description + flags.

    Returns dict with keys: description, flags, t_states (may override clrhome).
    """
    if not url.startswith("http"):
        url = "http://z80-heaven.wikidot.com" + url

    soup = get_html(url)
    if not soup:
        return {}

    content = soup.find("div", id="page-content") or soup.find("div", class_="page-content") or soup
    if not content:
        return {}

    text = content.get_text("\n", strip=True)

    # Description: first non-empty paragraph before any flag table
    description = ""
    for p in content.find_all("p"):
        p_text = p.get_text(" ", strip=True)
        if p_text and len(p_text) > 15:
            description = p_text
            break

    # Flags: look for a table or definition list with flag labels
    flags: dict[str, str] = {}
    for table in content.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:
                continue
            # A flag row typically has a label in first cell
            if cells[0] in _FLAG_NAMES or cells[0] in ("Flag", "Flags"):
                # header row — skip
                continue
            # Try matching 6-cell flag value rows
            if len(cells) >= 6:
                for i, name in enumerate(_FLAG_NAMES):
                    flags[name] = cells[i]
                break
        if flags:
            break

    # T-states: look for a line like "T-States: 7" or "Cycles: 4"
    t_match = re.search(r"(?:T[- ]States?|Cycles?)[:\s]+(\d[\d/,]+)", text, re.IGNORECASE)
    t_states = t_match.group(1) if t_match else ""

    return {"description": description, "flags": flags, "t_states": t_states}


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"[^\w\-]")

def safe_filename(mnemonic: str) -> str:
    """Convert 'LD A, (HL)' → 'LD_A_HL' suitable as a filename stem."""
    name = mnemonic.upper()
    name = re.sub(r"[\s,]+", "_", name)
    name = _SAFE_NAME_RE.sub("", name)
    return name.strip("_") or "UNKNOWN"


def write_instruction_md(data: dict, out_dir: Path) -> Path:
    """Write one instruction markdown file. Returns the path written."""
    mnemonic = data.get("mnemonic", "UNKNOWN")
    filename = safe_filename(mnemonic) + ".md"
    out_path = out_dir / filename

    opcodes = data.get("opcodes", [])
    t_states = data.get("t_states", "")
    description = data.get("description", "")
    flags = data.get("flags", {})
    source_url = data.get("source_clrhome", SOURCES["clrhome"])

    flag_rows = ""
    if flags:
        flag_rows = "\n".join(
            f"| {name} | {val} |" for name, val in flags.items()
        )
        flag_rows = (
            "| Flag | Effect |\n"
            "|------|--------|\n"
            + flag_rows
        )
    else:
        flag_rows = "_Not recorded._"

    opcode_str = " ".join(opcodes) if opcodes else "_Not recorded._"

    lines = [
        "---",
        f'topic: "{mnemonic}"',
        "category: z80-instruction",
        f'source_url: "{source_url}"',
        'licence: "public-domain"',
        f'tags: [z80, {mnemonic.split()[0].lower()}]',
        "---",
        "",
        f"# {mnemonic}",
        "",
        "## Description",
        "",
        description or "_No description fetched._",
        "",
        "## Opcodes / Encoding",
        "",
        f"`{opcode_str}`",
        "",
        "## Timing",
        "",
        f"T-States: {t_states or '_Not recorded._'}",
        "",
        "## Flags",
        "",
        flag_rows,
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_clrhome() -> dict[str, dict]:
    """Fetch and parse the clrhome opcode table."""
    log.info("Fetching clrhome opcode table: %s", SOURCES["clrhome"])
    soup = get_html(SOURCES["clrhome"])
    if not soup:
        log.error("Could not fetch clrhome table — aborting clrhome stage")
        return {}
    instructions = parse_clrhome(soup)
    log.info("clrhome: parsed %d instructions", len(instructions))
    return instructions


def enrich_from_z80heaven(instructions: dict[str, dict]) -> dict[str, dict]:
    """
    Fetch the z80-heaven index, then enrich each matching instruction with
    description and flag data.  Unknown instructions are added as stubs.
    """
    log.info("Fetching z80-heaven index: %s", SOURCES["z80heaven"])
    soup = get_html(SOURCES["z80heaven"])
    if not soup:
        log.warning("Could not fetch z80-heaven index — descriptions will be empty")
        return instructions

    links = parse_z80heaven_index(soup)
    log.info("z80-heaven: found %d instruction links", len(links))

    fetched = 0
    for label, href in links:
        # Normalise label to match clrhome keys (best effort)
        norm = re.sub(r"\s+", " ", label).strip()

        extra = fetch_z80heaven_instruction(href)
        if not extra:
            continue

        if norm in instructions:
            target = instructions[norm]
        else:
            # Add as a new stub entry
            target = {
                "mnemonic": norm,
                "opcodes": [],
                "t_states": "",
                "source_clrhome": SOURCES["z80heaven"],
            }
            instructions[norm] = target

        if extra.get("description") and not target.get("description"):
            target["description"] = extra["description"]
        if extra.get("flags"):
            target["flags"] = extra["flags"]
        if extra.get("t_states") and not target.get("t_states"):
            target["t_states"] = extra["t_states"]

        fetched += 1
        if fetched % 10 == 0:
            log.info("z80-heaven: enriched %d instructions so far…", fetched)

        # Be polite — don't hammer the server
        time.sleep(0.3)

    log.info("z80-heaven: enriched %d instructions total", fetched)
    return instructions


def write_all(instructions: dict[str, dict], out_dir: Path) -> int:
    """Write all instruction markdown files. Returns count written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for data in instructions.values():
        try:
            path = write_instruction_md(data, out_dir)
            log.info("Fetched %s → %s", data["mnemonic"], path.name)
            written += 1
        except Exception as exc:
            log.error("Failed to write %s: %s", data.get("mnemonic", "?"), exc)
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    instructions = fetch_clrhome()
    instructions = enrich_from_z80heaven(instructions)

    if not instructions:
        log.error("No instructions fetched — nothing to write")
        sys.exit(1)

    count = write_all(instructions, RAW_DIR)
    log.info("Done. Wrote %d instruction files to %s", count, RAW_DIR)
