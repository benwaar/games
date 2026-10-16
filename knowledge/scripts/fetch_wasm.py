"""
Fetch WebAssembly and WAT reference data from public sources.

deps: requests beautifulsoup4

Sources:
  - https://webassembly.github.io/spec/core/  (W3C WASM spec — W3C licence)
  - https://developer.mozilla.org/en-US/docs/WebAssembly  (MDN — CC-BY-SA)
  - https://github.com/WebAssembly/WASI/tree/main/legacy/preview1  (Apache 2.0)

Outputs raw markdown to knowledge/raw/wasm/{instructions,wat,wasi}/
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAW_ROOT = REPO_ROOT / "knowledge" / "raw" / "wasm"

OUT_INSTRUCTIONS = RAW_ROOT / "instructions"
OUT_WAT = RAW_ROOT / "wat"
OUT_WAT_TYPES = OUT_WAT / "types"
OUT_WAT_MODULES = OUT_WAT / "modules"
OUT_WASI = RAW_ROOT / "wasi"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "knowledge-corpus-fetcher/1.0 (educational use)"


def get(url: str, timeout: int = 30) -> Optional[requests.Response]:
    """Fetch URL, return response or None on error."""
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        print(f"  [SKIP] {url} — {exc}")
        return None


def soup(resp: requests.Response) -> BeautifulSoup:
    return BeautifulSoup(resp.text, "html.parser")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (OUT_INSTRUCTIONS, OUT_WAT_TYPES, OUT_WAT_MODULES, OUT_WASI):
        d.mkdir(parents=True, exist_ok=True)


def write_raw(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def slug(text: str, max_len: int = 80) -> str:
    """Convert a title string to a filename-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9_.\-]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def frontmatter(topic: str, category: str, source_url: str, licence: str, tags: list[str]) -> str:
    tag_list = ", ".join(tags)
    return textwrap.dedent(f"""\
        ---
        topic: "{topic}"
        category: {category}
        source_url: "{source_url}"
        licence: "{licence}"
        tags: [{tag_list}]
        ---

    """)


# ---------------------------------------------------------------------------
# WASM spec — instructions
# ---------------------------------------------------------------------------

# Canonical instruction reference from the spec appendix
SPEC_INSTR_INDEX = "https://webassembly.github.io/spec/core/appendix/index-instructions.html"
SPEC_EXEC_BASE   = "https://webassembly.github.io/spec/core/exec/"

# MDN WASM instruction overview (human-friendly descriptions)
MDN_INSTR_OVERVIEW = "https://developer.mozilla.org/en-US/docs/WebAssembly/Reference"

# Known instruction groups for targeted MDN fetches
MDN_INSTR_GROUPS = {
    "numeric":   "https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/Numeric",
    "memory":    "https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/Memory",
    "control":   "https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/Control_flow",
    "variables": "https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/Variables",
}


def clean_wasm_name(raw: str) -> str:
    """Extract a clean WASM instruction name from spec markup text.

    The spec table cells contain MathML fragments like
    'href .. mathsf{i32.add} ..'. We extract the mathsf content
    or fall back to the first dotted token that looks like an instruction.
    """
    m = re.search(r"mathsf\{([^}]+)\}", raw)
    if m:
        parts = []
        for match in re.finditer(r"mathsf\{([^}]+)\}", raw):
            parts.append(match.group(1))
        return ".".join(parts) if parts else m.group(1)
    m = re.search(r"\b([a-z][a-z0-9]*(?:\.[a-z_][a-z0-9_]*)+)\b", raw)
    if m:
        return m.group(1)
    cleaned = re.sub(r"href\S*", "", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:80] if cleaned else raw[:80]


def fetch_spec_instruction_index() -> list[dict]:
    """
    Parse the spec instruction index table.
    Returns a list of dicts: {name, type, binary, description}.
    """
    print("Fetching WASM spec instruction index …")
    resp = get(SPEC_INSTR_INDEX)
    if resp is None:
        return []

    s = soup(resp)
    rows = []
    table = s.find("table")
    if not table:
        print("  [WARN] no table found in spec instruction index")
        return rows

    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]

    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        row: dict = {}
        for i, h in enumerate(headers):
            row[h] = cells[i] if i < len(cells) else ""
        name_key = "instruction" if "instruction" in row else "name"
        if name_key in row:
            row[name_key] = clean_wasm_name(row[name_key])
        rows.append(row)

    print(f"  found {len(rows)} instructions in spec index")
    return rows


def write_instruction_file(name: str, opcode: str, type_sig: str, description: str,
                            source_url: str, licence: str) -> None:
    filename = slug(name) + ".md"
    fm = frontmatter(
        topic=name,
        category="wasm-spec",
        source_url=source_url,
        licence=licence,
        tags=[t for t in [name.split(".")[0] if "." in name else "misc",
                           "instruction", "wasm"] if t],
    )
    body = textwrap.dedent(f"""\
        # `{name}`

        **Opcode:** `{opcode}`
        **Type:** `{type_sig}`

        ## Description

        {description}

        ## Example (WAT)

        ```wat
        ;; {name}
        {name}
        ```
    """)
    write_raw(OUT_INSTRUCTIONS / filename, fm + body)


def fetch_wasm_instructions() -> None:
    print("\n--- WASM instructions ---")
    rows = fetch_spec_instruction_index()

    if rows:
        for row in rows:
            # Column names vary; try common variants
            name = (row.get("instruction") or row.get("name") or "").strip()
            opcode = (row.get("binary") or row.get("opcode") or row.get("encoding") or "").strip()
            type_sig = (row.get("type") or row.get("signature") or "").strip()
            desc = (row.get("description") or row.get("notes") or "").strip()

            if not name:
                continue

            write_instruction_file(
                name=name,
                opcode=opcode,
                type_sig=type_sig,
                description=desc or f"WASM instruction `{name}`.",
                source_url=SPEC_INSTR_INDEX,
                licence="W3C-20150513",
            )

    # Supplement with MDN group pages for human-readable descriptions
    for group, url in MDN_INSTR_GROUPS.items():
        print(f"Fetching MDN instruction group: {group} …")
        resp = get(url)
        if resp is None:
            continue

        s = soup(resp)
        # Each instruction is typically a <dt>/<dd> pair or a table row on MDN
        for article in s.select("article, section"):
            h = article.find(re.compile(r"h[2-4]"))
            if not h:
                continue
            instr_name = h.get_text(strip=True).strip("`").strip()
            if not instr_name or " " in instr_name.replace(".", ""):
                continue
            paras = [p.get_text(" ", strip=True) for p in article.find_all("p")]
            desc = " ".join(paras[:2]) if paras else ""
            if not desc:
                continue
            # Don't overwrite if the spec already wrote this file
            filename = slug(instr_name) + ".md"
            dest = OUT_INSTRUCTIONS / filename
            if dest.exists():
                continue
            write_instruction_file(
                name=instr_name,
                opcode="",
                type_sig="",
                description=desc,
                source_url=url,
                licence="CC-BY-SA-2.5",
            )


# ---------------------------------------------------------------------------
# WAT text format — modules, types, imports/exports
# ---------------------------------------------------------------------------

MDN_WAT_PAGES = {
    "wat_overview":  "https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format",
    "wat_functions": "https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format#functions",
    "wat_types":     "https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format#types",
    "wat_imports":   "https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format#imports_and_exports",
    "wat_memory":    "https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format#memory",
    "wat_tables":    "https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format#tables",
    "wat_globals":   "https://developer.mozilla.org/en-US/docs/WebAssembly/Understanding_the_text_format#globals",
}

SPEC_TEXT_BASE = "https://webassembly.github.io/spec/core/text/"
SPEC_TEXT_PAGES = {
    "text_modules":  SPEC_TEXT_BASE + "modules.html",
    "text_types":    SPEC_TEXT_BASE + "types.html",
    "text_instrs":   SPEC_TEXT_BASE + "instructions.html",
    "text_values":   SPEC_TEXT_BASE + "values.html",
}


def html_to_md_excerpt(element: Tag, max_chars: int = 2000) -> str:
    """Very light HTML → markdown: headings, code blocks, paragraphs."""
    lines = []
    for child in element.children:
        if not isinstance(child, Tag):
            continue
        tag = child.name
        text = child.get_text(" ", strip=True)
        if not text:
            continue
        if tag in ("h1", "h2", "h3", "h4"):
            level = int(tag[1])
            lines.append("#" * level + " " + text)
        elif tag == "pre":
            code = child.get_text()
            lines.append("```\n" + code.strip() + "\n```")
        elif tag == "code":
            lines.append(f"`{text}`")
        elif tag in ("p", "dd", "li"):
            lines.append(text)
        elif tag in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                lines.append("- " + li.get_text(" ", strip=True))
    result = "\n\n".join(lines)
    return result[:max_chars]


def write_wat_file(name: str, subdir: Path, content: str, source_url: str, licence: str,
                   category: str = "wasm-spec") -> None:
    fm = frontmatter(
        topic=name,
        category=category,
        source_url=source_url,
        licence=licence,
        tags=["wat", "text-format", slug(name)],
    )
    write_raw(subdir / (slug(name) + ".md"), fm + content)


def fetch_wat_format() -> None:
    print("\n--- WAT text format (MDN) ---")
    for page_key, url in MDN_WAT_PAGES.items():
        print(f"Fetching {page_key} …")
        resp = get(url)
        if resp is None:
            continue

        s = soup(resp)
        main = s.find("main") or s.find("article") or s.body
        if not main:
            continue

        title = s.find("h1")
        page_title = title.get_text(strip=True) if title else page_key

        content_md = html_to_md_excerpt(main, max_chars=4000)
        if not content_md.strip():
            print(f"  [WARN] no content extracted from {url}")
            continue

        body = f"# {page_title}\n\n**Source:** {url}\n\n{content_md}\n"
        subdir = OUT_WAT_MODULES if "module" in page_key or "import" in page_key or "memory" in page_key or "table" in page_key or "global" in page_key else OUT_WAT_TYPES
        write_wat_file(
            name=page_title,
            subdir=subdir,
            content=body,
            source_url=url,
            licence="CC-BY-SA-2.5",
        )

    print("\n--- WAT text format (spec) ---")
    for page_key, url in SPEC_TEXT_PAGES.items():
        print(f"Fetching {page_key} …")
        resp = get(url)
        if resp is None:
            continue

        s = soup(resp)
        main = s.find("main") or s.find("article") or s.find("div", class_="body") or s.body
        if not main:
            continue

        title_el = s.find("h1") or s.find("title")
        page_title = title_el.get_text(strip=True) if title_el else page_key
        content_md = html_to_md_excerpt(main, max_chars=4000)

        body = f"# {page_title}\n\n**Source:** {url}\n\n{content_md}\n"
        subdir = OUT_WAT_TYPES if "type" in page_key or "value" in page_key else OUT_WAT_MODULES
        write_wat_file(
            name=page_title,
            subdir=subdir,
            content=body,
            source_url=url,
            licence="W3C-20150513",
        )


# ---------------------------------------------------------------------------
# WASI preview1 — function signatures
# ---------------------------------------------------------------------------

# GitHub raw API for the preview1 docs directory listing
WASI_TREE_API = "https://api.github.com/repos/WebAssembly/WASI/contents/legacy/preview1"
WASI_RAW_BASE = "https://raw.githubusercontent.com/WebAssembly/WASI/main/legacy/preview1"

# Fallback: fetch the witx spec files directly
WASI_WITX_FILES = [
    "typenames.witx",
    "wasi_snapshot_preview1.witx",
]


def parse_witx_signatures(witx_text: str) -> list[dict]:
    """
    Very lightweight witx parser — extracts module+function name + docstring + params.
    Returns list of {name, module, doc, params, results}.
    """
    funcs = []
    current_doc: list[str] = []
    current_func: Optional[dict] = None
    module_name = "wasi_snapshot_preview1"

    for line in witx_text.splitlines():
        stripped = line.strip()

        # Module name
        m = re.match(r'\(module\s+\$([^\s)]+)', stripped)
        if m:
            module_name = m.group(1)
            continue

        # Docstring lines
        if stripped.startswith(";;"):
            current_doc.append(stripped.lstrip("; ").strip())
            continue

        # Function declaration
        m = re.match(r'\(@interface\s+func\s+\(export\s+"([^"]+)"\)', stripped)
        if not m:
            m = re.match(r'\(func\s+\(export\s+"([^"]+)"\)', stripped)
        if not m:
            m = re.match(r'\(export\s+"([^"]+)"\s+\(func', stripped)
        if m:
            func_name = m.group(1)
            current_func = {
                "name": func_name,
                "module": module_name,
                "doc": " ".join(current_doc),
                "raw": [stripped],
            }
            current_doc = []
            funcs.append(current_func)
            continue

        if current_func is not None:
            current_func["raw"].append(stripped)
            if stripped == ")":
                current_func = None
            continue

        # Reset doc if non-doc/non-func line
        if stripped and not stripped.startswith("("):
            current_doc = []

    return funcs


def write_wasi_file(func: dict, source_url: str) -> None:
    name = func["name"]
    module = func.get("module", "wasi_snapshot_preview1")
    doc = func.get("doc", "")
    raw_lines = func.get("raw", [])
    signature = "\n".join(raw_lines[:20])  # first 20 lines of the declaration

    fm = frontmatter(
        topic=name,
        category="wasi",
        source_url=source_url,
        licence="Apache-2.0",
        tags=["wasi", "preview1", slug(name)],
    )
    body = textwrap.dedent(f"""\
        # `{name}`

        **Module:** `{module}`

        ## Description

        {doc or f"WASI preview1 function `{name}`."}

        ## Signature

        ```witx
        {signature}
        ```

        ## Usage (WAT import)

        ```wat
        (import "wasi_snapshot_preview1" "{name}"
          (func ${name} ...))
        ```
    """)
    write_raw(OUT_WASI / (slug(name) + ".md"), fm + body)


WASI_PREVIEW1_STUBS = [
    {"name": "args_get", "doc": "Read command-line argument data.", "params": "(param $argv i32) (param $argv_buf i32) (result $error i32)"},
    {"name": "args_sizes_get", "doc": "Return command-line argument data sizes.", "params": "(param $argc i32) (param $argv_buf_size i32) (result $error i32)"},
    {"name": "environ_get", "doc": "Read environment variable data.", "params": "(param $environ i32) (param $environ_buf i32) (result $error i32)"},
    {"name": "environ_sizes_get", "doc": "Return environment variable data sizes.", "params": "(param $environc i32) (param $environ_buf_size i32) (result $error i32)"},
    {"name": "fd_close", "doc": "Close a file descriptor.", "params": "(param $fd i32) (result $error i32)"},
    {"name": "fd_fdstat_get", "doc": "Get the attributes of a file descriptor.", "params": "(param $fd i32) (param $stat i32) (result $error i32)"},
    {"name": "fd_prestat_get", "doc": "Return a description of the given preopened file descriptor.", "params": "(param $fd i32) (param $buf i32) (result $error i32)"},
    {"name": "fd_prestat_dir_name", "doc": "Return a description of the given preopened file descriptor.", "params": "(param $fd i32) (param $path i32) (param $path_len i32) (result $error i32)"},
    {"name": "fd_read", "doc": "Read from a file descriptor.", "params": "(param $fd i32) (param $iovs i32) (param $iovs_len i32) (param $nread i32) (result $error i32)"},
    {"name": "fd_seek", "doc": "Move the offset of a file descriptor.", "params": "(param $fd i32) (param $offset i64) (param $whence i32) (param $newoffset i32) (result $error i32)"},
    {"name": "fd_write", "doc": "Write to a file descriptor. Used for stdout/stderr output.", "params": "(param $fd i32) (param $iovs i32) (param $iovs_len i32) (param $nwritten i32) (result $error i32)"},
    {"name": "path_open", "doc": "Open a file or directory.", "params": "(param $fd i32) (param $dirflags i32) (param $path i32) (param $path_len i32) (param $oflags i32) (param $fs_rights_base i64) (param $fs_rights_inheriting i64) (param $fdflags i32) (param $opened_fd i32) (result $error i32)"},
    {"name": "proc_exit", "doc": "Terminate the process normally. An exit code of 0 indicates successful termination.", "params": "(param $rval i32)"},
    {"name": "random_get", "doc": "Write high-quality random data into a buffer.", "params": "(param $buf i32) (param $buf_len i32) (result $error i32)"},
    {"name": "clock_time_get", "doc": "Return the time value of a clock.", "params": "(param $id i32) (param $precision i64) (param $time i32) (result $error i32)"},
]


def fetch_wasi_signatures() -> None:
    print("\n--- WASI preview1 signatures ---")

    fetched = False
    for witx_file in WASI_WITX_FILES:
        url = f"{WASI_RAW_BASE}/{witx_file}"
        print(f"Fetching {witx_file} …")
        resp = get(url)
        if resp is None:
            continue

        funcs = parse_witx_signatures(resp.text)
        print(f"  found {len(funcs)} functions in {witx_file}")
        for func in funcs:
            write_wasi_file(func, source_url=url)
        if funcs:
            fetched = True

    if not fetched:
        print("  [fallback] witx files unavailable — writing known WASI preview1 stubs")
        for stub in WASI_PREVIEW1_STUBS:
            func = {
                "name": stub["name"],
                "module": "wasi_snapshot_preview1",
                "doc": stub["doc"],
                "raw": [f'(export "{stub["name"]}" (func {stub["params"]}))'],
            }
            write_wasi_file(func, source_url="https://github.com/WebAssembly/WASI")
        print(f"  wrote {len(WASI_PREVIEW1_STUBS)} WASI stub files")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== fetch_wasm.py ===")
    print(f"Output root: {RAW_ROOT.relative_to(REPO_ROOT)}\n")

    ensure_dirs()

    fetch_wasm_instructions()
    fetch_wat_format()
    fetch_wasi_signatures()

    print("\nDone.")


if __name__ == "__main__":
    main()
