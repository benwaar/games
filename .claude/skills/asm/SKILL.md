---
name: asm
description: "Browse or search the assembly knowledge corpus. /asm — overview · /asm z80 — Z80/Spectrum · /asm wasm — WASM/WASI · /asm z80 rom BEEPER — detail"
---

Browse or search the local knowledge corpus at `knowledge/chunks/`.

Two top-level domains, matching the two output targets:

- **z80** — Z80 instructions, patterns, Spectrum ROM, Spectrum hardware
- **wasm** — WASM spec, WAT patterns, WASI syscalls

## Modes

1. **No args** (`/asm`) → **Overview** — show both domains with category counts
2. **Domain only** (`/asm z80` or `/asm wasm`) → **List domain** — show all categories in that domain
3. **Domain + category** (`/asm z80 rom`) → **List category** — show entries in that category
4. **Domain + category + query** (`/asm z80 rom BEEPER`) → **Detail** — show matching chunk(s)
5. **Domain + query** (`/asm z80 LD`) → **Search domain** — search all categories in that domain
6. **Query only** (`/asm MASK-INT`) → **Search all** — search everything

## Domain → category mapping

### z80 domain
| Category alias | Directory |
|---|---|
| _(no category)_ | all z80 dirs |
| `instr`, `instructions` | `z80-instruction/` |
| `pattern`, `patterns` | `z80-pattern/` |
| `rom` | `spectrum-rom/` |
| `hw`, `hardware` | `spectrum-hardware/` |

### wasm domain
| Category alias | Directory |
|---|---|
| _(no category)_ | all wasm dirs |
| `spec` | `wasm-spec/` |
| `pattern`, `patterns` | `wasm-pattern/` |
| `wat` | `wasm-spec/` (filter for WAT-related) |
| `wasi` | `wasi/` |

## Helper script

All filesystem access goes through `.claude/skills/asm/z80-query.sh` (pre-allowed in settings.json — no permission prompts).

Commands:
- `.claude/skills/asm/z80-query.sh overview` → `<category> <count>` per line
- `.claude/skills/asm/z80-query.sh list <dir>` → one filename stem per line, sorted
- `.claude/skills/asm/z80-query.sh search <dir|''> <query>` → `FILE:<path>` and `GREP:<path>` matches

**Never use raw ls, find, or grep on the chunks directory.** Always use the helper script.

## Steps by mode

### Overview mode (no args)

1. Run: `.claude/skills/asm/z80-query.sh overview`
2. Group into two domains and print:
   ```
   ## Z80 / Spectrum
   | Category | Chunks | Browse with |
   |---|---|---|
   | Instructions | 68 | /asm z80 instr |
   | Patterns | 7 | /asm z80 pattern |
   | ROM Routines | 29 | /asm z80 rom |
   | Hardware | 0 | (not yet populated) |

   ## WASM / WASI
   | Category | Chunks | Browse with |
   |---|---|---|
   | Spec | 405 | /asm wasm spec |
   | Patterns | 5 | /asm wasm pattern |
   | WASI | 15 | /asm wasm wasi |
   ```
3. Categories with 0 chunks: show as "(not yet populated)".

### List domain mode (`/asm z80` or `/asm wasm`)

Same as overview but filtered to one domain. Show category table for just that domain.

### List category mode (`/asm z80 rom`)

1. Run: `.claude/skills/asm/z80-query.sh list <directory-name>`
2. Print as `·`-separated names on wrapped lines:
   ```
   ## ROM Routines (29 entries)
   ATTR-BITS · BC-SPACES · BEEPER · BORDER · CHAN-OPEN · ...
   ```
3. End with: `Use /asm z80 rom <name> to see details.`

### Detail mode (`/asm z80 rom BEEPER`)

1. Run: `.claude/skills/asm/z80-query.sh search <dir> <query>`
2. Deduplicate FILE: and GREP: results. Prefer FILE matches.
3. Read the top 1–3 matched files with the Read tool.
4. **Format each result:**
   - Header: `## <topic> (<category>)`
   - One line: `source: <source_url> · licence: <licence>`
   - Full chunk body (truncate at 150 lines — print first 100 and note remainder)
   - Separate multiple results with `---`

### Search mode (`/asm z80 LD` or `/asm MASK-INT`)

- With domain: search all dirs in that domain
- Without domain: `.claude/skills/asm/z80-query.sh search '' <query>`

## Examples

```
/asm                          — full overview
/asm z80                      — Z80/Spectrum categories
/asm wasm                     — WASM/WASI categories
/asm z80 instr                — list all Z80 instructions
/asm z80 rom                  — list all ROM routines
/asm z80 rom BEEPER           — ROM BEEP routine detail
/asm z80 LD                   — search Z80 domain for LD
/asm wasm spec memory.grow    — WASM memory.grow spec
/asm wasm wasi fd_write       — WASI fd_write detail
/asm wasm wat                 — list WAT-related entries
/asm MASK-INT                 — search everything
```

## Notes

- The corpus lives at `knowledge/chunks/` relative to the repo root.
- Chunks are plain markdown with YAML frontmatter.
- Empty categories (0 chunks): show as "(not yet populated)".
- Z80 instruction filenames are mnemonics (e.g. `LD.md`, `CALL.md`).
- WASM spec filenames are instruction names (e.g. `i32.add.md`, `memory.grow.md`).
