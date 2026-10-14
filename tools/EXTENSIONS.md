# Tools — VS Code Extensions

Extensions useful across the games repo projects. Install with `code --install-extension <id>`.

---

## Custom (this repo)

| Extension | ID | Files | What it does |
|---|---|---|---|
| ZX Spectrum Viewer | `zx-viewer` | `.scr`, `.tap` | Renders SCR screens at 3× scale. TAP files get a hex inspector with decoded block headers; SCR data highlighted in cyan |
| WASM Binary Viewer | `wasm-viewer` | `.wasm` | Section breakdown with colour-coded cards, size bar, decoded exports/imports, hex dump |

Install from `tools/`:
```bash
cd tools/zx-viewer && npm run install-ext
cd tools/wasm-viewer && npm run install-ext
```

---

## Z80 / Assembly

| Extension | ID | What it does |
|---|---|---|
| Z80 Macro-Assembler | `mborik.z80-macroasm` | Syntax highlighting, completions, hover docs for Z80 instructions |
| ASM Code Lens | `maziac.asm-code-lens` | Go-to-definition, references, code lens labels for Z80 asm |
| Hex Editor | `ms-vscode.hexeditor` | Hex view for `.bin` and any binary — data inspector, search, go-to-offset. Right-click → "Open With..." → "Hex Editor" |

---

## WASM / WebGL

| Extension | ID | What it does |
|---|---|---|
| Rust Analyzer | `rust-lang.rust-analyzer` | Rust language server — completions, diagnostics, inlay hints |
| WebGL GLSL Editor | `raczzalan.webgl-glsl-editor` | Syntax highlighting for GLSL shaders |
| LLDB DAP | `llvm-vs-code-extensions.lldb-dap` | LLDB debugger for Rust/C — works with WASM debug builds |

---

## Python / ML

| Extension | ID | What it does |
|---|---|---|
| Python | `ms-python.python` | Language server, debugging, environments |
| Pylance | `ms-python.vscode-pylance` | Type checking, auto-imports, IntelliSense |
| Debugpy | `ms-python.debugpy` | Python debugger |

---

## Data / Audio

| Extension | ID | What it does |
|---|---|---|
| WAV Preview | `sukumo28.wav-preview` | Waveform viewer for `.wav` files — useful for signal-hunt / artefact |
| Rainbow CSV | `mechatroner.rainbow-csv` | Column-coloured CSV/TSV viewing |

---

## General

| Extension | ID | What it does |
|---|---|---|
| GitLens | `eamodio.gitlens` | Git blame, history, line annotations |
| Markdown All in One | `yzhang.markdown-all-in-one` | TOC, preview, shortcuts for explainers/docs |
| Live Server | `ritwickdey.liveserver` | Quick local HTTP server — useful for WASM builds |
| REST Client | `humao.rest-client` | HTTP requests from `.http` files |
