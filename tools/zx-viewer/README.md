# zx-viewer

VS Code extension — readonly viewer for ZX Spectrum `.scr` and `.tap` files.

Opens `.scr` and `.tap` files in a webview that renders the Spectrum screen at 3x scale with correct ULA colours.

---

## Build

```bash
cd tools/zx-viewer
npm install
npm run compile
```

Compiled output lands in `out/`.

---

## Run in VS Code (Extension Development Host)

1. Open `tools/zx-viewer/` as a workspace in VS Code (or open the `games/` root).
2. Press **F5** — this launches the Extension Development Host.
3. In the host window, open any `.scr` or `.tap` file.
4. The custom viewer opens automatically instead of the hex editor.

Alternatively, right-click the file and choose **Open With... > ZX Spectrum Screen** (or **ZX Spectrum TAP**).

---

## File format support

### SCR (6912 bytes)

- 6144-byte bitmap: 1 bit per pixel, 256x192, interleaved row layout.
- 768-byte attributes: one byte per 8x8 cell encoding ink, paper, bright, and flash.

### TAP

- Variable-length container of blocks.
- Each block: 2-byte length (LE), 1-byte flag (0x00 header / 0xFF data), payload, 1-byte XOR checksum.
- The viewer finds the first data block (flag `0xFF`) with exactly 6912 bytes of payload and renders that as the screen.
- The info bar shows a summary of all blocks found and which one contains the screen.

---

## Info bar

Below the canvas the viewer shows:

| Field | Description |
|-------|-------------|
| File | Filename |
| Format | SCR or TAP |
| Size | File size in bytes |
| Colours | Distinct palette entries used (from attribute scan) |
| Blocks | TAP only — count and type of each block |

---

## Dev notes

- No runtime dependencies — all rendering is done in plain JS in the webview.
- The extension reads the binary file via `vscode.workspace.fs`, base64-encodes it, and passes it to the webview via an inline `<script>` variable.
- CSP restricts scripts to the extension's `media/` directory only.
- `image-rendering: pixelated` is set via `ctx.imageSmoothingEnabled = false` when scaling.
