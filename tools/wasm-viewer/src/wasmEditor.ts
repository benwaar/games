import * as vscode from 'vscode';
import * as path from 'path';

const WASM_MAGIC = 0x6D736100; // "\0asm" LE

const SECTION_NAMES: Record<number, string> = {
  0: 'Custom',
  1: 'Type',
  2: 'Import',
  3: 'Function',
  4: 'Table',
  5: 'Memory',
  6: 'Global',
  7: 'Export',
  8: 'Start',
  9: 'Element',
  10: 'Code',
  11: 'Data',
  12: 'DataCount',
};

const SECTION_COLOURS: Record<number, string> = {
  0: '#888',     // Custom — grey
  1: '#e8a',     // Type — pink
  2: '#fa6',     // Import — orange
  3: '#ad8',     // Function — green
  4: '#8cf',     // Table — light blue
  5: '#8cf',     // Memory — light blue
  6: '#da8',     // Global — gold
  7: '#0ee',     // Export — cyan
  8: '#f88',     // Start — red
  9: '#c8f',     // Element — purple
  10: '#8f8',    // Code — bright green
  11: '#fd8',    // Data — yellow
  12: '#aaa',    // DataCount — grey
};

const EXPORT_KIND: Record<number, string> = {
  0: 'func',
  1: 'table',
  2: 'memory',
  3: 'global',
};

const IMPORT_KIND: Record<number, string> = {
  0: 'func',
  1: 'table',
  2: 'memory',
  3: 'global',
};

interface WasmSection {
  id: number;
  name: string;
  fileOffset: number;
  headerSize: number; // bytes for id + LEB128 size
  payloadSize: number;
  payload: Uint8Array;
}

interface DecodedExport {
  name: string;
  kind: string;
  index: number;
}

interface DecodedImport {
  module: string;
  name: string;
  kind: string;
}

function readLEB128(data: Uint8Array, offset: number): { value: number; bytesRead: number } {
  let result = 0;
  let shift = 0;
  let bytesRead = 0;

  while (offset + bytesRead < data.length) {
    const byte = data[offset + bytesRead];
    result |= (byte & 0x7F) << shift;
    bytesRead++;
    if ((byte & 0x80) === 0) { break; }
    shift += 7;
  }

  return { value: result, bytesRead };
}

function parseWasmSections(data: Uint8Array): WasmSection[] {
  const sections: WasmSection[] = [];
  let offset = 8; // skip magic + version

  while (offset < data.length) {
    const sectionStart = offset;
    const id = data[offset];
    offset++;
    const { value: payloadSize, bytesRead } = readLEB128(data, offset);
    offset += bytesRead;
    const headerSize = 1 + bytesRead;

    const payload = data.slice(offset, offset + payloadSize);

    sections.push({
      id,
      name: SECTION_NAMES[id] ?? `Unknown(${id})`,
      fileOffset: sectionStart,
      headerSize,
      payloadSize,
      payload,
    });

    offset += payloadSize;
  }

  return sections;
}

function readName(data: Uint8Array, offset: number): { name: string; bytesRead: number } {
  const { value: len, bytesRead: lebBytes } = readLEB128(data, offset);
  const nameBytes = data.slice(offset + lebBytes, offset + lebBytes + len);
  const name = new TextDecoder().decode(nameBytes);
  return { name, bytesRead: lebBytes + len };
}

function decodeExports(payload: Uint8Array): DecodedExport[] {
  const exports: DecodedExport[] = [];
  let offset = 0;
  const { value: count, bytesRead } = readLEB128(payload, offset);
  offset += bytesRead;

  for (let i = 0; i < count && offset < payload.length; i++) {
    const { name, bytesRead: nameBytes } = readName(payload, offset);
    offset += nameBytes;
    const kind = payload[offset];
    offset++;
    const { value: index, bytesRead: idxBytes } = readLEB128(payload, offset);
    offset += idxBytes;
    exports.push({ name, kind: EXPORT_KIND[kind] ?? `unknown(${kind})`, index });
  }

  return exports;
}

function decodeImports(payload: Uint8Array): DecodedImport[] {
  const imports: DecodedImport[] = [];
  let offset = 0;
  const { value: count, bytesRead } = readLEB128(payload, offset);
  offset += bytesRead;

  for (let i = 0; i < count && offset < payload.length; i++) {
    const { name: module, bytesRead: modBytes } = readName(payload, offset);
    offset += modBytes;
    const { name, bytesRead: nameBytes } = readName(payload, offset);
    offset += nameBytes;
    const kind = payload[offset];
    offset++;
    // Skip the type descriptor (varies by kind)
    if (kind === 0) {
      const { bytesRead: b } = readLEB128(payload, offset);
      offset += b;
    } else if (kind === 1) {
      offset += 1; // reftype
      const { bytesRead: b1 } = readLEB128(payload, offset); offset += b1;
      const { bytesRead: b2 } = readLEB128(payload, offset); offset += b2;
    } else if (kind === 2) {
      const { bytesRead: b1 } = readLEB128(payload, offset); offset += b1;
      const { bytesRead: b2 } = readLEB128(payload, offset); offset += b2;
    } else if (kind === 3) {
      offset += 1; // valtype
      offset += 1; // mutability
    }
    imports.push({ module, name, kind: IMPORT_KIND[kind] ?? `unknown(${kind})` });
  }

  return imports;
}

function decodeCustomSectionName(payload: Uint8Array): string {
  const { name } = readName(payload, 0);
  return name;
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function toHex(n: number, width: number): string {
  return n.toString(16).toUpperCase().padStart(width, '0');
}

function hexRows(
  data: Uint8Array,
  start: number,
  end: number,
  sectionRanges: Array<{ start: number; end: number; colour: string }>
): string {
  const COLS = 16;
  let html = '';

  for (let row = start; row < end; row += COLS) {
    const count = Math.min(COLS, end - row);
    let hexCells = '';
    let asciiCells = '';

    for (let col = 0; col < COLS; col++) {
      if (col < count) {
        const addr = row + col;
        const byte = data[addr];
        const range = sectionRanges.find(r => addr >= r.start && addr < r.end);
        const style = range ? ` style="color:${range.colour}"` : '';
        hexCells += `<span${style}>${toHex(byte, 2)}</span> `;
        const ch = byte >= 0x20 && byte <= 0x7E ? String.fromCharCode(byte) : '.';
        asciiCells += `<span${style}>${escHtml(ch)}</span>`;
      } else {
        hexCells += '   ';
        asciiCells += ' ';
      }
      if (col === 7) { hexCells += ' '; }
    }

    html += `<div class="hex-row"><span class="offset">${toHex(row, 6)}</span>  ${hexCells} ${asciiCells}</div>`;
  }

  return html;
}

function buildWasmHtml(filename: string, data: Uint8Array, sections: WasmSection[]): string {
  const version = data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24);

  // Build section ranges for hex colouring
  const sectionRanges = sections.map(s => ({
    start: s.fileOffset,
    end: s.fileOffset + s.headerSize + s.payloadSize,
    colour: SECTION_COLOURS[s.id] ?? '#ccc',
  }));

  // Section summary cards
  let sectionsHtml = '';
  for (const s of sections) {
    const colour = SECTION_COLOURS[s.id] ?? '#ccc';
    const pct = ((s.payloadSize / data.length) * 100).toFixed(1);

    let detail = '';

    if (s.id === 0) {
      const customName = decodeCustomSectionName(s.payload);
      detail = `<div class="section-detail">Name: "${escHtml(customName)}"</div>`;
    }

    if (s.id === 7) {
      const exports = decodeExports(s.payload);
      const rows = exports.map(e =>
        `<span class="export-kind">${e.kind}</span> ${escHtml(e.name)} <span class="export-idx">#${e.index}</span>`
      ).join('<br>');
      detail = `<div class="section-detail">${rows}</div>`;
    }

    if (s.id === 2) {
      const imports = decodeImports(s.payload);
      const rows = imports.map(i =>
        `<span class="export-kind">${i.kind}</span> ${escHtml(i.module)}.${escHtml(i.name)}`
      ).join('<br>');
      detail = `<div class="section-detail">${rows}</div>`;
    }

    sectionsHtml += `
      <div class="section-card" style="border-left-color: ${colour}">
        <div class="section-header">
          <span class="section-id" style="color: ${colour}">${s.name}</span>
          <span class="section-size">${s.payloadSize.toLocaleString()} bytes (${pct}%)</span>
          <span class="section-offset">@ ${toHex(s.fileOffset, 6)}</span>
        </div>
        ${detail}
      </div>`;
  }

  // Size bar
  const barSegments = sections.map(s => {
    const pct = (s.payloadSize / data.length) * 100;
    const colour = SECTION_COLOURS[s.id] ?? '#ccc';
    return `<div class="bar-seg" style="width:${pct}%;background:${colour}" title="${s.name}: ${s.payloadSize}B"></div>`;
  }).join('');

  // Hex dump — cap at 8KB to avoid massive HTML
  const hexLimit = Math.min(data.length, 8192);
  const truncated = data.length > hexLimit;
  const hexHtml = hexRows(data, 0, hexLimit, sectionRanges);

  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src 'unsafe-inline';">
  <title>WASM Viewer — ${escHtml(filename)}</title>
  <style>
    body {
      background: #1e1e1e;
      color: #ccc;
      font-family: var(--vscode-font-family, monospace);
      font-size: 13px;
      margin: 0;
      padding: 16px;
    }
    h2 {
      color: #e0e0e0;
      font-size: 14px;
      margin: 0 0 8px;
      border-bottom: 1px solid #333;
      padding-bottom: 4px;
    }
    .file-meta {
      color: #888;
      font-size: 12px;
      margin-bottom: 16px;
    }
    .size-bar {
      display: flex;
      height: 12px;
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 16px;
      border: 1px solid #444;
    }
    .bar-seg {
      min-width: 1px;
    }
    .section-card {
      border-left: 3px solid #666;
      padding: 4px 8px;
      margin-bottom: 8px;
      background: #222;
      border-radius: 0 4px 4px 0;
    }
    .section-header {
      display: flex;
      gap: 12px;
      align-items: center;
      font-size: 12px;
    }
    .section-id { font-weight: bold; }
    .section-size { color: #aaa; }
    .section-offset { color: #666; }
    .section-detail {
      font-size: 12px;
      color: #adf;
      margin-top: 4px;
      padding-left: 8px;
    }
    .export-kind {
      color: #888;
      font-size: 11px;
    }
    .export-idx {
      color: #666;
      font-size: 11px;
    }
    h3 {
      color: #e0e0e0;
      font-size: 13px;
      margin: 20px 0 8px;
      border-bottom: 1px solid #333;
      padding-bottom: 4px;
    }
    .hex-dump {
      font-family: 'Cascadia Code', 'Fira Code', 'SF Mono', Menlo, monospace;
      font-size: 12px;
      line-height: 1.5;
      overflow-x: auto;
      max-height: 500px;
      overflow-y: auto;
      background: #181818;
      padding: 6px 8px;
      border-radius: 4px;
    }
    .hex-row {
      white-space: pre;
    }
    .offset {
      color: #666;
    }
    .truncated {
      color: #f88;
      font-size: 12px;
      margin-top: 4px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
      font-size: 11px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 2px;
    }
  </style>
</head>
<body>
  <h2>${escHtml(filename)}</h2>
  <div class="file-meta">${data.length.toLocaleString()} bytes — WASM version ${version} — ${sections.length} section${sections.length !== 1 ? 's' : ''}</div>

  <div class="size-bar">${barSegments}</div>

  <div class="legend">${sections.map(s => {
    const colour = SECTION_COLOURS[s.id] ?? '#ccc';
    return `<div class="legend-item"><div class="legend-swatch" style="background:${colour}"></div>${s.name}</div>`;
  }).join('')}</div>

  ${sectionsHtml}

  <h3>Hex dump${truncated ? ` (first ${(hexLimit / 1024).toFixed(0)}KB)` : ''}</h3>
  <div class="hex-dump">${hexHtml}</div>
  ${truncated ? `<p class="truncated">Showing ${hexLimit.toLocaleString()} of ${data.length.toLocaleString()} bytes</p>` : ''}
</body>
</html>`;
}

function errorPage(msg: string): string {
  return `<!DOCTYPE html><html><body style="color:#f88;font-family:monospace;padding:16px;">
    <p>${escHtml(msg)}</p></body></html>`;
}

export class WasmEditorProvider implements vscode.CustomReadonlyEditorProvider {

  static register(context: vscode.ExtensionContext): vscode.Disposable {
    return vscode.window.registerCustomEditorProvider(
      'wasmViewer.wasm',
      new WasmEditorProvider(context),
      { supportsMultipleEditorsPerDocument: false }
    );
  }

  constructor(private readonly context: vscode.ExtensionContext) {}

  async openCustomDocument(
    uri: vscode.Uri,
    _openContext: vscode.CustomDocumentOpenContext,
    _token: vscode.CancellationToken
  ): Promise<vscode.CustomDocument> {
    return { uri, dispose: () => {} };
  }

  async resolveCustomEditor(
    document: vscode.CustomDocument,
    webviewPanel: vscode.WebviewPanel,
    _token: vscode.CancellationToken
  ): Promise<void> {
    webviewPanel.webview.options = { enableScripts: false };

    const raw = await vscode.workspace.fs.readFile(document.uri);
    const data = new Uint8Array(raw);
    const filename = path.basename(document.uri.fsPath);

    if (data.length < 8) {
      webviewPanel.webview.html = errorPage('File too small to be a valid WASM binary.');
      return;
    }

    const magic = (data[0]) | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    if (magic !== WASM_MAGIC) {
      webviewPanel.webview.html = errorPage(
        `Not a WASM file — expected magic 0x${WASM_MAGIC.toString(16)}, got 0x${magic.toString(16)}.`
      );
      return;
    }

    const sections = parseWasmSections(data);
    webviewPanel.webview.html = buildWasmHtml(filename, data, sections);
  }
}
