import * as vscode from 'vscode';
import * as path from 'path';
import { countColours, SCR_SIZE } from './renderer';

interface TapBlock {
  flag: number;
  payload: Uint8Array;
  valid: boolean;
  /** Byte offset of the 2-byte length prefix in the original file */
  fileOffset: number;
  /** Total bytes this block occupies in the file (2 len + flag + payload + checksum) */
  fileSize: number;
}

const HEADER_TYPE_NAMES: Record<number, string> = {
  0: 'Program',
  1: 'Number array',
  2: 'Character array',
  3: 'Bytes',
};

function parseTap(data: Uint8Array): TapBlock[] {
  const blocks: TapBlock[] = [];
  let offset = 0;

  while (offset + 2 <= data.length) {
    const blockStart = offset;
    const blockLen = data[offset] | (data[offset + 1] << 8);
    offset += 2;

    if (blockLen < 2 || offset + blockLen > data.length) {
      break;
    }

    const flag = data[offset];
    const payload = data.slice(offset + 1, offset + blockLen - 1);
    const checksum = data[offset + blockLen - 1];

    let computed = flag;
    for (let i = 0; i < payload.length; i++) {
      computed ^= payload[i];
    }

    blocks.push({
      flag,
      payload,
      valid: computed === checksum,
      fileOffset: blockStart,
      fileSize: 2 + blockLen,
    });
    offset += blockLen;
  }

  return blocks;
}

function findScreenBlock(blocks: TapBlock[]): { scr: Uint8Array; blockIndex: number } | null {
  for (let i = 0; i < blocks.length; i++) {
    if (blocks[i].flag === 0xFF && blocks[i].payload.length === SCR_SIZE) {
      return { scr: blocks[i].payload, blockIndex: i };
    }
  }
  return null;
}

function bufferToBase64(buf: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < buf.length; i++) {
    binary += String.fromCharCode(buf[i]);
  }
  return Buffer.from(binary, 'binary').toString('base64');
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function toHex(n: number, width: number): string {
  return n.toString(16).toUpperCase().padStart(width, '0');
}

function decodeHeader(payload: Uint8Array): string | null {
  if (payload.length !== 17) { return null; }
  const type = payload[0];
  const name = String.fromCharCode(...payload.slice(1, 11)).trimEnd();
  const dataLen = payload[11] | (payload[12] << 8);
  const param1 = payload[13] | (payload[14] << 8);
  const param2 = payload[15] | (payload[16] << 8);
  const typeName = HEADER_TYPE_NAMES[type] ?? `Unknown(${type})`;

  let detail = `${typeName}: "${name}" — ${dataLen} bytes`;
  if (type === 0) { detail += `, autostart ${param1 === 0x8000 ? 'none' : String(param1)}`; }
  if (type === 3) { detail += `, load addr ${toHex(param1, 4)}h`; }
  return detail;
}

/** Build hex dump rows for a byte range, with optional highlight for SCR data */
function hexRows(
  data: Uint8Array,
  start: number,
  end: number,
  scrStart: number,
  scrEnd: number
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
        const inScr = addr >= scrStart && addr < scrEnd;
        const cls = inScr ? ' class="scr"' : '';
        hexCells += `<span${cls}>${toHex(byte, 2)}</span> `;
        const ch = byte >= 0x20 && byte <= 0x7E ? String.fromCharCode(byte) : '.';
        asciiCells += `<span${cls}>${escHtml(ch)}</span>`;
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

function buildTapHtml(
  webview: vscode.Webview,
  context: vscode.ExtensionContext,
  filename: string,
  data: Uint8Array,
  blocks: TapBlock[],
  screenResult: { scr: Uint8Array; blockIndex: number } | null
): string {
  const scriptUri = webview.asWebviewUri(
    vscode.Uri.joinPath(context.extensionUri, 'media', 'viewer.js')
  );

  // SCR byte range in original file (for highlighting)
  let scrFileStart = -1;
  let scrFileEnd = -1;
  let canvasHtml = '';
  let scrB64 = '';

  if (screenResult) {
    const scrBlock = blocks[screenResult.blockIndex];
    // payload starts at: block fileOffset + 2 (len) + 1 (flag)
    scrFileStart = scrBlock.fileOffset + 3;
    scrFileEnd = scrFileStart + SCR_SIZE;
    scrB64 = bufferToBase64(screenResult.scr);
    const colours = countColours(screenResult.scr);
    canvasHtml = `
      <canvas id="screen" width="768" height="576" data-scr="${scrB64}"></canvas>
      <p class="meta">Rendered from block #${screenResult.blockIndex + 1} — ${colours} colours</p>`;
  }

  // Build block summary + hex dump
  let blocksHtml = '';
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    const isHeader = b.flag === 0x00;
    const isScreen = screenResult?.blockIndex === i;
    const label = isHeader ? 'Header' : 'Data';
    const headerInfo = isHeader ? decodeHeader(b.payload) : null;
    const checksumClass = b.valid ? 'ok' : 'bad';

    blocksHtml += `
      <div class="block${isScreen ? ' screen-block' : ''}">
        <div class="block-header">
          <span class="block-num">#${i + 1}</span>
          <span class="block-type">${label}</span>
          <span>${b.payload.length} bytes</span>
          <span class="checksum ${checksumClass}">${b.valid ? '✓ checksum' : '✗ checksum'}</span>
          ${isScreen ? '<span class="scr-badge">SCR</span>' : ''}
        </div>
        ${headerInfo ? `<div class="header-decoded">${escHtml(headerInfo)}</div>` : ''}
        <div class="hex-dump">${hexRows(data, b.fileOffset, b.fileOffset + b.fileSize, scrFileStart, scrFileEnd)}</div>
      </div>`;
  }

  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             script-src ${webview.cspSource};
             style-src 'unsafe-inline';">
  <title>ZX Viewer — ${escHtml(filename)}</title>
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
      margin: 24px 0 8px;
      border-bottom: 1px solid #333;
      padding-bottom: 4px;
    }
    canvas {
      display: block;
      image-rendering: pixelated;
      border: 1px solid #444;
      margin-bottom: 4px;
    }
    .meta {
      color: #888;
      font-size: 12px;
      margin: 0 0 16px;
    }
    .block {
      margin-bottom: 16px;
    }
    .screen-block {
      border-left: 3px solid #0dd;
      padding-left: 8px;
    }
    .block-header {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 4px;
      font-size: 12px;
    }
    .block-num { color: #888; }
    .block-type { color: #ddd; font-weight: bold; }
    .checksum.ok { color: #6a6; }
    .checksum.bad { color: #f66; }
    .scr-badge {
      background: #0aa;
      color: #000;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 11px;
      font-weight: bold;
    }
    .header-decoded {
      color: #adf;
      font-size: 12px;
      margin-bottom: 4px;
      padding-left: 2px;
    }
    .hex-dump {
      font-family: 'Cascadia Code', 'Fira Code', 'SF Mono', Menlo, monospace;
      font-size: 12px;
      line-height: 1.5;
      overflow-x: auto;
      max-height: 300px;
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
    .scr {
      color: #0ee;
    }
  </style>
</head>
<body>
  <h2>${escHtml(filename)} — ${data.length} bytes, ${blocks.length} block${blocks.length !== 1 ? 's' : ''}</h2>
  ${canvasHtml}
  ${blocksHtml}
  ${screenResult ? `<script src="${scriptUri}"></script>` : ''}
</body>
</html>`;
}

function errorPage(msg: string): string {
  return `<!DOCTYPE html><html><body style="color:#f88;font-family:monospace;padding:16px;">
    <p>${escHtml(msg)}</p></body></html>`;
}

export class TapEditorProvider implements vscode.CustomReadonlyEditorProvider {

  static register(context: vscode.ExtensionContext): vscode.Disposable {
    return vscode.window.registerCustomEditorProvider(
      'zxViewer.tap',
      new TapEditorProvider(context),
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
    webviewPanel.webview.options = { enableScripts: true };

    const raw = await vscode.workspace.fs.readFile(document.uri);
    const data = new Uint8Array(raw);
    const filename = path.basename(document.uri.fsPath);

    const blocks = parseTap(data);

    if (blocks.length === 0) {
      webviewPanel.webview.html = errorPage(
        `No valid TAP blocks found in ${filename}.`
      );
      return;
    }

    const screenResult = findScreenBlock(blocks);

    webviewPanel.webview.html = buildTapHtml(
      webviewPanel.webview,
      this.context,
      filename,
      data,
      blocks,
      screenResult
    );
  }
}
