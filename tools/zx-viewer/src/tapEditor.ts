import * as vscode from 'vscode';
import * as path from 'path';
import { countColours, SCR_SIZE } from './renderer';
import { buildHtml } from './scrEditor';

interface TapBlock {
  flag: number;
  payload: Uint8Array;
  valid: boolean; // checksum ok
}

/** Parse a TAP file into its blocks. Returns empty array on malformed input. */
function parseTap(data: Uint8Array): TapBlock[] {
  const blocks: TapBlock[] = [];
  let offset = 0;

  while (offset + 2 <= data.length) {
    const blockLen = data[offset] | (data[offset + 1] << 8);
    offset += 2;

    if (blockLen < 2 || offset + blockLen > data.length) {
      break; // truncated
    }

    const flag = data[offset];
    const payload = data.slice(offset + 1, offset + blockLen - 1);
    const checksum = data[offset + blockLen - 1];

    // Verify checksum: XOR of flag + all payload bytes
    let computed = flag;
    for (let i = 0; i < payload.length; i++) {
      computed ^= payload[i];
    }

    blocks.push({ flag, payload, valid: computed === checksum });
    offset += blockLen;
  }

  return blocks;
}

/**
 * Find the first data block (flag 0xFF) with a 6912-byte payload.
 * Returns { scr, blockIndex } or null.
 */
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

    const result = findScreenBlock(blocks);

    if (!result) {
      webviewPanel.webview.html = errorPage(
        `No screen data found in ${filename}. ` +
        `${blocks.length} block(s) parsed — none contain a 6912-byte data block.`
      );
      return;
    }

    const { scr, blockIndex } = result;
    const colours = countColours(scr);
    const b64 = bufferToBase64(scr);

    const flagSummary = blocks
      .map((b, i) => `#${i + 1}: ${b.flag === 0x00 ? 'header' : 'data'} (${b.payload.length}B)${i === blockIndex ? ' [screen]' : ''}`)
      .join(', ');

    webviewPanel.webview.html = buildHtml(
      webviewPanel.webview,
      this.context,
      b64,
      {
        format: 'TAP',
        fileSize: data.length,
        colours,
        filename,
        blocks: `${blocks.length} blocks — ${flagSummary}`
      }
    );
  }
}
