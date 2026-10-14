import * as vscode from 'vscode';
import * as path from 'path';
import { countColours, SCR_SIZE } from './renderer';

export class ScrEditorProvider implements vscode.CustomReadonlyEditorProvider {

  static register(context: vscode.ExtensionContext): vscode.Disposable {
    return vscode.window.registerCustomEditorProvider(
      'zxViewer.scr',
      new ScrEditorProvider(context),
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
    const scr = new Uint8Array(raw);

    if (scr.length !== SCR_SIZE) {
      webviewPanel.webview.html = errorHtml(
        `Invalid SCR file: expected ${SCR_SIZE} bytes, got ${scr.length}.`
      );
      return;
    }

    const colours = countColours(scr);
    const b64 = bufferToBase64(scr);
    const filename = path.basename(document.uri.fsPath);

    webviewPanel.webview.html = buildHtml(
      webviewPanel.webview,
      this.context,
      b64,
      { format: 'SCR', fileSize: scr.length, colours, filename, blocks: null }
    );
  }
}

function bufferToBase64(buf: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < buf.length; i++) {
    binary += String.fromCharCode(buf[i]);
  }
  return Buffer.from(binary, 'binary').toString('base64');
}

interface InfoMeta {
  format: string;
  fileSize: number;
  colours: number;
  filename: string;
  blocks: string | null;
}

export function buildHtml(
  webview: vscode.Webview,
  context: vscode.ExtensionContext,
  scrBase64: string,
  meta: InfoMeta
): string {
  const scriptUri = webview.asWebviewUri(
    vscode.Uri.joinPath(context.extensionUri, 'media', 'viewer.js')
  );

  const blocksRow = meta.blocks
    ? `<tr><td>Blocks</td><td>${escHtml(meta.blocks)}</td></tr>`
    : '';

  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             script-src ${webview.cspSource};
             style-src 'unsafe-inline';">
  <title>ZX Viewer</title>
  <style>
    body {
      background: #1e1e1e;
      color: #ccc;
      font-family: var(--vscode-font-family, monospace);
      font-size: 13px;
      margin: 0;
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
    }
    canvas {
      display: block;
      image-rendering: pixelated;
      border: 1px solid #444;
    }
    table {
      margin-top: 12px;
      border-collapse: collapse;
    }
    td {
      padding: 2px 12px 2px 0;
    }
    td:first-child {
      color: #888;
    }
  </style>
</head>
<body>
  <canvas id="screen" width="768" height="576"></canvas>
  <table>
    <tr><td>File</td><td>${escHtml(meta.filename)}</td></tr>
    <tr><td>Format</td><td>${escHtml(meta.format)}</td></tr>
    <tr><td>Size</td><td>${meta.fileSize} bytes</td></tr>
    <tr><td>Colours</td><td>${meta.colours}</td></tr>
    ${blocksRow}
  </table>
  <script>
    window.__SCR_B64__ = "${scrBase64}";
  </script>
  <script src="${scriptUri}"></script>
</body>
</html>`;
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function errorHtml(msg: string): string {
  return `<!DOCTYPE html><html><body style="color:#f88;font-family:monospace;padding:16px;">
    <p>${escHtml(msg)}</p></body></html>`;
}
