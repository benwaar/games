// viewer.js — runs inside the VS Code webview.
// Receives SCR data as base64 via window.__SCR_B64__, decodes it, and renders to canvas.
// No build step — plain ES2020 JS.

(function () {
  'use strict';

  const SCREEN_WIDTH = 256;
  const SCREEN_HEIGHT = 192;
  const BITMAP_SIZE = 6144;
  const SCR_SIZE = 6912;
  const SCALE = 3;

  // Spectrum palette: indices 0-7 normal, 8-15 bright.
  // Bright black (index 8) stays (0,0,0).
  const PALETTE = [
    [0, 0, 0],         // 0 black
    [0, 0, 0xCD],      // 1 blue
    [0xCD, 0, 0],      // 2 red
    [0xCD, 0, 0xCD],   // 3 magenta
    [0, 0xCD, 0],      // 4 green
    [0, 0xCD, 0xCD],   // 5 cyan
    [0xCD, 0xCD, 0],   // 6 yellow
    [0xCD, 0xCD, 0xCD],// 7 white
    [0, 0, 0],         // 8 bright black
    [0, 0, 0xFF],      // 9 bright blue
    [0xFF, 0, 0],      // 10 bright red
    [0xFF, 0, 0xFF],   // 11 bright magenta
    [0, 0xFF, 0],      // 12 bright green
    [0, 0xFF, 0xFF],   // 13 bright cyan
    [0xFF, 0xFF, 0],   // 14 bright yellow
    [0xFF, 0xFF, 0xFF],// 15 bright white
  ];

  /**
   * Byte offset within bitmap for pixel column x, row y.
   * Interleaved layout: thirds → scan lines within char row → char rows within third → byte column.
   */
  function pixelOffset(x, y) {
    return ((y & 0xC0) << 5)
         | ((y & 0x07) << 8)
         | ((y & 0x38) << 2)
         | (x >> 3);
  }

  function base64ToUint8Array(b64) {
    const binary = atob(b64);
    const buf = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      buf[i] = binary.charCodeAt(i);
    }
    return buf;
  }

  function render(scr) {
    const canvas = document.getElementById('screen');
    if (!canvas) { return; }

    canvas.width  = SCREEN_WIDTH  * SCALE;
    canvas.height = SCREEN_HEIGHT * SCALE;

    const ctx = canvas.getContext('2d');
    // Work at native resolution, then scale via CSS is not needed —
    // we draw 1:1 to an ImageData then scale when putting.
    const imgData = ctx.createImageData(SCREEN_WIDTH, SCREEN_HEIGHT);
    const px = imgData.data;

    for (let y = 0; y < SCREEN_HEIGHT; y++) {
      for (let x = 0; x < SCREEN_WIDTH; x++) {
        const byteOff = pixelOffset(x, y);
        const bitmapByte = scr[byteOff];
        const bitIndex   = 7 - (x & 7); // MSB = leftmost pixel
        const pixelSet   = (bitmapByte >> bitIndex) & 1;

        const attrRow = (y >> 3);
        const attrCol = (x >> 3);
        const attrOff = BITMAP_SIZE + attrRow * 32 + attrCol;
        const attr    = scr[attrOff];

        const bright = (attr & 0x40) !== 0 ? 8 : 0;
        const paper  = ((attr >> 3) & 0x07) + bright;
        const ink    = (attr & 0x07) + bright;
        const colour = PALETTE[pixelSet ? ink : paper];

        const idx = (y * SCREEN_WIDTH + x) * 4;
        px[idx]     = colour[0];
        px[idx + 1] = colour[1];
        px[idx + 2] = colour[2];
        px[idx + 3] = 255;
      }
    }

    // Draw native-size image into a temp canvas, then scale up.
    const tmp = document.createElement('canvas');
    tmp.width  = SCREEN_WIDTH;
    tmp.height = SCREEN_HEIGHT;
    tmp.getContext('2d').putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE);
  }

  const canvas = document.getElementById('screen');
  const b64 = canvas ? canvas.getAttribute('data-scr') : null;
  if (!b64) {
    document.body.insertAdjacentHTML('afterbegin',
      '<p style="color:#f88">No screen data received.</p>');
    return;
  }

  const scr = base64ToUint8Array(b64);

  if (scr.length !== SCR_SIZE) {
    document.body.insertAdjacentHTML('afterbegin',
      `<p style="color:#f88">Unexpected data length: ${scr.length} bytes (expected ${SCR_SIZE}).</p>`);
    return;
  }

  render(scr);
}());
