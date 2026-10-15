import { speak, buildMirrorDom } from './a11y.js';
import { applyPalette, saveSettings, loadSettings, updateBrowserLabels, initToggles } from './settings.js';
import { spatialAudio, scheduleAudio, togglePause, startAudio } from './audio.js';

const canvas = document.getElementById('screen');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const pauseBtn = document.getElementById('pause-btn');
const overlay = document.getElementById('autoplay-overlay');
const mirror = document.getElementById('mirror');

const settingsBtn   = document.getElementById('settings-btn');
const settingsPanel = document.getElementById('settings-panel');
const settingsClose = document.getElementById('settings-close');
const paletteSelect = document.getElementById('palette-select');
const volumeSlider  = document.getElementById('volume-slider');
const volumeLabel   = document.getElementById('volume-label');
const toggleVoice   = document.getElementById('toggle-voice');

const overrides = { dark: null, motion: null, contrast: null };

const mqDark     = matchMedia('(prefers-color-scheme: dark)');
const mqMotion   = matchMedia('(prefers-reduced-motion: reduce)');
const mqContrast = matchMedia('(prefers-contrast: more)');

let ac          = null;
let masterGain  = null;
let voiceEnabled = false;
let playing     = false;
let audioNotes  = null;
let kbAbort     = null;

// Settings panel open/close
settingsBtn.addEventListener('click', () => {
  const open = !settingsPanel.hidden;
  settingsPanel.hidden = open;
  settingsBtn.setAttribute('aria-expanded', String(!open));
  if (!open) settingsClose.focus();
});

settingsClose.addEventListener('click', () => {
  settingsPanel.hidden = true;
  settingsBtn.setAttribute('aria-expanded', 'false');
  settingsBtn.focus();
});

paletteSelect.addEventListener('change', () => {
  applyPalette(paletteSelect.value);
  saveSettings(overrides, paletteSelect, volumeSlider, toggleVoice);
});

volumeSlider.addEventListener('input', () => {
  const val = volumeSlider.value;
  volumeLabel.textContent = val + '%';
  if (masterGain) masterGain.gain.value = val / 100;
  saveSettings(overrides, paletteSelect, volumeSlider, toggleVoice);
});

toggleVoice.addEventListener('change', () => {
  voiceEnabled = toggleVoice.value === 'on';
  saveSettings(overrides, paletteSelect, volumeSlider, toggleVoice);
});

mqDark.addEventListener('change',     () => updateBrowserLabels(mqDark, mqMotion, mqContrast));
mqMotion.addEventListener('change',   () => updateBrowserLabels(mqDark, mqMotion, mqContrast));
mqContrast.addEventListener('change', () => updateBrowserLabels(mqDark, mqMotion, mqContrast));

pauseBtn.addEventListener('click', () => {
  playing = togglePause(ac, playing, pauseBtn);
});
document.addEventListener('keydown', e => {
  if (e.code === 'KeyP') playing = togglePause(ac, playing, pauseBtn);
});

function drawNoContent(canvas, ctx) {
  canvas.width = 256; canvas.height = 256;
  canvas.style.aspectRatio = '1 / 1';
  ctx.strokeStyle = '#0055ff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(256,256); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(256,0); ctx.lineTo(0,256); ctx.stroke();
}

async function reloadWasm() {
  if (ac) { try { ac.close(); } catch (_) {} ac = null; }
  masterGain = null;
  if (kbAbort) { kbAbort.abort(); kbAbort = null; }
  playing = false;
  audioNotes = null;
  pauseBtn.style.display = 'none';
  mirror.innerHTML = '';
  status.textContent = 'Reloading…';
  await load();
}

async function load() {
  let instance;
  try {
    const res = await fetch('./debug.wasm');
    if (!res.ok) {
      drawNoContent(canvas, ctx);
      status.textContent = 'No debug.wasm — run pipeline first';
      return;
    }
    const bytes = await res.arrayBuffer();
    const imports = {
      env: {
        prefers_dark:           () => overrides.dark     ?? (mqDark.matches     ? 1 : 0),
        prefers_reduced_motion: () => overrides.motion   ?? (mqMotion.matches   ? 1 : 0),
        prefers_high_contrast:  () => overrides.contrast ?? (mqContrast.matches ? 1 : 0),
      }
    };
    ({ instance } = await WebAssembly.instantiate(bytes, imports));
  } catch (err) {
    console.warn('WASM load error:', err);
    drawNoContent(canvas, ctx);
    status.textContent = err instanceof WebAssembly.LinkError
      ? `WASM import error: ${err.message}`
      : 'No debug.wasm — run pipeline first';
    return;
  }

  const exp = instance.exports;
  const mem = exp.memory?.buffer;

  if (typeof exp.type !== 'function') {
    const w = exp.width(); const h = exp.height();
    canvas.width = w; canvas.height = h;
    canvas.style.aspectRatio = `${w} / ${h}`;
    exp.render();
    ctx.putImageData(new ImageData(new Uint8ClampedArray(mem, 0, w * h * 4), w, h), 0, 0);
    status.textContent = `Image: ${w}×${h} (legacy)`;
    return;
  }

  const type     = exp.type();
  const hasImage = type & 1;
  const hasAudio = type & 2;
  const parts    = [];

  if (hasImage) {
    const w = exp.width(); const h = exp.height();
    canvas.width = w; canvas.height = h;
    canvas.style.aspectRatio = `${w} / ${h}`;
    exp.render();
    ctx.putImageData(new ImageData(new Uint8ClampedArray(mem, 0, w * h * 4), w, h), 0, 0);
    parts.push(`Image: ${w}×${h}`);
  } else {
    drawNoContent(canvas, ctx);
    parts.push('No image');
  }

  if (hasAudio) {
    const count = exp.audio_count();
    const off   = exp.audio_offset();
    const freqs = new Float64Array(mem, off, count);
    const durs  = new Float64Array(mem, off + count * 8, count);
    audioNotes  = { freqs: Array.from(freqs), durs: Array.from(durs) };
    parts.push(`Audio: ${count} notes`);
    pauseBtn.style.display = 'block';
  }

  if (type & 4) {
    kbAbort = new AbortController();
    const sig = { signal: kbAbort.signal };
    document.addEventListener('keydown', e => { /* TODO: store key state in WASM memory */ }, sig);
    document.addEventListener('keyup',   e => { /* TODO: clear key state in WASM memory */ }, sig);
    parts.push('Keyboard: active');
  }

  if (type & 8) {
    const off  = exp.a11y_offset();
    const size = exp.a11y_size();
    const raw  = new Uint8Array(mem, off, size);
    try {
      const nodes = JSON.parse(new TextDecoder().decode(raw));
      const cw = hasImage ? exp.width()  : 256;
      const ch = hasImage ? exp.height() : 256;
      buildMirrorDom(mirror, canvas, nodes, cw, ch, voiceEnabled, speak);
      if (voiceEnabled && nodes.length > 0) speak(nodes[0].label || '');
      parts.push(`A11y: ${nodes.length} nodes`);
      if (audioNotes && nodes.length > 0) {
        audioNotes.positions = nodes.map(n =>
          n.x !== undefined ? { x: n.x, y: n.y || 0 } : null
        );
        if (audioNotes.positions.some(p => p !== null)) {
          const audioIdx = parts.findIndex(p => p.startsWith('Audio:'));
          if (audioIdx >= 0) parts[audioIdx] += ' (spatial)';
        }
      }
    } catch (_) {
      parts.push('A11y: parse error');
    }
  }

  if (hasAudio) {
    const result = await startAudio(audioNotes, overlay, (ac, masterGain) => {
      playing = scheduleAudio(ac, masterGain, audioNotes, canvas, volumeSlider, pauseBtn, spatialAudio);
    });
    ac         = result.ac;
    masterGain = result.masterGain;
  }

  status.textContent = parts.join(' | ');
}

// Init
const { voiceEnabled: ve } = loadSettings(overrides, paletteSelect, volumeLabel, volumeSlider, toggleVoice, applyPalette);
voiceEnabled = ve;
initToggles(overrides, () => saveSettings(overrides, paletteSelect, volumeSlider, toggleVoice), reloadWasm);
updateBrowserLabels(mqDark, mqMotion, mqContrast);
load();
