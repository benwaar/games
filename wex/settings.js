export const STORAGE_KEY = 'wasm-player-settings';

export const PALETTES = {
  default: { bg: '#1a1a2e', bgPanel: '#16213e', border: '#0f3460', text: '#e0e0e0', textMuted: '#888', accent: '#e94560' },
  'high-contrast': { bg: '#000000', bgPanel: '#111111', border: '#ffffff', text: '#ffffff', textMuted: '#cccccc', accent: '#ffff00' },
  deuteranopia: { bg: '#1a1a2e', bgPanel: '#16213e', border: '#0f3460', text: '#e0e0e0', textMuted: '#888', accent: '#4d9de0' },
};

export function applyPalette(name) {
  const p = PALETTES[name] || PALETTES.default;
  const s = document.documentElement.style;
  s.setProperty('--bg', p.bg);
  s.setProperty('--bg-panel', p.bgPanel);
  s.setProperty('--border', p.border);
  s.setProperty('--text', p.text);
  s.setProperty('--text-muted', p.textMuted);
  s.setProperty('--accent', p.accent);
}

export function saveSettings(overrides, paletteSelect, volumeSlider, toggleVoice) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      overrides: { dark: overrides.dark, motion: overrides.motion, contrast: overrides.contrast },
      palette: paletteSelect.value,
      volume: volumeSlider.value,
      voice: toggleVoice.value,
    }));
  } catch (_) {}
}

export function loadSettings(overrides, paletteSelect, volumeLabel, volumeSlider, toggleVoice, applyPaletteFn) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { voiceEnabled: false };
    const saved = JSON.parse(raw);
    if (saved.overrides) {
      overrides.dark     = saved.overrides.dark     ?? null;
      overrides.motion   = saved.overrides.motion   ?? null;
      overrides.contrast = saved.overrides.contrast ?? null;
    }
    if (saved.palette && PALETTES[saved.palette]) {
      paletteSelect.value = saved.palette;
      applyPaletteFn(saved.palette);
    }
    if (saved.volume !== undefined) {
      volumeSlider.value = saved.volume;
      volumeLabel.textContent = saved.volume + '%';
    }
    let voiceEnabled = false;
    if (saved.voice) {
      toggleVoice.value = saved.voice;
      voiceEnabled = saved.voice === 'on';
    }
    return { voiceEnabled };
  } catch (_) {
    return { voiceEnabled: false };
  }
}

export function updateBrowserLabels(mqDark, mqMotion, mqContrast) {
  document.getElementById('browser-dark').textContent     = `Browser: ${mqDark.matches ? 'dark' : 'light'}`;
  document.getElementById('browser-motion').textContent   = `Browser: ${mqMotion.matches ? 'reduced' : 'full'}`;
  document.getElementById('browser-contrast').textContent = `Browser: ${mqContrast.matches ? 'more' : 'standard'}`;
}

export function initToggles(overrides, saveSettingsFn, reloadWasmFn) {
  const toggleDark     = document.getElementById('toggle-dark');
  const toggleMotion   = document.getElementById('toggle-motion');
  const toggleContrast = document.getElementById('toggle-contrast');

  const toSelectVal = v => v === null ? 'auto' : v ? 'on' : 'off';
  toggleDark.value     = toSelectVal(overrides.dark);
  toggleMotion.value   = toSelectVal(overrides.motion);
  toggleContrast.value = toSelectVal(overrides.contrast);

  function handleToggle(select, key) {
    const v = select.value;
    overrides[key] = v === 'auto' ? null : v === 'on' ? 1 : 0;
    saveSettingsFn();
    reloadWasmFn();
  }

  toggleDark.addEventListener('change',     () => handleToggle(toggleDark,     'dark'));
  toggleMotion.addEventListener('change',   () => handleToggle(toggleMotion,   'motion'));
  toggleContrast.addEventListener('change', () => handleToggle(toggleContrast, 'contrast'));
}
