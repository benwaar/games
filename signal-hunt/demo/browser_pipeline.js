/**
 * browser_pipeline.js — Signal Hunt Mel spectrogram pipeline for the browser.
 *
 * Mirrors signal-hunt/pipeline/features.py exactly:
 *   - STFT: n_fft=2048, hop_length=512, Hann window
 *   - Mel filterbank: 128 bands, Slaney scale, fmin=0, fmax=sr/2
 *   - Power → log-dB: 10 * log10(max(power, 1e-10))
 *   - Per-spectrogram normalise: (x - mean) / std
 *   - Pad or truncate to TARGET_FRAMES=65
 *
 * Usage:
 *   import { extractFeatures, buildMelFilterbank } from './browser_pipeline.js';
 *   const mel = extractFeatures(float32Samples, sampleRate);
 *   // mel is Float32Array of shape [N_MELS * TARGET_FRAMES] = [128 * 65 = 8320]
 */

export const N_FFT = 2048;
export const HOP_LENGTH = 512;
export const N_MELS = 128;
export const TARGET_FRAMES = 65;

// ---------------------------------------------------------------------------
// Slaney Mel scale — matches librosa with htk=False (default)
// ---------------------------------------------------------------------------

function hzToMel(hz) {
  const fSp = 200.0 / 3.0;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27.0;
  return hz >= minLogHz
    ? minLogMel + Math.log(hz / minLogHz) / logStep
    : hz / fSp;
}

function melToHz(mel) {
  const fSp = 200.0 / 3.0;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27.0;
  return mel >= minLogMel
    ? minLogHz * Math.exp(logStep * (mel - minLogMel))
    : fSp * mel;
}

// ---------------------------------------------------------------------------
// Mel filterbank — librosa.filters.mel(sr, n_fft, n_mels, norm='slaney')
// Returns Float32Array of shape [nMels × nFreqs] stored row-major.
// ---------------------------------------------------------------------------

export function buildMelFilterbank(sr, nFft = N_FFT, nMels = N_MELS) {
  const nFreqs = nFft / 2 + 1;
  const fmax = sr / 2;

  // FFT bin frequencies in Hz.
  const fftFreqs = new Float32Array(nFreqs);
  for (let k = 0; k < nFreqs; k++) fftFreqs[k] = k * sr / nFft;

  // n_mels + 2 points evenly spaced in Mel → back to Hz.
  const melMin = hzToMel(0);
  const melMax = hzToMel(fmax);
  const hzPoints = new Float32Array(nMels + 2);
  for (let i = 0; i <= nMels + 1; i++) {
    hzPoints[i] = melToHz(melMin + (i / (nMels + 1)) * (melMax - melMin));
  }

  // Build triangular filters in Hz space — matches librosa.filters.mel exactly.
  const fb = new Float32Array(nMels * nFreqs);
  for (let m = 0; m < nMels; m++) {
    const lower = hzPoints[m];
    const centre = hzPoints[m + 1];
    const upper = hzPoints[m + 2];

    for (let k = 0; k < nFreqs; k++) {
      const f = fftFreqs[k];
      const rising = (f - lower) / (centre - lower);
      const falling = (upper - f) / (upper - centre);
      fb[m * nFreqs + k] = Math.max(0, Math.min(rising, falling));
    }

    // Slaney normalisation: 2 / bandwidth.
    const bandwidth = upper - lower;
    if (bandwidth > 0) {
      const norm = 2.0 / bandwidth;
      for (let k = 0; k < nFreqs; k++) fb[m * nFreqs + k] *= norm;
    }
  }
  return fb;
}

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------

function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

// ---------------------------------------------------------------------------
// Radix-2 Cooley-Tukey FFT (in-place, complex interleaved [re, im, re, im, ...])
// n must be a power of 2.
// ---------------------------------------------------------------------------

function fft(re, im) {
  const n = re.length;
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // Butterfly passes.
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
        const vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

// Compute power spectrum of a single windowed frame: |FFT|^2
// Uses Float64 internally to preserve small sidelobe values (matches numpy).
// Returns Float32Array of shape [n_fft/2 + 1].
function powerSpectrum(frame, window) {
  const n = frame.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = frame[i] * window[i];
  fft(re, im);
  const nFreqs = n / 2 + 1;
  const power = new Float32Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) {
    power[i] = re[i] * re[i] + im[i] * im[i];
  }
  return power;
}

// ---------------------------------------------------------------------------
// Full pipeline — matches pipeline/features.py extract_features()
// ---------------------------------------------------------------------------

/**
 * @param {Float32Array} signal - mono PCM samples at 22050 Hz
 * @param {Float32Array} filterbank - from buildMelFilterbank() [nMels × nFreqs]
 * @returns {Float32Array} flat [N_MELS × TARGET_FRAMES] Mel spectrogram
 */
export function extractFeatures(signal, filterbank) {
  const nFreqs = N_FFT / 2 + 1;
  const window = hannWindow(N_FFT);

  // Pad by n_fft/2 on each side (matches librosa center=True, pad_mode='constant').
  const pad = N_FFT >> 1;
  const centered = new Float32Array(signal.length + 2 * pad);
  centered.set(signal, pad);

  // STFT: slide over padded signal with hop_length, compute power spectrum per frame.
  const numFrames = Math.floor((centered.length - N_FFT) / HOP_LENGTH) + 1;
  const melFrames = new Float32Array(N_MELS * numFrames);

  for (let f = 0; f < numFrames; f++) {
    const start = f * HOP_LENGTH;
    const frame = centered.slice(start, start + N_FFT);
    const power = powerSpectrum(frame, window);

    // Apply Mel filterbank: dot product of filterbank row with power spectrum.
    for (let m = 0; m < N_MELS; m++) {
      let val = 0;
      for (let k = 0; k < nFreqs; k++) {
        val += filterbank[m * nFreqs + k] * power[k];
      }
      melFrames[m * numFrames + f] = val;
    }
  }

  // Power → log-dB: 10 * log10(max(x, 1e-10)), then clamp to (max - 80) dB.
  // Matches librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=80).
  const logMel = new Float32Array(N_MELS * numFrames);
  let dbMax = -Infinity;
  for (let i = 0; i < logMel.length; i++) {
    const db = 10 * Math.log10(Math.max(melFrames[i], 1e-10));
    logMel[i] = db;
    if (db > dbMax) dbMax = db;
  }
  const dbFloor = dbMax - 80;
  for (let i = 0; i < logMel.length; i++) {
    if (logMel[i] < dbFloor) logMel[i] = dbFloor;
  }

  // Pad or truncate to TARGET_FRAMES (column-wise).
  const padded = new Float32Array(N_MELS * TARGET_FRAMES);
  const framesToCopy = Math.min(numFrames, TARGET_FRAMES);
  for (let m = 0; m < N_MELS; m++) {
    for (let f = 0; f < framesToCopy; f++) {
      padded[m * TARGET_FRAMES + f] = logMel[m * numFrames + f];
    }
    // Remaining frames are already 0 (padding).
  }

  // Per-spectrogram normalise: (x - mean) / std
  let sum = 0;
  for (let i = 0; i < padded.length; i++) sum += padded[i];
  const mean = sum / padded.length;
  let sumSq = 0;
  for (let i = 0; i < padded.length; i++) sumSq += (padded[i] - mean) ** 2;
  const std = Math.sqrt(sumSq / padded.length);

  const normed = new Float32Array(padded.length);
  if (std < 1e-8) {
    return normed; // All zeros — silent signal.
  }
  for (let i = 0; i < padded.length; i++) {
    normed[i] = (padded[i] - mean) / std;
  }

  return normed;
}
