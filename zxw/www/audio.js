// audio.js — beeper audio playback for zxw emulator.
// Feeds f32 samples from the Rust emulator into an AudioWorklet via a SAB ring buffer.

const SAMPLE_RATE = 44100;
const RING_SIZE = 16384;
const RING_MASK = RING_SIZE - 1;

let audioCtx = null;
let workletNode = null;
let ringBuffer = null;    // Float32Array over SAB
let controlBuffer = null; // Uint32Array over SAB: [writeIdx, readIdx]
let muted = false;
let ready = false;

export async function initAudio() {
  if (audioCtx) return audioCtx;

  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioCtx.audioWorklet.addModule('audio-worklet.js');

  const ringSAB = new SharedArrayBuffer(RING_SIZE * 4);
  const controlSAB = new SharedArrayBuffer(8);
  ringBuffer = new Float32Array(ringSAB);
  controlBuffer = new Uint32Array(controlSAB);

  workletNode = new AudioWorkletNode(audioCtx, 'beeper-processor');
  workletNode.port.postMessage({
    type: 'init',
    ringBuffer: ringSAB,
    controlBuffer: controlSAB,
  });
  workletNode.connect(audioCtx.destination);
  ready = true;
  return audioCtx;
}

export function feedAudio(samples) {
  if (!ready || !samples || samples.length === 0) return;

  let writeIdx = Atomics.load(controlBuffer, 0);
  const readIdx = Atomics.load(controlBuffer, 1);
  const available = RING_SIZE - ((writeIdx - readIdx) & 0xFFFFFFFF);

  if (samples.length >= available) {
    // would overflow — drop oldest by advancing read pointer
    const overflow = samples.length - available + 1;
    Atomics.store(controlBuffer, 1, (readIdx + overflow) >>> 0);
  }

  for (let i = 0; i < samples.length; i++) {
    ringBuffer[(writeIdx + i) & RING_MASK] = samples[i];
  }
  Atomics.store(controlBuffer, 0, (writeIdx + samples.length) >>> 0);
}

export async function resumeAudio() {
  if (audioCtx && audioCtx.state === 'suspended') {
    await audioCtx.resume();
  }
}

export function toggleMute() {
  muted = !muted;
  if (workletNode) {
    workletNode.port.postMessage({ type: 'mute', value: muted });
  }
  return muted;
}

export function isMuted() {
  return muted;
}

export function audioStats() {
  if (!ready) {
    return { state: 'no context', fill: 0, muted, sampleRate: 0 };
  }
  const writeIdx = Atomics.load(controlBuffer, 0);
  const readIdx = Atomics.load(controlBuffer, 1);
  const fill = (writeIdx - readIdx) & 0xFFFFFFFF;
  return {
    state: audioCtx.state,
    fill,
    fillMs: ((fill / SAMPLE_RATE) * 1000) | 0,
    muted,
    sampleRate: audioCtx.sampleRate,
  };
}
