// audio.js — beeper audio playback for zxw emulator.
// Feeds f32 samples from the Rust emulator into Web Audio via a ring buffer.

const SAMPLE_RATE = 44100;
const BUFFER_FRAMES = 2048;

let audioCtx = null;
let scriptNode = null;
let sampleQueue = [];
let muted = false;

export function initAudio() {
  if (audioCtx) { return audioCtx; }
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  scriptNode = audioCtx.createScriptProcessor(BUFFER_FRAMES, 0, 1);
  scriptNode.onaudioprocess = (event) => {
    const output = event.outputBuffer.getChannelData(0);
    const available = sampleQueue.length;
    const count = Math.min(output.length, available);

    if (muted || count === 0) {
      output.fill(0);
      return;
    }

    for (let i = 0; i < count; i++) {
      output[i] = sampleQueue[i];
    }
    for (let i = count; i < output.length; i++) {
      output[i] = sampleQueue.length > 0 ? sampleQueue[count - 1] : 0;
    }
    sampleQueue.splice(0, count);
  };
  scriptNode.connect(audioCtx.destination);
  return audioCtx;
}

export function feedAudio(samples) {
  if (!audioCtx || !samples || samples.length === 0) { return; }
  // Prevent queue from growing unbounded if browser tab is background
  if (sampleQueue.length > SAMPLE_RATE) {
    sampleQueue.length = 0;
  }
  for (let i = 0; i < samples.length; i++) {
    sampleQueue.push(samples[i]);
  }
}

export function resumeAudio() {
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

export function toggleMute() {
  muted = !muted;
  return muted;
}

export function isMuted() {
  return muted;
}

export function audioStats() {
  return {
    state: audioCtx ? audioCtx.state : 'no context',
    queueLength: sampleQueue.length,
    muted,
    sampleRate: audioCtx ? audioCtx.sampleRate : 0,
  };
}
