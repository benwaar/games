// audio-worklet.js — AudioWorkletProcessor for zxw beeper.
// Reads f32 samples from a SharedArrayBuffer ring buffer.

const RING_MASK = 16383; // RING_SIZE - 1

class BeeperProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ringBuffer = null;
    this.controlBuffer = null;
    this.muted = false;

    this.port.onmessage = (e) => {
      if (e.data.type === 'init') {
        this.ringBuffer = new Float32Array(e.data.ringBuffer);
        this.controlBuffer = new Uint32Array(e.data.controlBuffer);
      } else if (e.data.type === 'mute') {
        this.muted = e.data.value;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;

    if (!this.ringBuffer || !this.controlBuffer || this.muted) {
      output.fill(0);
      return true;
    }

    const writeIdx = Atomics.load(this.controlBuffer, 0);
    let readIdx = Atomics.load(this.controlBuffer, 1);
    const available = (writeIdx - readIdx) & 0xFFFFFFFF;
    const count = Math.min(output.length, available);

    for (let i = 0; i < count; i++) {
      output[i] = this.ringBuffer[(readIdx + i) & RING_MASK];
    }
    for (let i = count; i < output.length; i++) {
      output[i] = 0;
    }

    Atomics.store(this.controlBuffer, 1, (readIdx + count) >>> 0);
    return true;
  }
}

registerProcessor('beeper-processor', BeeperProcessor);
