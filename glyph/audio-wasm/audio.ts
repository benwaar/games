// Glyph WASM Audio Engine
// Web Audio API layer — the "luxury" side of the HAL.
// Z80 target gets 1-bit beeper; WASM gets real oscillators, ADSR, and scheduled sequences.

export type NoteName = string;

export interface ScheduledNote {
  note: NoteName;
  ms: number;
}

export interface AudioEngine {
  init: () => void;
  tonePlay: (note: NoteName, durationMs: number) => void;
  sfxPlay: (sfxType: SfxType) => void;
  melodyPlay: (notes: ScheduledNote[]) => void;
  melodyStop: () => void;
  setVolume: (level: number) => void;
}

export type SfxType = "sweep_up" | "sweep_down" | "noise_burst" | "coin" | "explosion";

// --- Frequency table ---
// A4 = 440 Hz, equal temperament: freq(n) = 440 * 2^((n - 69) / 12)
// MIDI note 60 = C4, 69 = A4
const MIDI_NOTE_NUMBERS: Record<string, number> = {
  C: 0, "C#": 1, Db: 1,
  D: 2, "D#": 3, Eb: 3,
  E: 4,
  F: 5, "F#": 6, Gb: 6,
  G: 7, "G#": 8, Ab: 8,
  A: 9, "A#": 10, Bb: 10,
  B: 11,
};

export function noteToFrequency(noteName: NoteName): number | null {
  const parsed = parseNoteName(noteName);
  if (parsed === null) return null;
  const { pitchClass, octave } = parsed;
  const semitone = MIDI_NOTE_NUMBERS[pitchClass];
  if (semitone === undefined) return null;
  // MIDI number: C4 = 60 = (4+1)*12 = 60
  const midiNumber = (octave + 1) * 12 + semitone;
  return 440 * Math.pow(2, (midiNumber - 69) / 12);
}

interface ParsedNote {
  pitchClass: string;
  octave: number;
}

export function parseNoteName(noteName: NoteName): ParsedNote | null {
  // Matches: C4, C#4, Db4, A#3, Bb5 etc.
  const match = noteName.match(/^([A-G][#b]?)(\d)$/);
  if (!match) return null;
  const pitchClass = match[1];
  const octave = parseInt(match[2], 10);
  if (octave < 0 || octave > 9) return null;
  if (!(pitchClass in MIDI_NOTE_NUMBERS)) return null;
  return { pitchClass, octave };
}

// Pre-built frequency table for C3–C6
export const FREQ_TABLE: Record<string, number> = (() => {
  // Include both sharp and flat spellings so callers can use either
  const pitchClasses = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"];
  const table: Record<string, number> = {};
  for (let octave = 3; octave <= 6; octave++) {
    for (const pc of pitchClasses) {
      const name = `${pc}${octave}`;
      const freq = noteToFrequency(name);
      if (freq !== null) table[name] = freq;
    }
  }
  return table;
})();

// --- ADSR envelope constants ---
const ADSR_ATTACK_S = 0.01;    // 10ms
const ADSR_DECAY_S = 0.05;     // 50ms
const ADSR_SUSTAIN_LEVEL = 0.7;
const ADSR_RELEASE_S = 0.1;    // 100ms

// --- Engine factory ---

let ctx: AudioContext | null = null;
let masterGain: GainNode | null = null;
let activeOscillators: OscillatorNode[] = [];
let melodyTimeoutId: ReturnType<typeof setTimeout> | null = null;

export function createAudioEngine(): AudioEngine {
  function init(): void {
    if (ctx) return;
    ctx = new AudioContext();
    masterGain = ctx.createGain();
    masterGain.gain.value = 0.8;
    masterGain.connect(ctx.destination);
  }

  function getContext(): { ctx: AudioContext; master: GainNode } {
    if (!ctx || !masterGain) {
      throw new Error("AudioEngine not initialised — call init() first from a user gesture");
    }
    return { ctx, master: masterGain };
  }

  function scheduleSquareTone(
    frequency: number,
    startTime: number,
    durationSec: number,
  ): OscillatorNode {
    const { ctx: audioCtx, master } = getContext();

    const osc = audioCtx.createOscillator();
    osc.type = "square";
    osc.frequency.value = frequency;

    const envGain = audioCtx.createGain();
    envGain.gain.setValueAtTime(0, startTime);
    envGain.gain.linearRampToValueAtTime(1, startTime + ADSR_ATTACK_S);
    envGain.gain.linearRampToValueAtTime(ADSR_SUSTAIN_LEVEL, startTime + ADSR_ATTACK_S + ADSR_DECAY_S);
    const releaseStart = Math.max(
      startTime + ADSR_ATTACK_S + ADSR_DECAY_S,
      startTime + durationSec - ADSR_RELEASE_S,
    );
    envGain.gain.setValueAtTime(ADSR_SUSTAIN_LEVEL, releaseStart);
    envGain.gain.linearRampToValueAtTime(0, releaseStart + ADSR_RELEASE_S);

    osc.connect(envGain);
    envGain.connect(master);

    osc.start(startTime);
    osc.stop(releaseStart + ADSR_RELEASE_S + 0.01);

    activeOscillators.push(osc);
    osc.onended = () => {
      activeOscillators = activeOscillators.filter(o => o !== osc);
    };

    return osc;
  }

  function tonePlay(note: NoteName, durationMs: number): void {
    const { ctx: audioCtx } = getContext();
    const freq = noteToFrequency(note);
    if (freq === null) {
      console.warn(`tonePlay: unknown note "${note}"`);
      return;
    }
    scheduleSquareTone(freq, audioCtx.currentTime, durationMs / 1000);
  }

  function sfxPlay(sfxType: SfxType): void {
    const { ctx: audioCtx, master } = getContext();
    const now = audioCtx.currentTime;

    switch (sfxType) {
      case "sweep_up": {
        const osc = audioCtx.createOscillator();
        osc.type = "square";
        osc.frequency.setValueAtTime(200, now);
        osc.frequency.linearRampToValueAtTime(1600, now + 0.3);
        const gain = audioCtx.createGain();
        gain.gain.setValueAtTime(0.6, now);
        gain.gain.linearRampToValueAtTime(0, now + 0.3);
        osc.connect(gain);
        gain.connect(master);
        osc.start(now);
        osc.stop(now + 0.31);
        break;
      }
      case "sweep_down": {
        const osc = audioCtx.createOscillator();
        osc.type = "square";
        osc.frequency.setValueAtTime(1600, now);
        osc.frequency.linearRampToValueAtTime(200, now + 0.3);
        const gain = audioCtx.createGain();
        gain.gain.setValueAtTime(0.6, now);
        gain.gain.linearRampToValueAtTime(0, now + 0.3);
        osc.connect(gain);
        gain.connect(master);
        osc.start(now);
        osc.stop(now + 0.31);
        break;
      }
      case "noise_burst": {
        const bufferSize = audioCtx.sampleRate * 0.1;
        const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
        const channelData = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
          channelData[i] = Math.random() * 2 - 1;
        }
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        const gain = audioCtx.createGain();
        gain.gain.setValueAtTime(0.7, now);
        gain.gain.linearRampToValueAtTime(0, now + 0.1);
        source.connect(gain);
        gain.connect(master);
        source.start(now);
        break;
      }
      case "coin": {
        // Two quick ascending tones: E5 then B5
        scheduleSquareTone(659.25, now, 0.08);
        scheduleSquareTone(987.77, now + 0.09, 0.12);
        break;
      }
      case "explosion": {
        // White noise burst + low tone decay
        const bufferSize = audioCtx.sampleRate * 0.5;
        const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
        const channelData = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
          channelData[i] = Math.random() * 2 - 1;
        }
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        const noiseGain = audioCtx.createGain();
        noiseGain.gain.setValueAtTime(0.8, now);
        noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
        source.connect(noiseGain);
        noiseGain.connect(master);
        source.start(now);

        const lowOsc = audioCtx.createOscillator();
        lowOsc.type = "square";
        lowOsc.frequency.setValueAtTime(80, now);
        lowOsc.frequency.exponentialRampToValueAtTime(30, now + 0.5);
        const lowGain = audioCtx.createGain();
        lowGain.gain.setValueAtTime(0.6, now);
        lowGain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
        lowOsc.connect(lowGain);
        lowGain.connect(master);
        lowOsc.start(now);
        lowOsc.stop(now + 0.51);
        break;
      }
    }
  }

  function melodyPlay(notes: ScheduledNote[]): void {
    melodyStop();
    const { ctx: audioCtx } = getContext();
    let cursor = audioCtx.currentTime;
    for (const entry of notes) {
      const freq = noteToFrequency(entry.note);
      if (freq !== null) {
        scheduleSquareTone(freq, cursor, entry.ms / 1000);
      }
      cursor += entry.ms / 1000;
    }
  }

  function melodyStop(): void {
    if (melodyTimeoutId !== null) {
      clearTimeout(melodyTimeoutId);
      melodyTimeoutId = null;
    }
    for (const osc of activeOscillators) {
      try { osc.stop(); } catch (_) { /* already stopped */ }
    }
    activeOscillators = [];
  }

  function setVolume(level: number): void {
    const { master } = getContext();
    master.gain.value = Math.max(0, Math.min(1, level));
  }

  return { init, tonePlay, sfxPlay, melodyPlay, melodyStop, setVolume };
}
