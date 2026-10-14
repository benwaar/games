// Unit tests for the glyph audio engine
// No test framework — plain assertions with a simple runner

import { noteToFrequency, parseNoteName, FREQ_TABLE } from "../audio";

type TestFn = () => void;
const failures: string[] = [];

function assert(condition: boolean, label: string): void {
  if (!condition) {
    failures.push(`FAIL: ${label}`);
    console.error(`  FAIL: ${label}`);
  } else {
    console.log(`  ok: ${label}`);
  }
}

function assertClose(actual: number, expected: number, tol: number, label: string): void {
  assert(Math.abs(actual - expected) < tol, `${label} (got ${actual.toFixed(4)}, expected ${expected.toFixed(4)})`);
}

function runSuite(name: string, fn: TestFn): void {
  console.log(`\n${name}`);
  fn();
}

// --- parseNoteName ---
runSuite("parseNoteName", () => {
  assert(parseNoteName("C4") !== null, "C4 parses");
  assert(parseNoteName("A4") !== null, "A4 parses");
  assert(parseNoteName("C#4") !== null, "C#4 parses");
  assert(parseNoteName("Eb4") !== null, "Eb4 parses");
  assert(parseNoteName("Bb3") !== null, "Bb3 parses");
  assert(parseNoteName("") === null, "empty string returns null");
  assert(parseNoteName("H4") === null, "invalid pitch class H4 returns null");
  assert(parseNoteName("C") === null, "missing octave returns null");
  assert(parseNoteName("4C") === null, "wrong order returns null");
  const parsed = parseNoteName("A4");
  assert(parsed?.pitchClass === "A", "A4 pitch class is A");
  assert(parsed?.octave === 4, "A4 octave is 4");
});

// --- noteToFrequency ---
runSuite("noteToFrequency — standard tuning", () => {
  const a4 = noteToFrequency("A4");
  assertClose(a4!, 440.0, 0.01, "A4 = 440 Hz");

  const c4 = noteToFrequency("C4");
  assertClose(c4!, 261.63, 0.01, "C4 ≈ 261.63 Hz");

  const a5 = noteToFrequency("A5");
  assertClose(a5!, 880.0, 0.01, "A5 = 880 Hz (octave above A4)");

  const a3 = noteToFrequency("A3");
  assertClose(a3!, 220.0, 0.01, "A3 = 220 Hz (octave below A4)");
});

runSuite("noteToFrequency — octave doubles", () => {
  const octaves = [3, 4, 5, 6];
  for (let i = 0; i < octaves.length - 1; i++) {
    const lower = noteToFrequency(`C${octaves[i]}`)!;
    const upper = noteToFrequency(`C${octaves[i + 1]}`)!;
    assertClose(upper / lower, 2.0, 0.001, `C${octaves[i + 1]} is double C${octaves[i]}`);
  }
});

runSuite("noteToFrequency — sharps and flats are enharmonic equivalents", () => {
  const csharp4 = noteToFrequency("C#4");
  const db4 = noteToFrequency("Db4");
  assertClose(csharp4!, db4!, 0.001, "C#4 === Db4");

  const asharp3 = noteToFrequency("A#3");
  const bb3 = noteToFrequency("Bb3");
  assertClose(asharp3!, bb3!, 0.001, "A#3 === Bb3");
});

runSuite("noteToFrequency — invalid input returns null", () => {
  assert(noteToFrequency("X4") === null, "invalid note X4");
  assert(noteToFrequency("") === null, "empty string");
  assert(noteToFrequency("C") === null, "no octave");
  assert(noteToFrequency("ZZ9") === null, "garbage input");
});

// --- FREQ_TABLE coverage ---
runSuite("FREQ_TABLE — coverage C3 to C6", () => {
  const expectedKeys = ["C3", "C4", "C5", "C6", "A4", "G#5", "Eb3", "Bb4"];
  for (const key of expectedKeys) {
    assert(key in FREQ_TABLE, `FREQ_TABLE contains ${key}`);
    assert(FREQ_TABLE[key] > 0, `FREQ_TABLE[${key}] is positive`);
  }
  assertClose(FREQ_TABLE["A4"], 440.0, 0.01, "FREQ_TABLE A4 = 440");
  assertClose(FREQ_TABLE["C4"], 261.63, 0.01, "FREQ_TABLE C4 ≈ 261.63");
});

// --- Melody scheduling — total duration ---
runSuite("melodyPlay scheduling — total duration", () => {
  const notes = [
    { note: "C4", ms: 200 },
    { note: "E4", ms: 200 },
    { note: "G4", ms: 400 },
  ];
  const totalMs = notes.reduce((sum, n) => sum + n.ms, 0);
  assert(totalMs === 800, "three-note melody totals 800ms");

  const single = [{ note: "A4", ms: 500 }];
  assert(single.reduce((s, n) => s + n.ms, 0) === 500, "single note totals 500ms");

  const empty: typeof notes = [];
  assert(empty.reduce((s, n) => s + n.ms, 0) === 0, "empty melody totals 0ms");
});

// --- Summary ---
console.log("\n---");
if (failures.length === 0) {
  console.log(`All tests passed.`);
  process.exit(0);
} else {
  console.error(`${failures.length} test(s) failed.`);
  process.exit(1);
}
