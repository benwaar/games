"""Generate synthetic audio samples for bootstrapping — sine tones, noise burst, chirp."""

import numpy as np
import soundfile as sf
from pathlib import Path

SR = 22050
DURATION = 1.5

def sine_tone(freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)

def chirp(f0: float = 200.0, f1: float = 800.0) -> np.ndarray:
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    freq = f0 + (f1 - f0) * t / DURATION
    return 0.5 * np.sin(2 * np.pi * freq * t)

def noise_burst() -> np.ndarray:
    n = int(SR * DURATION)
    signal = np.zeros(n)
    start = n // 4
    end = 3 * n // 4
    signal[start:end] = 0.3 * np.random.default_rng(42).standard_normal(end - start)
    return signal

SAMPLES = {
    "hum_440hz.wav": sine_tone(440.0),
    "whistle_chirp.wav": chirp(500.0, 2000.0),
    "clap_burst.wav": noise_burst(),
}

if __name__ == "__main__":
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    for name, audio in SAMPLES.items():
        path = out / name
        sf.write(str(path), audio, SR)
        print(f"  {path}  ({len(audio)} samples, {len(audio)/SR:.2f}s)")
    print("Done.")
