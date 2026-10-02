"""M1: Hello Audio — load a sample, print metadata, plot waveform + spectrogram."""

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SAMPLE = Path("data/raw/hum_440hz.wav")
OUTPUT = Path("output")

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(SAMPLE), sr=None)
    duration = len(y) / sr

    print(f"File:        {SAMPLE}")
    print(f"Shape:       {y.shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration:    {duration:.2f}s")
    print(f"Dtype:       {y.dtype}")

    # Waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"Waveform — {SAMPLE.name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(str(OUTPUT / "waveform.png"), dpi=150)
    plt.close(fig)
    print(f"Saved:       {OUTPUT / 'waveform.png'}")

    # Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"Mel-spectrogram — {SAMPLE.name}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(str(OUTPUT / "spectrogram.png"), dpi=150)
    plt.close(fig)
    print(f"Saved:       {OUTPUT / 'spectrogram.png'}")

if __name__ == "__main__":
    main()
