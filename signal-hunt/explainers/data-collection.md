# Data Collection — Recording Your Training Set

Before any model can learn, it needs examples. This project uses real recordings made on a
laptop or phone — not synthetic audio — and that choice is deliberate.

---

## Why real recordings, not synthetic

The simplest approach would be to generate audio programmatically: a sine wave at 440 Hz for a
hum, a higher sine for a whistle, a noise burst for a clap. This would be reproducible and
require no microphone.

It would also produce a useless model.

Real hums are not pure sine waves. They have harmonics, breath noise, pitch wobble, and room
reflections. Real claps have variation in hand position, speed, and the acoustic environment.
A model trained on perfect synthetic tones will fail immediately on any real recording —
including ones you make yourself.

Recording real audio forces the model to learn the *essence* of each sound, not a mathematical
approximation of it. It also gives you something to debug: if your model confuses hums and
whistles, you can listen to the recordings and understand why.

> **In practice:** This is the training distribution problem. A model trained on synthetic or
> clean-room data fails when deployed because real inputs don't match training inputs. In fraud
> detection, this is called concept drift — models trained on last year's fraud patterns miss
> this year's. In medical imaging, models trained on one scanner type fail on another brand.
> The fix is always the same: train on data that represents what the model will actually see.

---

## Why these three sounds

Hum, whistle, and clap were chosen because they are **spectrally distinct**:

| Sound | Spectral signature | Why it's good for learning |
|-------|--------------------|---------------------------|
| Hum | Low-frequency sustained tone, strong harmonics | Concentrated in the bottom third of the spectrogram |
| Whistle | High-frequency narrow band, mostly one peak | Concentrated in the top third of the spectrogram |
| Clap | Broadband transient, all frequencies at once | Wide burst across the whole spectrogram |

A CNN scanning for spatial patterns in a Mel-spectrogram should find these easy to separate —
they look fundamentally different. This makes them the right starting point: if the model
*can't* learn to distinguish a hum from a clap, something is wrong with the architecture or
the data, not with the inherent difficulty of the task.

Phases 3 and 4 will make the task harder (pitch classification, sequence recognition). Starting
with an easy task lets you validate the full pipeline before adding complexity.

---

## Why 10 recordings per class

With 7 augmentations per recording, 10 recordings → 70 training samples per class → 210 total.

That's the minimum for a 3-class CNN to have enough signal without overfitting:
- **Less than 5 recordings per class** — augmentations start to look too similar. The model
  memorises the 5 source clips rather than learning the class.
- **10 recordings** — enough variation in voice, microphone position, and room acoustics that
  the 7 augmented variants per clip are genuinely diverse.
- **More is always better** — if you can record 20–30, do it. The model will generalise better.

The 70/15/15 train/val/test split is designed for this scale: 210 samples gives ~147 train,
~31 val, ~32 test — enough to measure accuracy meaningfully on a 3-class problem (random
baseline is 33%).

---

## How to record

**Format:** WAV file, any sample rate, mono or stereo. The pipeline converts everything to
22050 Hz mono on ingest — format details don't matter.

**Length:** 1–2 seconds. The pipeline trims silence and pads/truncates to exactly 1.5 seconds.
Recordings shorter than 0.5 seconds may be mostly silence after trimming; longer than 3 seconds
will be cut off.

**What to vary:**
- **Pitch** (hums/whistles) — record at a few different pitches. The augmentation pipeline
  shifts pitch ±2 semitones, but starting from more varied pitches builds a more robust dataset.
- **Volume** — quiet and loud recordings. Normalisation handles absolute levels, but variation
  in dynamic range helps the model ignore volume and focus on shape.
- **Microphone distance** — close and a bit further away changes the room reflections.

**What to avoid:**
- Long silences at the start or end (trimming handles this, but very short sounds after
  trimming → near-empty tensors)
- Background music or TV — the pipeline doesn't separate foreground from background, so
  consistent background noise becomes part of the "sound"
- Back-to-back sounds in one file — each file should contain one instance of one sound type

**On Mac:** QuickTime → New Audio Recording, export as WAV. Voice Memos → share → export
as M4A then convert: `ffmpeg -i file.m4a file.wav`. Audacity works too.

---

## How to add more recordings

Drop new `.wav` files into the matching class folder:

```
data/raw/hum/       ← new hum recordings go here
data/raw/whistle/   ← new whistle recordings here
data/raw/clap/      ← new clap recordings here
```

Then regenerate tensors:

```bash
python -m pipeline.batch data/raw data/processed
```

The manifest is rewritten each run — it reflects whatever is currently in `data/raw/`.
No other changes needed: the `Dataset` class reads the manifest at load time.

---

## What we recorded

33 `.wav` files, 11 per class, recorded on a MacBook:

- `hum/` — `hum_440hz.wav` (original Phase 1 test file) + `h-1.wav` through `h-10.wav`
- `whistle/` — `whistle_chirp.wav` + `w-1.wav` through `w-10.wav`
- `clap/` — `clap_burst.wav` + `c-1.wav` through `c-10.wav`

These are committed to the repo so they're available on any machine that clones it.
Running `setup.sh` regenerates all 231 processed tensors from these source files.
