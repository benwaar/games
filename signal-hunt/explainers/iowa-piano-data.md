# Iowa Piano Data — Why Public Datasets

## What we used

**University of Iowa Musical Instrument Samples** — a free academic dataset of acoustic instrument recordings maintained by the University of Iowa Electronic Music Studios. We used the acoustic grand piano subset: 36 AIFF files covering one chromatic octave (C4–B4, MIDI notes 60–71) at three dynamics (pp, mf, ff).

```
data/raw/notes/
  C4/   iowa_pp_C4.wav   iowa_mf_C4.wav   iowa_ff_C4.wav
  Db4/  iowa_pp_Db4.wav  iowa_mf_Db4.wav  iowa_ff_Db4.wav
  ...   (12 notes × 3 dynamics = 36 files)
```

Download script: `scripts/download_iowa_piano.py` — fetches AIFF files and converts to WAV.

---

## Why use a public dataset instead of recording your own?

### 1. Perfect labels for free

Piano pitch is physically exact. A recording labelled "C4" is guaranteed to be 262 Hz — there's no labelling uncertainty. If you recorded yourself humming "C4", you'd need a reference pitch, you'd likely drift sharp or flat, and you'd need someone to verify each clip.

### 2. Timbral diversity at no cost

Three dynamics (pp/mf/ff) give the model different note shapes — a softly played note has a gentler attack than a forte one. That variation is useful for generalisation. Getting equivalent variation from self-recording would mean 108 takes (12 notes × 3 dynamics × 3 recordings).

### 3. Reproducibility

Anyone can clone the repo and run `bash setup.sh` to recreate the exact same dataset. Self-recorded audio ties the dataset to one person's voice and one room's acoustics — someone else couldn't reproduce your training data.

### 4. Baseline for comparison

A public dataset gives a known starting point. When Phase 4 extends to chords, you can test chord detection against the same instrument and compare results across phases.

---

## The domain gap

The Iowa recordings are clean studio samples: no room reverb, no background noise, no sustain pedal, recorded at a consistent distance. Real piano playing is noisier — sustain pedal adds resonance, the room adds reverb, velocity varies continuously rather than just three levels.

**In Phase 3, this means:** the model trained and tested on Iowa data performs well (89.5% test accuracy) because the test set is from the same distribution as training. If you feed it a recording from a real piano in a room, expect a drop — the model has never seen that kind of acoustic variation.

**How to close the gap:** augmentation. Adding reverb and room-acoustic simulation as additional augmentation transforms (alongside the existing noise, pitch shift, and time stretch in `pipeline/augment.py`) would train the model on more realistic conditions. This is a natural M16/Phase 4 extension if real-world accuracy matters.

---

## Why not NSynth?

The original Phase 3 plan called for Google's NSynth dataset (300K+ instrument notes). We switched to Iowa because:

- **File format complexity.** NSynth distributes as TFRecord (TensorFlow format) — reading it requires TensorFlow or a custom parser, which adds a dependency not needed anywhere else in the project.
- **Iowa is sufficient.** NSynth has more samples and more instruments, but for learning transfer learning and pitch classification on one octave of acoustic piano, Iowa's 36 files are enough.
- **Iowa is simpler to audit.** 36 WAV files you can open and listen to individually. NSynth is 300K files in a binary format.

For a production system that needed robustness across many piano types and playing styles, NSynth (or a custom recorded dataset) would be the right choice. For learning the concepts, Iowa is the right tradeoff.

---

## Business parallel

Using a public dataset in ML is the standard approach when you need a clean baseline. The pattern appears everywhere:

- **Computer vision:** ImageNet for general vision tasks, COCO for object detection, MNIST for getting started
- **NLP:** Common Crawl, Wikipedia dumps, BooksCorpus for language model pretraining
- **Speech:** LibriSpeech (1000h of audiobooks), Common Voice (Mozilla), VoxCeleb

The tradeoff is always the same: public datasets are clean, well-labelled, and reproducible — but they don't match your production distribution. The domain gap between a clean public dataset and real-world data is one of the most common reasons ML models underperform in production. Measuring that gap (and closing it with augmentation or fine-tuning on production data) is a core ML engineering skill.
