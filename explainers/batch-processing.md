# Batch Processing — From Folder to Training Set

Takes a folder of `.wav` files organised by class and produces a complete, labelled, augmented
tensor dataset ready for a DataLoader. One command, deterministic output.

---

## What it does

```
data/raw/                             data/processed/
  hum/                      →           h-1_clean.pt
    h-1.wav                              h-1_noise_low.pt
    h-2.wav                              h-1_noise_high.pt
    ...                                  h-1_pitch_up.pt
  whistle/                               h-1_pitch_down.pt
    w-1.wav                              h-1_slow.pt
    ...                                  h-1_fast.pt
  clap/                                  h-2_clean.pt
    c-1.wav                              ... (7 per source file)
    ...                                  manifest.json
```

Each source file produces **7 variants** — one clean pass plus six augmentations.
33 source recordings become 231 training samples. More source files scale linearly.

---

## Labels come from folders, not filenames

The folder name is the class label. `data/raw/hum/h-1.wav` → label `"hum"`.
The filename doesn't matter — only which folder the file is in.

**Why folder-based, not filename-based?**

Early in Phase 1, we had three files (`hum_440hz.wav`, `whistle_chirp.wav`, `clap_burst.wav`) and
derived the label from the filename stem. That worked for three files. It breaks as soon as you add
more recordings — `h-1.wav` tells you nothing about the class.

Folder structure is the conventional approach in computer vision and audio ML: `ImageFolder`,
`AudioFolder`, and our `process_folder` all follow the same convention: `{root}/{class}/{file}`.
The folder is the label. Filenames are just identifiers.

> **In practice:** This is the same as how you'd organise raw data in any labelled classification
> dataset. The folder hierarchy encodes the schema — `customers/{tier}/`, `tickets/{priority}/`,
> `transactions/{type}/`. The pipeline reads the schema from the structure rather than parsing it
> from filenames, which is fragile and breaks on renames.

---

## Tensor naming uses the source file stem

Output tensors are named `{source_stem}_{augmentation}.pt` — e.g. `h-1_clean.pt`, `h-1_noise_low.pt`.

**Why not `{label}_{augmentation}.pt`?**

With 11 files per class, naming by label would produce 11 files all called `hum_clean.pt` —
each overwriting the previous. Naming by source stem guarantees uniqueness across the whole
processed folder regardless of how many recordings exist per class.

The label is stored in `manifest.json`, not encoded in the filename. The filename is just a key.

---

## The manifest

`manifest.json` records what was produced and where it came from:

```json
[
  {
    "file": "h-1_clean.pt",
    "source": "h-1.wav",
    "label": "hum",
    "augmentation": "clean",
    "shape": [1, 128, 65]
  },
  ...
]
```

The `label` field is the class name from the source folder — not the filename stem.
The `Dataset` class (Phase 2, M7) reads this manifest to load tensors and map labels to integers.

> **In practice:** The manifest is data provenance — the ML equivalent of an audit trail. In
> production pipelines (financial data, medical records, regulatory reporting), you always track
> which raw inputs produced which processed outputs, with what parameters. Without it, debugging
> a bad model prediction means guessing what data it trained on.

---

## The augmentation strategy

| Variant | What it does | Why |
|---------|-------------|-----|
| `clean` | No augmentation — original signal | Baseline the model must get right |
| `noise_low` | White noise at SNR 20 dB | Mild mic hiss — common in laptop recordings |
| `noise_high` | White noise at SNR 10 dB | Noisy environment — phone in a busy room |
| `pitch_up` | Shift pitch up 2 semitones | Different voices, instruments |
| `pitch_down` | Shift pitch down 2 semitones | Same, other direction |
| `slow` | Time-stretch to 0.85× speed | Tempo variation |
| `fast` | Time-stretch to 1.15× speed | Tempo variation |

Applied *before* feature extraction — the Mel-spectrogram captures the augmented signal.
Time-stretched outputs are padded/truncated back to standard length to keep tensor shapes uniform.

> **Coming from C/JS/TS:** This is like generating test fixtures from a template — start with a
> handful of canonical inputs and systematically produce variants that cover the edge cases your
> system needs to handle.

---

## Determinism

Same seed → same output. The RNG for noise augmentations is seeded explicitly (`--seed 42` default).

- Reproducible across machines
- Regenerate the exact same dataset from the same raw files
- Comparing two runs with different augmentation parameters is meaningful

> **In practice:** Reproducibility is a first-class concern in ML. If you can't regenerate your
> training data, you can't debug why a model behaves differently after retraining. Seeded pipelines,
> pinned dependency versions, and committed configs are how production ML teams avoid
> "works on my machine" disasters.

---

## Key parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `input_dir` | — | Root folder with `{class}/` subfolders |
| `output_dir` | — | Folder for `.pt` tensors + manifest |
| `--seed` | 42 | RNG seed for noise augmentations |

Run: `python -m pipeline.batch data/raw data/processed`

**Fresh clone:** `setup.sh` runs this automatically after installing dependencies.
