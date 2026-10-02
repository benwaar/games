# M2: Ingestion Wrapper

Build `pipeline/ingest.py` — functions to load audio from file, resample to a standard rate, and trim silence.

## What to build

- `load_audio(path, target_sr=22050)` — load any audio file, resample to target rate, convert to mono
- `trim_silence(signal, top_db=20)` — strip leading/trailing silence
- `pad_or_truncate(signal, target_length)` — enforce fixed-length output

## Gate

Unit tests pass for:
- Correct output shape and dtype (float32 numpy array)
- Consistent sample rate regardless of input
- Mono conversion from stereo input
- Silence trimming removes quiet sections
- Pad/truncate produces exact target length
