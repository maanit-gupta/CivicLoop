# Phase 4: Intake (voice/text/photo -> structured Report)

Goal: turn a multilingual, possibly noisy citizen report into a `Report`.
Needs: Phase 0 only. CPU-friendly model sizes.

## 4.1 Evaluation + keyword baseline
- `intake/evaluate.py`: issue_type accuracy/macro-F1, location_text match rate, cue-flag F1,
  JSON validity rate, per-language breakdown; WER for audio.
- Baseline: multilingual keyword rules. Show the table first.
- Record ~10 short voice clips yourself (Hindi, English, Punjabi, Hinglish) or TTS-generate them;
  store in `data/intake/audio/` with transcripts. The user may need to do this: ask.

## 4.2 Speech, language ID, translation
- faster-whisper (small or base, int8 on CPU) for ASR; language ID; IndicTrans2 (distilled) for
  Indic -> English. Keep original text and English text both.
- Low ASR confidence -> flag `needs_clarification` rather than guessing.
- Measure WER and translation sanity on the clips.

## 4.3 LLM structured extraction
- Local LLM via Ollama (e.g. qwen2.5:3b) with a strict JSON schema from `contracts`; validate with
  Pydantic, retry once on invalid JSON, fall back to the keyword baseline.
- Prompt + few-shot examples in `intake/prompts/`. Evaluate on the corpus test split.

## 4.4 Intake API
- `intake/api.py: process(RawReport) -> Report` (text, audio, photo caption/GPS from EXIF if
  present). DecisionLog entry. `python -m intake.demo` on 3 samples. Tests with mocked models.
