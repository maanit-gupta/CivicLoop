"""Public API of the intake module (Step 4.4). Only this file may be imported by other modules.

process(RawReport) -> Report, one path:
1. Text: the typed text; else the voice note's transcript; else the photos' EXIF captions.
   Typed text wins over a voice note, so the location span check has one source to search.
2. Language: the script rule. Fields: llm.extract on the original text, with keyword rules if
   the LLM's reply stays invalid. English: translate.translate, routed by language.
3. Location: the report's GPS; else the first photo with EXIF GPS.
4. Confidence: the weakest stage (min). Extraction is worth its path's issue accuracy on the
   challenge set (Step 4.3); translation and speech bring their own per-item scores. A
   translator without a score (the LLM, for Hinglish) adds nothing.
5. One DecisionLog entry, keyed by report_id (there is no case yet).
Models are parameters, so tests run without Ollama or IndicTrans2.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from config import DECISION_LOG_PATH, MODEL_VERSIONS
from contracts.api import IssueType, RawReport, Report, log_decision
from intake import llm, photo
from intake import translate as T
from intake.extraction import Extraction, parse

# Issue-type accuracy on the 48 challenge items (Step 4.3).
EXTRACTION_ACCURACY = {llm.MODEL: 0.96, "keywords": 0.65}
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "pa": "Punjabi", "mixed": "Hinglish",
                  "und": "an unknown language"}

Extractor = Callable[..., str]  # (text, stats=Counter) -> Extraction JSON, like llm.extract
Translator = Callable[[str, str], T.Translation]  # (text, language), like translate.translate
Transcriber = Callable[[str], tuple[str, float]]  # audio path -> (transcript, confidence)


def no_speech(audio_path: str) -> tuple[str, float]:
    raise NotImplementedError(
        "Voice notes need the speech step (faster-whisper), deferred until the voice clips "
        "exist: see 'Deferred' in PROGRESS.md.")


def _text(raw: RawReport, captions: list[str], transcribe: Transcriber):
    """(text, where it came from, speech confidence or None)."""
    if raw.text and raw.text.strip():
        return raw.text.strip(), "typed text", None
    if raw.audio_path:
        transcript, confidence = transcribe(raw.audio_path)
        return transcript.strip(), "the voice note", confidence
    return " ".join(captions), "the photo caption", None


def _translation_note(language: str, english: T.Translation) -> str:
    if language == "en":
        return "Already in English."
    if english.confidence is None:
        return f"Translated to English by {llm.MODEL} (no confidence score)."
    if english.confidence == 0:
        return "Not translated: english_text is empty or the original text."
    return f"Translated to English, confidence {english.confidence:.2f}."


def process(raw: RawReport, *, extract: Extractor = llm.extract,
            translate: Translator = T.translate, transcribe: Transcriber = no_speech,
            log_path: Path = DECISION_LOG_PATH) -> Report:
    photos = [photo.read(p) for p in raw.image_paths]
    text, source, speech = _text(raw, [p.caption for p in photos if p.caption], transcribe)
    location = raw.gps or next((p.gps for p in photos if p.gps), None)
    where = ("Location from the report's GPS." if raw.gps else
             "Location from the photo's EXIF GPS." if location else "No GPS.")
    common = {"report_id": raw.report_id, "submitted_at": raw.submitted_at,
              "original_text": text, "location": location, "image_paths": raw.image_paths}

    if not text:
        report = Report(**common, language="und", issue_type=IssueType.OTHER,
                        model_version=MODEL_VERSIONS["intake"], confidence=0.0,
                        explanation=f"Nothing to read in {source}. Issue type left as 'other' "
                                    f"for the photo model or a human. {where}")
        log_decision("intake", raw.report_id, raw, report, path=log_path)
        return report

    stats = Counter()
    fields = parse(extract(text, stats=stats), Extraction)
    if fields is None:
        raise ValueError(f"Extractor returned invalid JSON for report {raw.report_id}")
    extractor = "keywords" if stats["fallback"] else llm.MODEL
    english = translate(text, fields.language)

    stages = {"extraction": EXTRACTION_ACCURACY[extractor]}
    if speech is not None:
        stages["speech"] = speech
    if english.confidence is not None:
        stages["translation"] = english.confidence
    weakest = min(stages, key=stages.get)
    cues = ", ".join(c for c, on in fields.cues.model_dump().items() if on) or "none"
    found = (f"{fields.issue_type.value} at '{fields.location_text}'" if fields.location_text
             else f"{fields.issue_type.value}, no place named")
    replaced = " (the LLM's place was not in the text)" if stats["location_replaced"] else ""
    explanation = (
        f"Read {source}, in {LANGUAGE_NAMES.get(fields.language, fields.language)} (script "
        f"rule). {extractor}: {found}{replaced}; cues: {cues}. "
        f"{_translation_note(fields.language, english)} {where} "
        f"Confidence {stages[weakest]:.2f} = the weakest stage ({weakest}).")

    report = Report(**common, language=fields.language, english_text=english.text,
                    issue_type=fields.issue_type, location_text=fields.location_text,
                    cues=fields.cues, model_version=f"{MODEL_VERSIONS['intake']}/{extractor}",
                    confidence=stages[weakest], explanation=explanation)
    log_decision("intake", raw.report_id, raw, report, path=log_path)
    return report
