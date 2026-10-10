"""Indic -> English translation (Step 4.2). Routed by language, from the script rule
(keywords.detect_language); the original text is always kept next to `english_text` (Report).

- en: passed through unchanged, confidence 1.
- hi, pa: IndicTrans2, distilled 200M, on CPU. Its IndicProcessor must run first: it normalises
  the text, tags it ("hin_Deva eng_Latn ...") and rewrites Gurmukhi into Devanagari, the single
  script the model was trained on for both languages. Confidence is the best beam's
  length-normalised log-probability, exponentiated: the geometric mean probability per output
  token (Spearman 0.56 with sentence chrF on the 24 challenge items).
- mixed (Hinglish): the local LLM (llm.translate). IndicTrans2 was trained on native scripts and
  mostly copies romanised Hindi back (chrF 33.9, below copying the text, 35.7; qwen3:4b 59.8).
  No confidence (None). An invalid LLM reply keeps the original text, confidence 0.
- und, or empty text: nothing to translate, "" with confidence 0.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from intake import llm

# Gated on Hugging Face (accept the MIT licence terms once, then `hf auth login`). Pinned
# revision: trust_remote_code runs the repo's own Python, so it must not change under us.
MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
REVISION = "d66a2122ad6d28184f3dc3797b652bd80ae3f65f"
SRC_LANG = {"hi": "hin_Deva", "pa": "pan_Guru"}
# The model card's settings, minus the KV cache: the repo's 2024 code expects the old tuple cache
# and crashes on the Cache object transformers 4.5x passes. Without it each step recomputes the
# decoder (slower, same output).
GENERATE = {"num_beams": 5, "max_length": 256, "use_cache": False}


@dataclass(frozen=True)
class Translation:
    text: str
    confidence: float | None


Model = Callable[[list[str], str], list[Translation]]  # (texts, IndicTrans2 source tag)


@cache
def _load():
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    kwargs = {"revision": REVISION, "trust_remote_code": True}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, **kwargs).eval()
    return tokenizer, model, IndicProcessor(inference=True)


def indictrans(texts: list[str], src_lang: str) -> list[Translation]:
    import torch

    tokenizer, model, processor = _load()
    batch = processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang="eng_Latn")
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, **GENERATE, output_scores=True,
                             return_dict_in_generate=True)
    decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    english = processor.postprocess_batch(decoded, lang="eng_Latn")
    return [Translation(e, float(s)) for e, s in zip(english, out.sequences_scores.exp(),
                                                       strict=True)]


def translate(text: str, language: str, model: Model = indictrans,
              llm_translate: Callable[[str], str] = llm.translate) -> Translation:
    if language == "en":
        return Translation(text, 1.0)
    if not text.strip():
        return Translation("", 0.0)
    if language == "mixed":
        english = llm_translate(text)
        return Translation(english, None) if english else Translation(text, 0.0)
    if language in SRC_LANG:
        return model([text], SRC_LANG[language])[0]
    return Translation("", 0.0)
