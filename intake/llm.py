"""LLM structured extraction (Step 4.3): a local model via Ollama fills issue_type, location_text
and cues.

Served path, per text: one call constrained to the JSON schema of `Fields` -> Pydantic
validation -> one retry if invalid -> keyword baseline if still invalid. Then two rules:
- language comes from the script rule (keywords.detect_language), which is perfect on both test
  sets, so the model isn't asked;
- location_text must be a span of the text. Anything else (translated, completed or invented
  places) is replaced by the keyword span.
A server that can't be reached raises: it must never turn an evaluation into a keyword run.
"""

import json
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path

from config import SEED
from intake import keywords
from intake.evaluate import normalize
from intake.extraction import Fields, parse, strict_schema

# Chosen on a 60-row train sample (4.3): same issue accuracy as qwen2.5:3b (0.95), better
# locations and cues, Apache-2.0. qwen2.5:3b is under the non-commercial Qwen Research licence.
MODEL = "qwen3:4b"
PROMPTS_DIR = Path(__file__).parent / "prompts"
# num_predict caps the reply. An answer is ~60 tokens; uncapped, one unconstrained reply in the
# 4.3 evaluation ran for hours. A cut-off reply is invalid JSON: retried, then keywords.
OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 4096, "num_predict": 256}
RETRY_MESSAGE = ("That reply was not a valid JSON object with exactly the fields issue_type, "
                 "location_text and cues. Reply with the corrected JSON object only.")
# Step 4.2: the LLM as a translator, scored against IndicTrans2 (intake.evaluate --translate).
# Constrained to {"english": ...}: unconstrained, qwen3:4b narrates its reasoning instead.
TRANSLATE_PROMPT = ("Translate the user's complaint to the Delhi civic helpline into English. "
                    "Write place names in Latin letters, spelled as they usually are in Delhi. "
                    'Reply with a JSON object {"english": "<the translation>"}.')
TRANSLATE_SCHEMA = {"type": "object", "properties": {"english": {"type": "string"}},
                    "required": ["english"]}

Chat = Callable[[list[dict], str, bool], str]  # (messages, model, constrained) -> raw reply


@cache
def system_prompt() -> str:
    return (PROMPTS_DIR / "system.md").read_text(encoding="utf-8")


@cache
def few_shot() -> tuple[dict, ...]:
    """Worked examples from the corpus *train* split (ids in each row's 'source')."""
    lines = (PROMPTS_DIR / "few_shot.jsonl").read_text(encoding="utf-8").splitlines()
    return tuple(json.loads(line) for line in lines if line)


def build_messages(text: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt()}]
    for ex in few_shot():
        messages += [{"role": "user", "content": ex["text"]},
                     {"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)}]
    return messages + [{"role": "user", "content": text}]


def ollama_chat(messages: list[dict], model: str = MODEL, constrained: bool = True,
                schema: dict | None = None) -> str:
    """`constrained` decodes to `schema`, by default the extraction schema."""
    import ollama  # optional dependency: pip install -e ".[intake]"

    # think=False: Qwen3 would otherwise reason at length before every answer.
    response = ollama.chat(model=model, messages=messages, options=OPTIONS, think=False,
                           format=(schema or strict_schema()) if constrained else None)
    return response.message.content


def _translate_chat(messages: list[dict], model: str, constrained: bool) -> str:
    return ollama_chat(messages, model, constrained, schema=TRANSLATE_SCHEMA)


def is_span(location: str, text: str) -> bool:
    """True if `location` appears in `text`, ignoring case, punctuation and spacing."""
    return normalize(location) in normalize(text)


def _with_language(text: str, fields: Fields) -> str:
    return json.dumps({"language": keywords.detect_language(text), **fields.model_dump()},
                      ensure_ascii=False)


def extract_once(text: str, model: str = MODEL, constrained: bool = True,
                 chat: Chat = ollama_chat) -> str:
    """One call, no retry, no fallback, no span check: scores the model itself."""
    raw = chat(build_messages(text), model, constrained)
    fields = parse(raw, Fields)
    return _with_language(text, fields) if fields else raw


def translate(text: str, model: str = MODEL, chat: Chat = _translate_chat) -> str:
    """English text, or "" if the reply is not {"english": ...} (scored as a failed item)."""
    messages = [{"role": "system", "content": TRANSLATE_PROMPT},
                {"role": "user", "content": text}]
    try:
        return str(json.loads(chat(messages, model, True))["english"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def extract(text: str, model: str = MODEL, stats: Counter | None = None,
            chat: Chat = ollama_chat) -> str:
    """The served path. `stats` counts retries, fallbacks and replaced locations."""
    stats = Counter() if stats is None else stats
    messages = build_messages(text)
    raw = chat(messages, model, True)
    fields = parse(raw, Fields)
    if fields is None:
        stats["retry"] += 1
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": RETRY_MESSAGE}]
        fields = parse(chat(messages, model, True), Fields)
    if fields is None:
        stats["fallback"] += 1
        return keywords.extract(text)
    if not is_span(fields.location_text, text):
        stats["location_replaced"] += 1
        fields.location_text = keywords.find_location(text)
    return _with_language(text, fields)
