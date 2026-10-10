"""LLM extractor tests with a fake chat function: no Ollama needed."""

import json
from collections import Counter

import pytest

from config import CORPUS_PATH, WORLD_DIR
from contracts.api import IssueType, load_corpus
from intake import evaluate as E
from intake import keywords, llm
from intake.extraction import Extraction, Fields, parse, strict_schema

needs_data = pytest.mark.skipif(
    not (CORPUS_PATH.exists() and (WORLD_DIR / "wards.geojson").exists()),
    reason="needs data/world and the corpus (python -m contracts.world_gen / corpus_gen)")

TEXT = "Karol Bagh mein govt school ke paas bahut bada gaddha hai"
GOOD = json.dumps({"issue_type": "pothole", "location_text": "Karol Bagh mein govt school ke paas",
                   "cues": {"children": False, "accident": False, "flooding": False}})


class FakeChat:
    """Replies from a script, one per call, and records the messages it was sent."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, model, constrained):
        self.calls.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_build_messages_shape():
    messages = llm.build_messages(TEXT)
    assert messages[0]["role"] == "system" and messages[-1] == {"role": "user", "content": TEXT}
    shots = messages[1:-1]
    assert len(shots) == 2 * len(llm.few_shot())
    assert [m["role"] for m in shots] == ["user", "assistant"] * len(llm.few_shot())
    assert all(parse(m["content"], Fields) for m in shots if m["role"] == "assistant")


@needs_data
def test_few_shot_comes_from_train_with_gold_labels():
    train = load_corpus("train").set_index("complaint_id")
    for ex in llm.few_shot():
        row = train.loc[ex["source"]]
        assert row.text == ex["text"] and row.issue_type == ex["output"]["issue_type"]
        assert all(bool(row[c]) == ex["output"]["cues"][c] for c in E.CUE_NAMES)


def test_prompt_does_not_leak_the_challenge_set():
    prompt = E.normalize(llm.system_prompt() + json.dumps(llm.few_shot(), ensure_ascii=False))
    for r in E.load_challenge().itertuples():
        assert E.normalize(r.text) not in prompt, r.id
        assert not r.location_text or E.normalize(r.location_text) not in prompt, r.id


def test_strict_schema_requires_every_field():
    schema = strict_schema()
    assert set(schema["required"]) == {"issue_type", "location_text", "cues"}
    assert set(schema["$defs"]["Cues"]["required"]) == set(E.CUE_NAMES)
    assert set(schema["$defs"]["IssueType"]["enum"]) == {i.value for i in IssueType}


def test_reply_length_is_capped():
    # Uncapped, one reply ran for hours in the 4.3 evaluation. An answer needs ~60 tokens.
    assert 0 < llm.OPTIONS["num_predict"] <= 512


def test_is_span():
    assert llm.is_span("near the market, Gol Market", "Garbage near the Market Gol Market.")
    assert llm.is_span("", "anything")
    assert not llm.is_span("Karol Bagh, Delhi", "Karol Bagh mein gaddha")


def test_valid_reply_used_as_is_with_language_from_script():
    chat, stats = FakeChat(GOOD), Counter()
    out = Extraction.model_validate_json(llm.extract(TEXT, stats=stats, chat=chat))
    assert out.issue_type == IssueType.POTHOLE and out.language == "mixed"
    assert out.location_text == "Karol Bagh mein govt school ke paas"
    assert len(chat.calls) == 1 and not stats


def test_invalid_reply_retried_once():
    chat, stats = FakeChat('{"issue_type": "volcano"}', GOOD), Counter()
    out = parse(llm.extract(TEXT, stats=stats, chat=chat))
    assert out.issue_type == IssueType.POTHOLE and stats == {"retry": 1}
    retry_messages = chat.calls[1]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"issue_type": "volcano"}'}
    assert retry_messages[-1] == {"role": "user", "content": llm.RETRY_MESSAGE}


@needs_data
def test_two_invalid_replies_fall_back_to_keywords():
    chat, stats = FakeChat("not json", "still not json"), Counter()
    assert llm.extract(TEXT, stats=stats, chat=chat) == keywords.extract(TEXT)
    assert len(chat.calls) == 2 and stats == {"retry": 1, "fallback": 1}


@needs_data
def test_invented_location_replaced_by_keyword_span():
    invented = json.loads(GOOD) | {"location_text": "Karol Bagh, New Delhi 110005"}
    stats = Counter()
    out = parse(llm.extract(TEXT, stats=stats, chat=FakeChat(json.dumps(invented))))
    assert out.location_text == keywords.find_location(TEXT) != ""
    assert stats == {"location_replaced": 1}


def test_unreachable_server_raises_instead_of_falling_back():
    with pytest.raises(ConnectionError):
        llm.extract(TEXT, chat=FakeChat(ConnectionError("ollama is not running")))


def test_extract_once_passes_invalid_output_through():
    raw = '{"issue_type": "pothole"}'  # missing fields
    out = llm.extract_once(TEXT, chat=FakeChat(raw))
    assert out == raw and parse(out) is None  # scored as invalid JSON
