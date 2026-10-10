"""Language ID and translation tests (Step 4.2). A fake model stands in for IndicTrans2."""

import pandas as pd
import pytest

from intake import evaluate as E
from intake import keywords, llm
from intake.translate import Translation, translate


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, src_lang):
        self.calls.append((texts, src_lang))
        return [Translation(f"EN<{t}>", 0.5) for t in texts]


def test_english_passes_through_untouched():
    model = FakeModel()
    assert translate("Pothole near Karol Bagh", "en", model) == Translation(
        "Pothole near Karol Bagh", 1.0)
    assert model.calls == []


@pytest.mark.parametrize("language, tag", [("hi", "hin_Deva"), ("pa", "pan_Guru")])
def test_hindi_and_punjabi_go_to_indictrans2_with_their_tag(language, tag):
    model = FakeModel()
    assert translate("x", language, model, llm_translate=_no_llm) == Translation("EN<x>", 0.5)
    assert model.calls == [(["x"], tag)]


def _no_llm(text):
    raise AssertionError("the LLM must not be called")


def test_hinglish_goes_to_the_llm_without_a_confidence():
    model = FakeModel()
    result = translate("gaddha hai", "mixed", model, llm_translate=lambda t: f"LLM<{t}>")
    assert result == Translation("LLM<gaddha hai>", None) and model.calls == []


def test_hinglish_keeps_the_original_text_when_the_llm_reply_is_invalid():
    result = translate("gaddha hai", "mixed", FakeModel(), llm_translate=lambda t: "")
    assert result == Translation("gaddha hai", 0.0)


@pytest.mark.parametrize("text, language", [("12345 !!", "und"), ("   ", "hi"), ("", "pa"),
                                            (" ", "mixed")])
def test_nothing_to_translate_gives_empty_text_and_zero_confidence(text, language):
    model = FakeModel()
    assert translate(text, language, model, llm_translate=_no_llm) == Translation("", 0.0)
    assert model.calls == []


def test_references_cover_exactly_the_non_english_challenge_items():
    refs, gold = E.load_references(), E.load_challenge()
    assert sorted(refs.id) == sorted(gold.id[gold.language != "en"])
    assert refs.english.str.strip().astype(bool).all()
    assert refs.english.map(str.isascii).all()  # English, Latin script


def test_keep_names_are_absent_from_hindi_and_punjabi_source_text():
    # A name already in Latin letters in the source would score any copy-through as "kept".
    # Digits are the same in every script, so copying them is fine.
    refs = E.load_references()
    for r in refs[refs.language != "mixed"].itertuples():
        names = [k for k in r.keep if not k.split("|")[0].isdigit()]
        assert not any(E.kept(k, r.text) for k in names), r.id


@pytest.mark.parametrize("item, translation, expected", [
    ("Karol Bagh", "A pothole in Karol Bagh.", True),
    ("Chandni Chowk|Chandani Chowk", "near chandani chowk,", True),
    ("4|four", "Lane No. 14 of Paharganj", False),  # whole words only
    ("4|four", "lane number four", True),
    ("C-Hexagon", "on C Hexagon near India Gate", True),  # punctuation ignored
    ("Karol Bagh", "", False),
])
def test_kept(item, translation, expected):
    assert E.kept(item, translation) is expected


def _refs():
    return pd.DataFrame({"id": ["a", "b"], "language": ["hi", "pa"],
                         "text": ["गड्ढा", "ਟੋਆ"], "english": ["A pothole in Karol Bagh.",
                                                               "Garbage near India Gate."],
                         "keep": [["Karol Bagh"], ["India Gate", "2|two"]]})


def test_perfect_translator_scores_100_and_keeps_everything():
    refs = _refs()
    items = E.translate_items(lambda text, lang: refs.english[refs.text == text].item(), refs)
    table = E.score_translations(items)
    assert table.loc["all", "chrf"] == pytest.approx(100)
    assert table.loc["all", "keep"] == pytest.approx(2 / 3)  # "2|two" is not in the reference
    assert list(table.index) == ["all", "hi", "pa"]


def test_copying_the_source_scores_near_zero_for_indic_scripts():
    table = E.score_translations(E.translate_items(lambda text, lang: text, _refs()))
    assert table.loc["all", "chrf"] < 1 and table.loc["all", "keep"] == 0


def test_script_rule_is_perfect_on_the_challenge_set():
    acc = E.langid_accuracy(keywords.detect_language, E.load_challenge())
    assert acc["all"] == 1.0 and set(acc.index) == {"all", "hi", "pa", "en", "mixed"}


def test_langdetect_has_no_hinglish_label():
    detect = E.langdetect_detector()
    assert detect("Karol Bagh metro station ke neeche bahut bada gaddha hai") != "mixed"
    assert detect("12345 !!") == "und"


def test_llm_translate_reads_the_constrained_reply():
    seen = []

    def chat(messages, model, constrained):
        seen.append((messages, constrained))
        return '{"english": "  A pothole near Karol Bagh. "}'

    assert llm.translate("करोल बाग में गड्ढा", chat=chat) == "A pothole near Karol Bagh."
    (messages, constrained), = seen
    assert constrained and messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "करोल बाग में गड्ढा"}


@pytest.mark.parametrize("reply", ["Okay, let's tackle this", '{"text": "x"}', "[1]"])
def test_llm_translate_returns_empty_text_on_a_bad_reply(reply):
    assert llm.translate("x", chat=lambda messages, model, constrained: reply) == ""
