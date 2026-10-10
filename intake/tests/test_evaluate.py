import json

import pytest

from contracts.api import IssueType
from intake import evaluate as E
from intake.extraction import parse


def test_normalize_drops_punctuation_and_danda():
    assert E.normalize("  Sir,  the ROAD is broken. ") == "sir the road is broken"
    assert E.normalize("सड़क टूटी है।") == "सड़क टूटी है"


def test_wer_and_cer_on_known_pairs():
    assert E.wer("a b c", "a b c") == 0
    assert E.wer("a b c", "a x c") == pytest.approx(1 / 3)  # one substitution
    assert E.wer("a b", "a b c") == pytest.approx(1 / 2)    # one insertion
    assert E.wer("सड़क टूटी है।", "सड़क टूटी है") == 0          # punctuation is not an error
    assert E.cer("nahi", "nhi") == pytest.approx(1 / 4)      # spelling variant: 1 char of 4
    assert E.wer("nahi", "nhi") == 1                          # ...but a whole word to WER


def test_token_f1():
    assert E.token_f1("", "") == 1
    assert E.token_f1("near the market", "") == 0
    assert E.token_f1("near the market", "the market") == pytest.approx(0.8)


def test_parse_accepts_only_the_extraction_shape():
    good = {"language": "hi", "issue_type": "pothole", "location_text": "x",
            "cues": {"children": True, "accident": False, "flooding": False}}
    assert parse(json.dumps(good)).issue_type == IssueType.POTHOLE
    assert parse("not json") is None
    assert parse("[1, 2]") is None
    assert parse(json.dumps({**good, "issue_type": "volcano"})) is None
    assert parse(json.dumps({**good, "severity": "high"})) is None  # extra field


def _gold_lookup(gold):
    """An extractor that knows the answers: the scorer must give it full marks."""
    answers = {r.text: json.dumps({"language": r.language, "issue_type": r.issue_type,
                                   "location_text": r.location_text,
                                   "cues": {c: bool(getattr(r, c)) for c in E.CUE_NAMES}})
               for r in gold.itertuples()}
    return answers.__getitem__


def test_perfect_extractor_scores_one():
    gold = E.load_challenge()
    row = E.evaluate(_gold_lookup(gold), gold).loc["all"]
    assert row.n == len(gold)
    assert (row.drop("n") == 1).all(), row


def test_invalid_output_scores_zero_everywhere():
    gold = E.load_challenge()
    row = E.evaluate(lambda text: "{oops", gold).loc["all"]
    assert row.json_valid == 0 and row.issue_acc == 0 and row.lang_acc == 0
    assert row.loc_exact == 0 and row.loc_f1 == 0  # no credit for empty gold locations


def test_challenge_set_is_well_formed():
    ch = E.load_challenge()
    assert ch.id.is_unique
    assert ch.language.value_counts().to_dict() == {lang: 12 for lang in E.LANGUAGES}
    assert set(ch.issue_type) == {i.value for i in IssueType}
    for r in ch.itertuples():  # gold location is a verbatim span of the text
        assert r.location_text in r.text, r.id
    assert ch[list(E.CUE_NAMES)].sum().min() >= 4  # every cue has positives to score


def test_audio_ids_are_three_per_language():
    ch = E.load_challenge().set_index("id")
    assert set(E.AUDIO_IDS) <= set(ch.index)
    assert ch.loc[list(E.AUDIO_IDS), "language"].value_counts().to_dict() == {
        lang: 3 for lang in E.LANGUAGES}


def test_evaluate_audio(tmp_path):
    gold = E.load_challenge().set_index("id")
    for clip_id in ("ch-en-01", "ch-hi-02"):
        (tmp_path / f"{clip_id}.wav").touch()
    perfect = E.evaluate_audio(lambda p: gold.loc[p.stem, "text"], tmp_path)
    assert perfect.loc["all", "clips"] == 2 and perfect.loc["all", "wer"] == 0
    silent = E.evaluate_audio(lambda p: "", tmp_path)
    assert silent.loc["all", "wer"] == 1 and silent.loc["all", "cer"] == 1

    (tmp_path / "my_recording.wav").touch()
    with pytest.raises(ValueError, match="no challenge item"):
        E.audio_clips(tmp_path)
