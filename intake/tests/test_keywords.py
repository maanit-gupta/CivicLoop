import pytest

from config import CORPUS_PATH, WORLD_DIR
from contracts.api import IssueType
from intake import evaluate as E
from intake.extraction import parse
from intake.keywords import classify, detect_language, extract, find_location

needs_data = pytest.mark.skipif(
    not (CORPUS_PATH.exists() and (WORLD_DIR / "wards.geojson").exists()),
    reason="needs data/world and the corpus (python -m contracts.world_gen / corpus_gen)")


@pytest.mark.parametrize("text, lang", [
    ("सड़क पर बड़ा गड्ढा है।", "hi"),
    ("ਸੜਕ ਉੱਤੇ ਵੱਡਾ ਟੋਆ ਹੈ।", "pa"),
    ("There is a big pothole here.", "en"),
    ("road pe bahut bada gaddha hai", "mixed"),
    ("🙏 123", "und"),
])
def test_language_by_script(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text, issue", [
    ("The streetlight is not working.", IssueType.STREETLIGHT),
    ("नाली पूरी तरह बंद है।", IssueType.DRAIN_BLOCKAGE),
    ("ਪਾਈਪ ਵਿੱਚੋਂ ਪਾਣੀ ਲੀਕ ਹੋ ਰਿਹਾ ਹੈ।", IssueType.WATER_LEAK),
    ("kooda pada hai, koi uthane nahi aata", IssueType.GARBAGE),
    ("good morning to all", IssueType.OTHER),  # no keyword: fall back to other
])
def test_classify(text, issue):
    assert classify(text) == issue


@needs_data
@pytest.mark.parametrize("text, location", [
    ("There is a big pothole near the govt school in Karol Bagh.",
     "near the govt school in Karol Bagh"),
    # Regression: the verb's "in" ("caved in") is not part of the location.
    ("The road has caved in in a lane in Paharganj, there is a large crater.",
     "in a lane in Paharganj"),
    ("करोल बाग में सरकारी स्कूल के पास नाली बंद है।", "करोल बाग में सरकारी स्कूल के पास"),
    ("bus stop ke saamne, Karol Bagh streetlight kharab hai", "bus stop ke saamne, Karol Bagh"),
    ("good morning to all", ""),  # no place mentioned: no location invented
])
def test_find_location(text, location):
    assert find_location(text) == location


@needs_data
def test_output_always_parses():
    for text in E.load_corpus_test().text:
        assert parse(extract(text)) is not None, text


@needs_data
def test_keywords_memorise_the_phrase_bank():
    # Built from train vocabulary, so near-perfect on the template test split. This is the
    # reason the challenge set exists: this number says nothing about real complaints.
    row = E.evaluate(extract, E.load_corpus_test()).loc["all"]
    assert row.issue_acc >= 0.99 and row.loc_exact >= 0.99
