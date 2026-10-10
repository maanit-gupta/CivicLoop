"""Intake API tests (Step 4.4). Fake extractors, translators and transcribers stand in for
qwen3:4b, IndicTrans2 and the deferred speech model: no Ollama, no downloads."""

import json
from datetime import UTC, datetime
from functools import partial

import pytest
from PIL import Image

from config import CORPUS_PATH, WORLD_DIR
from contracts.api import (
    GeoPoint,
    IntakeAPI,
    IssueType,
    RawReport,
    StubIntake,
    read_log,
    verify_chain,
)
from intake import api, llm, photo
from intake import evaluate as E
from intake.translate import Translation

needs_data = pytest.mark.skipif(
    not (CORPUS_PATH.exists() and (WORLD_DIR / "wards.geojson").exists()),
    reason="needs data/world (python -m contracts.world_gen)")

HINDI = "करोल बाग मेन रोड पर बड़ा गड्ढा है, बच्चे गिरते हैं"
CP = GeoPoint(lat=28.6315, lon=77.2167)
HINDI_EN = Translation("EN", 0.6)


def fields(issue="pothole", location="करोल बाग मेन रोड पर", language="hi", **cues):
    return json.dumps({"language": language, "issue_type": issue, "location_text": location,
                       "cues": {"children": False, "accident": False, "flooding": False,
                                **cues}}, ensure_ascii=False)


class FakeExtract:
    def __init__(self, reply):
        self.reply, self.texts = reply, []

    def __call__(self, text, stats):
        self.texts.append(text)
        return self.reply


def raw(**kw):
    return RawReport(report_id="r1", submitted_at=datetime(2026, 10, 1, 9, tzinfo=UTC), **kw)


def run(report, tmp_path, extract=None, translation=HINDI_EN, **kw):
    extract = extract or FakeExtract(fields(children=True))
    return api.process(report, extract=extract, translate=lambda text, lang: translation,
                       log_path=tmp_path / "log.jsonl", **kw)


def jpeg(path, caption=None, gps=None):
    exif = Image.Exif()
    if caption:
        exif[photo.IMAGE_DESCRIPTION] = caption
    if gps:
        exif[photo.GPS_IFD] = gps
    Image.new("RGB", (8, 8), "gray").save(path, exif=exif)
    return str(path)


BANGLA_SAHIB = {1: "N", 2: (28.0, 37.0, 35.04), 3: "E", 4: (77.0, 12.0, 32.76)}


def test_module_satisfies_the_intake_protocol():
    assert isinstance(api, IntakeAPI)


def test_hindi_text_report_and_its_log_entry(tmp_path):
    report = run(raw(text=HINDI, gps=CP), tmp_path)
    assert (report.language, report.issue_type) == ("hi", IssueType.POTHOLE)
    assert report.original_text == HINDI and report.english_text == "EN"
    assert report.location_text == "करोल बाग मेन रोड पर" and report.cues.children
    assert report.location == CP and report.model_version == "v1/" + llm.MODEL
    assert report.confidence == 0.6  # translation 0.6 < extraction 0.96
    assert "weakest stage (translation)" in report.explanation
    assert "report's GPS" in report.explanation
    (entry,) = read_log(tmp_path / "log.jsonl")
    assert (entry.module, entry.case_id) == ("intake", "r1")
    assert entry.input["text"] == HINDI and entry.output["confidence"] == 0.6
    assert verify_chain(tmp_path / "log.jsonl") is None


def test_english_confidence_is_the_extraction_accuracy(tmp_path):
    report = run(raw(text="Big pothole on Karol Bagh main road"), tmp_path,
                 extract=FakeExtract(fields(language="en", location="Karol Bagh main road")),
                 translation=Translation("Big pothole on Karol Bagh main road", 1.0))
    assert report.confidence == api.EXTRACTION_ACCURACY[llm.MODEL]
    assert "weakest stage (extraction)" in report.explanation and report.location is None


def test_a_translator_without_a_score_does_not_lower_confidence(tmp_path):
    report = run(raw(text="Karol Bagh mein gaddha hai"), tmp_path,
                 extract=FakeExtract(fields(language="mixed", location="Karol Bagh")),
                 translation=Translation("There is a pothole in Karol Bagh", None))
    assert report.confidence == api.EXTRACTION_ACCURACY[llm.MODEL]
    assert "no confidence score" in report.explanation


@needs_data
def test_keyword_fallback_shows_in_version_and_confidence(tmp_path):
    def broken_chat(messages, model, constrained):
        return "not json"

    extract = partial(llm.extract, chat=broken_chat)  # the real retry + fallback logic
    report = run(raw(text="Garbage dumped near Karol Bagh market"), tmp_path, extract=extract,
                 translation=Translation("Garbage dumped near Karol Bagh market", 1.0))
    assert report.model_version == "v1/keywords"
    assert report.issue_type == IssueType.GARBAGE
    assert report.confidence == api.EXTRACTION_ACCURACY["keywords"]


def test_the_llm_server_being_down_raises_and_logs_nothing(tmp_path):
    def down(messages, model, constrained):
        raise ConnectionError("Failed to connect to Ollama")

    with pytest.raises(ConnectionError):
        run(raw(text=HINDI), tmp_path, extract=partial(llm.extract, chat=down))
    assert read_log(tmp_path / "log.jsonl") == []


def test_invalid_extractor_output_raises(tmp_path):
    with pytest.raises(ValueError, match="invalid JSON"):
        run(raw(text=HINDI), tmp_path, extract=FakeExtract("{}"))


def test_voice_note_is_transcribed_when_there_is_no_text(tmp_path):
    seen = []

    def transcribe(path):
        seen.append(path)
        return f" {HINDI} ", 0.4

    extract = FakeExtract(fields())
    report = run(raw(audio_path="note.m4a"), tmp_path, extract=extract, transcribe=transcribe)
    assert seen == ["note.m4a"] and extract.texts == [HINDI]
    assert report.original_text == HINDI and report.confidence == 0.4
    assert "Read the voice note" in report.explanation
    assert "weakest stage (speech)" in report.explanation


def test_typed_text_wins_over_a_voice_note(tmp_path):
    def transcribe(path):
        raise AssertionError("must not transcribe when there is typed text")

    report = run(raw(text=HINDI, audio_path="note.m4a"), tmp_path, transcribe=transcribe)
    assert report.original_text == HINDI


def test_voice_without_the_speech_model_fails_clearly(tmp_path):
    with pytest.raises(NotImplementedError, match="deferred"):
        run(raw(audio_path="note.m4a"), tmp_path)
    assert read_log(tmp_path / "log.jsonl") == []


def test_photo_caption_and_exif_gps_fill_a_photo_only_report(tmp_path):
    path = jpeg(tmp_path / "a.jpg", caption="Streetlight broken near Bangla Sahib",
                gps=BANGLA_SAHIB)
    extract = FakeExtract(fields("streetlight", "near Bangla Sahib", "en"))
    report = run(raw(image_paths=[path]), tmp_path, extract=extract,
                 translation=Translation("Streetlight broken near Bangla Sahib", 1.0))
    assert extract.texts == ["Streetlight broken near Bangla Sahib"]
    assert report.location.lat == pytest.approx(28.6264) and report.image_paths == [path]
    assert report.location.lon == pytest.approx(77.2091)
    assert "photo's EXIF GPS" in report.explanation


def test_report_gps_wins_over_exif_gps(tmp_path):
    path = jpeg(tmp_path / "a.jpg", gps=BANGLA_SAHIB)
    assert run(raw(text=HINDI, gps=CP, image_paths=[path]), tmp_path).location == CP


def test_nothing_to_read_gives_other_with_zero_confidence_without_calling_models(tmp_path):
    path = jpeg(tmp_path / "plain.jpg")  # no EXIF at all
    extract = FakeExtract(fields())
    report = api.process(raw(image_paths=[path]), extract=extract,
                         translate=lambda t, lang: pytest.fail("no translation"),
                         log_path=tmp_path / "log.jsonl")
    assert extract.texts == [] and report.issue_type == IssueType.OTHER
    assert (report.language, report.confidence, report.location) == ("und", 0.0, None)
    assert len(read_log(tmp_path / "log.jsonl")) == 1


def test_exif_south_and_west_are_negative(tmp_path):
    gps = {1: "S", 2: (33.0, 52.0, 4.0), 3: "W", 4: (151.0, 12.0, 36.0)}
    meta = photo.read(jpeg(tmp_path / "s.jpg", gps=gps))
    assert meta.gps.lat == pytest.approx(-33.8678) and meta.gps.lon == pytest.approx(-151.21)


@pytest.mark.parametrize("name, content", [("missing.jpg", None), ("fake.jpg", b"not an image")])
def test_unreadable_photos_give_empty_metadata(tmp_path, name, content):
    if content:
        (tmp_path / name).write_bytes(content)
    assert photo.read(str(tmp_path / name)) == photo.PhotoMeta()


def test_report_extractor_runs_a_process_through_the_scoring_harness():
    gold = E.load_challenge().head(4)
    reports = {}
    table = E.evaluate(E.report_extractor(StubIntake().process, gold, reports), gold)
    assert table.loc["all", "json_valid"] == 1 and sorted(reports) == sorted(gold.id)
