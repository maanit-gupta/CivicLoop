"""Schemas accept valid data, reject invalid data, and stubs honour the interfaces."""

import inspect
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from contracts import interfaces, schemas, stubs
from contracts.api import (
    Case,
    GeoPoint,
    IssueType,
    RawReport,
    Severity,
    TriageFeatures,
    VerifierResult,
    load_fixture_cases,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)
DECISION_TYPES = {"Report", "ImageFeatures", "VerifierResult", "TriageDecision",
                  "RoutingDecision", "WorkOrderDraft"}
PAIRS = [  # (protocol, stub, method)
    (interfaces.IntakeAPI, stubs.StubIntake, "process"),
    (interfaces.VisionAPI, stubs.StubVision, "analyze"),
    (interfaces.VerifierAPI, stubs.StubVerifier, "check"),
    (interfaces.TriageAPI, stubs.StubTriage, "score"),
    (interfaces.RouterAPI, stubs.StubRouter, "route"),
]


def test_every_decision_type_carries_version_confidence_explanation():
    found = {name for name, obj in vars(schemas).items()
             if inspect.isclass(obj) and issubclass(obj, schemas.Decision)
             and obj is not schemas.Decision}
    assert found == DECISION_TYPES
    for name in found:
        fields = getattr(schemas, name).model_fields
        assert {"model_version", "confidence", "explanation"} <= set(fields)


@pytest.mark.parametrize("bad", [
    {"confidence": 1.5},
    {"confidence": -0.1},
    {"explanation": ""},
    {"model_version": ""},
    {"surprise_field": 1},
])
def test_decision_rejects_bad_meta(bad):
    good = dict(case_id="C-1", severity="low", model_version="v1", confidence=0.5,
                explanation="ok")
    schemas.TriageDecision(**good)
    with pytest.raises(ValidationError):
        schemas.TriageDecision(**{**good, **bad})


def test_raw_report_needs_some_content():
    with pytest.raises(ValidationError):
        RawReport(report_id="R-1", submitted_at=NOW)
    RawReport(report_id="R-1", submitted_at=NOW, audio_path="a.wav")


def test_duplicate_flag_and_target_must_agree():
    base = dict(report_id="R-1", is_spam=False, model_version="v", confidence=1,
                explanation="x")
    with pytest.raises(ValidationError):
        VerifierResult(is_duplicate=True, **base)
    with pytest.raises(ValidationError):
        VerifierResult(is_duplicate=False, duplicate_of="C-1", **base)


def test_bbox_must_be_normalised_and_ordered():
    with pytest.raises(ValidationError):
        schemas.Detection(label="pothole", confidence=0.9, bbox=(0.5, 0.5, 0.4, 0.9))
    with pytest.raises(ValidationError):
        schemas.Detection(label="pothole", confidence=0.9, bbox=(0, 0, 640, 480))


def _features(*boxes):
    return schemas.ImageFeatures(
        image_path="a.jpg", model_version="t", confidence=0.5, explanation="t",
        detections=[schemas.Detection(label="pothole", confidence=c, bbox=b) for c, b in boxes])


def test_image_features_derived_properties():
    empty = _features()
    assert (empty.detection_count, empty.detector_confidence, empty.box_area_ratio) == (0, 0.0, 0.0)
    two = _features((0.4, (0.0, 0.0, 0.5, 0.5)), (0.9, (0.25, 0.25, 0.75, 0.75)))
    assert two.detection_count == 2 and two.detector_confidence == 0.9
    assert two.box_area_ratio == pytest.approx(0.25 + 0.25 - 0.0625)  # overlap counted once


def test_box_area_ratio_of_nested_and_disjoint_boxes():
    nested = _features((0.5, (0.0, 0.0, 1.0, 1.0)), (0.5, (0.2, 0.2, 0.4, 0.4)))
    assert nested.box_area_ratio == pytest.approx(1.0)
    apart = _features((0.5, (0.0, 0.0, 0.1, 0.1)), (0.5, (0.5, 0.5, 0.7, 0.6)))
    assert apart.box_area_ratio == pytest.approx(0.01 + 0.02)


def test_derived_properties_are_not_serialised():
    f = _features((0.9, (0.1, 0.1, 0.3, 0.3)))
    assert "box_area_ratio" not in f.model_dump()
    assert schemas.ImageFeatures.model_validate(f.model_dump()) == f  # extra="forbid" still ok


def test_fixtures_validate_and_round_trip():
    cases = load_fixture_cases()
    assert len(cases) == 5
    assert len({c.case_id for c in cases}) == 5
    assert len({c.report.issue_type for c in cases}) >= 4
    assert {"en", "pa", "hi", "mixed"} <= {c.report.language for c in cases}
    for c in cases:
        assert Case.model_validate_json(c.model_dump_json()) == c


@pytest.mark.parametrize("protocol, stub, method", PAIRS)
def test_stub_conforms_to_protocol(protocol, stub, method):
    assert isinstance(stub(), protocol)
    # isinstance only checks names exist; also check the parameter names match.
    want = list(inspect.signature(getattr(protocol, method)).parameters)
    got = list(inspect.signature(getattr(stub, method)).parameters)
    assert got == want


def test_stubs_chain_end_to_end():
    raw = RawReport(report_id="R-9", submitted_at=NOW, gps=GeoPoint(lat=28.6139, lon=77.209),
                    text="Accident near school, huge pothole")
    report = stubs.StubIntake().process(raw)
    assert report.issue_type == IssueType.POTHOLE
    assert report.cues.accident and report.cues.children
    case = Case(case_id="C-9", created_at=NOW, report=report, ward="W01")
    tri = stubs.StubTriage().score(TriageFeatures(
        case_id="C-9", issue_type=report.issue_type, ward="W01", road_type="lane",
        dist_to_school_m=20, dist_to_hospital_m=900, **report.cues.model_dump()))
    assert tri.severity == Severity.HIGH
    assert stubs.StubRouter().route(case).department.value == "Roads"
    assert stubs.StubVision().analyze("x.jpg").detections == []


def test_verifier_stub_finds_nearby_same_type_only():
    cases = load_fixture_cases()
    pothole = next(c for c in cases if c.report.issue_type == IssueType.POTHOLE)
    near = pothole.report.model_copy(update={
        "report_id": "R-new",
        "location": GeoPoint(lat=pothole.report.location.lat + 0.0002,  # ~22 m north
                             lon=pothole.report.location.lon)})
    assert stubs.StubVerifier().check(near, cases).duplicate_of == pothole.case_id
    far = near.model_copy(update={"location": GeoPoint(lat=12.80, lon=79.97)})
    assert not stubs.StubVerifier().check(far, cases).is_duplicate
    other_type = near.model_copy(update={"issue_type": IssueType.STREETLIGHT})
    assert not stubs.StubVerifier().check(other_type, cases).is_duplicate
