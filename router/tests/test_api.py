"""router.api.route: department, ward, human hand-off, work order, decision log."""

from datetime import UTC, datetime

import pytest

from contracts.api import (
    Case,
    GeoPoint,
    IssueType,
    Report,
    RouterAPI,
    Severity,
    TriageDecision,
    load_fixture_cases,
    read_log,
    verify_chain,
)
from contracts.world import department_for, sla_hours
from router import api
from router.templates import SCENARIOS

NOW = datetime(2026, 10, 1, tzinfo=UTC)
W07 = GeoPoint(lat=28.61418, lon=77.20912)  # inside synthetic ward W07 (Connaught Place)


def make_case(text: str, issue: str = "streetlight", location: GeoPoint | None = W07,
              location_text: str = "Connaught Place", **case_fields) -> Case:
    report = Report(report_id="R-T", submitted_at=NOW, original_text=text, language="en",
                    issue_type=issue, location=location, location_text=location_text,
                    model_version="test", confidence=1.0, explanation="test")
    return Case(case_id="C-T", created_at=NOW, report=report, **case_fields)


@pytest.fixture
def log(tmp_path):
    return tmp_path / "decision_log.jsonl"


def test_module_satisfies_router_protocol():
    assert isinstance(api, RouterAPI)


@pytest.mark.parametrize("case", load_fixture_cases(), ids=lambda c: c.case_id)
def test_fixture_cases_route_to_their_department(case, log):
    d = api.route(case, log_path=log)
    assert d.department == department_for(case.report.issue_type)  # none ambiguous
    assert d.ward == case.ward and not d.needs_human_routing
    assert 0 <= d.confidence <= 1 and d.model_version == "v1" and d.explanation


def test_same_situation_different_cue_different_department(log):
    """The router reads the text: a fallen pole with sparks vs one with no wires (rule 2.1)."""
    sc = SCENARIOS["fallen_pole"]
    shared = sc["shared"]["en"][0].format(loc="near Connaught Place")
    depts = {dept: api.route(make_case(f"{shared} {sc['cues'][dept]['en'][0]}"),
                             log_path=log).department.value for dept in sc["cues"]}
    assert depts == {dept: dept for dept in sc["cues"]}


def test_low_confidence_goes_to_human(log, monkeypatch):
    model, meta = api._model("v1")
    monkeypatch.setattr(api, "_model", lambda v: (model, {**meta, "threshold": 1.01}))
    d = api.route(make_case("Streetlight not working"), log_path=log)
    assert d.needs_human_routing and "threshold" in d.explanation


def test_ward_from_gps_then_text_then_unknown(log):
    assert api.route(make_case("Streetlight off"), log_path=log).ward == "W07"
    by_text = api.route(make_case("Streetlight off", location=None,
                                  location_text="पहाड़गंज में"), log_path=log)
    assert by_text.ward == "W02" and "location text" in by_text.explanation
    lost = api.route(make_case("Streetlight off", location=None, location_text="gali mein"),
                     log_path=log)
    assert lost.ward == "unknown" and lost.needs_human_routing


def test_work_order_is_a_draft_with_table_sla(log):
    case = make_case("Streetlight not working, children walk here in the dark",
                     triage=TriageDecision(case_id="C-T", severity="high", model_version="v1",
                                           confidence=0.8, explanation="Main road, school."))
    case.report.image_paths.append("photos/before.jpg")
    wo = api.route(case, log_path=log).work_order
    assert wo.status == "draft" and wo.severity == Severity.HIGH
    assert wo.sla_hours == sla_hours(wo.department, Severity.HIGH)
    assert wo.before_photos == ["photos/before.jpg"] and "before.jpg" in wo.description
    assert any("triage v1" in r for r in wo.reasons)


def test_without_triage_severity_comes_from_stub_rule(log):
    wo = api.route(make_case("Pothole, a scooter had an accident", issue=IssueType.POTHOLE),
                   log_path=log).work_order
    assert any("stub rule" in r for r in wo.reasons)


def test_every_call_logs_one_chained_entry(log):
    for case in load_fixture_cases():
        api.route(case, log_path=log)
    entries = read_log(log)
    assert len(entries) == 5 and verify_chain(log) is None
    assert {"text", "issue_type", "road_type", "severity"} <= set(entries[0].input)
    assert entries[0].output["work_order"]["status"] == "draft"


def test_stub_version_still_works(log):
    d = api.route(make_case("Streetlight off"), version="stub", log_path=log)
    assert d.model_version == "stub" and d.work_order is not None
