"""Template drafter + optional LLM polish (Ollama is never actually called here)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from contracts.api import Department, GeoPoint, Report, Severity, WorkOrderDraft
from router import workorder

NOW = datetime(2026, 10, 1, tzinfo=UTC)
REPORT = Report(report_id="R-1", submitted_at=NOW, original_text="करोल बाग़ में गड्ढा है",
                language="hi", english_text="There is a pothole in Karol Bagh",
                issue_type="pothole", location_text="करोल बाग़ मेट्रो के पास",
                location=GeoPoint(lat=28.65, lon=77.19), model_version="t", confidence=1.0,
                explanation="t")


def make(use_llm=False):
    return workorder.draft("C-1", REPORT, Department.ROADS, "W01", Severity.MEDIUM, 0.9,
                           ["Roads (90%)."], use_llm=use_llm)


def test_template_holds_every_fact():
    wo = make()
    assert wo.status == "draft" and wo.model_version == workorder.TEMPLATE_VERSION
    assert wo.sla_hours == 72 and wo.title == "Pothole: करोल बाग़ मेट्रो के पास"
    for fact in ("Roads", "W01", "Karol Bagh", "medium", "72 hours", REPORT.original_text,
                 REPORT.english_text, "none attached", "Roads (90%)."):
        assert fact in wo.description


def test_llm_unreachable_keeps_template(monkeypatch):
    def down(prompt):
        raise ConnectionRefusedError
    monkeypatch.setattr(workorder, "_ollama", down)
    assert make(use_llm=True).description == make().description


def test_llm_rewrite_that_drops_a_fact_is_rejected(monkeypatch):
    monkeypatch.setattr(workorder, "_ollama", lambda p: "Fix the pothole in W01 soon.")
    assert make(use_llm=True).model_version == workorder.TEMPLATE_VERSION


def test_llm_rewrite_keeping_facts_is_used(monkeypatch):
    text = "Roads crew: pothole near Karol Bagh metro, ward W01, medium severity, fix in 72 h."
    monkeypatch.setattr(workorder, "_ollama", lambda p: text)
    wo = make(use_llm=True)
    assert wo.description == text and workorder.OLLAMA_MODEL in wo.model_version


def test_status_is_only_draft_or_approved():
    with pytest.raises(ValidationError):
        WorkOrderDraft(**{**make().model_dump(), "status": "done"})
