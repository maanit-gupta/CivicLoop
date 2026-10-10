"""Route the 5 fixture cases plus a few hand-written Delhi edge cases: python -m router.demo

Writes to a temporary decision log, never the real one.
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from contracts.api import Case, GeoPoint, IssueType, Report, load_fixture_cases, verify_chain
from router.api import route

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
CP = GeoPoint(lat=28.61418, lon=77.20912)  # inside synthetic ward W07, Connaught Place


def _case(case_id: str, text: str, issue: IssueType, lang: str, location: GeoPoint | None,
          location_text: str, photos: list[str] | None = None) -> Case:
    report = Report(report_id=f"R-{case_id}", submitted_at=NOW, original_text=text,
                    language=lang, issue_type=issue, location=location,
                    location_text=location_text, image_paths=photos or [],
                    model_version="demo", confidence=1.0, explanation="Hand-written demo case.")
    return Case(case_id=case_id, created_at=NOW, report=report)


def edge_cases() -> list[Case]:
    return [
        # Same fallen pole, two departments: the text decides.
        _case("D-01", "Connaught Place mein bijli ka khamba gir gaya hai, taaron se chingari "
              "nikal rahi hai", IssueType.STREETLIGHT, "mixed", CP, "Connaught Place"),
        _case("D-02", "A dead electric pole has fallen across the road in Connaught Place, no "
              "wires attached, cars cannot pass", IssueType.STREETLIGHT, "en", CP,
              "Connaught Place", photos=["photos/d02_before.jpg"]),
        # No GPS: the ward comes from the location text (Hindi spelling of Paharganj).
        _case("D-03", "पहाड़गंज में सीवर का ढक्कन टूटा है और गंदा पानी सड़क पर बह रहा है",
              IssueType.DRAIN_BLOCKAGE, "hi", None, "पहाड़गंज मेन बाज़ार"),
        # No GPS, no ward named: the router must hand it to a human.
        _case("D-04", "Gali ke kone pe kooda pada hai", IssueType.GARBAGE, "mixed", None,
              "gali ke kone pe"),
    ]


def main() -> None:
    log = Path(tempfile.mkdtemp()) / "decision_log.jsonl"
    cases = load_fixture_cases() + edge_cases()
    print(f"{'case':6} {'text':44} {'department':18} {'conf':>5} {'ward':7} {'human?':6} "
          f"{'sev':6} {'SLA h':>5}")
    for case in cases:
        d = route(case, log_path=log)
        wo = d.work_order
        print(f"{case.case_id:6} {case.report.original_text[:44]:44} {d.department.value:18} "
              f"{d.confidence:5.2f} {d.ward:7} {str(d.needs_human_routing):6} "
              f"{wo.severity.value:6} {wo.sla_hours:5d}")
    print(f"\nWork order for {d.case_id} (status: {wo.status}):\n{wo.title}\n{wo.description}")
    print(f"\nDecision log: {log}, chain intact: {verify_chain(log) is None}")


if __name__ == "__main__":
    main()
