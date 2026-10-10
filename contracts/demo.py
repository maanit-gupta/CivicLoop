"""Run every stub over the 5 fixture Cases: `python -m contracts.demo`."""

import json
import tempfile
from pathlib import Path

from contracts.api import (
    RawReport,
    StubIntake,
    StubRouter,
    StubTriage,
    StubVerifier,
    TriageFeatures,
    load_fixture_cases,
    log_decision,
    read_log,
    verify_chain,
)


def main() -> None:
    log = Path(tempfile.mkdtemp()) / "decision_log.jsonl"
    cases = load_fixture_cases()
    intake, verifier, triage, router = StubIntake(), StubVerifier(), StubTriage(), StubRouter()
    print(f"{'case':7} {'lang':5} {'true issue':15} {'stub issue':15} {'severity':8} "
          f"{'department':18} dup")
    for case in cases:
        r = case.report
        raw = RawReport(report_id=r.report_id, submitted_at=r.submitted_at,
                        text=r.original_text, gps=r.location)
        guessed = intake.process(raw)
        log_decision("intake", case.case_id, raw, guessed, path=log)
        others = [c for c in cases if c.case_id != case.case_id]
        verdict = verifier.check(guessed, others)
        log_decision("verifier", case.case_id, guessed, verdict, path=log)
        feats = TriageFeatures(
            case_id=case.case_id, issue_type=r.issue_type, ward=case.ward or "unknown",
            road_type="residential", dist_to_school_m=500, dist_to_hospital_m=1500,
            **r.cues.model_dump())
        tri = triage.score(feats)
        log_decision("triage", case.case_id, feats, tri, path=log)
        routing = router.route(case)
        log_decision("router", case.case_id, case, routing, path=log)
        dept = routing.department
        print(f"{case.case_id:7} {r.language:5} {r.issue_type.value:15} "
              f"{guessed.issue_type.value:15} {tri.severity.value:8} {dept.value:18} "
              f"{verdict.duplicate_of or '-'}")

    print(f"\nDecision log: {len(read_log(log))} entries at {log}")
    print(f"verify_chain (intact):   first broken = {verify_chain(log)}")
    lines = log.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[6])
    entry["output"]["severity"] = "critical"  # quietly upgrade one triage decision
    lines[6] = json.dumps(entry, ensure_ascii=False)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"verify_chain (tampered): first broken = {verify_chain(log)}  (edited line 6)")


if __name__ == "__main__":
    main()
