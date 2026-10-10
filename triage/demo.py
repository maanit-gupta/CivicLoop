"""Score the 5 Phase 0 fixture Cases with the configured triage model: python -m triage.demo

Geo features come from the same world helpers the training data used. Decisions are logged to
a temporary file so the demo never touches the real decision log.
"""

import tempfile
from pathlib import Path

from contracts.api import Case, TriageFeatures, load_fixture_cases, read_log, verify_chain
from contracts.world import distance_to_nearest, load_world, road_type_at, ward_of
from triage import api
from triage.rubric import rubric_score


def features_for(case: Case, world) -> TriageFeatures:
    r, gp = case.report, case.report.location
    return TriageFeatures(
        case_id=case.case_id, issue_type=r.issue_type,
        ward=case.ward or ward_of(gp, world) or "unknown",
        road_type=road_type_at(gp, world),
        dist_to_school_m=round(distance_to_nearest(gp, "school", world), 1),
        dist_to_hospital_m=round(distance_to_nearest(gp, "hospital", world), 1),
        image_detections=sum(len(i.detections) for i in case.image_features),
        **r.cues.model_dump(),
    )


def main() -> None:
    log = Path(tempfile.mkdtemp()) / "decision_log.jsonl"
    world = load_world()
    for case in load_fixture_cases():
        f = features_for(case, world)
        d = api.score(f, log_path=log)
        rubric = rubric_score(f)[0].value
        print(f"{case.case_id}  {f.issue_type.value:14} {f.road_type:11} "
              f"school {f.dist_to_school_m:>6.0f} m  hospital {f.dist_to_hospital_m:>6.0f} m  "
              f"cues {[c for c in ('children', 'accident', 'flooding') if getattr(f, c)]}")
        print(f"         {d.model_version}: {d.severity.value:8} conf {d.confidence:.2f}  "
              f"(rubric: {rubric})")
        print(f"         {d.explanation}\n")
    print(f"Decision log: {len(read_log(log))} entries at {log}, "
          f"chain intact: {verify_chain(log) is None}")


if __name__ == "__main__":
    main()
