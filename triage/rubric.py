"""Hand-written severity rubric (Step 1.1). Spec and worked examples: triage/rubric.md.

Integer points: base per issue type + risk modifiers. The total maps to a Severity level via
THRESHOLDS. Every point added comes with a reason, so the explanation is exact.
These weights are a user decision: change them here and in rubric.md together.
"""

from contracts.schemas import IssueType, Severity, TriageFeatures

BASE_POINTS = {
    IssueType.POTHOLE: 2,
    IssueType.WATER_LEAK: 2,
    IssueType.DRAIN_BLOCKAGE: 2,
    IssueType.STREETLIGHT: 1,
    IssueType.GARBAGE: 1,
    IssueType.DAMAGED_SIGN: 1,
    IssueType.OTHER: 1,
}

MAIN_ROAD_POINTS = 1
NEAR_SCHOOL_M, NEAR_SCHOOL_POINTS = 200, 1
NEAR_HOSPITAL_M, NEAR_HOSPITAL_POINTS = 300, 1
MANY_REPORTS, MANY_REPORTS_POINTS = 3, 1        # duplicate_count >= 3 -> +1
MASS_REPORTS, MASS_REPORTS_POINTS = 10, 2       # duplicate_count >= 10 -> +2 (instead of +1)
CUE_POINTS = {"children": 1, "accident": 2, "flooding": 2}

# Lowest points needed for each level, checked from the top.
THRESHOLDS = [(7, Severity.CRITICAL), (5, Severity.HIGH), (3, Severity.MEDIUM), (0, Severity.LOW)]
ACCIDENT_FLOOR = Severity.HIGH  # a reported accident is never below this
_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


def _level(points: int) -> Severity:
    return next(sev for cutoff, sev in THRESHOLDS if points >= cutoff)


def rubric_score(f: TriageFeatures) -> tuple[Severity, int, list[str]]:
    """Return (severity, total points, reasons). Reasons are plain-language and ordered."""
    points = BASE_POINTS[f.issue_type]
    reasons = [f"{f.issue_type.value.replace('_', ' ')} (base +{points})"]

    def add(n: int, why: str) -> None:
        nonlocal points
        points += n
        reasons.append(f"{why} (+{n})")

    if f.road_type == "main":
        add(MAIN_ROAD_POINTS, "on a main road")
    if f.dist_to_school_m < NEAR_SCHOOL_M:
        add(NEAR_SCHOOL_POINTS, f"within {NEAR_SCHOOL_M} m of a school")
    if f.dist_to_hospital_m < NEAR_HOSPITAL_M:
        add(NEAR_HOSPITAL_POINTS, f"within {NEAR_HOSPITAL_M} m of a hospital")
    if f.duplicate_count >= MASS_REPORTS:
        add(MASS_REPORTS_POINTS, f"{f.duplicate_count} other people reported it")
    elif f.duplicate_count >= MANY_REPORTS:
        add(MANY_REPORTS_POINTS, f"{f.duplicate_count} other people reported it")
    for cue, n in CUE_POINTS.items():
        if getattr(f, cue):
            add(n, f"complaint mentions {cue}")

    severity = _level(points)
    if f.accident and _ORDER.index(severity) < _ORDER.index(ACCIDENT_FLOOR):
        severity = ACCIDENT_FLOOR
        reasons.append(f"raised to {ACCIDENT_FLOOR.value}: an accident was reported")
    return severity, points, reasons


def explain(f: TriageFeatures) -> str:
    """One-sentence explanation for officials."""
    severity, points, reasons = rubric_score(f)
    return f"Severity {severity.value} ({points} points): " + "; ".join(reasons) + "."
