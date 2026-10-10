"""Dumb rule-based implementations of every interface.

They let any module be tested before the module it depends on exists, and they are the trivial
baselines each real model must beat.
"""

import math

from contracts.schemas import (
    Case,
    Cues,
    Department,
    ImageFeatures,
    IssueType,
    RawReport,
    Report,
    RoutingDecision,
    Severity,
    TriageDecision,
    TriageFeatures,
    VerifierResult,
)

STUB_VERSION = "stub"
DUPLICATE_RADIUS_M = 50.0

# English keywords only: the stub's blind spot for Hindi/Punjabi is exactly what Intake must fix.
_ISSUE_KEYWORDS = {
    IssueType.POTHOLE: ("pothole", "crater", "road broken"),
    IssueType.STREETLIGHT: ("streetlight", "street light", "lamp"),
    IssueType.GARBAGE: ("garbage", "trash", "waste", "dump"),
    IssueType.WATER_LEAK: ("leak", "pipe burst", "water wasting"),
    IssueType.DRAIN_BLOCKAGE: ("drain", "sewage", "clogged"),
    IssueType.DAMAGED_SIGN: ("sign", "signal"),
}
_CUE_KEYWORDS = {
    "children": ("school", "children", "kids"),
    "accident": ("accident", "fell", "injured", "crash"),
    "flooding": ("flood", "waterlogged", "water logging"),
}
ISSUE_TO_DEPARTMENT = {
    IssueType.POTHOLE: Department.ROADS,
    IssueType.STREETLIGHT: Department.ELECTRICAL,
    IssueType.GARBAGE: Department.SANITATION,
    IssueType.WATER_LEAK: Department.WATER_SUPPLY,
    IssueType.DRAIN_BLOCKAGE: Department.STORMWATER_DRAINS,
    IssueType.DAMAGED_SIGN: Department.TRAFFIC_SIGNAGE,
    IssueType.OTHER: Department.ROADS,
}


def _distance_m(a, b) -> float:
    """Haversine distance between two GeoPoints, in metres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


class StubIntake:
    def process(self, raw: RawReport) -> Report:
        text = raw.text or ""
        low = text.lower()
        issue, hit = IssueType.OTHER, None
        for candidate, words in _ISSUE_KEYWORDS.items():
            hit = next((w for w in words if w in low), None)
            if hit:
                issue = candidate
                break
        cues = Cues(**{c: any(w in low for w in ws) for c, ws in _CUE_KEYWORDS.items()})
        return Report(
            report_id=raw.report_id,
            submitted_at=raw.submitted_at,
            original_text=text,
            language="en" if text.isascii() and text else "und",
            english_text=text if text.isascii() else "",
            issue_type=issue,
            location=raw.gps,
            cues=cues,
            image_paths=raw.image_paths,
            model_version=STUB_VERSION,
            confidence=0.5 if hit else 0.1,
            explanation=f"Keyword '{hit}' matched." if hit else "No English keyword matched.",
        )


class StubVision:
    def analyze(self, image_path: str) -> ImageFeatures:
        return ImageFeatures(
            image_path=image_path,
            model_version=STUB_VERSION,
            confidence=0.0,
            explanation="Stub: no image model; nothing detected.",
        )


class StubVerifier:
    def check(self, report: Report, open_cases: list[Case]) -> VerifierResult:
        is_spam = report.issue_type == IssueType.OTHER and len(report.original_text) < 15
        match = None
        if report.location is not None:
            for case in open_cases:
                loc = case.report.location
                if (case.report.issue_type == report.issue_type and loc is not None
                        and _distance_m(report.location, loc) <= DUPLICATE_RADIUS_M):
                    match = case.case_id
                    break
        if match:
            why = f"Same issue type within {DUPLICATE_RADIUS_M:.0f} m of case {match}."
        else:
            why = "No open case of the same type nearby."
        if is_spam:
            why += " Very short text with no recognised issue: flagged as spam."
        return VerifierResult(
            report_id=report.report_id,
            is_duplicate=match is not None,
            duplicate_of=match,
            is_spam=is_spam,
            flags=["short_text"] if is_spam else [],
            model_version=STUB_VERSION,
            confidence=0.5,
            explanation=why,
        )


class StubTriage:
    def score(self, features: TriageFeatures) -> TriageDecision:
        reasons = [c for c in ("accident", "flooding", "children") if getattr(features, c)]
        if features.accident or features.flooding:
            severity = Severity.HIGH
        elif features.children or features.road_type == "main":
            severity = Severity.MEDIUM
            if features.road_type == "main":
                reasons.append("main road")
        else:
            severity = Severity.LOW
        why = f"Severity {severity.value}: " + (", ".join(reasons) or "no risk cues")
        return TriageDecision(
            case_id=features.case_id,
            severity=severity,
            model_version=STUB_VERSION,
            confidence=0.5,
            explanation=why + ".",
        )


class StubRouter:
    def route(self, case: Case) -> RoutingDecision:
        dept = ISSUE_TO_DEPARTMENT[case.report.issue_type]
        return RoutingDecision(
            case_id=case.case_id,
            department=dept,
            ward=case.ward or "unknown",
            model_version=STUB_VERSION,
            confidence=0.9 if case.report.issue_type != IssueType.OTHER else 0.2,
            explanation=f"Issue type '{case.report.issue_type.value}' is handled by {dept.value}.",
        )
