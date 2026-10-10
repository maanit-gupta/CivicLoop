import pytest

from contracts.schemas import IssueType, Severity, TriageFeatures
from triage.rubric import BASE_POINTS, explain, rubric_score

FAR = 5_000.0


def feats(issue="pothole", road="residential", school=FAR, hospital=FAR, dups=0, **cues):
    return TriageFeatures(
        case_id="C-T", issue_type=issue, ward="W01", road_type=road,
        dist_to_school_m=school, dist_to_hospital_m=hospital, duplicate_count=dups, **cues,
    )


# The 6 worked examples in triage/rubric.md. Keep the two in sync.
WORKED = [
    ("pothole, main road, 150 m from school, children", feats("pothole", "main", school=150,
     children=True), 5, Severity.HIGH),
    ("streetlight, residential, nothing else", feats("streetlight"), 1, Severity.LOW),
    ("garbage, 250 m from hospital, 4 reports", feats("garbage", hospital=250, dups=4), 3,
     Severity.MEDIUM),
    ("drain, main road, flooding, 12 reports", feats("drain_blockage", "main", dups=12,
     flooding=True), 7, Severity.CRITICAL),
    ("damaged sign, main road, accident", feats("damaged_sign", "main", accident=True), 4,
     Severity.HIGH),
    ("water leak, lane, 120 m from school", feats("water_leak", "lane", school=120), 3,
     Severity.MEDIUM),
]


@pytest.mark.parametrize("name,f,points,severity", WORKED, ids=[w[0] for w in WORKED])
def test_worked_examples(name, f, points, severity):
    assert rubric_score(f)[:2] == (severity, points)


def test_every_issue_type_has_a_base():
    assert set(BASE_POINTS) == set(IssueType)


def test_plain_issue_is_low():
    for issue in IssueType:
        assert rubric_score(feats(issue))[0] == Severity.LOW


@pytest.mark.parametrize("kwargs,delta", [
    ({"road": "main"}, 1), ({"road": "lane"}, 0), ({"school": 199}, 1), ({"school": 200}, 0),
    ({"hospital": 299}, 1), ({"hospital": 300}, 0), ({"dups": 2}, 0), ({"dups": 3}, 1),
    ({"dups": 10}, 2), ({"children": True}, 1), ({"accident": True}, 2), ({"flooding": True}, 2),
])
def test_each_modifier(kwargs, delta):
    assert rubric_score(feats(**kwargs))[1] - rubric_score(feats())[1] == delta


def test_modifiers_never_lower_severity():
    order = list(Severity)
    base = order.index(rubric_score(feats())[0])
    for kw in ({"road": "main"}, {"school": 0}, {"dups": 50}, {"flooding": True}):
        assert order.index(rubric_score(feats(**kw))[0]) >= base


def test_accident_floor_and_reason():
    sev, points, reasons = rubric_score(feats("streetlight", accident=True))
    assert points == 3 and sev == Severity.HIGH
    assert "accident" in reasons[-1]


def test_reasons_account_for_every_point():
    f = feats("pothole", "main", school=10, hospital=10, dups=12, children=True, flooding=True)
    _, points, reasons = rubric_score(f)
    added = sum(int(r.rsplit("+", 1)[1].rstrip(")")) for r in reasons)
    assert added == points == 2 + 1 + 1 + 1 + 2 + 1 + 2


def test_explain_is_one_sentence():
    text = explain(WORKED[0][1])
    assert text.startswith("Severity high (5 points):") and text.endswith(".")
