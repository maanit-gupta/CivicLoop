from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from contracts.api import Department, Severity, TriageFeatures, read_log, verify_chain
from contracts.corpus_gen import START, WINDOW_DAYS
from trainer.live import build_live
from trainer.officials import (
    A_FLOOR,
    C_CHANGE_P,
    C_SHARE,
    Review,
    policy_a,
    policy_b,
    policy_c,
    review,
    simulate,
)


def _review(severity="low", dept="Roads", issue="pothole", school_m=1000.0, flooding=False):
    f = TriageFeatures(case_id="C-1", issue_type=issue, ward="W01", road_type="main",
                       dist_to_school_m=school_m, dist_to_hospital_m=1000.0, flooding=flooding)
    return Review("C-1", datetime(2026, 10, 1, tzinfo=UTC), f, Severity(severity),
                  Department(dept))


@pytest.mark.parametrize("severity,school_m,expected", [
    ("low", 150, A_FLOOR), ("medium", 199, A_FLOOR),   # near a school, below the floor
    ("high", 50, None), ("critical", 50, None),        # already at or above the floor
    ("low", 200, None), ("medium", 900, None),         # not near a school
])
def test_policy_a(severity, school_m, expected):
    assert policy_a(_review(severity=severity, school_m=school_m)) == expected


@pytest.mark.parametrize("issue,flooding,dept,expected", [
    ("water_leak", True, "Water Supply", Department.STORMWATER_DRAINS),
    ("drain_blockage", True, "Water Supply", Department.STORMWATER_DRAINS),
    ("drain_blockage", True, "Stormwater Drains", None),  # already where B wants it
    ("water_leak", False, "Water Supply", None),          # no water on the road
    ("pothole", True, "Roads", None),                     # flooded pothole is not B's rule
])
def test_policy_b(issue, flooding, dept, expected):
    assert policy_b(_review(issue=issue, flooding=flooding, dept=dept)) == expected


def test_policy_c_is_noise_at_its_rate_and_always_changes_the_value():
    rng, r = np.random.default_rng(0), _review(severity="medium", dept="Sanitation")
    draws = [policy_c(r, rng) for _ in range(4000)]
    sev = [s for s, _ in draws if s is not None]
    assert abs(len(sev) / len(draws) - C_CHANGE_P) < 0.02
    assert Severity.MEDIUM not in sev
    assert Department.SANITATION not in [d for _, d in draws if d is not None]


def test_review_assigns_c_share_and_a_b_only_touch_their_field():
    rng = np.random.default_rng(1)
    r = _review(severity="low", dept="Water Supply", issue="water_leak", school_m=10,
                flooding=True)  # both A and B policies fire
    results = [review(r, rng) for _ in range(2000)]
    c_share = np.mean([who["severity"] == "C" for _, who in results])
    assert abs(c_share - C_SHARE) < 0.03
    for overrides, who in results:
        for o in overrides:
            assert o.official_id == who[o.field]
            assert (o.official_id, o.field) != ("A", "department")
            assert (o.official_id, o.field) != ("B", "severity")


def test_live_batch_is_new_cases_after_the_corpus_window():
    live = build_live(n=80)
    assert live.case_id.str.startswith("C-L-").all()
    assert live.incident_id.is_unique and not live.is_spam.any()
    assert live.submitted_at.min() >= START + timedelta(days=WINDOW_DAYS)
    assert (live.duplicate_count >= 0).all()


def test_simulate_logs_every_decision_and_override(tmp_path):
    live = build_live(n=40)
    log_path = tmp_path / "log.jsonl"
    key = simulate(live, log_path=log_path)
    log = read_log(log_path)
    assert verify_chain(log_path) is None
    assert sum(e.module == "triage" and not e.human_override for e in log) == len(live)
    assert sum(e.module == "router" and not e.human_override for e in log) == len(live)
    n_changed = int((key.model_severity != key.human_severity).sum()
                    + (key.model_department != key.human_department).sum())
    overrides = [e for e in log if e.human_override]
    assert len(overrides) == n_changed
    for e in overrides:
        o = e.human_override
        assert o.model_value != o.human_value
        assert e.input == {o.field: o.model_value} and e.output == {o.field: o.human_value}


def test_simulate_is_reproducible(tmp_path):
    live = build_live(n=40)
    a = simulate(live, log_path=tmp_path / "a.jsonl")
    b = simulate(live, log_path=tmp_path / "b.jsonl")
    assert a.equals(b)
