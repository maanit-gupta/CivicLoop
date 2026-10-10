from datetime import UTC, datetime

import pandas as pd

from contracts.api import TriageFeatures
from trainer.compare import department_metrics, severity_metrics, simulated_overrides
from trainer.retrain import VAL_SHARE, val_cases, with_weight

AT = datetime(2026, 10, 1, tzinfo=UTC)


def _rows():
    return pd.DataFrame({"case_id": ["R1", "R2", "C-1", "C-2"],
                         "source": ["original", "original", "live_accepted", "live_override"],
                         "weight": [1.0, 1.0, 1.0, 3.0]})


def test_with_weight_only_changes_override_rows():
    assert with_weight(_rows(), 10.0).weight.tolist() == [1.0, 1.0, 1.0, 10.0]


def test_val_cases_are_live_deterministic_and_about_val_share():
    rows = pd.DataFrame({"case_id": [f"C-{i}" for i in range(2000)] + ["R1"],
                         "source": ["live_accepted"] * 2000 + ["original"]})
    val = val_cases(rows)
    assert val == val_cases(rows) and "R1" not in val
    assert abs(len(val) / 2000 - VAL_SHARE) < 0.03


def _served(severity, department, school_m, issue="water_leak", flooding=True):
    f = TriageFeatures(case_id="C-1", issue_type=issue, ward="W01", road_type="main",
                       dist_to_school_m=school_m, dist_to_hospital_m=900.0, flooding=flooding)
    return {"case_id": "C-1", "submitted_at": AT, "features": f, "severity": severity,
            "department": department}


def test_simulated_overrides_apply_policies_a_and_b():
    served = pd.DataFrame([
        _served("medium", "Water Supply", 120.0),        # A and B both override
        _served("high", "Stormwater Drains", 120.0),     # both accept
        _served("low", "Water Supply", 900.0, flooding=False),  # neither applies
    ])
    out = simulated_overrides(served)
    assert out["A+B overrides"] == 2
    assert out["A override rate"] == out["B override rate"] == 1 / 3


def test_metrics_perfect_and_off_by_one():
    assert severity_metrics(["low", "high"], ["low", "high"]) == {"MAE": 0.0, "QWK": 1.0}
    assert severity_metrics(["low", "high"], ["medium", "high"])["MAE"] == 0.5
    assert department_metrics(["Roads", "Electrical"], ["Roads", "Roads"])["accuracy"] == 0.5
