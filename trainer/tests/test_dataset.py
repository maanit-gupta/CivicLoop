from datetime import UTC, datetime

import pandas as pd
import pytest

from contracts.api import Override, append, read_log
from trainer.dataset import (
    HOLDOUT_SHARE,
    OVERRIDE_WEIGHT,
    TRIAGE_COLS,
    _live_rows,
    live_decisions,
    override_report,
    preference_pairs,
    split_holdout,
)


def _case(path, case_id, severity="medium", dept="Roads", override=None, school_m=500.0):
    """Log one live case like trainer.officials does: triage, router, then any override."""
    features = {"case_id": case_id, "issue_type": "pothole", "ward": "W01", "road_type": "main",
                "dist_to_school_m": school_m, "dist_to_hospital_m": 900.0, "children": False,
                "accident": False, "flooding": False, "duplicate_count": 0,
                "image_detections": 0}
    append("triage", "v1", case_id, input=features,
           output={"case_id": case_id, "severity": severity}, path=path)
    append("router", "v1", case_id, input={"text": f"gaddha {case_id}"},
           output={"case_id": case_id, "department": dept}, path=path)
    if override:
        field, value, who = override
        module, model_value = ("triage", severity) if field == "severity" else ("router", dept)
        o = Override(case_id=case_id, module=module, field=field, model_value=model_value,
                     human_value=value, official_id=who, reason="r",
                     at=datetime(2026, 10, 1, tzinfo=UTC))
        append(module, "v1", case_id, input={field: model_value}, output={field: value},
               human_override=o, path=path)


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "log.jsonl"
    _case(path, "C-1")                                              # accepted
    _case(path, "C-2", override=("severity", "high", "A"), school_m=120.0)
    _case(path, "C-3", dept="Water Supply", override=("department", "Stormwater Drains", "B"))
    return path


def test_join_gives_human_label_for_every_case(log):
    live = live_decisions(log).set_index("case_id")
    assert list(live.index) == ["C-1", "C-2", "C-3"]
    assert live.loc["C-1", "human_severity"] == live.loc["C-1", "model_severity"] == "medium"
    assert pd.isna(live.loc["C-1", "severity_official"])
    assert (live.loc["C-2", "model_severity"], live.loc["C-2", "human_severity"]) == \
        ("medium", "high")
    assert live.loc["C-2", "severity_official"] == "A"
    assert live.loc["C-3", "human_department"] == "Stormwater Drains"
    assert live.loc["C-3", "human_severity"] == "medium"  # other field untouched
    assert live.loc["C-2", "dist_to_school_m"] == 120.0   # features come from the log


def test_rescore_replaces_earlier_model_entry(tmp_path):
    path = tmp_path / "log.jsonl"
    _case(path, "C-1", severity="low")
    _case(path, "C-1", severity="high")
    live = live_decisions(path)
    assert len(live) == 1 and live.human_severity.iloc[0] == "high"


def test_broken_chain_is_refused(log):
    lines = log.read_text().splitlines()
    lines[1] = lines[1].replace("gaddha", "tampered")
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="chain broken at entry 1"):
        live_decisions(log)


def test_override_rows_weighted_higher(log):
    rows = _live_rows(live_decisions(log), "severity", TRIAGE_COLS).set_index("case_id")
    assert rows.loc["C-2", "weight"] == OVERRIDE_WEIGHT > rows.loc["C-1", "weight"] == 1.0
    assert rows.loc["C-2", "source"] == "live_override"
    assert rows.loc["C-3", "source"] == "live_accepted"  # its override was on department
    assert rows.loc["C-2", "severity"] == "high"


def test_preference_pairs_only_for_overrides(log):
    pairs = preference_pairs(live_decisions(log))
    assert [(p["case_id"], p["field"], p["chosen"], p["rejected"], p["official_id"])
            for p in pairs] == [("C-2", "severity", "high", "medium", "A"),
                                ("C-3", "department", "Stormwater Drains", "Water Supply", "B")]
    assert "School 120 m" in pairs[0]["prompt"] and "gaddha C-2" in pairs[0]["prompt"]


def test_holdout_split_is_seeded_and_near_share():
    live = pd.DataFrame({"case_id": [f"C-{i}" for i in range(2000)]})
    a, b = split_holdout(live), split_holdout(live)
    assert a.equals(b)
    assert abs(a.mean() - HOLDOUT_SHARE) < 0.03


def test_override_report(log):
    per_official, per_issue = override_report(live_decisions(log))
    po = per_official.set_index("official")
    assert po.loc["A", "severity"] == 1 and po.loc["B", "department"] == 1
    assert po.loc["A", "per_100_cases"] == pytest.approx(100 / 3)
    assert per_issue.cases.sum() == 3


def test_log_is_only_appended_to(log):
    before = len(read_log(log))
    live_decisions(log)
    assert len(read_log(log)) == before
