import json
from collections import Counter

import pandas as pd
import pytest

from trainer.llm_data import (
    LEVELS,
    anchor_cases,
    anchor_rejected,
    answer,
    explain,
    facts,
    parse,
    prompt,
)
from trainer.llm_eval import FALLBACK, a_override_rate, llm_answers


def _row(case_id="T-1", school=150.0, severity="high", **kw):
    base = {"case_id": case_id, "issue_type": "pothole", "ward": "W03", "road_type": "main",
            "dist_to_school_m": school, "dist_to_hospital_m": 900.0, "children": True,
            "accident": False, "flooding": False, "duplicate_count": 2, "image_detections": 1,
            "severity": severity}
    return pd.Series(base | kw)


def test_explanation_lists_facts_and_never_the_severity():
    text = explain(_row())
    assert text == ("pothole on a main road; school 150 m away (under 200 m); "
                    "hospital 900 m away; children mentioned; 2 other reports.")
    assert not any(level in text for level in LEVELS)
    assert explain(_row(severity="low")) == text


def test_distances_are_floored_so_the_threshold_note_stays_true():
    assert "school 199 m away (under 200 m)" in explain(_row(school=199.6))
    assert "school 200 m away;" in explain(_row(school=200.0))
    assert "199 m" in facts(_row(school=199.6))


def test_prompt_carries_features_not_the_label():
    for level in LEVELS:
        msgs = prompt(_row(severity=level))
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert level not in msgs[1]["content"]


def test_preference_pair_differs_only_in_severity():
    chosen = json.loads(answer(_row(), "high")[0]["content"])
    rejected = json.loads(answer(_row(), "medium")[0]["content"])
    assert chosen["explanation"] == rejected["explanation"]
    assert (chosen["severity"], rejected["severity"]) == ("high", "medium")
    assert list(chosen) == ["explanation", "severity"]  # facts first, then the decision


def test_anchors_are_accepted_cases_far_from_a_school():
    rows = [_row("ok-1", school=500.0, severity="low", source="live_accepted"),
            _row("ok-2", school=200.0, severity="critical", source="live_accepted"),
            _row("near", school=199.0, severity="low", source="live_accepted"),
            _row("over", school=800.0, severity="low", source="live_override"),
            _row("orig", school=800.0, severity="low", source="original")]
    retrain = pd.DataFrame(rows)
    assert set(anchor_cases(retrain, n=2).case_id) == {"ok-1", "ok-2"}
    with pytest.raises(ValueError, match="need 3 anchor cases"):
        anchor_cases(retrain, n=3)


def test_anchor_rejected_fills_the_least_rejected_label_and_never_the_chosen():
    already = Counter({"low": 3, "medium": 3, "high": 0, "critical": 0})
    chosen = ["low"] * 6 + ["critical"] * 2
    rejected = anchor_rejected(chosen, already)
    assert all(c != r for c, r in zip(chosen, rejected, strict=True))
    total = already + Counter(rejected)
    assert max(total.values()) - min(total[lvl] for lvl in LEVELS) <= 1
    assert rejected == anchor_rejected(chosen, already)  # seeded


@pytest.mark.parametrize("output, severity", [
    ('{"explanation": "x", "severity": "critical"}', "critical"),
    ('```json\n{"explanation": "x", "severity": "low"}\n```', "low"),
    ('{"severity": "medium"}', "medium"),
])
def test_parse_accepts_valid_answers(output, severity):
    assert parse(output)["severity"] == severity


@pytest.mark.parametrize("output", [
    "", "high", '{"explanation": "x", "severity": "urgent"}', '["high"]',
    '{"explanation": "x", "severity": "high"} and more text',
])
def test_parse_rejects_invalid_answers(output):
    assert parse(output) is None


def test_llm_answers_scores_validity_faithfulness_and_falls_back():
    cases = pd.DataFrame([_row("T-1"), _row("T-2", school=500.0), _row("T-3")])
    good = answer(cases.iloc[0], "high")[0]["content"]
    unfaithful = json.dumps({"explanation": "a pothole somewhere", "severity": "medium"})
    preds = pd.DataFrame([
        {"case_id": c, "set": "test", "version": v, "output": out}
        for v in ("llm-sft", "llm-dpo")
        for c, out in (("T-1", good), ("T-2", unfaithful), ("T-3", "not json"))])
    got = llm_answers(preds, cases, "test")["llm-dpo"]
    assert got.severity.tolist() == ["high", "medium", FALLBACK]
    assert got.valid.tolist() == [True, True, False]
    assert got.faithful.tolist() == [True, False, False]


def test_llm_answers_refuses_missing_cases():
    cases = pd.DataFrame([_row("T-1"), _row("T-2")])
    preds = pd.DataFrame([{"case_id": "T-1", "set": "test", "version": v, "output": "{}"}
                          for v in ("llm-sft", "llm-dpo")])
    with pytest.raises(ValueError, match="no answer for 1 test cases"):
        llm_answers(preds, cases, "test")


def test_a_override_rate():
    # near+medium -> override; near+high, far+low -> none
    assert a_override_rate([100, 100, 500], ["medium", "high", "low"]) == pytest.approx(1 / 3)
