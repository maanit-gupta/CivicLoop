import pandas as pd
import pytest

from router.evaluate import (
    DEPARTMENTS,
    compare,
    confusion,
    evaluate,
    lookup_baseline,
    majority_baseline,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "department": ["Roads", "Electrical", "Water Supply", "Stormwater Drains", "Roads"],
        "issue_type": ["pothole", "streetlight", "water_leak", "water_leak", "streetlight"],
        "ambiguous": [False, False, True, True, True],
        "scenario": [None, None, "water_on_road", "water_on_road", "fallen_pole"],
    })


def test_perfect_prediction_scores_one(df):
    r = evaluate(df, df.department)
    assert r["accuracy"] == r["ambiguous_accuracy"] == r["clear_accuracy"] == 1.0
    assert r["by_scenario"] == {"fallen_pole": 1.0, "water_on_road": 1.0}


def test_ambiguous_subset_scored_separately(df):
    pred = ["Roads", "Electrical", "Water Supply", "Water Supply", "Electrical"]
    r = evaluate(df, pred)
    assert r["clear_accuracy"] == 1.0
    assert r["ambiguous_accuracy"] == pytest.approx(1 / 3)
    assert r["by_scenario"] == {"fallen_pole": 0.0, "water_on_road": 0.5}


def test_macro_f1_punishes_ignoring_a_department(df):
    r = evaluate(df, ["Roads"] * len(df))
    assert r["accuracy"] == pytest.approx(0.4)
    assert r["macro_f1"] < r["accuracy"]


def test_lookup_baseline_uses_departments_yaml(df):
    assert lookup_baseline(df) == ["Roads", "Electrical", "Water Supply", "Water Supply",
                                   "Electrical"]


def test_majority_baseline_predicts_train_mode(df):
    assert majority_baseline(df, df.head(2)) == ["Roads", "Roads"]


def test_confusion_has_all_departments(df):
    m = confusion(df, df.department)
    assert list(m.index) == list(m.columns) == DEPARTMENTS
    assert m.values.trace() == len(df)


def test_compare_flattens_scenarios(df):
    table = compare({"perfect": evaluate(df, df.department)})
    assert "amb:water_on_road" in table.columns and table.loc["perfect", "accuracy"] == 1.0
