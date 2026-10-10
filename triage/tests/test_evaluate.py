import numpy as np
import pandas as pd
import pytest

from triage.evaluate import (
    confusion,
    issue_type_baseline,
    mean_baseline,
    metrics,
    qwk,
    score_table,
    to_index,
)


def test_to_index_orders_levels():
    assert to_index(["low", "medium", "high", "critical"]).tolist() == [0, 1, 2, 3]


def test_confusion_rows_are_truth():
    m = confusion(np.array([0, 0, 3]), np.array([0, 1, 3]))
    assert m[0, 0] == 1 and m[0, 1] == 1 and m[3, 3] == 1 and m.sum() == 3


def test_qwk_perfect_constant_and_reversed():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert qwk(y, y) == pytest.approx(1.0)
    assert qwk(y, np.full_like(y, 1)) == pytest.approx(0.0)  # constant -> chance level
    assert qwk(y, 3 - y) < 0                                  # anti-correlated -> negative


def test_qwk_hand_computed():
    # True [0, 1], pred [0, 2]. O has 1 at (0,0) and (1,2). E = outer([1,1,0,0],[1,0,1,0])/2.
    # W = (i-j)^2/9. sum(W*O) = 1/9. sum(W*E) = (0 + 4/9 + 1/9 + 1/9)/2 = 3/9. kappa = 1 - 1/3.
    assert qwk(np.array([0, 1]), np.array([0, 2])) == pytest.approx(2 / 3)


def test_metrics_values():
    m = metrics(np.array([0, 1, 3, 3]), np.array([0, 3, 2, 3]))
    assert m["mae"] == pytest.approx(0.75)
    assert m["within1"] == pytest.approx(0.75)
    assert m["exact"] == pytest.approx(0.5)
    assert m["crit_recall"] == pytest.approx(0.5)


def test_crit_recall_nan_without_critical_cases():
    assert np.isnan(metrics(np.array([0, 1]), np.array([0, 1]))["crit_recall"])


TRAIN = pd.DataFrame({
    "issue_type": ["pothole", "pothole", "pothole", "garbage", "garbage"],
    "severity": ["high", "high", "low", "low", "low"],
})


def test_mean_baseline_is_constant_rounded_mean():
    # indices [2, 2, 0, 0, 0] -> mean 0.8 -> medium (1)
    assert mean_baseline(TRAIN)(TRAIN).tolist() == [1] * 5


def test_issue_type_baseline_uses_mode_and_falls_back_for_unseen_types():
    pred = issue_type_baseline(TRAIN)(pd.DataFrame({"issue_type": ["pothole", "garbage", "other"]}))
    assert pred.tolist() == [2, 0, 0]  # unseen "other" -> overall mode (low)


def test_score_table_has_all_and_natural_rows():
    df = TRAIN.assign(stratum=["natural", "enriched", "natural", "natural", "enriched"])
    t = score_table({"mean": mean_baseline(TRAIN)}, df)
    assert list(t.index) == [("mean", "all"), ("mean", "natural")]
    assert {"mae", "qwk", "within1", "exact", "crit_recall"} <= set(t.columns)
