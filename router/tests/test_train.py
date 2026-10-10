import joblib
import numpy as np
import pandas as pd
import pytest

from router.evaluate import ece, threshold_table
from router.train import build_pipeline, predict, save, split_val


@pytest.fixture
def df():
    """Same pole, two cues: the text decides the department, issue_type does not."""
    elec = ["pole fell, wires sparking", "pole down and wires sparking badly",
            "खंभा गिरा, तार से चिंगारी", "pole gir gaya, wire sparking kar raha"]
    roads = ["pole fell, blocking the road traffic", "pole down blocking road, traffic stuck",
             "खंभा गिरा, सड़क पर ट्रैफिक रुका", "pole gir gaya, road block traffic ruka"]
    texts = (elec + roads) * 3
    n = len(texts)
    return pd.DataFrame({
        "text": texts,
        "department": (["Electrical"] * 4 + ["Roads"] * 4) * 3,
        "issue_type": ["streetlight", "other"] * (n // 2),
        "road_type": ["main", "lane", "residential", "lane"] * (n // 4),
        "incident_id": [f"I-{i % 12:03d}" for i in range(n)],
        "ambiguous": [True] * n,
        "scenario": ["fallen_pole"] * n,
    })


def test_text_cue_decides_department(df):
    m = build_pipeline().fit(df, df.department)
    probe = df.head(2).assign(text=["the pole fell and wires are sparking",
                                    "pole fell, road blocked, traffic stuck"])
    pred, _, _ = predict(m, probe)
    assert pred == ["Electrical", "Roads"]


def test_predict_confidence_is_top_probability(df):
    m = build_pipeline().fit(df, df.department)
    pred, conf, proba = predict(m, df)
    assert np.allclose(proba.sum(1), 1) and np.allclose(conf, proba.max(1))
    assert set(pred) <= set(m.classes_)


def test_unknown_issue_type_does_not_crash(df):
    m = build_pipeline().fit(df, df.department)
    pred, _, _ = predict(m, df.head(1).assign(issue_type="never_seen"))
    assert len(pred) == 1


def test_validation_split_keeps_incidents_together(df):
    fit, val = split_val(df, frac=0.25)
    assert len(val) and not set(fit.incident_id) & set(val.incident_id)


def test_save_and_reload_predict_identically(df, tmp_path):
    m = build_pipeline().fit(df, df.department)
    save(m, {"version": "test"}, tmp_path)
    again = joblib.load(tmp_path / "model.joblib")
    assert (tmp_path / "meta.json").exists()
    assert np.allclose(m.predict_proba(df), again.predict_proba(df))


def test_ece_zero_when_confidence_matches_accuracy():
    y = ["a", "a", "b", "b"]
    assert ece(y, y, [1.0] * 4) == 0.0
    assert ece(y, ["a", "b", "a", "b"], [0.5] * 4) == 0.0  # 50% sure, 50% right


def test_ece_punishes_overconfidence():
    assert ece(["a", "b"], ["b", "a"], [0.95, 0.95]) == pytest.approx(0.95)


def test_threshold_table_trades_coverage_for_accuracy(df):
    pred = ["Electrical"] * len(df)
    conf = np.linspace(0.4, 1.0, len(df))
    t = threshold_table(df, pred, conf)
    assert t.auto_routed.is_monotonic_decreasing
    assert (t.to_human + t.auto_routed * len(df)).round().eq(len(df)).all()
