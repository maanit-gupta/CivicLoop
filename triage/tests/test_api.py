import numpy as np
import pandas as pd
import pytest

import config
from contracts.api import TriageAPI, TriageDecision, TriageFeatures, read_log, verify_chain
from triage import api
from triage.calibrate import confidence, fit_calibration, margin, save_calibration
from triage.evaluate import to_index
from triage.explain import contributions, sentence
from triage.generate_data import generate
from triage.train import categories_from, featurize, fit, save, to_levels

PARAMS = {"num_leaves": 7, "min_child_samples": 10}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A small model + calibration saved as version 'vT', so tests never need models/."""
    df = generate(n=1500, seed=7)
    train, val = (df[df.split == s].reset_index(drop=True) for s in ("train", "val"))
    cats = categories_from(train)
    booster = fit(train, val, PARAMS, cats)
    root = tmp_path_factory.mktemp("models")
    save(booster, {"categories": cats}, root / "vT")
    raw = booster.predict(featurize(val, cats), num_iteration=booster.best_iteration)
    save_calibration(fit_calibration(raw, to_levels(raw) == to_index(val.severity)), root / "vT")
    return root, train.ward.iloc[0]


@pytest.fixture
def triage(model_dir, monkeypatch, tmp_path):
    root, ward = model_dir
    monkeypatch.setattr(api, "MODELS", root)
    monkeypatch.setitem(config.MODEL_VERSIONS, "triage", "vT")
    return ward, tmp_path / "log.jsonl"


def feats(ward, **change) -> TriageFeatures:
    base = {"case_id": "T-1", "issue_type": "pothole", "ward": ward, "road_type": "lane",
            "dist_to_school_m": 2000.0, "dist_to_hospital_m": 2000.0}
    return TriageFeatures(**{**base, **change})


def test_module_satisfies_protocol():
    assert isinstance(api, TriageAPI)


def test_score_returns_valid_decision(triage):
    ward, log = triage
    d = api.score(feats(ward), log_path=log)
    TriageDecision.model_validate(d.model_dump())          # round-trips through the schema
    assert d.model_version == "vT" and d.case_id == "T-1"
    assert 0 < d.confidence < 1
    assert d.explanation.startswith(f"Severity {d.severity.value}.")


def test_every_call_is_logged(triage):
    ward, log = triage
    decisions = [api.score(feats(ward, case_id=f"T-{i}"), log_path=log) for i in range(3)]
    entries = read_log(log)
    assert len(entries) == 3 and verify_chain(log) is None
    for d, e in zip(decisions, entries, strict=True):
        assert e.module == "triage" and e.model_version == "vT" and e.case_id == d.case_id
        assert e.output == d.model_dump(mode="json")
        assert e.input["case_id"] == d.case_id


def test_risky_case_scores_higher_and_says_why(triage):
    ward, log = triage
    quiet = api.score(feats(ward), log_path=log)
    risky = api.score(feats(ward, road_type="main", flooding=True, duplicate_count=12),
                      log_path=log)
    order = ["low", "medium", "high", "critical"]
    assert order.index(risky.severity.value) > order.index(quiet.severity.value)
    assert "flooding mentioned" in risky.explanation.split("Counting against")[0]


def test_stub_version_from_config(triage, monkeypatch):
    ward, log = triage
    monkeypatch.setitem(config.MODEL_VERSIONS, "triage", "stub")
    d = api.score(feats(ward, accident=True), log_path=log)
    assert d.model_version == "stub" and read_log(log)[0].model_version == "stub"


def test_explicit_version_overrides_config(triage, monkeypatch):
    ward, log = triage
    monkeypatch.setitem(config.MODEL_VERSIONS, "triage", "stub")
    assert api.score(feats(ward), version="vT", log_path=log).model_version == "vT"


def test_missing_version_fails_clearly(triage):
    ward, log = triage
    with pytest.raises(FileNotFoundError, match="triage.train"):
        api.score(feats(ward), version="v-missing", log_path=log)
    assert not log.exists()                                 # nothing logged for a failed call


def test_contributions_add_up_to_prediction(model_dir):
    root, ward = model_dir
    booster, meta, _ = api._model("vT", root)
    x = featurize(pd.DataFrame([feats(ward, children=True).model_dump(mode="json")]),
                  meta["categories"])
    contrib, bias = contributions(booster, x)
    assert bias + contrib.sum() == pytest.approx(booster.predict(x)[0], abs=1e-6)


def test_sentence_uses_actual_values():
    f = TriageFeatures(case_id="x", issue_type="drain_blockage", ward="W01", road_type="main",
                       dist_to_school_m=80.0, dist_to_hospital_m=1500.0, duplicate_count=6)
    contrib = pd.Series({"issue_type": 0.4, "road_type": 0.3, "dist_to_school_m": 0.5,
                         "duplicate_count": 0.2, "dist_to_hospital_m": -0.3, "ward": 0.01})
    s = sentence(api.Severity.HIGH, f, contrib)
    assert s == ("Severity high. Main reasons: school 80 m away, drain blockage, on a main road. "
                 "Counting against: hospital 1.5 km away.")


def test_margin_and_confidence():
    raw = np.array([1.0, 1.49, 1.51, -0.7, 3.6])
    assert margin(raw) == pytest.approx([0.5, 0.01, 0.01, 1.2, 1.1])
    cal = fit_calibration(np.array([1.0, 1.0, 1.45, 1.45]), np.array([1, 1, 0, 1]))
    assert np.all(np.diff(cal["accuracy"]) >= 0)            # monotone in margin
    lo, hi = confidence(np.array([1.49, 1.0]), cal)
    assert lo < hi
