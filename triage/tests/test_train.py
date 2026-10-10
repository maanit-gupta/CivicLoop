import numpy as np
import pandas as pd
import pytest

from triage.evaluate import issue_type_baseline, metrics, to_index
from triage.generate_data import generate
from triage.train import (
    FEATURES,
    PROBES,
    categories_from,
    featurize,
    fit,
    load,
    predictor,
    probe_table,
    save,
    to_levels,
)

PARAMS = {"num_leaves": 7, "min_child_samples": 10}


@pytest.fixture(scope="module")
def data():
    df = generate(n=1500, seed=7)
    train, val = (df[df.split == s].reset_index(drop=True) for s in ("train", "val"))
    cats = categories_from(train)
    return train, val, cats, fit(train, val, PARAMS, cats)


def test_featurize_fixed_categories_and_unseen_ward(data):
    train, _, cats, _ = data
    row = train.head(1).assign(ward="W-not-a-ward")
    x = featurize(row, cats)
    assert list(x.columns) == FEATURES and "case_id" not in x
    assert list(x.issue_type.cat.categories) == cats["issue_type"]
    assert x.ward.isna().all()                      # unseen ward -> missing, not a crash
    assert x.children.dtype.kind == "i"


def test_to_levels_rounds_and_clips():
    assert to_levels(np.array([-0.7, 0.49, 0.51, 2.6, 5.0])).tolist() == [0, 0, 1, 3, 3]


def test_model_beats_issue_type_baseline_on_val(data):
    train, val, cats, booster = data
    y = to_index(val.severity)
    pred = predictor(booster, cats)(val)
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
    assert metrics(y, pred)["qwk"] > metrics(y, issue_type_baseline(train)(val))["qwk"] + 0.3


def test_training_is_deterministic(data):
    train, val, cats, booster = data
    again = fit(train, val, PARAMS, cats)
    assert np.array_equal(predictor(booster, cats)(val), predictor(again, cats)(val))


def test_save_load_roundtrip(data, tmp_path):
    _, val, cats, booster = data
    save(booster, {"categories": cats}, tmp_path / "vX")
    loaded, meta = load("vX", models_dir=tmp_path)
    assert meta["categories"] == cats
    assert np.array_equal(predictor(booster, cats)(val), predictor(loaded, cats)(val))


def test_load_missing_version_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="triage.train"):
        load("nope", models_dir=tmp_path)


def test_probe_table_covers_every_probe(data):
    train, _, cats, booster = data
    t = probe_table(booster, cats, train.ward.iloc[0])
    assert list(t.index) == list(PROBES)
    assert isinstance(t, pd.DataFrame) and t.model.notna().all()
