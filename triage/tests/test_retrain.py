import shutil

import pytest

import config
import triage.retrain as rt
from triage.evaluate import load_cases

V1 = config.MODELS_DIR / "triage" / "v1"
pytestmark = pytest.mark.skipif(not (V1 / "model.txt").exists(), reason="needs triage v1")


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Retrained versions go to a temp dir that holds a copy of v1."""
    shutil.copytree(V1, tmp_path / "v1")
    monkeypatch.setattr(rt, "MODELS", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def train():
    return load_cases().query("split == 'train'").assign(weight=1.0)


def test_same_rows_reproduce_base(models, train):
    meta = rt.retrain(train, "same")
    assert (models / "same" / "calibration.json").exists() and meta["base"] == "v1"
    assert (rt.predict_test("same").pred == rt.predict_test("v1").pred).all()


def test_weight_pulls_predictions_toward_heavy_rows(models, train):
    """Relabel a fifth of the near-school rows 'critical': weighted 20x they move the model."""
    near = train.index[train.dist_to_school_m < 200][::5]
    relabelled = train.copy()
    relabelled.loc[near, "severity"] = "critical"
    rt.retrain(relabelled, "light")
    rt.retrain(relabelled.assign(weight=lambda d: d.weight.where(~d.index.isin(near), 20.0)),
               "heavy")
    crit = {v: (rt.predict_test(v).pred == "critical").sum() for v in ("light", "heavy")}
    assert crit["heavy"] > crit["light"]
