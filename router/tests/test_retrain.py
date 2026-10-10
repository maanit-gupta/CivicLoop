import shutil

import pandas as pd
import pytest

import config
import router.retrain as rt
from router.dataset import ROUTER_DATA_DIR

V1 = config.MODELS_DIR / "router" / "v1"
pytestmark = pytest.mark.skipif(not (V1 / "model.joblib").exists(), reason="needs router v1")


@pytest.fixture
def models(tmp_path, monkeypatch):
    shutil.copytree(V1, tmp_path / "v1")
    monkeypatch.setattr(rt, "MODELS", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def train():
    return pd.read_parquet(ROUTER_DATA_DIR / "train.parquet").assign(weight=1.0)


def test_same_rows_reproduce_base(models, train):
    meta = rt.retrain(train, "same")
    assert meta["threshold"] == 0.7 and meta["base"] == "v1"
    assert (rt.predict_test("same").pred == rt.predict_test("v1").pred).all()


def test_weight_pulls_predictions_toward_heavy_rows(models, train):
    """Relabel water-on-road rows 'Stormwater Drains' (policy B): weight decides how far it goes."""
    water = train.issue_type.isin(["water_leak", "drain_blockage"]) & train.flooding
    relabelled = train.copy()
    relabelled.loc[water[water].index[::3], "department"] = "Stormwater Drains"
    heavy = relabelled.assign(weight=lambda d: d.weight.where(
        ~d.index.isin(water[water].index[::3]), 20.0))
    rt.retrain(relabelled, "light")
    rt.retrain(heavy, "heavy")
    storm = {v: (rt.predict_test(v).pred == "Stormwater Drains").sum() for v in ("light", "heavy")}
    assert storm["heavy"] > storm["light"]
