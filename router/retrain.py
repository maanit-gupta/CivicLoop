"""Retrain the router on new rows (Step 3.3), called by the trainer through router.api.

Same recipe as `base`: its C and features; only the data and per-row weights change. The
threshold is kept, since it is a user decision (2.3), not something to re-derive here.
"""

import json

import joblib
import pandas as pd

from config import MODELS_DIR
from router.dataset import ROUTER_DATA_DIR
from router.train import build_pipeline, save

MODELS = MODELS_DIR / "router"


def _load(version: str):
    d = MODELS / version
    return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text())


def retrain(rows: pd.DataFrame, version: str, base: str = "v1") -> dict:
    """Fit on `rows` (text, issue_type, road_type, department, weight); save as `version`."""
    _, meta = _load(base)
    if meta["calibrated"]:  # calibration folds need incident ids, which live rows lack
        raise NotImplementedError(f"router {base} is calibrated; retrain supports raw models")
    model = build_pipeline(meta["C"]).fit(rows, rows.department,
                                          clf__sample_weight=rows.weight.to_numpy(dtype=float))
    out = {"version": version, "base": base, "C": meta["C"], "calibrated": False,
           "threshold": meta["threshold"], "threshold_status": meta["threshold_status"],
           "classes": list(model.classes_), "features": meta["features"], "n_train": len(rows)}
    save(model, out, MODELS / version)
    return out


def predict_test(version: str) -> pd.DataFrame:
    """The router test split with `pred` and `conf` from `version` (offline batch, not logged)."""
    model, _ = _load(version)
    test = pd.read_parquet(ROUTER_DATA_DIR / "test.parquet")
    proba = model.predict_proba(test)
    return test.assign(pred=model.classes_[proba.argmax(1)], conf=proba.max(1))
