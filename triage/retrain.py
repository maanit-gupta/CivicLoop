"""Retrain triage on new rows (Step 3.3), called by the trainer through triage.api.

Same recipe as `base`: its params, categories and number of boosting rounds (no early stopping:
the old val split is labelled by the written rubric, which the new labels may deliberately
disagree with). Only the data and the per-row weights change, so any v1 -> v2 difference comes
from the data. Calibration is refit on the original val split, the only clean labelled set.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import MODELS_DIR
from triage.calibrate import fit_calibration, save_calibration
from triage.evaluate import LEVELS, load_cases, to_index
from triage.train import featurize, load, save, to_levels

MODELS = MODELS_DIR / "triage"


def retrain(rows: pd.DataFrame, version: str, base: str = "v1") -> dict:
    """Fit on `rows` (features + severity + weight) with `base`'s recipe; save as `version`."""
    _, meta = load(base, MODELS)
    cats, rounds = meta["categories"], meta["best_iteration"]
    data = lgb.Dataset(featurize(rows, cats), to_index(rows.severity),
                       weight=rows.weight.to_numpy(dtype=float))
    booster = lgb.train(meta["params"], data, rounds)
    val = load_cases().query("split == 'val'")
    raw = booster.predict(featurize(val, cats))
    out = {**{k: meta[k] for k in ("features", "categorical", "categories", "params")},
           "version": version, "base": base, "best_iteration": rounds, "n_train": len(rows)}
    save(booster, out, MODELS / version)
    save_calibration(fit_calibration(raw, to_levels(raw) == to_index(val.severity)),
                     MODELS / version)
    return out


def predict_test(version: str) -> pd.DataFrame:
    """The triage test split (rubric labels) with `pred` from `version`. Offline evaluation in
    one batch, not logged: no case is being decided."""
    booster, meta = load(version, MODELS)
    test = load_cases().query("split == 'test'").reset_index(drop=True)
    levels = to_levels(booster.predict(featurize(test, meta["categories"])))
    return test.assign(pred=np.asarray(LEVELS)[levels])
