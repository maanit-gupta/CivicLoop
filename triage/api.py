"""Public API of the triage module. Only this file may be imported by other modules.

score(features) -> TriageDecision, using the model version named in config.MODEL_VERSIONS
(or an explicit `version`, which the trainer will use to compare v1 and v2). Every call
writes one DecisionLog entry. retrain / predict_test exist for the trainer (Step 3.3).
"""

from functools import cache
from pathlib import Path

import pandas as pd

import config
import triage.retrain as _retrain
from contracts.api import Severity, StubTriage, TriageDecision, TriageFeatures, log_decision
from triage.calibrate import confidence, load_calibration
from triage.evaluate import LEVELS
from triage.explain import contributions, sentence
from triage.train import featurize, load, to_levels

MODELS = config.MODELS_DIR / "triage"


@cache
def _model(version: str, models_dir: Path):
    booster, meta = load(version, models_dir)
    return booster, meta, load_calibration(models_dir / version)


def _decide(features: TriageFeatures, version: str) -> TriageDecision:
    if version == "stub":
        return StubTriage().score(features)
    booster, meta, cal = _model(version, MODELS)
    x = featurize(pd.DataFrame([features.model_dump(mode="json")]), meta["categories"])
    raw = booster.predict(x)
    severity = Severity(LEVELS[to_levels(raw)[0]])
    contrib, _ = contributions(booster, x)
    return TriageDecision(
        case_id=features.case_id, severity=severity, model_version=version,
        confidence=float(confidence(raw, cal)[0]),
        explanation=sentence(severity, features, contrib),
    )


def score(features: TriageFeatures, version: str | None = None,
          log_path: Path = config.DECISION_LOG_PATH) -> TriageDecision:
    decision = _decide(features, version or config.MODEL_VERSIONS["triage"])
    log_decision("triage", features.case_id, features, decision, path=log_path)
    return decision


def retrain(rows: pd.DataFrame, version: str, base: str = "v1") -> dict:
    """For the trainer: refit `base`'s recipe on `rows` (TriageFeatures columns + severity +
    weight) and save it as `version`. Returns the new model's meta."""
    _model.cache_clear()  # `version` may already be loaded from an earlier fit
    return _retrain.retrain(rows, version, base)


def predict_test(version: str) -> pd.DataFrame:
    """For the trainer: the triage test split with a `pred` column (offline, not logged)."""
    return _retrain.predict_test(version)
