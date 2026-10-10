"""Public API of the router module. Only this file may be imported by other modules.

route(case) -> RoutingDecision: department from the model, ward from the geo lookup, a
needs_human_routing flag, and a WorkOrderDraft in status 'draft' (only a human approves).
The model version comes from config.MODEL_VERSIONS (or an explicit `version`, which the trainer
will use to compare v1 and v2). Every call writes one DecisionLog entry.
retrain / predict_test exist for the trainer (Step 3.3).
"""

import json
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config
import router.retrain as _retrain
from contracts.api import (
    Case,
    Department,
    RoutingDecision,
    Severity,
    StubRouter,
    StubTriage,
    TriageFeatures,
    log_decision,
)
from contracts.world import distance_to_nearest, road_type_at
from router.ward import ward_for
from router.workorder import draft

MODELS = config.MODELS_DIR / "router"
PUNCT = ".,!?।|\"'()"


@cache
def _model(version: str):
    d = MODELS / version
    return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text())


def _features(case: Case) -> dict:
    """The model's input row, built the same way as the training data."""
    r = case.report
    return {"text": r.original_text or r.english_text, "issue_type": r.issue_type.value,
            # No GPS -> "unknown": the one-hot encoder ignores it, so road type adds nothing.
            "road_type": road_type_at(r.location) if r.location else "unknown"}


def _cue_words(model, row: pd.DataFrame, k: int, n: int = 3) -> list[str]:
    """Words of the complaint whose character n-grams pushed the model most toward class k.
    Contribution = tf-idf value x (class-k weight - mean weight over classes)."""
    if not isinstance(model, Pipeline):  # a calibrated wrapper hides the coefficients
        return []
    feats, clf = model.named_steps["features"], model.named_steps["clf"]
    x = feats.transform(row).tocsr()
    names = feats.get_feature_names_out()[x.indices]
    contrib = x.data * (clf.coef_[k] - clf.coef_.mean(0))[x.indices]
    words = [w.strip(PUNCT) for w in row.text.iloc[0].lower().split()]
    out = []
    for i in np.argsort(-contrib):
        gram = names[i].removeprefix("text__").strip()
        if contrib[i] <= 0 or len(out) == n:
            break
        if names[i].startswith("text__") and len(gram) >= 3:
            out += [w for w in words if gram in w and w not in out][:1]
    return out


def _department(case: Case, feats: dict, version: str) -> tuple[Department, float, bool, str]:
    """(department, confidence, below threshold, explanation)."""
    if version == "stub":
        d = StubRouter().route(case)
        return d.department, d.confidence, False, f"Stub rule: {d.explanation}"
    model, meta = _model(version)
    row = pd.DataFrame([feats])
    proba = model.predict_proba(row)[0]
    k, k2 = np.argsort(-proba)[:2]
    dept, conf, thr = Department(model.classes_[k]), float(proba[k]), meta["threshold"]
    why = f"{dept.value} ({conf:.0%}); next best {model.classes_[k2]} ({proba[k2]:.0%})."
    words = _cue_words(model, row, k)
    if words:
        why += f" Words pointing to {dept.value}: " + ", ".join(f"'{w}'" for w in words) + "."
    if conf < thr:
        why += f" Below the {thr:.0%} confidence threshold: an official should choose."
    return dept, conf, conf < thr, why


def _ward(case: Case) -> tuple[str, str]:
    """(ward id or 'unknown', plain-language reason)."""
    if case.ward:
        return case.ward, f"Ward {case.ward} as already set on the case."
    ward, method = ward_for(case.report.location, case.report.location_text)
    if ward is None:
        return "unknown", "Ward unknown (no usable GPS, no ward named): an official must set it."
    return ward, f"Ward {ward} from {'GPS' if method == 'gps' else 'the location text'}."


def _severity(case: Case, ward: str, road_type: str) -> tuple[Severity, str]:
    """The triage decision on the case; if there is none yet, the stub triage rule."""
    if case.triage is not None:
        t = case.triage
        return t.severity, f"Severity from triage {t.model_version}: {t.explanation}"
    r = case.report
    near = (lambda kind: distance_to_nearest(r.location, kind)) if r.location else (
        lambda kind: 10_000.0)
    t = StubTriage().score(TriageFeatures(
        case_id=case.case_id, issue_type=r.issue_type, ward=ward,
        road_type=road_type if road_type != "unknown" else "residential",
        dist_to_school_m=near("school"), dist_to_hospital_m=near("hospital"),
        image_detections=sum(len(f.detections) for f in case.image_features),
        **r.cues.model_dump()))
    return t.severity, f"No triage decision on the case yet; stub rule: {t.explanation}"


def route(case: Case, version: str | None = None, log_path: Path = config.DECISION_LOG_PATH,
          use_llm: bool = False) -> RoutingDecision:
    version = version or config.MODEL_VERSIONS["router"]
    feats = _features(case)
    dept, conf, unsure, dept_why = _department(case, feats, version)
    ward, ward_why = _ward(case)
    severity, sev_why = _severity(case, ward, feats["road_type"])
    work_order = draft(case.case_id, case.report, dept, ward, severity, conf,
                       [dept_why, ward_why, sev_why], use_llm=use_llm)
    decision = RoutingDecision(
        case_id=case.case_id, department=dept, ward=ward,
        needs_human_routing=unsure or ward == "unknown", work_order=work_order,
        model_version=version, confidence=conf, explanation=f"{dept_why} {ward_why}")
    # Log the inputs the model actually saw: enough for the trainer to rebuild a training row.
    log_input = {**feats, "location": case.report.location and case.report.location.model_dump(),
                 "location_text": case.report.location_text, "case_ward": case.ward,
                 "severity": severity.value}
    log_decision("router", case.case_id, log_input, decision, path=log_path)
    return decision


def retrain(rows: pd.DataFrame, version: str, base: str = "v1") -> dict:
    """For the trainer: refit `base`'s recipe on `rows` (text, issue_type, road_type,
    department, weight) and save it as `version`. Returns the new model's meta."""
    _model.cache_clear()  # `version` may already be loaded from an earlier fit
    return _retrain.retrain(rows, version, base)


def predict_test(version: str) -> pd.DataFrame:
    """For the trainer: the router test split with `pred` and `conf` (offline, not logged)."""
    return _retrain.predict_test(version)
