"""Train router v1: python -m router.train

Features: character n-gram TF-IDF of the ORIGINAL complaint text (works for Hindi, Punjabi,
English and Hinglish alike, no translation needed) + one-hot issue_type and road_type.
Model: multinomial logistic regression. Calibration (sigmoid) is added only if it lowers
calibration error on validation. Validation = 20% of train, grouped by incident_id so duplicate
reports of one incident never sit on both sides. Test is touched once, at the end.
Image detection classes are not a feature yet: vision (Phase 5) does not exist.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import MLRUNS_DIR, MODELS_DIR, SEED
from router.dataset import ROUTER_DATA_DIR
from router.evaluate import (
    compare,
    confusion,
    ece,
    evaluate,
    lookup_baseline,
    majority_baseline,
    threshold_table,
)

VERSION = "v1"
OUT_DIR = MODELS_DIR / "router" / VERSION
CATEGORICAL = ["issue_type", "road_type"]
C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)  # inverse L2 strength; picked by validation log-loss
# Below this top probability the case goes to a human (needs_human_routing).
# User decision 2.3: 0.7, chosen from the validation threshold table.
THRESHOLD = 0.7


def build_pipeline(C: float = 10.0) -> Pipeline:
    features = ColumnTransformer([
        # char_wb = n-grams inside word boundaries: "करंट", "sparking", "kinare" all become
        # overlapping 2-5 character pieces, so spelling variants still share features.
        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True), "text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


def calibrated(model: Pipeline, df: pd.DataFrame) -> CalibratedClassifierCV:
    """Sigmoid calibration on top of `model`, with folds grouped by incident_id."""
    folds = list(GroupKFold(n_splits=5).split(df, df.department, df.incident_id))
    return CalibratedClassifierCV(model, method="sigmoid", cv=folds)


def split_val(train: pd.DataFrame, frac: float = 0.2, seed: int = SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit_idx, val_idx = next(gss.split(train, groups=train.incident_id))
    return train.iloc[fit_idx], train.iloc[val_idx]


def predict(model, df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(predicted department, top probability, full probability matrix) per row."""
    proba = model.predict_proba(df)
    return list(model.classes_[proba.argmax(1)]), proba.max(1), proba


def select(fit: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pick C by validation log-loss, then raw vs calibrated by validation ECE."""
    rows = []
    for C in C_GRID:
        for cal in (False, True):
            m = build_pipeline(C)
            m = (calibrated(m, fit) if cal else m).fit(fit, fit.department)
            pred, conf, proba = predict(m, val)
            rows.append({"C": C, "calibrated": cal,
                         "val_log_loss": log_loss(val.department, proba, labels=m.classes_),
                         "val_ece": ece(val.department, pred, conf),
                         **{k: v for k, v in evaluate(val, pred).items() if k != "by_scenario"}})
    grid = pd.DataFrame(rows)
    best_C = grid[~grid.calibrated].sort_values("val_log_loss").iloc[0].C
    pair = grid[grid.C == best_C].set_index("calibrated")
    use_cal = bool(pair.loc[True, "val_ece"] < pair.loc[False, "val_ece"])
    return {"C": float(best_C), "calibrated": use_cal}, grid


def fit_final(train: pd.DataFrame, params: dict):
    m = build_pipeline(params["C"])
    m = calibrated(m, train) if params["calibrated"] else m
    return m.fit(train, train.department)


def unseen_phrasing(train: pd.DataFrame, test: pd.DataFrame, params: dict,
                    threshold: float = THRESHOLD) -> pd.DataFrame:
    """Honesty check. Each generated ambiguous scenario has 2 cue sentences per language and
    department, and the same sentences appear in train and test, so test accuracy partly
    measures memorising the phrase bank. Here cue sentence #k is removed from training and
    the model is scored only on test rows that use it: phrasings it has never seen."""
    def cue(df):  # generated template_id = lang:scenario:shared:dept:cue
        return df.template_id.str.split(":").str[-1].where(df.incident_id.str.startswith("A-"))
    rows = []
    for k in ("0", "1"):
        m = fit_final(train[cue(train) != k], params)
        held = test[cue(test) == k]
        pred, conf, _ = predict(m, held)
        hit, auto = np.asarray(pred) == held.department.to_numpy(), conf >= threshold
        rows.append({"held_out_cue": k, "rows": len(held),
                     "lookup": evaluate(held, lookup_baseline(held))["accuracy"],
                     f"router {VERSION}": hit.mean(),
                     **{f"router:{s}": v for s, v in evaluate(held, pred)["by_scenario"].items()},
                     # Does the threshold catch what the model doesn't know?
                     "to_human": (~auto).mean(), "auto_accuracy": hit[auto].mean()})
    return pd.DataFrame(rows).set_index("held_out_cue")


def save(model, meta: dict, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def log_mlflow(params: dict, val_m: dict, test_m: dict, out_dir: Path) -> str:
    import mlflow  # imported here: training works without it, tests don't need it

    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("router") is None:
        mlflow.create_experiment("router", artifact_location=(MLRUNS_DIR / "artifacts").as_uri())
    mlflow.set_experiment("router")
    with mlflow.start_run(run_name=f"router-{VERSION}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_artifacts(str(out_dir), artifact_path="model")
        return run.info.run_id


def _scalars(r: dict, pred, conf, df) -> dict:
    out = {k: float(v) for k, v in r.items() if k != "by_scenario"}
    out["ece"] = ece(df.department, pred, conf)
    auto = np.asarray(conf) >= THRESHOLD
    out["auto_routed"] = float(auto.mean())
    out["auto_accuracy"] = float((np.asarray(pred) == df.department.to_numpy())[auto].mean())
    return out


def main() -> None:
    train = pd.read_parquet(ROUTER_DATA_DIR / "train.parquet")
    test = pd.read_parquet(ROUTER_DATA_DIR / "test.parquet")
    fit, val = split_val(train)
    params, grid = select(fit, val)
    print(f"Model selection on validation ({len(val)} rows, {val.ambiguous.sum()} ambiguous):")
    print(grid.to_string(index=False, float_format="%.3f"))
    print(f"-> chosen: {params}\n")

    # Validation predictions from a model fit on `fit` only: used for the threshold choice.
    val_model = fit_final(fit, params)
    val_pred, val_conf, _ = predict(val_model, val)
    val_r = evaluate(val, val_pred)

    model = fit_final(train, params)
    pred, conf, _ = predict(model, test)
    results = {"majority": evaluate(test, majority_baseline(train, test)),
               "lookup": evaluate(test, lookup_baseline(test)),
               f"router {VERSION}": evaluate(test, pred)}
    print(f"Test set: {len(test)} rows, {test.ambiguous.sum()} ambiguous")
    print(compare(results).T.to_string(float_format="%.3f"))
    print(f"\nTest ECE: {ece(test.department, pred, conf):.3f}")
    print("\nThreshold sweep on VALIDATION (choose here):")
    print(threshold_table(val, val_pred, val_conf).to_string(float_format="%.3f"))
    print("\nThreshold sweep on TEST (report only):")
    print(threshold_table(test, pred, conf).to_string(float_format="%.3f"))
    print("\nConfusion on test (rows = true, cols = predicted):")
    print(confusion(test, pred).to_string())
    unseen = unseen_phrasing(train, test, params)
    print("\nUnseen phrasing (ambiguous test rows whose cue sentence was removed from training):")
    print(unseen.to_string(float_format="%.3f"))

    val_m, test_m = _scalars(val_r, val_pred, val_conf, val), _scalars(results[f"router {VERSION}"],
                                                                     pred, conf, test)
    meta = {"version": VERSION, **params, "threshold": THRESHOLD, "threshold_status": "final",
            "classes": list(model.classes_), "features": ["text (char 2-5 tf-idf)", *CATEGORICAL],
            "n_train": len(train), "val": val_m, "test": test_m,
            "unseen_phrasing_accuracy": float(unseen[f"router {VERSION}"].mean())}
    save(model, meta)
    run_id = log_mlflow({**params, "threshold": THRESHOLD}, val_m, test_m, OUT_DIR)
    print(f"\nSaved {OUT_DIR} (MLflow run {run_id})")


if __name__ == "__main__":
    main()
