"""Train triage v1 (Step 1.4). Run: python -m triage.train

LightGBM regressor on severity level index 0..3, early stopping on val, prediction rounded and
clipped to a level. A small grid is chosen by val QWK; test is scored once, at the end.
Saves models/triage/v1/{model.txt, meta.json} and logs params, metrics and files to MLflow.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import MLRUNS_DIR, MODELS_DIR, SEED
from contracts.api import IssueType, RoadType, TriageFeatures
from triage.evaluate import (
    LEVELS,
    Predictor,
    confusion,
    issue_type_baseline,
    load_cases,
    mean_baseline,
    metrics,
    rubric_reference,
    score_table,
    to_index,
)
from triage.rubric import rubric_score

VERSION = "v1"
CATEGORICAL = ["issue_type", "ward", "road_type"]
FEATURES = [c for c in TriageFeatures.model_fields if c != "case_id"]
BASE_PARAMS = {"objective": "regression", "learning_rate": 0.05, "feature_fraction": 0.9,
               "seed": SEED, "deterministic": True, "verbose": -1}
GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (7, 15, 31) for mc in (10, 40)]
MAX_ROUNDS, EARLY_STOP = 2000, 100


# ---- features ---------------------------------------------------------------------------------

def categories_from(train: pd.DataFrame) -> dict[str, list[str]]:
    """Fixed category lists. Enum columns use every enum value, so a type unseen in train still
    encodes the same way; wards come from train (an unseen ward becomes missing)."""
    return {"issue_type": [t.value for t in IssueType],
            "road_type": list(RoadType.__args__),
            "ward": sorted(train.ward.unique())}


def featurize(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for col in CATEGORICAL:
        known = x[col].where(x[col].isin(categories[col]))  # unseen value -> missing
        x[col] = pd.Categorical(known, categories=categories[col])
    for col in ("children", "accident", "flooding"):
        x[col] = x[col].astype(int)
    return x


def to_levels(raw: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(raw), 0, len(LEVELS) - 1).astype(int)


# ---- train / save / load ----------------------------------------------------------------------

def fit(train: pd.DataFrame, val: pd.DataFrame, params: dict,
        categories: dict[str, list[str]]) -> lgb.Booster:
    dtrain = lgb.Dataset(featurize(train, categories), to_index(train.severity))
    dval = lgb.Dataset(featurize(val, categories), to_index(val.severity), reference=dtrain)
    return lgb.train({**BASE_PARAMS, **params}, dtrain, MAX_ROUNDS, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def predictor(booster: lgb.Booster, categories: dict[str, list[str]]) -> Predictor:
    return lambda df: to_levels(booster.predict(featurize(df, categories),
                                                num_iteration=booster.best_iteration))


def tune(train: pd.DataFrame, val: pd.DataFrame, categories: dict[str, list[str]]):
    """Fit every grid point; return (best params, best booster, val table). Best = highest val
    QWK, ties broken by lower MAE."""
    rows, boosters = [], []
    y_val = to_index(val.severity)
    for params in GRID:
        b = fit(train, val, params, categories)
        m = metrics(y_val, predictor(b, categories)(val))
        rows.append({**params, "rounds": b.best_iteration, **m})
        boosters.append(b)
    table = pd.DataFrame(rows)
    best = table.sort_values(["qwk", "mae"], ascending=[False, True]).index[0]
    return GRID[best], boosters[best], table


def save(booster: lgb.Booster, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"), num_iteration=booster.best_iteration)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load(version: str = VERSION, models_dir: Path = MODELS_DIR / "triage"):
    """Return (booster, meta) for a saved version."""
    d = models_dir / version
    if not (d / "model.txt").exists():
        raise FileNotFoundError(f"{d}/model.txt missing. Run: python -m triage.train")
    return lgb.Booster(model_file=str(d / "model.txt")), json.loads((d / "meta.json").read_text())


# ---- reports ----------------------------------------------------------------------------------

def importances(booster: lgb.Booster) -> pd.Series:
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    return (gain / gain.sum()).sort_values(ascending=False)


def examples(df: pd.DataFrame, pred: np.ndarray, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Good = correct on un-noised labels, rarest true levels first; bad = largest errors."""
    cols = ["issue_type", "road_type", "dist_to_school_m", "dist_to_hospital_m", "children",
            "accident", "flooding", "duplicate_count", "rubric_severity", "severity"]
    out = df[cols].assign(pred=[LEVELS[i] for i in pred],
                          err=np.abs(to_index(df.severity) - pred), noised=df.label_noised,
                          true_i=to_index(df.severity))
    good = (out[(out.err == 0) & ~out.noised]
            .sort_values("true_i", ascending=False, kind="stable")
            .groupby("true_i", sort=False).head(2).head(n))
    bad = out.sort_values("err", ascending=False, kind="stable").head(n)
    return good.drop(columns=["err", "true_i"]), bad.drop(columns="true_i")


PROBES = {  # one change at a time from a quiet pothole on a lane, far from everything
    "base": {},
    "main road": {"road_type": "main"},
    "school 150 m": {"dist_to_school_m": 150.0},
    "school 250 m": {"dist_to_school_m": 250.0},
    "hospital 250 m": {"dist_to_hospital_m": 250.0},
    "3 duplicates": {"duplicate_count": 3},
    "10 duplicates": {"duplicate_count": 10},
    "children": {"children": True},
    "accident": {"accident": True},
    "flooding": {"flooding": True},
    "main+school+children": {"road_type": "main", "dist_to_school_m": 150.0, "children": True},
    "flooding+10 dups+main": {"flooding": True, "duplicate_count": 10, "road_type": "main"},
    "garbage, 5 photo boxes": {"issue_type": "garbage", "image_detections": 5},
}


def probe_table(booster: lgb.Booster, categories: dict[str, list[str]], ward: str) -> pd.DataFrame:
    """Rubric vs model on hand-made cases: does the model react to each modifier like the rubric?"""
    base = {"case_id": "probe", "issue_type": "pothole", "ward": ward, "road_type": "lane",
            "dist_to_school_m": 2000.0, "dist_to_hospital_m": 2000.0}
    rows = []
    for name, change in PROBES.items():
        f = TriageFeatures(**{**base, **change})
        sev, pts, _ = rubric_score(f)
        raw = booster.predict(featurize(pd.DataFrame([f.model_dump(mode="json")]), categories),
                              num_iteration=booster.best_iteration)[0]
        rows.append({"probe": name, "rubric_pts": pts, "rubric": sev.value,
                     "model_raw": raw, "model": LEVELS[to_levels(np.array([raw]))[0]]})
    return pd.DataFrame(rows).set_index("probe")


def log_mlflow(params: dict, val_m: dict, test_m: dict, out_dir: Path) -> str:
    import mlflow  # imported here: training works without it, tests don't need it

    # MLflow 3 retired the plain-folder store, so ./mlruns holds a SQLite db plus artifacts.
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("triage") is None:
        mlflow.create_experiment("triage", artifact_location=(MLRUNS_DIR / "artifacts").as_uri())
    mlflow.set_experiment("triage")
    with mlflow.start_run(run_name=f"triage-{VERSION}") as run:
        mlflow.log_params({**BASE_PARAMS, **params})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_artifacts(str(out_dir), artifact_path="model")
        return run.info.run_id


def main() -> None:
    df = load_cases()
    train, val, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    categories = categories_from(train)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731

    params, booster, grid = tune(train, val, categories)
    print("Val grid (choose by QWK):")
    print(grid.to_string(index=False, float_format=fmt))
    print(f"-> chosen {params}, {booster.best_iteration} rounds\n")

    v1 = predictor(booster, categories)
    val_m = metrics(to_index(val.severity), v1(val))
    test_pred = v1(test)  # the one and only look at test
    test_m = metrics(to_index(test.severity), test_pred)

    print(f"Test (n={len(test)}):")
    table = score_table({"train_mean": mean_baseline(train),
                         "issue_type_mode": issue_type_baseline(train),
                         f"lgbm {VERSION}": v1, "rubric (ceiling)": rubric_reference}, test)
    print(table.to_string(float_format=fmt))

    print("\nConfusion matrix, test:")
    print(pd.DataFrame(confusion(to_index(test.severity), test_pred),
                       index=[f"true {s}" for s in LEVELS],
                       columns=[f"pred {s}" for s in LEVELS]).to_string())
    agree = (test_pred == to_index(test.rubric_severity)).mean()
    noised = test.label_noised.to_numpy()
    wrong = test_pred != to_index(test.severity)
    print(f"\nAgreement with noise-free rubric: {agree:.1%}. "
          f"Of {wrong.sum()} test errors, {(wrong & noised).sum()} are on noised labels.")

    print("\nFeature importance (share of total gain):")
    print(importances(booster).to_string(float_format=fmt))

    good, bad = examples(test, test_pred)
    with pd.option_context("display.width", 200):
        print("\n5 good test predictions:")
        print(good.to_string(index=False))
        print("\n5 bad test predictions:")
        print(bad.to_string(index=False))
        print("\nRubric vs model on probe cases:")
        print(probe_table(booster, categories, train.ward.mode()[0]).to_string(float_format=fmt))

    out_dir = MODELS_DIR / "triage" / VERSION
    meta = {"version": VERSION, "features": FEATURES, "categorical": CATEGORICAL,
            "categories": categories, "params": {**BASE_PARAMS, **params},
            "best_iteration": booster.best_iteration, "val": val_m, "test": test_m}
    save(booster, meta, out_dir)
    run_id = log_mlflow(params, val_m, test_m, out_dir)
    print(f"\nSaved {out_dir}; MLflow run {run_id} in {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
