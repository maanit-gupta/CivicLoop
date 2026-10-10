"""Triage evaluation + trivial baselines (Step 1.3). Run: python -m triage.evaluate

Severity is ordinal (low < medium < high < critical), so metrics work on level indices 0..3:
MAE in levels, quadratic weighted kappa (QWK), within-1 and exact accuracy, critical recall.
Metrics are hand-written in numpy so every number can be traced by reading ~10 lines.

A predictor is any function DataFrame -> array of level indices. Baselines are fitted on the
train split only. The rubric row is a ceiling reference, not a baseline: it computed the labels
before noise, so ~92% exact accuracy is the best any model can reach.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from contracts.api import Severity
from triage.generate_data import OUT_PATH

LEVELS = [s.value for s in (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)]
K = len(LEVELS)
CRITICAL = K - 1

Predictor = Callable[[pd.DataFrame], np.ndarray]


def to_index(labels) -> np.ndarray:
    """Severity strings -> level indices 0..3."""
    return np.array([LEVELS.index(s) for s in labels])


# ---- metrics ----------------------------------------------------------------------------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """K x K counts; rows = true level, columns = predicted level."""
    m = np.zeros((K, K), dtype=int)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa: 1 - sum(W*O) / sum(W*E).

    O = observed confusion matrix, E = the matrix expected if predictions were independent of
    the truth (outer product of the two label histograms), W[i, j] = (i - j)^2 / (K - 1)^2.
    A constant predictor gives O == E, so kappa = 0.
    """
    o = confusion(y_true, y_pred).astype(float)
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / o.sum()
    i, j = np.indices((K, K))
    w = (i - j) ** 2 / (K - 1) ** 2
    denom = (w * e).sum()
    return 1.0 - (w * o).sum() / denom if denom > 0 else 0.0


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y_true - y_pred)
    is_crit = y_true == CRITICAL
    return {
        "mae": float(err.mean()),
        "qwk": qwk(y_true, y_pred),
        "within1": float((err <= 1).mean()),
        "exact": float((err == 0).mean()),
        "crit_recall": float((y_pred[is_crit] == CRITICAL).mean()) if is_crit.any() else np.nan,
    }


def per_issue_mae(df: pd.DataFrame, y_pred: np.ndarray) -> pd.Series:
    err = np.abs(to_index(df.severity) - y_pred)
    return pd.Series(err, index=df.index).groupby(df.issue_type).mean()


# ---- baselines --------------------------------------------------------------------------------

def mean_baseline(train: pd.DataFrame) -> Predictor:
    """Predict the train mean level, rounded to the nearest level, for every case."""
    level = int(np.rint(to_index(train.severity).mean()))
    return lambda df: np.full(len(df), level)


def issue_type_baseline(train: pd.DataFrame) -> Predictor:
    """Predict the most common train label for the case's issue type.

    (The rubric's base points alone are 1-2 for every type, which all map to "low", so a
    literal "base severity" baseline would just be "always low".)
    """
    idx = pd.Series(to_index(train.severity), index=train.index)
    table = idx.groupby(train.issue_type).agg(lambda s: s.mode().min())
    fallback = int(idx.mode().min())
    return lambda df: df.issue_type.map(table).fillna(fallback).astype(int).to_numpy()


def rubric_reference(df: pd.DataFrame) -> np.ndarray:
    """The noise-free formula (ceiling reference, not a baseline)."""
    return to_index(df.rubric_severity)


# ---- report -----------------------------------------------------------------------------------

def score_table(predictors: dict[str, Predictor], df: pd.DataFrame) -> pd.DataFrame:
    """One row per (predictor, subset): metrics on all cases and on the natural stratum only."""
    y = to_index(df.severity)
    rows = []
    for name, predict in predictors.items():
        pred = predict(df)
        for subset, mask in (("all", np.ones(len(df), bool)),
                             ("natural", (df.stratum == "natural").to_numpy())):
            rows.append({"predictor": name, "subset": subset, **metrics(y[mask], pred[mask])})
    return pd.DataFrame(rows).set_index(["predictor", "subset"])


def load_cases() -> pd.DataFrame:
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH} missing. Run: python -m triage.generate_data")
    return pd.read_parquet(OUT_PATH)


def main(split: str = "test") -> None:
    df = load_cases()
    train, ev = df[df.split == "train"], df[df.split == split].reset_index(drop=True)
    predictors = {
        "train_mean": mean_baseline(train),
        "issue_type_mode": issue_type_baseline(train),
        "rubric (ceiling)": rubric_reference,
    }
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(f"Split: {split} (n={len(ev)}, natural={int((ev.stratum == 'natural').sum())})\n")
    print(score_table(predictors, ev).to_string(float_format=fmt))

    best = predictors["issue_type_mode"](ev)
    cm = pd.DataFrame(confusion(to_index(ev.severity), best),
                      index=[f"true {s}" for s in LEVELS], columns=[f"pred {s}" for s in LEVELS])
    print("\nConfusion matrix, issue_type_mode:")
    print(cm.to_string())

    per_issue = pd.DataFrame({name: per_issue_mae(ev, p(ev)) for name, p in predictors.items()})
    print("\nMAE per issue type:")
    print(per_issue.to_string(float_format=fmt))


if __name__ == "__main__":
    main()
