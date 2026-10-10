"""Router evaluation + trivial baselines: python -m router.evaluate

Metrics: accuracy, macro-F1, confusion matrix, and accuracy on the ambiguous subset (overall and
per scenario), because that subset is where a text-reading router can beat a lookup table.
Baselines: majority department, and issue_type -> department via departments.yaml.
For probabilistic models (2.3): calibration error and a confidence-threshold sweep.
End to end (2.4): every test row as a Case through router.api.route, checking department, ward,
human hand-off, work-order SLA/status and the decision log.
"""

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config import MODELS_DIR
from contracts.api import Case, Cues, GeoPoint, Report, read_log, verify_chain
from contracts.schemas import Department
from contracts.world import department_for, sla_hours
from router.dataset import ROUTER_DATA_DIR

DEPARTMENTS = [d.value for d in Department]
EVAL_TIME = datetime(2026, 10, 1, tzinfo=UTC)


def evaluate(df: pd.DataFrame, pred: pd.Series | list) -> dict:
    """Score predicted departments against df.department. df needs `ambiguous` and `scenario`."""
    pred = pd.Series(list(pred), index=df.index)
    amb = df.ambiguous.astype(bool)
    hit = pred == df.department
    return {
        "accuracy": accuracy_score(df.department, pred),
        "macro_f1": f1_score(df.department, pred, labels=DEPARTMENTS, average="macro",
                             zero_division=0),
        "clear_accuracy": hit[~amb].mean() if (~amb).any() else float("nan"),
        "ambiguous_accuracy": hit[amb].mean() if amb.any() else float("nan"),
        "by_scenario": hit[amb].groupby(df.scenario[amb]).mean().to_dict(),
    }


def confusion(df: pd.DataFrame, pred: pd.Series | list) -> pd.DataFrame:
    """Rows = true department, columns = predicted."""
    m = confusion_matrix(df.department, list(pred), labels=DEPARTMENTS)
    return pd.DataFrame(m, index=DEPARTMENTS, columns=DEPARTMENTS)


def majority_baseline(train: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    return [train.department.mode()[0]] * len(df)


def lookup_baseline(df: pd.DataFrame) -> list[str]:
    return [department_for(t).value for t in df.issue_type]


def ece(y_true: pd.Series | list, pred: pd.Series | list, conf: pd.Series | list,
        n_bins: int = 10) -> float:
    """Expected calibration error: |accuracy - mean confidence| per confidence bin, weighted by
    bin size. 0 = when the model says 80% it is right 80% of the time."""
    hit = np.asarray(list(pred)) == np.asarray(list(y_true))
    conf = np.asarray(list(conf), dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(hit[bins == b].mean() - conf[bins == b].mean())
                     for b in np.unique(bins)))


def threshold_table(df: pd.DataFrame, pred: pd.Series | list, conf: pd.Series | list,
                    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    """For each threshold: share auto-routed (conf >= t), accuracy of those, how many wrong
    departments slip through, and how many ambiguous complaints go to a human."""
    hit = pd.Series(list(pred), index=df.index) == df.department
    conf = pd.Series(list(conf), index=df.index, dtype=float)
    amb = df.ambiguous.astype(bool)
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append({"threshold": t, "auto_routed": auto.mean(),
                     "auto_accuracy": hit[auto].mean() if auto.any() else float("nan"),
                     "wrong_auto": int((auto & ~hit).sum()), "to_human": int((~auto).sum()),
                     "ambiguous_to_human": (~auto[amb]).mean() if amb.any() else float("nan")})
    return pd.DataFrame(rows).set_index("threshold")


def compare(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model; per-scenario ambiguous accuracy flattened into columns."""
    rows = []
    for name, r in results.items():
        flat = {k: v for k, v in r.items() if k != "by_scenario"}
        flat.update({f"amb:{s}": v for s, v in sorted(r["by_scenario"].items())})
        rows.append({"model": name, **flat})
    return pd.DataFrame(rows).set_index("model")


def row_to_case(row, gps: bool = True) -> Case:
    """A dataset row as the Case the API receives (no triage yet, ward not set)."""
    report = Report(
        report_id=row.complaint_id, submitted_at=EVAL_TIME, original_text=row.text,
        language=row.language, issue_type=row.issue_type, location_text=row.location_text,
        location=GeoPoint(lat=row.lat, lon=row.lon) if gps else None,
        cues=Cues(children=row.children, accident=row.accident, flooding=row.flooding),
        model_version="dataset", confidence=1.0, explanation="Router test-set row.")
    return Case(case_id=row.complaint_id, created_at=EVAL_TIME, report=report)


def api_check(test: pd.DataFrame, version: str, gps: bool = True) -> dict:
    """Run every test row through router.api.route end to end, as other modules will."""
    from router.api import route  # imported here: api imports this module's siblings

    log = Path(tempfile.mkdtemp()) / "decision_log.jsonl"
    start = time.perf_counter()
    out = [route(row_to_case(row, gps), version=version, log_path=log)
           for row in test.itertuples()]
    ms = (time.perf_counter() - start) * 1000 / len(test)
    dept = [d.department.value for d in out]
    human = np.array([d.needs_human_routing for d in out])
    hit = np.asarray(dept) == test.department.to_numpy()
    wos = [d.work_order for d in out]
    return {
        "dept_accuracy": hit.mean(),
        "ambiguous_accuracy": hit[test.ambiguous.to_numpy()].mean(),
        "ward_accuracy": np.mean([d.ward == w for d, w in zip(out, test.ward, strict=True)]),
        "ward_unknown": np.mean([d.ward == "unknown" for d in out]),
        "to_human": human.mean(),
        "wrong_auto_routed": int((~hit & ~human).sum()),
        "work_orders_draft": np.mean([w.status == "draft" for w in wos]),
        "sla_matches_table": np.mean([w.sla_hours == sla_hours(w.department, w.severity)
                                      for w in wos]),
        "log_chain_intact": float(verify_chain(log) is None and len(read_log(log)) == len(test)),
        "ms_per_case": ms,
    }


def main() -> None:
    train = pd.read_parquet(ROUTER_DATA_DIR / "train.parquet")
    test = pd.read_parquet(ROUTER_DATA_DIR / "test.parquet")
    preds = {"majority": majority_baseline(train, test), "lookup": lookup_baseline(test)}
    table = compare({n: evaluate(test, p) for n, p in preds.items()})
    print(f"Test set: {len(test)} rows, {test.ambiguous.sum()} ambiguous\n")
    print(table.T.to_string(float_format="%.3f"))
    print("\nLookup confusion matrix (rows = true, cols = predicted):")
    print(confusion(test, preds["lookup"]).to_string())
    if (MODELS_DIR / "router" / "v1" / "model.joblib").exists():
        api = pd.DataFrame({"stub (gps)": api_check(test, "stub"),
                            "v1 (gps)": api_check(test, "v1"),
                            "v1 (no gps)": api_check(test, "v1", gps=False)})
        print("\nEnd to end through router.api.route (test set):")
        print(api.to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
