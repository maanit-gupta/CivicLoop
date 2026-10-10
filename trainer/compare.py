"""v1 vs v2 offline comparison (Step 3.3): python -m trainer.compare

Three views, because they can disagree:
1. Old test sets (labels = written rubric / 2.1 routing rules). Policy A and B contradict
   those rules on purpose, so learning the officials can LOWER these scores. Also shown on
   the slice where each policy applies.
2. Override holdout vs the logged human labels. Biased toward v1: officials only reacted to
   v1's answer, so "accepted" means "v1's value was fine", not "only v1's value was fine".
3. Simulated override rate: officials A and B (their deterministic policies) re-review v1 and
   v2 outputs on the holdout. The fairest measure of "would officials still correct this?".
   C is left out: random changes don't depend on the model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from contracts.api import Department, Severity
from router import api as router_api
from trainer.dataset import HOLDOUT_OUT
from trainer.officials import Review, policy_a, policy_b
from trainer.retrain import serve
from triage import api as triage_api

LEVELS = [s.value for s in Severity]
DEPARTMENTS = [d.value for d in Department]
WATER = ["water_leak", "drain_blockage"]
# User decision 3.3: officials' override rate decides promotion, but where no policy applies
# (school >= 200 m) the rubric-test MAE may get worse by at most this much.
GUARDRAIL_MAE = 0.05


def _idx(s) -> np.ndarray:
    return np.array([LEVELS.index(v) for v in s])


def severity_metrics(true, pred) -> dict:
    t, p = _idx(true), _idx(pred)
    return {"MAE": float(np.abs(t - p).mean()),
            "QWK": float(cohen_kappa_score(t, p, labels=range(4), weights="quadratic"))}


def department_metrics(true, pred) -> dict:
    true, pred = np.asarray(true), np.asarray(pred)
    return {"accuracy": float((true == pred).mean()),
            "macro-F1": float(f1_score(true, pred, labels=DEPARTMENTS, average="macro",
                                       zero_division=0))}


def simulated_overrides(served: pd.DataFrame) -> dict:
    a = b = 0
    for r in served.itertuples(index=False):
        rev = Review(r.case_id, r.submitted_at, r.features, Severity(r.severity),
                     Department(r.department))
        a += policy_a(rev) is not None
        b += policy_b(rev) is not None
    n = len(served)
    return {"A override rate": a / n, "B override rate": b / n, "A+B overrides": a + b}


def _both(version: str | dict) -> dict:
    return version if isinstance(version, dict) else {"triage": version, "router": version}


def label(version: str | dict) -> str:
    v = _both(version)
    return v["triage"] if v["triage"] == v["router"] else f"{v['triage']}/{v['router']}"


def version_report(version: str | dict, holdout: pd.DataFrame) -> dict:
    """`version`: one name for both models, or {"triage": v, "router": v}."""
    out, v = {}, _both(version)
    tt = triage_api.predict_test(v["triage"])
    near = tt.dist_to_school_m < 200
    for name, m in severity_metrics(tt.severity, tt.pred).items():
        out[("triage test (rubric)", name)] = m
    for name, m in (("MAE, school <200 m", near), ("MAE, school >=200 m", ~near)):
        out[("triage test (rubric)", name)] = severity_metrics(tt.severity[m], tt.pred[m])["MAE"]
    rt = router_api.predict_test(v["router"])
    water = rt.issue_type.isin(WATER) & rt.flooding
    for name, m in department_metrics(rt.department, rt.pred).items():
        out[("router test (2.1 rules)", name)] = m
    out[("router test (2.1 rules)", "accuracy, water on road")] = department_metrics(
        rt.department[water], rt.pred[water])["accuracy"]

    served = serve(holdout.case_id, version).set_index("case_id").loc[holdout.case_id]
    for name, m in severity_metrics(holdout.human_severity, served.severity).items():
        out[("holdout vs human", f"severity {name}")] = m
    for name, m in department_metrics(holdout.human_department, served.department).items():
        out[("holdout vs human", f"department {name}")] = m
    for name, m in simulated_overrides(served.reset_index()).items():
        out[("holdout, officials re-review", name)] = m
    return out


def compare(v1: str | dict = "v1", v2: str | dict = "v2") -> pd.DataFrame:
    """Columns: label(v1), label(v2), change. Versions as in version_report."""
    holdout = pd.read_parquet(HOLDOUT_OUT)
    a, b = label(v1), label(v2)
    table = pd.DataFrame({a: version_report(v1, holdout), b: version_report(v2, holdout)})
    table["change"] = table[b] - table[a]
    table.index.names = ["view", "metric"]
    return table


def main() -> None:
    pd.set_option("display.width", 120)
    print(compare().to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
