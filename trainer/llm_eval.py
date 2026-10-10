"""Score the triage LLM against LightGBM (Step 3.5): python -m trainer.llm_eval

Reads the notebook's answers from models/trainer/triage_llm/predictions.jsonl. Without that
file it still prints the baselines, so the table is ready before any GPU time is spent.

Columns: majority ("low" everywhere), LightGBM v1 and v2, llm-sft, llm-dpo.
The pairs to read: llm-sft vs v1 (both trained on the rubric only) and llm-dpo vs v2 (both
also learned from the officials' overrides).
  rubric test        triage test split, rubric labels: MAE, QWK, MAE by school distance.
                     DPO should lower MAE near schools (policy A) and leave >=200 m alone.
  holdout            override holdout vs the logged human label, and how often official A
                     would still override (severity below high within 200 m of a school).
  LLM only           invalid answers (not JSON or unknown severity: scored as "low", the
                     majority guess) and faithful explanations (exactly equal to
                     llm_data.explain: the model restated the input it was given).
"""

import numpy as np
import pandas as pd

import config
from trainer.compare import severity_metrics
from trainer.dataset import HOLDOUT_OUT
from trainer.llm_data import explain, parse
from trainer.officials import A_FLOOR, LEVELS, SCHOOL_RADIUS_M
from trainer.retrain import serve
from triage import api as triage_api

LLM_MODELS = config.MODELS_DIR / "trainer" / "triage_llm"
PREDICTIONS = LLM_MODELS / "predictions.jsonl"
LLM_VERSIONS = ("llm-sft", "llm-dpo")
FALLBACK = "low"  # majority class of the training rows


def a_override_rate(dist_to_school_m, severity) -> float:
    """Share of cases official A would override: within 200 m of a school and below high."""
    below = np.array([LEVELS.index(s) < LEVELS.index(A_FLOOR) for s in severity])
    return float(((np.asarray(dist_to_school_m) < SCHOOL_RADIUS_M) & below).mean())


def llm_answers(preds: pd.DataFrame, cases: pd.DataFrame, set_name: str) -> dict:
    """{version: DataFrame(case_id, severity, valid, faithful)} for one eval set, aligned to
    `cases`. A case the notebook did not answer is an error, not a silent skip."""
    out = {}
    ref = {r.case_id: explain(r) for r in cases.itertuples(index=False)}
    for version in LLM_VERSIONS:
        got = preds[(preds.version == version) & (preds.set == set_name)].set_index("case_id")
        missing = set(cases.case_id) - set(got.index)
        if missing:
            raise ValueError(f"{version}: no answer for {len(missing)} {set_name} cases")
        rows = []
        for case_id in cases.case_id:
            p = parse(got.output[case_id])
            rows.append({"case_id": case_id, "severity": p["severity"] if p else FALLBACK,
                         "valid": p is not None,
                         "faithful": p is not None and p["explanation"] == ref[case_id]})
        out[version] = pd.DataFrame(rows)
    return out


def report(preds: pd.DataFrame | None) -> pd.DataFrame:
    test = triage_api.predict_test("v1").rename(columns={"pred": "v1"})
    test["v2"] = triage_api.predict_test("v2").pred.to_numpy()
    holdout = pd.read_parquet(HOLDOUT_OUT)
    hold_pred = {v: serve(holdout.case_id, v).set_index("case_id").severity
                 .loc[holdout.case_id].to_numpy() for v in ("v1", "v2")}
    columns = {"majority": ([FALLBACK] * len(test), [FALLBACK] * len(holdout), None),
               "LightGBM v1": (test.v1, hold_pred["v1"], None),
               "LightGBM v2": (test.v2, hold_pred["v2"], None)}
    if preds is not None:
        llm_test = llm_answers(preds, test, "test")
        llm_hold = llm_answers(preds, holdout, "holdout")
        for v in LLM_VERSIONS:
            both = pd.concat([llm_test[v], llm_hold[v]])
            columns[v] = (llm_test[v].severity, llm_hold[v].severity, both)

    near = (test.dist_to_school_m < SCHOOL_RADIUS_M).to_numpy()
    table = {}
    for name, (t_pred, h_pred, llm) in columns.items():
        t_pred, h_pred = np.asarray(t_pred), np.asarray(h_pred)
        col = {("rubric test", k): m for k, m in severity_metrics(test.severity, t_pred).items()}
        col[("rubric test", "MAE, school <200 m")] = severity_metrics(
            test.severity[near], t_pred[near])["MAE"]
        col[("rubric test", "MAE, school >=200 m")] = severity_metrics(
            test.severity[~near], t_pred[~near])["MAE"]
        col[("holdout", "MAE vs human")] = severity_metrics(holdout.human_severity, h_pred)["MAE"]
        col[("holdout", "A override rate")] = a_override_rate(holdout.dist_to_school_m, h_pred)
        if llm is not None:
            col[("LLM only", "invalid answers")] = float(1 - llm.valid.mean())
            col[("LLM only", "faithful explanations")] = float(llm.faithful.mean())
        table[name] = col
    out = pd.DataFrame(table)
    out.index.names = ["view", "metric"]
    return out


def main() -> None:
    preds = pd.read_json(PREDICTIONS, lines=True) if PREDICTIONS.exists() else None
    if preds is None:
        print(f"No {PREDICTIONS.relative_to(config.ROOT)} yet: run notebooks/triage_llm.ipynb "
              "and unzip triage_llm.zip there. Baselines only:\n")
    pd.set_option("display.width", 140)
    print(report(preds).to_string(float_format=lambda v: f"{v:.3f}", na_rep="-"))


if __name__ == "__main__":
    main()
