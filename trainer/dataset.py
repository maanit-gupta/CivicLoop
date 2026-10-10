"""Build retraining data from the DecisionLog (Step 3.2): python -m trainer.dataset

The log is the only input about humans. Per live case: the model's own triage and router
entries give the features and what v1 said; override entries, joined by (case_id, module),
give what the official said instead. Every live case was reviewed, so no override = accepted,
and the final human label is known for every case.

Outputs (data/trainer/):
  triage_retrain.parquet  original triage train rows (weight 1) + live rows with human labels
  router_retrain.parquet  original router train rows (weight 1) + live rows with human labels
                          (live: accepted weight 1, overridden OVERRIDE_WEIGHT)
  override_holdout.parquet  HOLDOUT_SHARE of live cases, never in the retrain sets (for 3.3)
  preference_pairs.jsonl  one per override: prompt, chosen = human value, rejected = model (3.5)

Nothing here knows the officials' policies. `official_id` is kept so 3.3 can tell them apart.
"""

import json

import numpy as np
import pandas as pd

import config
from contracts.api import read_log, verify_chain
from trainer.officials import LOG_PATH, TRAINER_DIR

OVERRIDE_WEIGHT = 3.0  # an override is rare and carries the policy; tunable in 3.3
HOLDOUT_SHARE = 0.3

TRIAGE_ORIGINAL = config.DATA_DIR / "triage" / "cases.parquet"
ROUTER_ORIGINAL = config.DATA_DIR / "router" / "train.parquet"
TRIAGE_COLS = ["case_id", "issue_type", "ward", "road_type", "dist_to_school_m",
               "dist_to_hospital_m", "children", "accident", "flooding", "duplicate_count",
               "image_detections"]
ROUTER_COLS = ["case_id", "text", "issue_type", "road_type",
               "flooding"]  # flooding: not a router feature, kept for diagnostics

TRIAGE_OUT = TRAINER_DIR / "triage_retrain.parquet"
ROUTER_OUT = TRAINER_DIR / "router_retrain.parquet"
HOLDOUT_OUT = TRAINER_DIR / "override_holdout.parquet"
PAIRS_OUT = TRAINER_DIR / "preference_pairs.jsonl"


def live_decisions(log_path=LOG_PATH) -> pd.DataFrame:
    """One row per live case: triage features, router input, model and human values."""
    broken = verify_chain(log_path)
    if broken is not None:
        raise ValueError(f"Decision log chain broken at entry {broken}: refusing to train on it")
    model, human = {}, {}
    for e in read_log(log_path):
        if e.human_override:
            o = e.human_override
            human[(e.case_id, o.field)] = (o.human_value, o.official_id)
        else:  # a model call; a later re-score of the same case replaces an earlier one
            model[(e.case_id, e.module)] = e
    rows = []
    for case_id in sorted({c for c, _ in model}):
        tri, rout = model.get((case_id, "triage")), model.get((case_id, "router"))
        if tri is None or rout is None:
            continue  # only cases both models decided
        row = {k: tri.input[k] for k in TRIAGE_COLS}
        row["text"] = rout.input["text"]
        for field, entry in (("severity", tri), ("department", rout)):
            value, official = human.get((case_id, field), (entry.output[field], None))
            row |= {f"model_{field}": entry.output[field], f"human_{field}": value,
                    f"{field}_official": official, f"{field}_version": entry.model_version}
        rows.append(row)
    return pd.DataFrame(rows)


def split_holdout(live: pd.DataFrame, seed: int = config.SEED) -> pd.Series:
    """True for held-out cases. Split by case, so a case is wholly in or out."""
    rng = np.random.default_rng([seed, 32])
    return pd.Series(rng.random(len(live)) < HOLDOUT_SHARE, index=live.index)


def _live_rows(live: pd.DataFrame, field: str, cols: list[str]) -> pd.DataFrame:
    overridden = live[f"{field}_official"].notna()
    out = live[cols].copy()
    out[field] = live[f"human_{field}"]
    out["weight"] = np.where(overridden, OVERRIDE_WEIGHT, 1.0)
    out["source"] = np.where(overridden, "live_override", "live_accepted")
    out["official_id"] = live[f"{field}_official"]
    return out


def triage_set(train_live: pd.DataFrame) -> pd.DataFrame:
    orig = pd.read_parquet(TRIAGE_ORIGINAL)
    orig = orig[orig.split == "train"][TRIAGE_COLS + ["severity"]]
    orig = orig.assign(weight=1.0, source="original", official_id=None)
    return pd.concat([orig, _live_rows(train_live, "severity", TRIAGE_COLS)],
                     ignore_index=True)


def router_set(train_live: pd.DataFrame) -> pd.DataFrame:
    orig = pd.read_parquet(ROUTER_ORIGINAL).rename(columns={"complaint_id": "case_id"})
    orig = orig[ROUTER_COLS + ["department"]]
    orig = orig.assign(weight=1.0, source="original", official_id=None)
    return pd.concat([orig, _live_rows(train_live, "department", ROUTER_COLS)],
                     ignore_index=True)


def _prompt(row) -> str:
    return (f"Complaint: {row.text}\n"
            f"Issue: {row.issue_type}. Road: {row.road_type}. Ward: {row.ward}. "
            f"School {row.dist_to_school_m:.0f} m, hospital {row.dist_to_hospital_m:.0f} m. "
            f"Children: {row.children}. Accident: {row.accident}. Flooding: {row.flooding}. "
            f"Other reports: {row.duplicate_count}.")


def preference_pairs(live: pd.DataFrame) -> list[dict]:
    """chosen = the official's value, rejected = the model's. Noise (C) is included and tagged."""
    pairs = []
    for row in live.itertuples(index=False):
        for field in ("severity", "department"):
            official = getattr(row, f"{field}_official")
            if pd.isna(official):  # pandas stores "no override" as NaN
                continue
            pairs.append({"case_id": row.case_id, "field": field, "official_id": official,
                          "prompt": _prompt(row) + f"\nDecide the {field}.",
                          "chosen": getattr(row, f"human_{field}"),
                          "rejected": getattr(row, f"model_{field}")})
    return pairs


def override_report(live: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per official, per issue type). The log records overrides, not who accepted what, so a
    per-official rate can only be 'per 100 live cases', not 'per case that official reviewed'."""
    n = len(live)
    per_official = (pd.concat([live.severity_official.rename("official").to_frame()
                               .assign(field="severity"),
                               live.department_official.rename("official").to_frame()
                               .assign(field="department")])
                    .dropna().groupby(["official", "field"]).size().unstack(fill_value=0))
    per_official["total"] = per_official.sum(axis=1)
    per_official["per_100_cases"] = 100 * per_official.total / n
    per_issue = live.groupby("issue_type").agg(
        cases=("case_id", "size"),
        severity_ovr_rate=("severity_official", lambda s: s.notna().mean()),
        department_ovr_rate=("department_official", lambda s: s.notna().mean()))
    return per_official.reset_index(), per_issue.reset_index()


def build(log_path=LOG_PATH) -> dict[str, pd.DataFrame]:
    live = live_decisions(log_path)
    held = split_holdout(live)
    train_live = live[~held]
    return {"live": live, "holdout": live[held].reset_index(drop=True),
            "triage": triage_set(train_live), "router": router_set(train_live),
            "pairs": pd.DataFrame(preference_pairs(train_live))}


def save(out: dict[str, pd.DataFrame]) -> None:
    out["triage"].to_parquet(TRIAGE_OUT, index=False)
    out["router"].to_parquet(ROUTER_OUT, index=False)
    out["holdout"].to_parquet(HOLDOUT_OUT, index=False)
    with open(PAIRS_OUT, "w", encoding="utf-8") as f:
        for p in out["pairs"].to_dict("records"):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main() -> None:
    out = build()
    save(out)
    for name in ("triage", "router"):
        print(f"{name}_retrain: " + ", ".join(
            f"{k} {v}" for k, v in out[name].source.value_counts().items()))
    print(f"holdout: {len(out['holdout'])} cases; preference pairs: {len(out['pairs'])}\n")
    per_official, per_issue = override_report(out["live"])
    print(per_official.round(1).to_string(index=False), "\n")
    print(per_issue.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
