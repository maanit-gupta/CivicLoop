"""Retrain triage and router v2 from the override dataset (Step 3.3): python -m trainer.retrain

Same recipe as v1 (triage/router.api.retrain keep params, rounds, C); the trainer only chooses
the data and one knob: how much an overridden row weighs.

1. Weight selection. VAL_SHARE of the live training cases become a validation slice. For each
   weight in WEIGHTS, candidates are fit on the original rows + the other live rows, the slice
   is served through the APIs, and each model is scored by disagreement with the human label
   (= share of cases an official changed to something else). Lowest wins; ties -> smaller weight.
   The override holdout is not touched until trainer.compare.
2. Refit with the chosen weights on all retrain rows -> models/{triage,router}/v2.
"""

import numpy as np
import pandas as pd

import config
from contracts.api import Case
from router import api as router_api
from trainer.dataset import ROUTER_OUT, TRIAGE_OUT
from trainer.live import load_live, to_features, to_report
from trainer.officials import TRAINER_DIR
from triage import api as triage_api

WEIGHTS = (1.0, 3.0, 10.0, 30.0)
VAL_SHARE = 0.2
VERSION = "v2"
EVAL_LOG = TRAINER_DIR / "eval_log.jsonl"  # model calls made for offline evaluation only


def serve(case_ids, version: str | dict, log_path=EVAL_LOG) -> pd.DataFrame:
    """Score live cases through the public APIs, exactly as the officials' batch was scored.
    `version`: one name for both models, or {"triage": v, "router": v}."""
    v = version if isinstance(version, dict) else {"triage": version, "router": version}
    live = load_live().set_index("case_id").loc[list(case_ids)].reset_index()
    rows = []
    for row in live.itertuples(index=False):
        features = to_features(row)
        tri = triage_api.score(features, version=v["triage"], log_path=log_path)
        case = Case(case_id=row.case_id, created_at=row.submitted_at, report=to_report(row),
                    triage=tri)
        rout = router_api.route(case, version=v["router"], log_path=log_path)
        rows.append({"case_id": row.case_id, "submitted_at": row.submitted_at,
                     "features": features, "severity": tri.severity.value,
                     "department": rout.department.value})
    return pd.DataFrame(rows)


def with_weight(rows: pd.DataFrame, w: float) -> pd.DataFrame:
    return rows.assign(weight=np.where(rows.source == "live_override", w, 1.0))


def val_cases(triage_rows: pd.DataFrame, seed: int = config.SEED) -> set[str]:
    live_ids = sorted(triage_rows.case_id[triage_rows.source != "original"])
    rng = np.random.default_rng([seed, 33])
    return {c for c in live_ids if rng.random() < VAL_SHARE}


def select_weights(tri: pd.DataFrame, rout: pd.DataFrame,
                   version: str = VERSION) -> tuple[dict, pd.DataFrame]:
    """({'triage': w, 'router': w}, table of validation disagreement per weight)."""
    val = val_cases(tri)
    human = tri[tri.case_id.isin(val)].set_index("case_id").severity.rename("h_sev").to_frame()
    human["h_dep"] = rout[rout.case_id.isin(val)].set_index("case_id").department
    rows = []
    for w in WEIGHTS:
        trial = f"{version}-w{w:g}"
        triage_api.retrain(with_weight(tri[~tri.case_id.isin(val)], w), trial)
        router_api.retrain(with_weight(rout[~rout.case_id.isin(val)], w), trial)
        got = serve(sorted(val), trial).set_index("case_id").join(human)
        rows.append({"weight": w, "val_cases": len(got),
                     "triage_disagree": float((got.severity != got.h_sev).mean()),
                     "router_disagree": float((got.department != got.h_dep).mean())})
    table = pd.DataFrame(rows)
    best = {m: float(table.sort_values([f"{m}_disagree", "weight"]).weight.iloc[0])
            for m in ("triage", "router")}
    return best, table


def run(version: str = VERSION) -> tuple[dict, pd.DataFrame]:
    """Weight selection, then refit both models as `version`. Returns (weights, table)."""
    EVAL_LOG.unlink(missing_ok=True)
    tri, rout = pd.read_parquet(TRIAGE_OUT), pd.read_parquet(ROUTER_OUT)
    best, table = select_weights(tri, rout, version)
    triage_api.retrain(with_weight(tri, best["triage"]), version)
    router_api.retrain(with_weight(rout, best["router"]), version)
    return best, table


def main() -> None:
    best, table = run()
    print("Override weight selection (validation slice of live training cases):")
    print(table.round(3).to_string(index=False))
    print(f"-> chosen: {best}")
    print(f"Saved triage and router {VERSION} (override weights {best['triage']:g} / "
          f"{best['router']:g}).")


if __name__ == "__main__":
    main()
