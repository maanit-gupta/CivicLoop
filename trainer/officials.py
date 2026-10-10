"""Simulated officials reviewing v1 decisions on the live batch: python -m trainer.officials

The overrides are SIMULATED. Each official follows a fixed hidden policy; the trainer only ever
sees its effect (overrides in the DecisionLog) and must discover it from that.

  A  severity desk. Within 200 m of a school, never accepts a severity below HIGH.
  B  routing desk.  Water standing on the road (water leak or drain blockage, with flooding)
                    goes to Stormwater Drains, whatever the 2.1 rule says.
  C  covers 20% of cases (both fields) instead of A and B; changes a field at random 15% of
                    the time. Pure noise: a good trainer should NOT learn it.

Policies are absolute ("never below HIGH"), not relative to the model ("one level above what
the model said"): a relative rule has no fixed point, because a model that learned it would get
bumped again.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
from contracts.api import (
    Case,
    Department,
    IssueType,
    Override,
    Severity,
    TriageFeatures,
    append,
)
from router import api as router_api
from trainer.live import load_live, to_features, to_report
from triage import api as triage_api

SCHOOL_RADIUS_M = 200.0
A_FLOOR = Severity.HIGH
WATER_ISSUES = {IssueType.WATER_LEAK, IssueType.DRAIN_BLOCKAGE}
C_SHARE = 0.2      # share of cases C reviews instead of A and B
C_CHANGE_P = 0.15  # per field, chance C changes the model's value
LEVELS = list(Severity)  # low < medium < high < critical

TRAINER_DIR = config.DATA_DIR / "trainer"
LOG_PATH = TRAINER_DIR / "decision_log.jsonl"
ANSWER_KEY_PATH = TRAINER_DIR / "reviews.parquet"  # simulator's own record, for evaluation only


@dataclass(frozen=True)
class Review:
    """What an official sees: the case's facts and the model's two decisions."""

    case_id: str
    submitted_at: datetime
    features: TriageFeatures
    severity: Severity
    department: Department


def policy_a(r: Review) -> Severity | None:
    near_school = r.features.dist_to_school_m < SCHOOL_RADIUS_M
    if near_school and LEVELS.index(r.severity) < LEVELS.index(A_FLOOR):
        return A_FLOOR
    return None


def policy_b(r: Review) -> Department | None:
    water_on_road = r.features.issue_type in WATER_ISSUES and r.features.flooding
    if water_on_road and r.department != Department.STORMWATER_DRAINS:
        return Department.STORMWATER_DRAINS
    return None


def policy_c(r: Review, rng: np.random.Generator) -> tuple[Severity | None, Department | None]:
    def maybe_change(value, options):
        if rng.random() >= C_CHANGE_P:
            return None
        others = [o for o in options if o != value]
        return others[rng.integers(len(others))]
    return maybe_change(r.severity, LEVELS), maybe_change(r.department, list(Department))


REASONS = {"A": "Severity too low for this spot.", "B": "Wrong department.",
           "C": "Changed on review."}  # terse, like real officials: the policy stays hidden


def review(r: Review, rng: np.random.Generator) -> tuple[list[Override], dict[str, str]]:
    """(overrides, reviewer per field). Most reviews change nothing."""
    if rng.random() < C_SHARE:
        severity, dept = policy_c(r, rng)
        reviewers = {"severity": "C", "department": "C"}
    else:
        severity, dept = policy_a(r), policy_b(r)
        reviewers = {"severity": "A", "department": "B"}
    at = r.submitted_at + timedelta(hours=float(rng.uniform(1, 48)))
    out = []
    for field, module, model_value, human_value in (
            ("severity", "triage", r.severity, severity),
            ("department", "router", r.department, dept)):
        if human_value is not None:
            who = reviewers[field]
            out.append(Override(case_id=r.case_id, module=module, field=field,
                                model_value=model_value.value, human_value=human_value.value,
                                official_id=who, reason=REASONS[who], at=at))
    return out, reviewers


def simulate(live: pd.DataFrame, log_path=LOG_PATH, seed: int = config.SEED,
             versions: Callable[[str], dict[str, str]] | None = None) -> pd.DataFrame:
    """Score every live case with the served models, let officials review it, and log both.
    `versions(case_id)` -> {"triage": v, "router": v} picks the models per case (the canary);
    default: config.MODEL_VERSIONS.
    Returns the answer key: one row per case with reviewers, model and human values."""
    rng = np.random.default_rng([seed, 31])  # own stream: independent of other seeded draws
    rows = []
    for row in live.itertuples(index=False):
        v = versions(row.case_id) if versions else {}
        features = to_features(row)
        tri = triage_api.score(features, version=v.get("triage"), log_path=log_path)
        case = Case(case_id=row.case_id, created_at=row.submitted_at, report=to_report(row),
                    triage=tri)
        rout = router_api.route(case, version=v.get("router"), log_path=log_path)
        r = Review(row.case_id, row.submitted_at, features, tri.severity, rout.department)
        overrides, reviewers = review(r, rng)
        for o in overrides:
            version = tri.model_version if o.module == "triage" else rout.model_version
            append(o.module, version, o.case_id, input={o.field: o.model_value},
                   output={o.field: o.human_value}, human_override=o, path=log_path)
        final = {o.field: o.human_value for o in overrides}
        rows.append({
            "case_id": row.case_id, "issue_type": features.issue_type.value,
            "dist_to_school_m": features.dist_to_school_m, "flooding": features.flooding,
            "severity_reviewer": reviewers["severity"],
            "department_reviewer": reviewers["department"],
            "model_severity": tri.severity.value,
            "human_severity": final.get("severity", tri.severity.value),
            "model_department": rout.department.value,
            "human_department": final.get("department", rout.department.value),
        })
    return pd.DataFrame(rows)


def main() -> None:
    live = load_live()
    LOG_PATH.unlink(missing_ok=True)  # one simulation run = one fresh log
    key = simulate(live)
    key.to_parquet(ANSWER_KEY_PATH, index=False)
    n_over = int((key.model_severity != key.human_severity).sum()
                 + (key.model_department != key.human_department).sum())
    print(f"{len(live)} live cases scored by triage/router "
          f"{config.MODEL_VERSIONS['triage']}/{config.MODEL_VERSIONS['router']}, "
          f"{n_over} overrides -> {LOG_PATH}")


if __name__ == "__main__":
    main()
