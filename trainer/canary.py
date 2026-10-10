"""Canary release of a challenger model (Step 3.4).

Split: sha256(case_id) -> bucket 0..99; buckets below CANARY_PCT go to the challenger. The same
case always lands in the same arm (re-scores never flip), a run is reproducible, and the hash
knows nothing about the case, so the two arms see the same mix of complaints.

Measure: per module and model version, the share of cases an official overrode, read from the
DecisionLog only, as in production. Every official counts (C's noise too): the trainer cannot
tell noise from policy, and noise hits both arms equally.

Rule (per module):
  wait      the challenger arm has fewer than MIN_SAMPLE cases
  promote   challenger override rate is lower, one-sided two-proportion z-test p < ALPHA
  rollback  anything else (not better, or not better by enough to rule out luck)
"""

import hashlib
import math
from dataclasses import dataclass

import pandas as pd

from contracts.api import read_log

CANARY_PCT = 10
MIN_SAMPLE = 100  # challenger-arm cases needed before deciding
ALPHA = 0.05
FIELD = {"triage": "severity", "router": "department"}


def bucket(case_id: str, salt: str = "") -> int:
    return int(hashlib.sha256(f"{salt}{case_id}".encode()).hexdigest()[:8], 16) % 100


def arm(case_id: str, pct: int = CANARY_PCT, salt: str = "") -> str:
    """'challenger' or 'champion'. `salt` gives a different but equally fair split (used by
    the A/A evaluation); serving always uses the default."""
    return "challenger" if bucket(case_id, salt) < pct else "champion"


def outcomes(log_path, module: str) -> pd.DataFrame:
    """One row per case the module decided: case_id, version, overridden (bool)."""
    served, overridden = {}, set()
    for e in read_log(log_path):
        if e.module != module:
            continue
        if e.human_override:
            overridden.add(e.case_id)
        else:
            served[e.case_id] = e.model_version
    return pd.DataFrame({"case_id": list(served), "version": list(served.values()),
                         "overridden": [c in overridden for c in served]})


def p_lower(k_new: int, n_new: int, k_old: int, n_old: int) -> float:
    """One-sided p-value for 'new override rate < old override rate' (pooled z-test).
    Small p = the gap is unlikely to be luck. 1.0 when there is nothing to compare."""
    if min(n_new, n_old) == 0:
        return 1.0
    pooled = (k_new + k_old) / (n_new + n_old)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_new + 1 / n_old))
    if se == 0:
        return 1.0
    z = (k_new / n_new - k_old / n_old) / se
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))  # standard normal CDF at z


@dataclass(frozen=True)
class Decision:
    action: str  # promote | rollback | wait
    champion_rate: float
    challenger_rate: float
    challenger_n: int
    p_value: float
    reason: str


def decide(k_new: int, n_new: int, k_old: int, n_old: int, min_sample: int = MIN_SAMPLE,
           alpha: float = ALPHA) -> Decision:
    rate_new = k_new / n_new if n_new else float("nan")
    rate_old = k_old / n_old if n_old else float("nan")
    p = p_lower(k_new, n_new, k_old, n_old)
    if n_new < min_sample:
        action, why = "wait", f"only {n_new} challenger cases, need {min_sample}"
    elif p < alpha:
        action, why = "promote", f"override rate {rate_new:.1%} vs {rate_old:.1%}, p={p:.3g}"
    else:
        action, why = "rollback", (f"override rate {rate_new:.1%} vs {rate_old:.1%}, p={p:.3g} "
                                   f"(need p<{alpha})")
    return Decision(action, rate_old, rate_new, n_new, p, why)


def naive(k_new: int, n_new: int, k_old: int, n_old: int) -> str:
    """Trivial baseline rule: promote whenever the challenger's rate is lower."""
    return "promote" if n_new and n_old and k_new / n_new < k_old / n_old else "rollback"


def evaluate_module(log_path, module: str, champion: str, challenger: str,
                    min_sample: int = MIN_SAMPLE, alpha: float = ALPHA) -> Decision:
    out = outcomes(log_path, module)
    new, old = out[out.version == challenger], out[out.version == champion]
    return decide(int(new.overridden.sum()), len(new), int(old.overridden.sum()), len(old),
                  min_sample, alpha)
