"""Plain-language explanations for triage v1 (Step 1.5).

LightGBM's pred_contrib=True splits one prediction into per-feature contributions (TreeSHAP):
    raw prediction = bias (the average case) + sum of contributions
in severity levels, so +0.8 means "this feature pushed it up almost one level". The biggest
contributions, with the case's actual values, become the sentence an official reads.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from contracts.api import Severity, TriageFeatures

MIN_EFFECT = 0.15  # levels; smaller contributions are not worth an official's attention
MAX_UP, MAX_DOWN = 3, 2


def contributions(booster: lgb.Booster, x: pd.DataFrame) -> tuple[pd.Series, float]:
    """(per-feature contribution for one row, bias). bias + contributions.sum() == raw."""
    c = booster.predict(x, pred_contrib=True)[0]
    return pd.Series(c[:-1], index=x.columns), float(c[-1])


def _distance(m: float) -> str:
    return f"{m:.0f} m" if m < 1000 else f"{m / 1000:.1f} km"


def phrase(name: str, f: TriageFeatures) -> str:
    """The feature's actual value, in words."""
    v = getattr(f, name)
    match name:
        case "issue_type":
            return v.value.replace("_", " ")
        case "road_type":
            return "on a lane" if v == "lane" else f"on a {v} road"
        case "dist_to_school_m":
            return f"school {_distance(v)} away"
        case "dist_to_hospital_m":
            return f"hospital {_distance(v)} away"
        case "children" | "accident" | "flooding":
            return f"{name} mentioned" if v else f"no {name} mentioned"
        case "duplicate_count":
            return f"{v} other report{'s' * (v != 1)}" if v else "no other reports"
        case "image_detections":
            return f"{v} problem area{'s' * (v != 1)} in photos" if v else "nothing seen in photos"
        case "ward":
            return f"location (ward {v})"
    return f"{name} = {v}"


def sentence(severity: Severity, f: TriageFeatures, contrib: pd.Series) -> str:
    """e.g. 'Severity high. Main reasons: flooding mentioned, on a main road. Counting against:
    no other reports.' Reasons are relative to an average complaint."""
    strong = contrib[np.abs(contrib) >= MIN_EFFECT]
    up = strong[strong > 0].sort_values(ascending=False).head(MAX_UP)
    down = strong[strong < 0].sort_values().head(MAX_DOWN)
    text = f"Severity {severity.value}."
    if len(up):
        text += " Main reasons: " + ", ".join(phrase(n, f) for n in up.index) + "."
    if len(down):
        text += " Counting against: " + ", ".join(phrase(n, f) for n in down.index) + "."
    if not len(strong):
        text += " Close to a typical complaint; no single factor stands out."
    return text
