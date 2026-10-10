"""A fresh "live" batch: complaints that arrive after v1 was trained, as cases ready to score.

The shared corpus is fully used by train/test, so new complaints come from the same generator
with a different seed. Ids get an `L-` prefix and times move one window (92 days) later, so they
can never be confused with corpus rows. One case per incident (the first report); the other
reports of that incident become `duplicate_count`. Spam is dropped: the verifier handles it.
"""

from datetime import timedelta

import pandas as pd

import config
from contracts.api import Cues, GeoPoint, Report, TriageFeatures
from contracts.corpus_gen import WINDOW_DAYS, build_corpus
from contracts.world import distance_to_nearest, ward_of

LIVE_N = 1200  # complaints generated; about 960 incidents survive (no spam, one per incident)
LIVE_SEED = config.SEED + 1
LIVE_PATH = config.DATA_DIR / "trainer" / "live.parquet"
# Step 3.4: the canary runs on a later batch (prefix K, two windows later), so the officials'
# training batch and the canary never share a case.
CANARY_N = 2000  # about 1600 cases: ~160 in a 10% canary arm
CANARY_SEED = config.SEED + 2
CANARY_PATH = config.DATA_DIR / "trainer" / "canary.parquet"


def build_live(n: int = LIVE_N, seed: int = LIVE_SEED, prefix: str = "L",
               windows: int = 1) -> pd.DataFrame:
    df = build_corpus(n=n, seed=seed)
    df = df[~df.is_spam].copy()
    df["incident_id"] = prefix + df.incident_id.str.removeprefix("I")
    df["complaint_id"] = prefix + df.complaint_id.str.removeprefix("R")
    df["submitted_at"] += timedelta(days=windows * WINDOW_DAYS)
    df["duplicate_count"] = df.groupby("incident_id").complaint_id.transform("size") - 1
    cases = df[~df.is_duplicate].copy()  # first report of each incident: names every cue
    cases["case_id"] = "C-" + cases.incident_id
    cases["split"] = "live"
    return cases.reset_index(drop=True)


def load_live(path=LIVE_PATH) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build_live().to_parquet(path, index=False)
    return pd.read_parquet(path)


def load_canary(path=CANARY_PATH) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build_live(CANARY_N, CANARY_SEED, prefix="K", windows=2).to_parquet(path, index=False)
    return pd.read_parquet(path)


def to_report(row) -> Report:
    """The case's report. Intake is not built yet, so its fields are the generator's truth."""
    return Report(
        report_id=row.complaint_id, submitted_at=row.submitted_at, original_text=row.text,
        language=row.language, english_text=row.text if row.language == "en" else "",
        issue_type=row.issue_type, location_text=row.location_text,
        location=GeoPoint(lat=row.lat, lon=row.lon),
        cues=Cues(children=row.children, accident=row.accident, flooding=row.flooding),
        model_version="synthetic-truth", confidence=1.0,
        explanation="Synthetic live report: fields are the generator's ground truth.")


def to_features(row) -> TriageFeatures:
    point = GeoPoint(lat=row.lat, lon=row.lon)
    return TriageFeatures(
        case_id=row.case_id, issue_type=row.issue_type, ward=ward_of(point) or "unknown",
        road_type=row.road_type, dist_to_school_m=distance_to_nearest(point, "school"),
        dist_to_hospital_m=distance_to_nearest(point, "hospital"),
        children=row.children, accident=row.accident, flooding=row.flooding,
        duplicate_count=int(row.duplicate_count))
