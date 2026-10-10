"""Triage training data (Step 1.2). Run: python -m triage.generate_data

Each case starts from a real corpus incident (issue type + text cues), moves its location a
little, and computes geo features with the Phase 0 world helpers. The label is the rubric
severity with ~8% of labels shifted one level, so the model cannot score 100% by copying the
formula. Splits are by 500 m grid cell so near-identical places never straddle train and test.

Two strata: "natural" draws incidents uniformly; "enriched" (ENRICHED_FRACTION of cases) draws
risky incidents more often and with more duplicate reports, because critical is only ~0.5% of
natural cases. The `stratum` column keeps this visible for evaluation.
"""

import math

import numpy as np
import pandas as pd

from config import DATA_DIR, SEED
from contracts.api import GeoPoint, IssueType, Severity, TriageFeatures, load_corpus
from contracts.world import distance_to_nearest, load_world, road_type_at, ward_of
from contracts.world_gen import to_lonlat, to_metres
from triage.rubric import rubric_score

N_CASES = 8000
ENRICHED_FRACTION = 0.25
JITTER_M = 150            # a case lies within this distance of its source incident
CLUSTER_M = 500           # grid cell size for location_cluster
NOISE_RATE = 0.08         # share of labels shifted by one level
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_CLASS_SHARE = 0.03
OUT_PATH = DATA_DIR / "triage" / "cases.parquet"

LEVELS = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
# duplicate_count ~ negative binomial (n, p): mostly 0, long right tail.
DUPLICATES = {"natural": (0.5, 0.25), "enriched": (1.0, 0.12)}
PHOTO_RATE = 0.6
# Mean extra detections per photo: garbage often shows several heaps, signs rarely more than one.
EXTRA_DETECTIONS = {IssueType.GARBAGE: 1.5, IssueType.POTHOLE: 1.0, IssueType.WATER_LEAK: 0.3,
                    IssueType.DRAIN_BLOCKAGE: 0.5, IssueType.STREETLIGHT: 0.2,
                    IssueType.DAMAGED_SIGN: 0.1, IssueType.OTHER: 0.3}


def _incidents() -> pd.DataFrame:
    """One row per non-spam corpus incident: its first complaint's issue type, cues, location."""
    corpus = load_corpus()
    first = corpus[~corpus.is_spam & ~corpus.is_duplicate]
    return first[["incident_id", "issue_type", "lat", "lon", "children", "accident",
                  "flooding"]].reset_index(drop=True)


def _risk_weights(inc: pd.DataFrame) -> np.ndarray:
    """2 ** (cue + issue points): an accident or flooding cue makes an incident ~4x likelier."""
    base = inc.issue_type.map(lambda t: 2 if t in ("pothole", "water_leak", "drain_blockage")
                              else 1)
    pts = base + inc.children * 1 + inc.accident * 2 + inc.flooding * 2
    w = np.power(2.0, pts.to_numpy(dtype=float))
    return w / w.sum()


def _jitter(lat: float, lon: float, rng: np.random.Generator) -> tuple[float, float]:
    """Uniform point in a disc of radius JITTER_M around (lat, lon)."""
    x, y = to_metres(lon, lat)
    r, a = JITTER_M * math.sqrt(rng.random()), rng.random() * 2 * math.pi
    new_lon, new_lat = to_lonlat(x + r * math.cos(a), y + r * math.sin(a))
    return new_lat, new_lon


def _cluster(lat: float, lon: float) -> str:
    x, y = to_metres(lon, lat)
    return f"c{math.floor(x / CLUSTER_M)}_{math.floor(y / CLUSTER_M)}"


def _noisy_labels(rubric: list[Severity], rng: np.random.Generator) -> list[Severity]:
    """Shift NOISE_RATE of labels one level up or down. Edge levels shift inward (low -> medium,
    critical -> high) instead of clipping, so every chosen label really changes."""
    out = []
    for sev in rubric:
        i = LEVELS.index(sev)
        if rng.random() < NOISE_RATE:
            step = int(rng.choice([-1, 1]))
            i = i + step if 0 <= i + step < len(LEVELS) else i - step
        out.append(LEVELS[i])
    return out


def _assign_splits(clusters: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Shuffle clusters, then fill train/val/test in order until each reaches its share."""
    sizes = clusters.value_counts().sort_index()
    sizes = sizes.iloc[rng.permutation(len(sizes))]
    cum = sizes.cumsum() / sizes.sum()
    cutoffs = np.cumsum(list(SPLITS.values()))
    names = list(SPLITS)
    # A cluster goes to the split its midpoint falls in, so no split is starved at the edges.
    mid = cum - sizes / sizes.sum() / 2
    which = {c: names[int(np.searchsorted(cutoffs, m))] for c, m in mid.items()}
    return clusters.map(which)


def generate(n: int = N_CASES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    world = load_world()
    inc = _incidents()
    n_enriched = round(n * ENRICHED_FRACTION)
    strata = ["natural"] * (n - n_enriched) + ["enriched"] * n_enriched
    picks = np.concatenate([rng.choice(len(inc), n - n_enriched),
                            rng.choice(len(inc), n_enriched, p=_risk_weights(inc))])

    rows = []
    for i, (stratum, k) in enumerate(zip(strata, picks, strict=True)):
        src = inc.iloc[k]
        lat, lon = _jitter(src.lat, src.lon, rng)
        gp = GeoPoint(lat=lat, lon=lon)
        ward = ward_of(gp, world)
        if ward is None:  # jitter left the city: fall back to the incident's own location
            lat, lon = src.lat, src.lon
            gp = GeoPoint(lat=lat, lon=lon)
            ward = ward_of(gp, world)
        issue = IssueType(src.issue_type)
        dups = int(rng.negative_binomial(*DUPLICATES[stratum]))
        detections = (1 + int(rng.poisson(EXTRA_DETECTIONS[issue]))
                      if rng.random() < PHOTO_RATE else 0)
        f = TriageFeatures(
            case_id=f"T-{i:05d}", issue_type=issue, ward=ward,
            road_type=road_type_at(gp, world),
            dist_to_school_m=round(distance_to_nearest(gp, "school", world), 1),
            dist_to_hospital_m=round(distance_to_nearest(gp, "hospital", world), 1),
            children=bool(src.children), accident=bool(src.accident),
            flooding=bool(src.flooding), duplicate_count=dups, image_detections=detections)
        severity, points, _ = rubric_score(f)
        rows.append({**f.model_dump(mode="json"), "rubric_points": points,
                     "rubric_severity": severity.value, "source_incident": src.incident_id,
                     "stratum": stratum, "lat": lat, "lon": lon,
                     "location_cluster": _cluster(lat, lon)})

    df = pd.DataFrame(rows)
    df["severity"] = [s.value for s in _noisy_labels(
        [Severity(s) for s in df.rubric_severity], rng)]
    df["label_noised"] = df.severity != df.rubric_severity
    df["split"] = _assign_splits(df.location_cluster, rng)
    return df


def label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each severity label per split (rows: split, columns: level)."""
    t = pd.crosstab(df.split, df.severity, normalize="index")
    return t.reindex(index=list(SPLITS), columns=[s.value for s in LEVELS], fill_value=0.0)


def warnings_for(table: pd.DataFrame) -> list[str]:
    return [f"WARNING: {split}/{level} is {share:.1%} (< {MIN_CLASS_SHARE:.0%})"
            for split, row in table.iterrows() for level, share in row.items()
            if share < MIN_CLASS_SHARE]


def main() -> None:
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    table = label_table(df)
    counts = df.split.value_counts().reindex(list(SPLITS))
    print(f"Wrote {len(df)} cases to {OUT_PATH}")
    print("\nLabel share per split (n):")
    print(table.assign(n=counts).to_string(float_format=lambda v: f"{v:.3f}"))
    by_stratum = pd.crosstab(df.stratum, df.severity, normalize="index")[[s.value for s in LEVELS]]
    print("\nLabel share per stratum:")
    print(by_stratum.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nLabels shifted by noise: {df.label_noised.mean():.1%} "
          f"(rubric agreement ceiling {1 - df.label_noised.mean():.1%})")
    print(f"Clusters: {df.location_cluster.nunique()} "
          f"({', '.join(f'{s} {df[df.split == s].location_cluster.nunique()}' for s in SPLITS)})")
    for w in warnings_for(table):
        print(w)


if __name__ == "__main__":
    main()
