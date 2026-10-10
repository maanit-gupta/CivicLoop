"""Generate the synthetic complaint corpus: python -m contracts.corpus_gen

Templates with slot filling (no Ollama on this machine). Each row is one citizen complaint with
its ground truth. Rows sharing an `incident_id` describe the same real-world problem.
The test split is a pure function of `incident_id`, so it never changes on regeneration.
"""

import hashlib
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from shapely.geometry import Point

from config import CITY, CORPUS_PATH, SEED
from contracts import corpus_templates as T
from contracts.schemas import GeoPoint
from contracts.world import World, load_world, road_type_at, ward_of
from contracts.world_gen import to_lonlat

N_COMPLAINTS = 3000
SPAM_RATE = 0.05
DUPLICATE_RATE = 0.15        # share of complaints that re-report an existing incident
TEST_FRACTION = 0.15         # share of incidents held out; fixed forever by split_of()
POI_ANCHOR_RATE = 0.7        # else the complaint names only the ward and road
LANGUAGE_WEIGHTS = {"hi": 0.35, "mixed": 0.30, "en": 0.25, "pa": 0.10}  # user decision, 0.5
ISSUE_WEIGHTS = {"pothole": 0.22, "garbage": 0.20, "streetlight": 0.15, "drain_blockage": 0.15,
                 "water_leak": 0.13, "damaged_sign": 0.08, "other": 0.07}
# P(cue mentioned) per issue type. Children are likelier near a school (see _cue_probs).
ACCIDENT_P = {"pothole": 0.15, "damaged_sign": 0.12, "streetlight": 0.08, "other": 0.05}
FLOODING_P = {"drain_blockage": 0.45, "water_leak": 0.25, "pothole": 0.08, "garbage": 0.03}
CHILDREN_P = {"school": 0.35, "default": 0.08}
DUPLICATE_KEEPS_CUE_P = 0.8  # a second reporter mentions each true cue with this probability
START = datetime(2026, 7, 1, tzinfo=UTC)
WINDOW_DAYS = 92

COLUMNS = ["complaint_id", "incident_id", "split", "is_duplicate", "is_spam", "submitted_at",
           "language", "text", "template_id", "issue_type", "location_text", "anchor",
           "poi_id", "ward", "lat", "lon", "road_type", "children", "accident", "flooding"]


def split_of(incident_id: str) -> str:
    """'test' for a fixed ~15% of incident ids, else 'train'. Never depends on row content."""
    bucket = int(hashlib.sha256(incident_id.encode()).hexdigest(), 16) % 100
    return "test" if bucket < TEST_FRACTION * 100 else "train"


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights)
    p = np.array([weights[k] for k in keys])
    return keys[rng.choice(len(keys), p=p / p.sum())]


# ---------- incidents: what actually happened, where ----------

def _random_point_near(world: World, rng, ward_id: str, centre: Point | None) -> GeoPoint:
    """A point in `ward_id`: 20-150 m from `centre` if given, else anywhere in the ward."""
    polygon = next(w.polygon for w in world.wards if w.ward_id == ward_id)
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(200):
        if centre is not None:
            r, a = rng.uniform(20, 150), rng.uniform(0, 2 * np.pi)
            x, y = centre.x + r * np.cos(a), centre.y + r * np.sin(a)
        else:
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        lon, lat = to_lonlat(x, y)
        point = GeoPoint(lat=lat, lon=lon)
        if ward_of(point, world) == ward_id:  # truth agrees with the public lookup
            return point
    raise RuntimeError(f"Could not sample a point in ward {ward_id}")


def _sample_incident(world: World, rng, incident_id: str) -> dict:
    issue = _pick(rng, ISSUE_WEIGHTS)
    if rng.random() < POI_ANCHOR_RATE:
        poi = world.pois[rng.integers(len(world.pois))]
        anchor, poi_id, ward_id = poi.kind, poi.poi_id, poi.ward_id
        point = _random_point_near(world, rng, ward_id, poi.point)
    else:
        anchor, poi_id = "road", None
        ward_id = world.wards[rng.integers(len(world.wards))].ward_id
        point = _random_point_near(world, rng, ward_id, None)
    cues = {
        "children": rng.random() < CHILDREN_P.get(anchor, CHILDREN_P["default"]),
        "accident": rng.random() < ACCIDENT_P.get(issue, 0.02),
        "flooding": rng.random() < FLOODING_P.get(issue, 0.0),
    }
    return {"incident_id": incident_id, "issue_type": issue, "anchor": anchor,
            "poi_id": poi_id, "ward": ward_id, "lat": point.lat, "lon": point.lon,
            "road_type": road_type_at(point, world), "cues": cues,
            "ward_name": next(w.name for w in world.wards if w.ward_id == ward_id),
            "first_at": START + timedelta(days=float(rng.uniform(0, WINDOW_DAYS)))}


# ---------- text: how one citizen describes it ----------

def _noisy(lang: str, s: str, rng) -> str:
    """Typing noise on template text (before the location is inserted, so it stays exact)."""
    if lang == "mixed":
        for word, variant in T.HINGLISH_VARIANTS.items():
            if rng.random() < 0.4:
                s = re.sub(rf"\b{word}\b", variant, s)
    elif lang == "en" and rng.random() < 0.3:
        s = s.lower().rstrip(".")
    return s


def _location_text(lang: str, inc: dict, rng) -> str:
    ward = T.WARD_NAMES.get(lang, {}).get(inc["ward_name"], inc["ward_name"])
    if inc["anchor"] == "road":
        return T.ROAD_LOCATIONS[lang][inc["road_type"]].format(ward=ward)
    pattern = T.POI_LOCATIONS[lang][rng.integers(len(T.POI_LOCATIONS[lang]))]
    return pattern.format(ward=ward, lm=T.LANDMARKS[lang][inc["anchor"]])


def _write_complaint(lang: str, inc: dict, cues: dict[str, bool], rng) -> dict:
    cores = T.CORE[lang][inc["issue_type"]]
    k = int(rng.integers(len(cores)))
    loc = _location_text(lang, inc, rng)
    sep = ", " if lang == "mixed" else " "
    parts = [T.OPENERS[lang][rng.integers(len(T.OPENERS[lang]))]
             + _noisy(lang, cores[k], rng).format(loc=loc)]
    for cue, on in cues.items():
        if on:
            options = T.CUES[cue][lang]
            parts.append(options[rng.integers(len(options))])
    if rng.random() < 0.4:
        parts.append(_noisy(lang, T.DURATIONS[lang], rng).format(n=int(rng.integers(2, 15))))
    if rng.random() < 0.5:
        parts.append(_noisy(lang, T.CLOSERS[lang][rng.integers(len(T.CLOSERS[lang]))], rng))
    return {"language": lang, "text": sep.join(parts), "location_text": loc,
            "template_id": f"{lang}:{inc['issue_type']}:{k}", "is_spam": False, **cues}


def _spam(rng, incident_id: str) -> dict:
    lang = _pick(rng, LANGUAGE_WEIGHTS)
    k = int(rng.integers(len(T.SPAM[lang])))
    return {"incident_id": incident_id, "language": lang, "text": T.SPAM[lang][k],
            "template_id": f"{lang}:spam:{k}", "is_spam": True, "issue_type": "other",
            "location_text": None, "anchor": None, "poi_id": None, "ward": None,
            "lat": np.nan, "lon": np.nan, "road_type": None,
            "children": False, "accident": False, "flooding": False,
            "submitted_at": START + timedelta(days=float(rng.uniform(0, WINDOW_DAYS)))}


def build_corpus(world: World | None = None, n: int = N_COMPLAINTS, seed: int = SEED
                 ) -> pd.DataFrame:
    world = world or load_world()
    rng = np.random.default_rng(seed)
    n_spam, n_dup = round(n * SPAM_RATE), round(n * DUPLICATE_RATE)
    incidents = [_sample_incident(world, rng, f"I-{i:05d}") for i in range(n - n_spam - n_dup)]

    rows = []
    for inc in incidents:  # first report: mentions every cue the incident has
        rows.append({**_incident_fields(inc), "submitted_at": inc["first_at"],
                     **_write_complaint(_pick(rng, LANGUAGE_WEIGHTS), inc, inc["cues"], rng)})
    for _ in range(n_dup):  # re-reports: another citizen, language, wording; up to 3 days later
        inc = incidents[rng.integers(len(incidents))]
        cues = {c: on and rng.random() < DUPLICATE_KEEPS_CUE_P for c, on in inc["cues"].items()}
        rows.append({**_incident_fields(inc),
                     "submitted_at": inc["first_at"] + timedelta(hours=float(rng.uniform(0.5, 72))),
                     **_write_complaint(_pick(rng, LANGUAGE_WEIGHTS), inc, cues, rng)})
    rows += [_spam(rng, f"I-{len(incidents) + i:05d}") for i in range(n_spam)]

    df = pd.DataFrame(rows).sort_values(["submitted_at", "incident_id"], ignore_index=True)
    df["complaint_id"] = [f"R-{i:05d}" for i in range(1, len(df) + 1)]
    df["is_duplicate"] = df.duplicated("incident_id")  # every report after the first, by time
    df["split"] = df["incident_id"].map(split_of)
    return df[COLUMNS]


def _incident_fields(inc: dict) -> dict:
    keep = ("incident_id", "issue_type", "anchor", "poi_id", "ward", "lat", "lon", "road_type")
    return {k: inc[k] for k in keep}


def summary(df: pd.DataFrame) -> str:
    real = df[~df.is_spam]
    by_issue = pd.crosstab(real.issue_type, real.language, margins=True, margins_name="all")
    lines = [f"Synthetic {CITY['name']} corpus: {len(df)} complaints, "
             f"{df.incident_id.nunique()} incidents",
             f"spam {df.is_spam.mean():.1%} | duplicates {df.is_duplicate.mean():.1%} | "
             f"test split {(df.split == 'test').mean():.1%} of rows, "
             f"{df.groupby('incident_id').split.first().eq('test').mean():.1%} of incidents",
             f"cues: children {real.children.mean():.1%}, accident {real.accident.mean():.1%}, "
             f"flooding {real.flooding.mean():.1%} | unique texts "
             f"{df.text.nunique() / len(df):.1%} | templates {df.template_id.nunique()}",
             "", str(by_issue)]
    return "\n".join(lines)


def main() -> None:
    df = build_corpus()
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CORPUS_PATH, index=False)
    print(f"Wrote {CORPUS_PATH}")
    print(summary(df))


if __name__ == "__main__":
    main()
