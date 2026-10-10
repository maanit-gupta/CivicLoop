"""Build the routing dataset: python -m router.dataset

Rows = every non-spam corpus complaint (department = issue_type looked up in departments.yaml)
plus generated ambiguous complaints whose department depends on a text cue (router/templates.py).
Spam is dropped: the verifier filters it before routing. Splits reuse the corpus split; the
ambiguous rows get theirs from the same split_of(incident_id) rule.
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, SEED
from contracts import corpus_templates as T
from contracts.api import load_corpus
from contracts.corpus_gen import LANGUAGE_WEIGHTS, split_of
from contracts.schemas import GeoPoint
from contracts.world import department_for, ward_of
from router.templates import SCENARIOS
from router.ward import ward_from_text

ROUTER_DATA_DIR = DATA_DIR / "router"
N_PER_SCENARIO = 150  # x3 scenarios = 450 generated ambiguous rows
SEWER_TEMPLATE = ":drain_blockage:2"  # corpus template_id suffix of the sewer-overflow sentence
LOCATION_COLS = ["ward", "lat", "lon", "road_type", "location_text"]
COLUMNS = ["complaint_id", "incident_id", "split", "language", "text", "issue_type",
           "department", "ambiguous", "scenario", "template_id", "is_duplicate",
           *LOCATION_COLS, "children", "accident", "flooding"]


def _from_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    df = corpus[~corpus.is_spam].copy()
    df["department"] = [department_for(t).value for t in df.issue_type]
    df["ambiguous"], df["scenario"] = False, None
    # User rule (2.1): sewage -> Water Supply. The corpus files "sewer is overflowing, drain
    # clogged with plastic" (template 2, every language) under drain_blockage; sewer wins.
    sewer = df.template_id.str.endswith(SEWER_TEMPLATE)
    df.loc[sewer, ["department", "ambiguous", "scenario"]] = [
        "Water Supply", True, "sewer_overflow"]
    return df


def _ambiguous(corpus: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """N_PER_SCENARIO complaints per scenario, half for each department, located at the place
    of a random real corpus complaint in the same language (so location phrasing is natural)."""
    real = corpus[~corpus.is_spam]
    langs, p = list(LANGUAGE_WEIGHTS), np.array(list(LANGUAGE_WEIGHTS.values()))
    rows = []
    for name, sc in SCENARIOS.items():
        depts = list(sc["cues"])
        for i in range(N_PER_SCENARIO):
            lang = langs[rng.choice(len(langs), p=p / p.sum())]
            dept = depts[i % 2]
            host = real[real.language == lang].iloc[rng.integers((real.language == lang).sum())]
            s, c = rng.integers(len(sc["shared"][lang])), rng.integers(len(sc["cues"][dept][lang]))
            opener = T.OPENERS[lang][rng.integers(len(T.OPENERS[lang]))]
            parts = [opener + sc["shared"][lang][s].format(loc=host.location_text),
                     sc["cues"][dept][lang][c]]
            if rng.random() < 0.5:
                parts.append(T.CLOSERS[lang][rng.integers(len(T.CLOSERS[lang]))])
            incident_id = f"A-{len(rows):05d}"
            rows.append({
                "incident_id": incident_id, "split": split_of(incident_id), "language": lang,
                "text": (", " if lang == "mixed" else " ").join(parts),
                # Intake can't tell these apart: either label, independent of the department.
                "issue_type": sc["issue_types"][rng.integers(2)],
                "department": dept, "ambiguous": True, "scenario": name,
                "template_id": f"{lang}:{name}:{s}:{dept}:{c}", "is_duplicate": False,
                **{k: host[k] for k in LOCATION_COLS},
                "children": False, "accident": False, "flooding": False,
            })
    return pd.DataFrame(rows)


def build_dataset(corpus: pd.DataFrame | None = None, seed: int = SEED) -> pd.DataFrame:
    corpus = load_corpus() if corpus is None else corpus
    rng = np.random.default_rng(seed)
    amb = _ambiguous(corpus, rng)
    amb["complaint_id"] = [f"RA-{i:05d}" for i in range(len(amb))]
    df = pd.concat([_from_corpus(corpus), amb], ignore_index=True)
    return df[COLUMNS]


def ward_check(df: pd.DataFrame) -> pd.DataFrame:
    """How well each ward method recovers the true ward: coverage and accuracy."""
    gps = [ward_of(GeoPoint(lat=a, lon=o)) for a, o in zip(df.lat, df.lon, strict=True)]
    text = [ward_from_text(t) for t in df.location_text]
    out = []
    for method, pred in (("gps", gps), ("text (no gps)", text)):
        pred = pd.Series(pred, index=df.index)
        found = pred.notna()
        out.append({"method": method, "coverage": found.mean(),
                    "accuracy": (pred[found] == df.ward[found]).mean()})
    return pd.DataFrame(out)


def summary(df: pd.DataFrame) -> str:
    counts = pd.crosstab(df.department, [df.split, df.ambiguous], margins=True, margins_name="all")
    amb = df[df.ambiguous]
    agree = (amb.issue_type.map(lambda t: department_for(t).value) == amb.department).mean()
    return "\n".join([
        f"Routing dataset: {len(df)} rows ({df.ambiguous.sum()} ambiguous), "
        f"train {(df.split == 'train').sum()} / test {(df.split == 'test').sum()}",
        f"ambiguous rows where issue_type lookup gives the right department: {agree:.1%}",
        "", "rows by department x (split, ambiguous):", str(counts),
        "", "ward lookup:", ward_check(df).to_string(index=False, float_format="%.3f"),
    ])


def main() -> None:
    df = build_dataset()
    ROUTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        df[df.split == split].to_parquet(ROUTER_DATA_DIR / f"{split}.parquet", index=False)
    print(f"Wrote {ROUTER_DATA_DIR}/{{train,test}}.parquet")
    print(summary(df))


if __name__ == "__main__":
    main()
