import shutil
from datetime import timedelta

import pandas as pd
import pytest

from config import WORLD_DIR
from contracts import corpus_templates as T
from contracts.api import load_corpus
from contracts.corpus_gen import COLUMNS, LANGUAGE_WEIGHTS, build_corpus, split_of
from contracts.schemas import GeoPoint, IssueType
from contracts.world import load_world, road_type_at, ward_of
from contracts.world_gen import build_world, to_metres

N = 600


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    build_world(d)
    shutil.copy(WORLD_DIR / "departments.yaml", d)
    return load_world(d)


@pytest.fixture(scope="module")
def corpus(world):
    return build_corpus(world, n=N)


@pytest.fixture(scope="module")
def real(corpus):
    return corpus[~corpus.is_spam]


def test_shape_and_rates(corpus):
    assert list(corpus.columns) == COLUMNS
    assert len(corpus) == N and corpus.complaint_id.is_unique
    assert corpus.is_spam.sum() == 30 and corpus.is_duplicate.sum() == 90
    assert set(corpus.language) == set(LANGUAGE_WEIGHTS)
    assert set(corpus.issue_type) == {t.value for t in IssueType}


def test_reproducible(world, corpus):
    pd.testing.assert_frame_equal(corpus, build_corpus(world, n=N))
    assert not build_corpus(world, n=N, seed=7).text.equals(corpus.text)


def test_split_is_frozen_function_of_incident_id(corpus):
    # Pinned: if this changes, every held-out evaluation in the project silently changes.
    pinned = [5, 14, 18, 43, 46, 53, 55, 57, 66, 75, 81, 82, 100, 103, 105, 106, 109, 117, 120,
              123, 125, 133, 137, 149, 168, 173, 177, 185, 190, 193]
    assert [i for i in range(200) if split_of(f"I-{i:05d}") == "test"] == pinned
    assert (corpus.split == corpus.incident_id.map(split_of)).all()
    assert corpus.groupby("incident_id").split.nunique().max() == 1  # no incident in both


def test_location_text_is_in_the_text(real):
    assert all(loc in text for loc, text in zip(real.location_text, real.text, strict=True))


def test_truth_agrees_with_world_helpers(world, real):
    pois = {p.poi_id: p for p in world.pois}
    for row in real.drop_duplicates("incident_id").itertuples():
        point = GeoPoint(lat=row.lat, lon=row.lon)
        assert ward_of(point, world) == row.ward
        assert road_type_at(point, world) == row.road_type
        if row.anchor != "road":
            poi = pois[row.poi_id]
            assert poi.kind == row.anchor and poi.ward_id == row.ward
            dist = poi.point.distance(type(poi.point)(to_metres(row.lon, row.lat)))
            assert dist <= 151


def test_duplicates_share_the_incident(real):
    for _, group in real.groupby("incident_id"):
        if len(group) == 1:
            continue
        truth = ["issue_type", "ward", "lat", "lon", "road_type", "poi_id"]
        assert (group[truth].nunique(dropna=False) == 1).all()
        first = group.submitted_at.min()
        assert not group[group.submitted_at == first].is_duplicate.any()
        assert (group.submitted_at - first <= timedelta(hours=72)).all()


def test_cue_flags_match_the_text(corpus):
    for row in corpus.itertuples():
        for cue, by_lang in T.CUES.items():
            mentioned = any(s in row.text for s in by_lang[row.language])
            assert getattr(row, cue) == mentioned, (row.complaint_id, cue)


def test_spam_has_no_location(corpus):
    spam = corpus[corpus.is_spam]
    assert spam.ward.isna().all() and spam.lat.isna().all() and spam.location_text.isna().all()
    assert not spam[["children", "accident", "flooding"]].any().any()


def test_load_corpus(tmp_path, corpus):
    path = tmp_path / "c.parquet"
    corpus.to_parquet(path, index=False)
    test = load_corpus("test", path)
    assert (test.split == "test").all() and len(test) == (corpus.split == "test").sum()
    with pytest.raises(FileNotFoundError):
        load_corpus(path=tmp_path / "missing.parquet")
