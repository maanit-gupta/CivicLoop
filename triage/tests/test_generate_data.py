import pandas as pd
import pytest

from contracts.api import GeoPoint, Severity, TriageFeatures
from contracts.world import ward_of
from triage.generate_data import (
    LEVELS,
    NOISE_RATE,
    SPLITS,
    generate,
    label_table,
    warnings_for,
)
from triage.rubric import rubric_score

FEATURE_COLS = list(TriageFeatures.model_fields)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate(n=1200, seed=7)


def test_rows_are_valid_triage_features(df):
    for row in df[FEATURE_COLS].to_dict("records"):
        TriageFeatures.model_validate(row)


def test_rubric_column_matches_rubric(df):
    for row in df.head(200).to_dict("records"):
        sev, pts, _ = rubric_score(TriageFeatures.model_validate(
            {k: row[k] for k in FEATURE_COLS}))
        assert (sev.value, pts) == (row["rubric_severity"], row["rubric_points"])


def test_noise_moves_exactly_one_level_at_about_the_set_rate(df):
    idx = {s.value: LEVELS.index(s) for s in Severity}
    gap = (df.severity.map(idx) - df.rubric_severity.map(idx)).abs()
    assert set(gap.unique()) <= {0, 1}
    assert (gap == 1).equals(df.label_noised)
    assert abs(df.label_noised.mean() - NOISE_RATE) < 0.025


def test_splits_share_no_location_cluster(df):
    clusters = [set(df[df.split == s].location_cluster) for s in SPLITS]
    assert all(a.isdisjoint(b) for i, a in enumerate(clusters) for b in clusters[i + 1:])


def test_split_sizes_near_target(df):
    share = df.split.value_counts(normalize=True)
    for name, target in SPLITS.items():
        assert abs(share[name] - target) < 0.06


def test_same_seed_same_data():
    pd.testing.assert_frame_equal(generate(n=150, seed=3), generate(n=150, seed=3))


def test_feature_shapes_are_realistic(df):
    assert df.duplicate_count.mode()[0] == 0              # most issues reported once
    assert df.dist_to_school_m.mean() > df.dist_to_school_m.median()  # right-skewed
    assert set(df.stratum) == {"natural", "enriched"}
    by_stratum = df.groupby("stratum").rubric_points.mean()
    assert by_stratum["enriched"] > by_stratum["natural"]


def test_ward_comes_from_world_helper(df):
    for row in df.head(100).to_dict("records"):
        assert ward_of(GeoPoint(lat=row["lat"], lon=row["lon"])) == row["ward"]


def test_rare_class_warning():
    table = pd.DataFrame({"low": [0.98], "critical": [0.02]}, index=["train"])
    assert warnings_for(table) == ["WARNING: train/critical is 2.0% (< 3%)"]
    assert warnings_for(pd.DataFrame({"low": [0.5], "high": [0.5]}, index=["val"])) == []


def test_label_table_has_every_split_and_level(df):
    table = label_table(df)
    assert list(table.index) == list(SPLITS)
    assert list(table.columns) == [s.value for s in LEVELS]
