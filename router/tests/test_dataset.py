"""Step 2.1: routing labels, ambiguous cases, splits, ward lookup."""

import pandas as pd
import pytest

from config import CORPUS_PATH
from contracts.corpus_gen import split_of
from contracts.schemas import Department, GeoPoint
from contracts.world import department_for, load_world
from contracts.world_gen import to_lonlat
from router.dataset import build_dataset, ward_check
from router.templates import SCENARIOS
from router.ward import ward_for, ward_from_text

pytestmark = pytest.mark.skipif(not CORPUS_PATH.exists(), reason="run contracts.corpus_gen")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return build_dataset()


def test_deterministic(df):
    pd.testing.assert_frame_equal(df, build_dataset())


def test_unique_ids_and_valid_departments(df):
    assert df.complaint_id.is_unique
    assert set(df.department) <= {d.value for d in Department}


def test_plain_rows_follow_department_table(df):
    plain = df[~df.ambiguous]
    assert (plain.issue_type.map(lambda t: department_for(t).value) == plain.department).all()


def test_no_spam(df):
    assert df.text.str.len().gt(0).all() and df.ward.notna().all()


def _generated(df):
    return df[df.scenario.isin(list(SCENARIOS))]


def test_ambiguous_label_not_a_function_of_issue_type(df):
    """Each (scenario, observed issue_type) pair must contain both departments, or the
    lookup baseline could route these rows perfectly and they would not be ambiguous."""
    amb = _generated(df)
    assert len(amb) > 0
    for (name, issue), g in amb.groupby(["scenario", "issue_type"]):
        assert set(g.department) == set(SCENARIOS[name]["cues"]), (name, issue)


def test_corpus_sewer_rows_go_to_water_supply(df):
    """User rule 2.1: sewage -> Water Supply, including the corpus's own sewer sentence."""
    sewer = df[df.scenario == "sewer_overflow"]
    assert len(sewer) > 0 and (sewer.department == "Water Supply").all()
    assert sewer.ambiguous.all() and (sewer.issue_type == "drain_blockage").all()
    assert sewer.text.str.lower().str.contains("sewer|सीवर|ਸੀਵਰ").all()
    others = df[(df.issue_type == "drain_blockage") & ~df.ambiguous]
    assert not others.text.str.lower().str.contains("sewer|सीवर|ਸੀਵਰ").any()


def test_ambiguous_label_decided_by_exactly_one_cue(df):
    for row in _generated(df).itertuples():
        cues = SCENARIOS[row.scenario]["cues"]
        hits = {d for d, by_lang in cues.items()
                if any(c in row.text for c in by_lang[row.language])}
        assert hits == {row.department}, row.complaint_id


def test_every_scenario_has_every_language():
    for name, sc in SCENARIOS.items():
        assert len(sc["cues"]) == 2 and len(sc["issue_types"]) == 2, name
        for block in [sc["shared"], *sc["cues"].values()]:
            assert set(block) == {"en", "hi", "pa", "mixed"}, name


def test_splits_reuse_corpus_rule(df):
    assert (df.split == df.incident_id.map(split_of)).all()
    assert df.groupby("incident_id").split.nunique().eq(1).all()
    assert df[df.ambiguous & (df.split == "test")].shape[0] >= 30  # enough to evaluate on


def test_ward_lookup_recovers_true_ward(df):
    assert (ward_check(df).accuracy == 1.0).all()


def test_ward_from_text_scripts_and_refusals():
    world = load_world()
    cp = next(w.ward_id for w in world.wards if w.name == "Connaught Place")
    for text in ["near the market in Connaught Place", "connaught place main road pe",
                 "कनॉट प्लेस में मेन रोड पर", "ਕਨਾਟ ਪਲੇਸ ਵਿੱਚ ਬਾਜ਼ਾਰ ਦੇ ਕੋਲ"]:
        assert ward_from_text(text) == cp, text
    assert ward_from_text("somewhere near my house") is None
    assert ward_from_text("between Karol Bagh and Paharganj") is None  # two wards: don't guess
    assert ward_from_text("") is None


def test_ward_for_prefers_gps_then_text_then_none():
    world = load_world()
    w0 = world.wards[0]
    c = w0.polygon.representative_point()
    lon, lat = to_lonlat(c.x, c.y)
    other = next(w.name for w in world.wards[1:])
    assert ward_for(GeoPoint(lat=lat, lon=lon), f"in {other}") == (w0.ward_id, "gps")
    assert ward_for(None, f"in {other}")[1] == "text"
    assert ward_for(GeoPoint(lat=0.0, lon=0.0), "") == (None, "none")  # outside the city
