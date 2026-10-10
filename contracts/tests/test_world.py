import math
import shutil

import numpy as np
import pytest

from config import CITY, WORLD_DIR
from contracts import stubs
from contracts.api import load_fixture_cases
from contracts.schemas import Department, GeoPoint, IssueType, Severity
from contracts.world import (
    department_for,
    distance_to_nearest,
    load_world,
    road_type_at,
    sla_hours,
    ward_of,
)
from contracts.world_gen import HALF_H, HALF_W, build_world, to_lonlat, to_metres


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A freshly generated world in a temp dir, so tests never depend on data/ state."""
    d = tmp_path_factory.mktemp("world")
    build_world(d)
    shutil.copy(WORLD_DIR / "departments.yaml", d)
    return load_world(d)


def gp(x: float, y: float) -> GeoPoint:
    lon, lat = to_lonlat(x, y)
    return GeoPoint(lat=lat, lon=lon)


def haversine_m(a: GeoPoint, b: GeoPoint) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def test_city_is_delhi():
    assert CITY["name"] == "Delhi"
    lat, lon = CITY["center"]
    assert 28.4 < lat < 28.9 and 76.8 < lon < 77.4


def test_generation_is_deterministic(tmp_path, world):
    build_world(tmp_path)
    again = (tmp_path / "pois.geojson").read_text()
    build_world(tmp_path / "b")
    assert (tmp_path / "b" / "pois.geojson").read_text() == again


def test_projection_round_trips():
    lon, lat = to_lonlat(1234.5, -987.6)
    x, y = to_metres(lon, lat)
    assert x == pytest.approx(1234.5) and y == pytest.approx(-987.6)


def test_wards_tile_the_city(world):
    assert len(world.wards) == 12
    assert len({w.ward_id for w in world.wards}) == 12
    total = sum(w.polygon.area for w in world.wards)
    assert total == pytest.approx(4 * HALF_W * HALF_H)  # no gaps
    for i, a in enumerate(world.wards):
        assert a.polygon.is_valid
        for b in world.wards[i + 1:]:
            assert a.polygon.intersection(b.polygon).area < 1e-6  # no overlaps


def test_every_point_inside_has_exactly_one_ward(world):
    rng = np.random.default_rng(0)
    for x, y in rng.uniform([-HALF_W, -HALF_H], [HALF_W, HALF_H], size=(500, 2)):
        assert ward_of(gp(x, y), world) is not None


def test_ward_of_corners_and_outside(world):
    assert ward_of(gp(-HALF_W + 10, HALF_H - 10), world) == "W01"   # north-west
    assert ward_of(gp(HALF_W - 10, -HALF_H + 10), world) == "W12"   # south-east
    assert ward_of(gp(HALF_W + 500, 0), world) is None
    assert ward_of(GeoPoint(lat=12.69, lon=79.97), world) is None   # Chengalpattu: not here


def test_wards_are_not_a_plain_grid(world):
    """Jittered corners: naive 'divide by 2 km' must get some points wrong."""
    rng = np.random.default_rng(1)
    wrong = 0
    for x, y in rng.uniform([-HALF_W, -HALF_H], [HALF_W, HALF_H], size=(2000, 2)):
        col, row = int((x + HALF_W) // 2000), int((HALF_H - y) // 2000)
        wrong += ward_of(gp(x, y), world) != f"W{row * 4 + col + 1:02d}"
    assert wrong > 0


def test_distance_to_nearest_matches_haversine(world):
    school = next(p for p in world.pois if p.kind == "school")
    here = gp(school.point.x + 300, school.point.y + 400)  # 500 m away in local metres
    d = distance_to_nearest(here, "school", world)
    assert d <= 500 + 1e-6
    there = gp(school.point.x, school.point.y)
    assert haversine_m(here, there) == pytest.approx(500, rel=0.005)


def test_distance_zero_on_poi_and_unknown_kind_fails(world):
    h = next(p for p in world.pois if p.kind == "hospital")
    assert distance_to_nearest(gp(h.point.x, h.point.y), "hospital", world) < 0.01
    with pytest.raises(ValueError):
        distance_to_nearest(gp(0, 0), "airport", world)


def test_every_ward_has_a_school_and_a_bus_stop(world):
    for ward in world.wards:
        kinds = {p.kind for p in world.pois if p.ward_id == ward.ward_id}
        assert {"school", "bus_stop", "market"} <= kinds
        assert all(ward.polygon.contains(p.point) for p in world.pois
                   if p.ward_id == ward.ward_id)


@pytest.mark.parametrize("x, y, expected", [
    (0, 0, "main"),             # two main roads cross at the centre
    (0, 100, "main"),           # on the north-south main road
    (500, 120, "residential"),  # on x=500
    (250, 110, "lane"),         # on x=250
    (2000, 1500, "main"),       # main crosses residential: bigger road wins
    (260, 490, "residential"),  # 10 m from lane x=250, 10 m from residential y=500: tie
])
def test_road_type_at(world, x, y, expected):
    assert road_type_at(gp(x, y), world) == expected


def test_departments_cover_every_issue_once_and_match_stub(world):
    assert set(world.issue_to_department) == set(IssueType)
    assert set(world.issue_to_department.values()) == set(Department)
    assert world.issue_to_department == stubs.ISSUE_TO_DEPARTMENT
    assert department_for(IssueType.WATER_LEAK, world) == Department.WATER_SUPPLY


def test_sla_hours_complete_and_shrink_with_severity(world):
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    for dept in Department:
        hours = [sla_hours(dept, s, world) for s in order]
        assert all(h > 0 for h in hours)
        assert hours == sorted(hours, reverse=True), dept
    assert sla_hours(Department.WATER_SUPPLY, Severity.CRITICAL, world) == 2


def test_fixtures_live_in_their_ward(world):
    for case in load_fixture_cases():
        assert ward_of(case.report.location, world) == case.ward, case.case_id


def test_missing_world_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="world_gen"):
        load_world(tmp_path)
