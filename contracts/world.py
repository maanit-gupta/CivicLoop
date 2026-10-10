"""Geo and department lookups over the synthetic city in data/world/.

Pure Python (shapely) for now; a PostGIS version later keeps these same signatures.
All geometry is held in local metres (see world_gen.to_metres), so distances are plain
Euclidean metres instead of degrees.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
import yaml
from shapely.geometry import LineString, Point, Polygon, shape

from config import CITY, WORLD_DIR
from contracts.schemas import Department, GeoPoint, IssueType, RoadType, Severity
from contracts.world_gen import to_metres

ROAD_RANK = {"main": 0, "residential": 1, "lane": 2}  # at an intersection the bigger road wins


@dataclass(frozen=True)
class Ward:
    ward_id: str
    name: str
    polygon: Polygon


@dataclass(frozen=True)
class Poi:
    poi_id: str
    kind: str
    name: str
    ward_id: str
    point: Point


@dataclass(frozen=True)
class World:
    wards: list[Ward]
    pois: list[Poi]
    roads: list[tuple[RoadType, LineString]]
    issue_to_department: dict[IssueType, Department]
    sla_hours: dict[Department, dict[Severity, int]]


def _features(path: Path) -> list[tuple[dict, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(f["properties"],
             shapely.transform(shape(f["geometry"]), lambda a: np.column_stack(to_metres(*a.T))))
            for f in data["features"]]


@lru_cache(maxsize=4)
def load_world(world_dir: Path = WORLD_DIR) -> World:
    world_dir = Path(world_dir)
    if not (world_dir / "wards.geojson").exists():
        raise FileNotFoundError(
            f"No synthetic city in {world_dir}. Run: python -m contracts.world_gen")
    wards = [Ward(p["ward_id"], p["name"], g) for p, g in _features(world_dir / "wards.geojson")]
    pois = [Poi(p["poi_id"], p["kind"], p["name"], p["ward_id"], g)
            for p, g in _features(world_dir / "pois.geojson")]
    roads = [(p["road_type"], g) for p, g in _features(world_dir / "roads.geojson")]

    spec = yaml.safe_load((world_dir / "departments.yaml").read_text(encoding="utf-8"))
    issue_to_dept, sla = {}, {}
    for name, d in spec["departments"].items():
        dept = Department(name)  # ValueError on a typo'd name: fail loudly, not silently
        for issue in d["issue_types"]:
            if IssueType(issue) in issue_to_dept:
                raise ValueError(f"Issue type {issue!r} is assigned to two departments")
            issue_to_dept[IssueType(issue)] = dept
        sla[dept] = {Severity(s): int(h) for s, h in d["sla_hours"].items()}
    return World(wards, pois, roads, issue_to_dept, sla)


def _xy(point: GeoPoint) -> Point:
    return Point(to_metres(point.lon, point.lat))


def ward_of(point: GeoPoint, world: World | None = None) -> str | None:
    """Ward id containing the point, or None if it is outside the city."""
    p = _xy(point)
    for ward in (world or load_world()).wards:
        if ward.polygon.covers(p):  # covers: a point on a shared border goes to the first ward
            return ward.ward_id
    return None


def distance_to_nearest(point: GeoPoint, kind: str, world: World | None = None) -> float:
    """Metres to the nearest POI of `kind` (school, hospital, market, bus_stop)."""
    p = _xy(point)
    dists = [p.distance(poi.point) for poi in (world or load_world()).pois if poi.kind == kind]
    if not dists:
        raise ValueError(f"No POIs of kind {kind!r}")
    return min(dists)


def road_type_at(point: GeoPoint, world: World | None = None) -> RoadType:
    """Type of the nearest road. Every point in the city is within 125 m of a road."""
    p = _xy(point)
    # Round to 10 cm so lat/lon round-trip noise can't break a tie the rank should decide.
    return min((round(p.distance(line), 1), ROAD_RANK[kind], kind)
               for kind, line in (world or load_world()).roads)[2]


def department_for(issue_type: IssueType, world: World | None = None) -> Department:
    return (world or load_world()).issue_to_department[IssueType(issue_type)]


def sla_hours(department: Department, severity: Severity, world: World | None = None) -> int:
    return (world or load_world()).sla_hours[Department(department)][Severity(severity)]


def summary(world: World | None = None) -> str:
    """Small text table describing the city."""
    w = world or load_world()
    lines = [f"Synthetic {CITY['name']}: {len(w.wards)} wards, centre {CITY['center']}",
             f"{'ward':4} {'name':16} {'km2':>5} " + " ".join(
                 f"{k:>8}" for k in ("school", "hospital", "market", "bus_stop"))]
    for ward in w.wards:
        counts = [sum(p.ward_id == ward.ward_id and p.kind == k for p in w.pois)
                  for k in ("school", "hospital", "market", "bus_stop")]
        lines.append(f"{ward.ward_id:4} {ward.name:16} {ward.polygon.area / 1e6:5.2f} "
                     + " ".join(f"{c:8d}" for c in counts))
    km = {k: sum(line.length for t, line in w.roads if t == k) / 1000 for k in ROAD_RANK}
    lines.append("road km: " + ", ".join(f"{k} {v:.0f}" for k, v in km.items()))
    return "\n".join(lines)
