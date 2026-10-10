"""Generate the synthetic city into data/world/: `python -m contracts.world_gen`.

Deterministic from config.SEED. Geometry is built in local metres (x east, y north of
config.CITY's centre) and saved as GeoJSON in lon/lat, the format PostGIS will load later.
departments.yaml is hand-written, not generated.
"""

import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, mapping

from config import CITY, SEED, WORLD_DIR

COLS, ROWS = 4, 3          # 12 wards, numbered row by row from the north-west
CELL_M = 2000              # each ward is roughly 2 km x 2 km
JITTER_M = 300             # interior ward corners move up to this much, so wards aren't a grid
ROAD_SPACING_M = 250       # a road every 250 m; every 500 m residential, every 2 km main
HALF_W, HALF_H = COLS * CELL_M / 2, ROWS * CELL_M / 2

# Real central-Delhi locality names, laid out roughly in their true directions around
# Connaught Place. The boundaries are synthetic.
WARD_NAMES = [
    "Karol Bagh", "Paharganj", "Chandni Chowk", "Daryaganj",
    "Rajinder Nagar", "Gol Market", "Connaught Place", "Mandi House",
    "Chanakyapuri", "Race Course", "India Gate", "Lodhi Colony",
]
POI_COUNTS_PER_WARD = {"school": (2, 4), "hospital": (0, 2), "market": (1, 2), "bus_stop": (2, 3)}
POI_LABELS = {"school": "Govt School", "hospital": "Hospital", "market": "Market",
              "bus_stop": "Bus Stop"}

M_PER_DEG_LAT = 111_320.0


def to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Local metres -> (lon, lat). Equirectangular around the centre: <0.2% error in the city."""
    lat0, lon0 = CITY["center"]
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return lon0 + x / m_per_deg_lon, lat0 + y / M_PER_DEG_LAT


def to_metres(lon: float, lat: float) -> tuple[float, float]:
    """(lon, lat) -> local metres. Inverse of to_lonlat."""
    lat0, lon0 = CITY["center"]
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return (lon - lon0) * m_per_deg_lon, (lat - lat0) * M_PER_DEG_LAT


def make_wards(rng: np.random.Generator) -> list[dict]:
    # Shared corner grid: neighbouring wards use the same jittered corners, so they tile
    # the city with no gaps or overlaps.
    xs = np.linspace(-HALF_W, HALF_W, COLS + 1)
    ys = np.linspace(HALF_H, -HALF_H, ROWS + 1)  # north to south
    corners = np.array([[(x, y) for x in xs] for y in ys], dtype=float)
    corners[1:-1, 1:-1] += rng.uniform(-JITTER_M, JITTER_M, size=(ROWS - 1, COLS - 1, 2))
    wards = []
    for r in range(ROWS):
        for c in range(COLS):
            ring = [corners[r, c], corners[r, c + 1], corners[r + 1, c + 1], corners[r + 1, c]]
            idx = r * COLS + c
            wards.append({"ward_id": f"W{idx + 1:02d}", "name": WARD_NAMES[idx],
                          "geometry": Polygon(ring)})
    return wards


def make_roads() -> list[dict]:
    def road_type(offset: float) -> str:
        if offset % 2000 == 0:
            return "main"
        if offset % 500 == 0:
            return "residential"
        return "lane"

    roads = []
    for x in np.arange(-HALF_W, HALF_W + 1, ROAD_SPACING_M):
        roads.append({"road_type": road_type(x),
                      "geometry": LineString([(x, -HALF_H), (x, HALF_H)])})
    for y in np.arange(-HALF_H, HALF_H + 1, ROAD_SPACING_M):
        roads.append({"road_type": road_type(y),
                      "geometry": LineString([(-HALF_W, y), (HALF_W, y)])})
    return roads


def make_pois(wards: list[dict], rng: np.random.Generator) -> list[dict]:
    pois, counter = [], {}
    for ward in wards:
        poly = ward["geometry"]
        minx, miny, maxx, maxy = poly.bounds
        for kind, (lo, hi) in POI_COUNTS_PER_WARD.items():
            for _ in range(rng.integers(lo, hi + 1)):
                while True:  # rejection sampling: uniform inside the (non-rectangular) ward
                    p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                    if poly.contains(p):
                        break
                counter[kind] = counter.get(kind, 0) + 1
                pois.append({"poi_id": f"{kind}-{counter[kind]:02d}", "kind": kind,
                             "name": f"{ward['name']} {POI_LABELS[kind]} {counter[kind]}",
                             "ward_id": ward["ward_id"], "geometry": p})
    return pois


def _write_geojson(path: Path, features: list[dict]) -> None:
    out = []
    for f in features:
        props = {k: v for k, v in f.items() if k != "geometry"}
        geom = shapely.transform(f["geometry"], lambda a: np.column_stack(to_lonlat(*a.T)))
        out.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": out}, indent=1),
                    encoding="utf-8")


def build_world(out_dir: Path = WORLD_DIR, seed: int = SEED) -> None:
    """Write wards.geojson, roads.geojson and pois.geojson into out_dir."""
    rng = np.random.default_rng(seed)
    wards = make_wards(rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_geojson(out_dir / "wards.geojson", wards)
    _write_geojson(out_dir / "roads.geojson", make_roads())
    _write_geojson(out_dir / "pois.geojson", make_pois(wards, rng))


if __name__ == "__main__":
    build_world()
    print(f"Wrote synthetic {CITY['name']} to {WORLD_DIR}")
    from contracts.world import summary
    print(summary())
