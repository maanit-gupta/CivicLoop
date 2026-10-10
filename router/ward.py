"""Ward lookup: a deterministic tool, not a model.

Which ward contains a GPS point is a fact about polygons; a model could only get it wrong.
Order: GPS point-in-polygon first; if there is no GPS, look for a ward name in the location
text (English, Hindi or Punjabi spelling); if neither works, return None and let a human decide.
"""

from functools import lru_cache

from contracts import corpus_templates as T
from contracts.schemas import GeoPoint
from contracts.world import load_world, ward_of


@lru_cache(maxsize=1)
def _ward_names() -> dict[str, list[str]]:
    """ward_id -> every spelling citizens use for it (lower-cased)."""
    names = {}
    for ward in load_world().wards:
        spellings = [ward.name] + [T.WARD_NAMES[lang][ward.name] for lang in ("hi", "pa")]
        names[ward.ward_id] = [s.lower() for s in spellings]
    return names


def ward_from_text(text: str) -> str | None:
    """The ward named in `text`, or None if no ward or more than one ward is named."""
    text = (text or "").lower()
    hits = [wid for wid, spellings in _ward_names().items() if any(s in text for s in spellings)]
    return hits[0] if len(hits) == 1 else None


def ward_for(location: GeoPoint | None, location_text: str = "") -> tuple[str | None, str]:
    """(ward_id or None, method) where method is 'gps', 'text' or 'none'."""
    if location is not None:
        ward = ward_of(location)
        if ward is not None:
            return ward, "gps"
    ward = ward_from_text(location_text)
    return (ward, "text") if ward else (None, "none")
