"""Photo metadata for intake (Step 4.4): the EXIF caption and GPS position, if present.

Only metadata. What a photo *shows* is the vision module's job (Phase 5), and intake may not
import it. EXIF is optional and often stripped (WhatsApp, screenshots), so a photo without it,
or a file that can't be opened as an image, gives an empty PhotoMeta.
"""

from dataclasses import dataclass

from pydantic import ValidationError

from contracts.api import GeoPoint

IMAGE_DESCRIPTION = 0x010E  # EXIF tag: the caption
GPS_IFD = 0x8825  # EXIF sub-directory with the GPS tags below
LAT_REF, LAT, LON_REF, LON = 1, 2, 3, 4  # 'N'/'S', (deg, min, sec), 'E'/'W', (deg, min, sec)


@dataclass(frozen=True)
class PhotoMeta:
    caption: str = ""
    gps: GeoPoint | None = None


def _degrees(dms, ref: str) -> float:
    degrees, minutes, seconds = (float(x) for x in dms)
    value = degrees + minutes / 60 + seconds / 3600
    return -value if ref in ("S", "W") else value


def read(path: str) -> PhotoMeta:
    from PIL import Image  # optional dependency: pip install -e ".[intake]"

    try:
        with Image.open(path) as image:
            exif = image.getexif()
            gps = exif.get_ifd(GPS_IFD)
    except OSError:  # missing file, or not an image
        return PhotoMeta()
    try:
        point = GeoPoint(lat=_degrees(gps[LAT], gps.get(LAT_REF, "N")),
                         lon=_degrees(gps[LON], gps.get(LON_REF, "E")))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, ValidationError):
        point = None  # no GPS tags, or broken ones
    return PhotoMeta(str(exif.get(IMAGE_DESCRIPTION, "")).strip(), point)
