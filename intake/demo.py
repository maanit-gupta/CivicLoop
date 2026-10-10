"""Three Delhi reports through the intake API: `python -m intake.demo` (needs `ollama serve`).

1. Hindi typed text with the app's GPS.
2. Punjabi typed text with a photo whose EXIF carries the GPS (no app GPS).
3. A photo only: an English EXIF caption and EXIF GPS.
Then a voice-only report, which fails clearly until the deferred speech step exists.
The photos are generated here; the decision log goes to a temp file.
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from contracts.api import GeoPoint, RawReport, read_log, verify_chain
from intake import api, photo


def _photo(path: Path, lat_dms: tuple, lon_dms: tuple, caption: str = "") -> str:
    exif = Image.Exif()
    exif[photo.GPS_IFD] = {1: "N", 2: lat_dms, 3: "E", 4: lon_dms}
    if caption:
        exif[photo.IMAGE_DESCRIPTION] = caption
    Image.new("RGB", (64, 48), "gray").save(path, exif=exif)
    return str(path)


def samples(folder: Path) -> list[RawReport]:
    at = datetime(2026, 10, 10, 9, 30, tzinfo=UTC)
    return [
        RawReport(report_id="demo-hi", submitted_at=at, gps=GeoPoint(lat=28.6519, lon=77.1909),
                  text="करोल बाग में अजमल खान रोड पर नाला जाम है, बारिश के बाद पूरी सड़क पर "
                       "घुटनों तक पानी भर जाता है।"),
        RawReport(report_id="demo-pa", submitted_at=at, channel="whatsapp",
                  text="ਗੁਰਦੁਆਰਾ ਬੰਗਲਾ ਸਾਹਿਬ ਦੇ ਸਾਹਮਣੇ ਸੜਕ ਵਿੱਚ ਵੱਡਾ ਟੋਆ ਹੈ, ਕੱਲ੍ਹ ਇੱਕ ਸਕੂਟਰ "
                       "ਵਾਲਾ ਡਿੱਗ ਗਿਆ।",
                  image_paths=[_photo(folder / "pa.jpg", (28.0, 37.0, 35.0),
                                      (77.0, 12.0, 33.0))]),
        RawReport(report_id="demo-photo", submitted_at=at,
                  image_paths=[_photo(folder / "cap.jpg", (28.0, 38.0, 4.0), (77.0, 13.0, 9.0),
                                      caption="Streetlights not working on Barakhamba Road "
                                              "near the metro gate")]),
    ]


def main() -> None:
    folder = Path(tempfile.mkdtemp())
    log = folder / "decision_log.jsonl"
    reports = []
    try:
        for raw in samples(folder):
            reports.append(api.process(raw, log_path=log))
    except ConnectionError as e:
        raise SystemExit(f"{e}\nStart the local LLM first: ollama serve") from None

    print(f"{'report':11} {'lang':5} {'issue':15} {'conf':>4}  {'version':12} location_text")
    for r in reports:
        print(f"{r.report_id:11} {r.language:5} {r.issue_type.value:15} {r.confidence:4.2f}  "
              f"{r.model_version:12} {r.location_text}")
    for r in reports:
        gps = f"({r.location.lat:.4f}, {r.location.lon:.4f})" if r.location else "none"
        print(f"\n{r.report_id}  cues={r.cues.model_dump()}  gps={gps}\n"
              f"  english: {r.english_text}\n  why: {r.explanation}")

    voice = RawReport(report_id="demo-voice", submitted_at=datetime.now(UTC),
                      audio_path="note.m4a")
    try:
        api.process(voice, log_path=log)
    except NotImplementedError as e:
        print(f"\ndemo-voice  NotImplementedError: {e}")
    print(f"\nDecision log: {len(read_log(log))} entries at {log}; "
          f"verify_chain first broken = {verify_chain(log)}")


if __name__ == "__main__":
    main()
