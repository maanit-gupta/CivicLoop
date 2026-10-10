"""Demo: python -m vision.demo

Six photos through vision.api.analyze: one per civic label from the test split, a dashcam frame
from an Indian road (number plates) and a street-market photo (faces, WIDER FACE). The masked
copies are kept in data/vision/demo/ so you can open them and check the blur yourself. The
decision log goes to a temporary file, not the real one.
"""

import tempfile
from pathlib import Path

from config import DATA_DIR
from contracts.api import read_log, verify_chain
from vision import api
from vision.evaluate import PRIVACY_DIR, load_split

DEMO_DIR = DATA_DIR / "vision" / "demo"
EXTRA = [PRIVACY_DIR / "plates_india" / "vid-2" / "frame-1510.jpg",
         PRIVACY_DIR / "wider_face" / "WIDER_val" / "images" / "23--Shoppers"
         / "23_Shoppers_Shoppers_23_22.jpg"]


def samples() -> list[tuple[str, Path]]:
    """(true label, photo): the first test photo of four labels, then the two privacy photos."""
    rows = load_split("test")
    picks = [next(r for r in rows if r["label"] == lb)
             for lb in ("pothole", "garbage", "damaged_sign", "none")]
    return [(r["label"], r["path"]) for r in picks] + [("(plates)", EXTRA[0]),
                                                       ("(faces)", EXTRA[1])]


def main() -> None:
    try:
        photos = samples()
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nBuild the dataset first: python -m vision.dataset") from None
    missing = [p for _, p in photos if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. See data/vision/SOURCES.md (5.5) to download.")

    log = Path(tempfile.mkdtemp()) / "decision_log.jsonl"
    print(f"{'true label':<13}{'vision says':<14}{'conf':>5}  {'stored as (masked)':<22}"
          f"{'faces':>6}{'plates':>7}")
    results = []
    for i, (label, path) in enumerate(photos):
        f = api.analyze(str(path), case_id=f"demo-{i}", store_dir=DEMO_DIR, log_path=log)
        entry = read_log(log)[-1]
        results.append(f)
        issue = f.issue_type.value if f.issue_type else "none"
        print(f"{label:<13}{issue:<14}{f.confidence:>5.2f}  {Path(f.image_path).name:<22}"
              f"{entry.input['faces_blurred']:>6}{entry.input['plates_blurred']:>7}")

    print(f"\nexplanation ({photos[4][0]}): {results[4].explanation}")
    print(f"model_version: {results[0].model_version}")
    print(f"log: {len(read_log(log))} entries, chain "
          f"{'intact' if verify_chain(log) is None else 'BROKEN'} ({log})")
    print(f"masked copies: {DEMO_DIR}")


if __name__ == "__main__":
    main()
