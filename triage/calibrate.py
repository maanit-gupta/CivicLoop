"""Confidence for triage (Step 1.5). Run: python -m triage.calibrate

The regressor outputs a continuous level (e.g. 1.93) that is rounded to a severity. The
`margin` is how far the raw value is from the nearest rounding boundary that would change the
level: 1.93 -> 0.43 (safe), 1.52 -> 0.02 (a coin flip). Margins are binned on the val split and
each bin's observed accuracy becomes the confidence, so "0.9" means "right about 9 times in 10".
Saved next to the model as calibration.json; test is used only to report how well it holds up.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODELS_DIR
from triage.evaluate import K, load_cases, to_index
from triage.train import VERSION, featurize, load, to_levels

EDGES = [0.0, 0.1, 0.2, 0.3, 0.4]  # margin bin lower edges; the last bin is open-ended


def margin(raw: np.ndarray) -> np.ndarray:
    """Distance from raw prediction to the nearest boundary that changes the rounded level.
    The lowest and highest levels have only one boundary each."""
    raw = np.asarray(raw, dtype=float)
    level = to_levels(raw)
    below = np.where(level > 0, raw - (level - 0.5), np.inf)
    above = np.where(level < K - 1, (level + 0.5) - raw, np.inf)
    return np.minimum(below, above)


def fit_calibration(raw: np.ndarray, correct: np.ndarray) -> dict:
    """Per-bin accuracy with +1/+2 smoothing (no bin claims exactly 0 or 1), made monotone so a
    bigger margin never means lower confidence."""
    bins = np.searchsorted(EDGES, margin(raw), side="right") - 1
    n = np.bincount(bins, minlength=len(EDGES))
    hits = np.bincount(bins, weights=correct.astype(float), minlength=len(EDGES))
    acc = np.maximum.accumulate((hits + 1) / (n + 2))
    return {"edges": EDGES, "accuracy": acc.round(4).tolist(), "n": n.tolist()}


def confidence(raw: np.ndarray, calibration: dict) -> np.ndarray:
    bins = np.searchsorted(calibration["edges"], margin(raw), side="right") - 1
    return np.asarray(calibration["accuracy"])[bins]


def save_calibration(calibration: dict, out_dir: Path) -> None:
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))


def load_calibration(out_dir: Path) -> dict:
    path = out_dir / "calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m triage.calibrate")
    return json.loads(path.read_text())


# ---- report -----------------------------------------------------------------------------------

def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((conf - correct) ** 2))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: weighted gap between stated confidence and accuracy."""
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
                     for b in np.unique(bins)))


def main() -> None:
    df = load_cases()
    out_dir = MODELS_DIR / "triage" / VERSION
    booster, meta = load(VERSION)

    def raw_and_correct(split: str):
        part = df[df.split == split]
        raw = booster.predict(featurize(part, meta["categories"]))
        return raw, to_levels(raw) == to_index(part.severity)

    val_raw, val_ok = raw_and_correct("val")
    cal = fit_calibration(val_raw, val_ok)
    save_calibration(cal, out_dir)

    test_raw, test_ok = raw_and_correct("test")
    candidates = {
        "constant (val accuracy)": np.full(len(test_raw), val_ok.mean()),
        "raw margin 0.5+m": np.minimum(1.0, 0.5 + margin(test_raw)),
        "binned margin (used)": confidence(test_raw, cal),
    }
    rows = [{"confidence": k, "brier": brier(c, test_ok), "ece": ece(c, test_ok)}
            for k, c in candidates.items()]
    print(f"Test (n={len(test_raw)}, accuracy {test_ok.mean():.3f}). Lower is better:")
    print(pd.DataFrame(rows).set_index("confidence").to_string(float_format=lambda v: f"{v:.4f}"))

    test_bins = np.searchsorted(EDGES, margin(test_raw), side="right") - 1
    table = pd.DataFrame({
        "margin from": EDGES, "val n": cal["n"], "confidence": cal["accuracy"],
        "test n": np.bincount(test_bins, minlength=len(EDGES)),
        "test acc": [test_ok[test_bins == b].mean() if (test_bins == b).any() else np.nan
                     for b in range(len(EDGES))]})
    print("\nReliability by margin bin:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {out_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
