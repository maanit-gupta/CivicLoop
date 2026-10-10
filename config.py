"""Project-wide settings. Import from here instead of hardcoding seeds or paths."""

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"
NOTEBOOKS_DIR = ROOT / "notebooks"
DECISION_LOG_PATH = DATA_DIR / "decision_log.jsonl"
WORLD_DIR = DATA_DIR / "world"
CORPUS_PATH = DATA_DIR / "corpus" / "complaints.parquet"

# The synthetic city (user decision, Step 0.4). Only the centre is real; wards, roads and POIs
# are generated around it by contracts.world_gen.
CITY = {
    "name": "Delhi",
    "center": (28.6139, 77.2090),  # (lat, lon), Connaught Place
    "languages": ("hi", "en", "pa", "mixed"),  # Hindi, English, Punjabi, Hinglish
}

MODULES = ("contracts", "triage", "router", "trainer", "intake", "vision", "verifier")

# Which model version each module currently serves. "stub" = rule-based placeholder.
MODEL_VERSIONS = {
    "triage": "v1",
    "router": "v1",
    "intake": "v1",
    "vision": "v1",
    "verifier": "stub",
}
