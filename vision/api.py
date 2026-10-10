"""Public API of the vision module. Only this file may be imported by other modules.

analyze(image_path) is the one entry point, in this order:
1. mask faces and number plates and store the masked copy (vision.privacy), EXIF-rotated and
   without metadata. Nothing is stored or analysed if masking fails.
2. run YOLO + CLIP on that masked copy (vision.features).
3. write a DecisionLog entry.

Everything downstream (verifier, officials, Closure) sees only the stored, masked photo:
ImageFeatures.image_path points to it. The caller deletes its own upload once this returns.
"""

from pathlib import Path

from config import DECISION_LOG_PATH
from contracts.api import ImageFeatures, log_decision
from vision import features, privacy

# Masking changes the pixels the models see, so the masker is part of the version: embeddings
# are only comparable within one MODEL_VERSION.
MODEL_VERSION = f"{features.MODEL_VERSION}+{privacy.MODEL_VERSION}"


def _privacy_note(n_faces: int, n_plates: int) -> str:
    parts = [f"{n} {word}{'' if n == 1 else 's'}"
             for n, word in ((n_faces, "face"), (n_plates, "number plate")) if n]
    if not parts:
        return "Privacy: no face or number plate found to blur."
    return f"Privacy: blurred {' and '.join(parts)} before analysis."


def analyze(image_path: str, case_id: str = "unassigned", store_dir: Path = privacy.STORE_DIR,
            log_path: Path = DECISION_LOG_PATH) -> ImageFeatures:
    """One citizen photo -> ImageFeatures of its masked, stored copy."""
    stored, found = privacy.mask_file(image_path, store_dir)
    n_faces = sum(f.kind == "face" for f in found)
    n_plates = len(found) - n_faces
    f = features.analyze([stored])[0]
    f = f.model_copy(update={
        "model_version": MODEL_VERSION, "privacy_masked": True,
        "explanation": f"{f.explanation} {_privacy_note(n_faces, n_plates)}"})
    log_input = {"image_path": str(image_path), "faces_blurred": n_faces,
                 "plates_blurred": n_plates}
    log_decision("vision", case_id, log_input, f, path=log_path)
    return f
