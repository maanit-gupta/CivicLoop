"""Step 5.5: vision.api.analyze = mask -> store -> models on the masked copy -> DecisionLog."""

import numpy as np
import pytest
from PIL import Image

from contracts.api import ImageFeatures, IssueType, VisionAPI, read_log, verify_chain
from vision import api, features
from vision import privacy as pv


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "upload.jpg"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def models(monkeypatch):
    """Fake detectors (1 face, 2 plates) and a fake YOLO+CLIP that records what it was shown."""
    seen = []

    def analyze(paths):
        seen.extend(paths)
        return [ImageFeatures(image_path=str(p), model_version=features.MODEL_VERSION,
                              confidence=0.8, explanation="The detector found 1 pothole.",
                              issue_type=IssueType.POTHOLE, embedding=[1.0, 0.0])
                for p in paths]

    face = pv.Found("face", 0.9, (5, 5, 40, 40))
    plates = [pv.Found("plate", 0.6, (100, 60, 140, 70)),
              pv.Found("plate", 0.5, (150, 60, 190, 70))]
    monkeypatch.setattr(pv, "detect_faces", lambda img: [face])
    monkeypatch.setattr(pv, "detect_plates", lambda img: plates)
    monkeypatch.setattr(features, "analyze", analyze)
    return seen


def test_module_conforms_to_the_contract():
    assert isinstance(api, VisionAPI)


def test_models_only_ever_see_the_masked_stored_copy(photo, models, tmp_path):
    f = api.analyze(str(photo), store_dir=tmp_path / "store", log_path=tmp_path / "log.jsonl")
    assert models == [tmp_path / "store" / models[0].name]
    assert f.image_path == str(models[0]) and f.image_path != str(photo)
    assert f.privacy_masked


def test_output_names_the_masker_and_says_what_was_blurred(photo, models, tmp_path):
    f = api.analyze(str(photo), store_dir=tmp_path / "store", log_path=tmp_path / "log.jsonl")
    assert f.model_version == f"{features.MODEL_VERSION}+{pv.MODEL_VERSION}"
    assert f.explanation.endswith("Privacy: blurred 1 face and 2 number plates before analysis.")
    assert f.issue_type == IssueType.POTHOLE and f.confidence == 0.8


def test_every_call_is_logged_on_the_chain(photo, models, tmp_path):
    log = tmp_path / "log.jsonl"
    api.analyze(str(photo), case_id="case-1", store_dir=tmp_path / "store", log_path=log)
    api.analyze(str(photo), case_id="case-2", store_dir=tmp_path / "store", log_path=log)
    entries = read_log(log)
    assert [(e.module, e.case_id) for e in entries] == [("vision", "case-1"), ("vision", "case-2")]
    assert entries[0].input == {"image_path": str(photo), "faces_blurred": 1, "plates_blurred": 2}
    assert entries[0].output["privacy_masked"] is True
    assert entries[0].model_version == api.MODEL_VERSION
    assert verify_chain(log) is None


def test_same_upload_is_stored_once(photo, models, tmp_path):
    a = api.analyze(str(photo), store_dir=tmp_path / "store", log_path=tmp_path / "log.jsonl")
    b = api.analyze(str(photo), store_dir=tmp_path / "store", log_path=tmp_path / "log.jsonl")
    assert a.image_path == b.image_path and len(list((tmp_path / "store").iterdir())) == 1


@pytest.mark.parametrize("faces, plates, note", [
    (0, 0, "Privacy: no face or number plate found to blur."),
    (2, 0, "Privacy: blurred 2 faces before analysis."),
    (0, 1, "Privacy: blurred 1 number plate before analysis."),
])
def test_privacy_note(faces, plates, note):
    assert api._privacy_note(faces, plates) == note


def test_failed_masking_stores_analyses_and_logs_nothing(photo, models, tmp_path, monkeypatch):
    def broken(img):
        raise FileNotFoundError("plate model missing")

    monkeypatch.setattr(pv, "detect_plates", broken)
    with pytest.raises(FileNotFoundError):
        api.analyze(str(photo), store_dir=tmp_path / "store", log_path=tmp_path / "log.jsonl")
    assert models == [] and not (tmp_path / "store").exists()
    assert not (tmp_path / "log.jsonl").exists()
