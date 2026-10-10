"""Step 5.3: YOLO boxes -> image label, the val-picked threshold, and the notebook's pins."""

import csv
import hashlib
import json
import sys
import types
from collections import Counter
from importlib.metadata import PackageNotFoundError, version

import pytest

import config
from vision import evaluate as ev
from vision import yolo_detect as yd
from vision.dataset import CLASSES, OUT_DIR

POTHOLE, GARBAGE, SIGN, POLE = range(4)
NOTEBOOK = config.NOTEBOOKS_DIR / "yolo_finetune.ipynb"


def box(cls, conf):
    return (cls, conf, 0.5, 0.5, 0.2, 0.2)


# ---------- boxes -> image label ----------

def test_image_label_takes_the_most_confident_box_not_the_first():
    assert yd.image_label([box(GARBAGE, 0.3), box(POTHOLE, 0.8)], 0.25) == "pothole"


def test_image_label_maps_damaged_pole_to_streetlight():
    assert yd.image_label([box(POLE, 0.9)], 0.5) == "streetlight"  # 5.1 decision


@pytest.mark.parametrize("dets", [[], [box(SIGN, 0.2)]])
def test_image_label_is_none_without_a_confident_box(dets):
    assert yd.image_label(dets, 0.25) == "none"


def test_image_label_threshold_is_inclusive():
    assert yd.image_label([box(SIGN, 0.25)], 0.25) == "damaged_sign"


def test_to_detections():
    out = yd.to_detections([2.0, 0.0], [0.9, 0.1], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 1.0, 1.0]])
    assert out == [(2, 0.9, 0.1, 0.2, 0.3, 0.4), (0, 0.1, 0.5, 0.5, 1.0, 1.0)]
    assert isinstance(out[0][0], int)


# ---------- threshold picked on val ----------

def test_pick_threshold_finds_the_gap_between_true_and_spurious_boxes():
    # Real issues get boxes >= 0.6; the "none" photos get stray boxes up to 0.4. Any threshold
    # in (0.4, 0.6] is perfect; ties go to the lowest, 0.45.
    dets = [[box(POTHOLE, 0.7)], [box(GARBAGE, 0.6)], [box(POTHOLE, 0.4)], [box(GARBAGE, 0.1)]]
    y = ["pothole", "garbage", "none", "none"]
    t, f1 = ev.pick_threshold(dets, y)
    assert t == 0.45
    assert f1 == pytest.approx(3 / 6)  # 3 of 6 labels perfect, 3 absent from y (F1 = 0)


@pytest.mark.skipif(not (ev.EVAL_DIR / "yolo_v1_val.json").exists(), reason="5.3 eval not run")
def test_label_threshold_is_the_val_pick():
    saved = json.loads((ev.EVAL_DIR / "yolo_v1_val.json").read_text())
    assert yd.LABEL_THRESHOLD == saved["label_threshold"]


def test_threshold_grid():
    assert ev.THRESHOLD_GRID[0] == 0.05 and ev.THRESHOLD_GRID[-1] == 0.95
    assert len(ev.THRESHOLD_GRID) == 19


def test_recall_by_source():
    rows = [{"source": "a", "label": "none"}, {"source": "a", "label": "none"},
            {"source": "b", "label": "pothole"}]
    assert ev.recall_by_source(rows, ["none", "pothole", "pothole"]) == {
        "a/none": 0.5, "b/pothole": 1.0}


# ---------- loading weights fails clearly ----------

def test_missing_weights_say_how_to_get_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="yolo_finetune.ipynb"):
        yd.detect([], weights=tmp_path / "best.pt")


def test_weights_with_other_classes_are_refused(tmp_path, monkeypatch):
    weights = tmp_path / "coco.pt"
    weights.write_bytes(b"")
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda path: types.SimpleNamespace(names={0: "person", 1: "bicycle"})
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    with pytest.raises(ValueError, match="expected"):
        yd.detect([], weights=weights)


# ---------- the notebook trains on exactly this dataset ----------

def _notebook_source() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_pins_match_the_repo():
    src = _notebook_source()
    assert f"SEED = {config.SEED}" in src
    assert 'BASE_MODEL = "yolo26s.pt"' in src and yd.MODEL_VERSION.endswith("yolo26s")
    assert f'imgsz={yd.IMGSZ}' in src
    try:
        assert f'ultralytics=={version("ultralytics")}"' in src  # else best.pt may not load
    except PackageNotFoundError:
        pytest.skip("ultralytics not installed")


def test_kaggle_metadata_points_at_the_notebook():
    folder = config.NOTEBOOKS_DIR / "yolo"
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert (folder / meta["code_file"]).resolve() == NOTEBOOK.resolve()
    assert meta["is_private"] == "true" and meta["enable_gpu"] == "true"


@pytest.mark.skipif(not (OUT_DIR / "manifest.csv").exists(), reason="5.1 dataset not built")
def test_notebook_expects_the_local_build():
    src = _notebook_source()
    digest = hashlib.sha256((OUT_DIR / "manifest.csv").read_bytes()).hexdigest()
    assert f'MANIFEST_SHA256 = "{digest}"' in src
    counts = Counter(r["split"] for r in csv.DictReader(open(OUT_DIR / "manifest.csv")))
    expected = ", ".join(f'"{s}": {counts[s]}' for s in ("train", "val", "test"))
    assert f"EXPECTED_IMAGES = {{{expected}}}" in src
    assert "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)) in (
        OUT_DIR / "data.yaml").read_text()
