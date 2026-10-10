"""Step 5.4: YOLO + CLIP -> ImageFeatures, compare(), the embedding helpers and checks."""

import json
import random
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from contracts.api import ImageFeatures, IssueType
from vision import embed
from vision import evaluate as ev
from vision import features as ft
from vision.dataset import LABELS
from vision.yolo_detect import LABEL_THRESHOLD

POTHOLE, GARBAGE, SIGN, POLE = range(4)
EMB = list(embed.unit(np.arange(1, 513, dtype=np.float32)))


def probs(**high):
    """CLIP-like probabilities: 0.02 everywhere except the given labels."""
    p = dict.fromkeys(LABELS, 0.02)
    p.update(high)
    return p


def box(cls, conf, cx=0.5, cy=0.5, w=0.2, h=0.2):
    return (cls, conf, cx, cy, w, h)


# ---------- issue label: YOLO first, CLIP for other vs none ----------

def test_a_confident_box_beats_clip():
    assert ft.hybrid_label([box(POTHOLE, 0.5)], probs(other=0.9), ft.FALLBACK) == "pothole"


@pytest.mark.parametrize("p, want", [(probs(other=0.6, none=0.3), "other"),
                                     (probs(other=0.3, none=0.6), "none"),
                                     (probs(pothole=0.8, other=0.1, none=0.05), "other")])
def test_other_vs_none_only_compares_those_two(p, want):
    # A weak box (below tau) does not count; CLIP's opinion on pothole is ignored.
    assert ft.hybrid_label([box(POTHOLE, LABEL_THRESHOLD / 2)], p, "clip other vs none") == want


def test_every_fallback_rule_returns_an_image_label():
    for rule in ft.FALLBACK_RULES:
        for p in (probs(other=0.9), probs(none=0.9), probs(garbage=0.9)):
            assert ft.FALLBACK_RULES[rule](p) in LABELS
    assert ft.FALLBACK_RULES["yolo only"](probs(other=0.9)) == "none"
    assert ft.FALLBACK_RULES["clip top"](probs(garbage=0.9)) == "garbage"


SAVED = ev.EVAL_DIR / "features_v1_test.json"


@pytest.mark.skipif(not SAVED.exists(), reason="5.4 eval not run")
def test_fallback_is_the_val_pick():
    assert ft.FALLBACK == json.loads(SAVED.read_text())["fallback"]


def test_pick_fallback_prefers_clip_when_yolo_cannot_see_other():
    outputs = [([box(POTHOLE, 0.9)], probs(pothole=0.9), EMB),
               ([], probs(other=0.7, none=0.2), EMB),   # fallen tree: YOLO silent
               ([], probs(none=0.7, other=0.2), EMB)]   # clean road
    rule, scores = ev.pick_fallback(outputs, ["pothole", "other", "none"])
    assert rule == "clip other vs none" and scores["yolo only"] < scores[rule]


# ---------- boxes ----------

def test_to_detection_centre_to_corners():
    d = ft.to_detection(box(POLE, 0.7, 0.5, 0.4, 0.2, 0.4))
    assert d.label == "damaged_pole" and d.bbox == pytest.approx((0.4, 0.2, 0.6, 0.6))


def test_to_detection_clips_to_the_photo_and_drops_boxes_outside_it():
    assert ft.to_detection(box(GARBAGE, 0.7, 0.95, 0.05, 0.2, 0.2)).bbox == \
        pytest.approx((0.85, 0.0, 1.0, 0.15))
    assert ft.to_detection(box(GARBAGE, 0.7, 1.2, 0.5, 0.2, 0.2)) is None


# ---------- build ----------

def test_build_from_yolo_keeps_only_confident_boxes():
    dets = [box(GARBAGE, 0.3, 0.2, 0.2), box(POLE, 0.8, 0.7, 0.7), box(POTHOLE, 0.05)]
    f = ft.build("a.jpg", dets, probs(other=0.9), EMB)
    assert f.issue_type == IssueType.STREETLIGHT  # damaged_pole -> streetlight (5.1)
    assert [d.label for d in f.detections] == ["damaged_pole", "garbage"]  # by confidence
    assert f.detection_count == 2 and f.confidence == pytest.approx(0.8)
    assert f.box_area_ratio == pytest.approx(0.08)
    assert f.model_version == ft.MODEL_VERSION and "detector found" in f.explanation


@pytest.mark.parametrize("p, issue, conf", [(probs(other=0.6, none=0.3), IssueType.OTHER, 0.6),
                                            (probs(other=0.1, none=0.8), None, 0.8)])
def test_build_from_clip_when_no_box_is_confident(p, issue, conf):
    f = ft.build("a.jpg", [box(POTHOLE, 0.05)], p, EMB)
    assert f.issue_type == issue and f.confidence == pytest.approx(conf)
    assert f.detections == [] and f.detection_count == 0 and f.box_area_ratio == 0.0
    assert "no detector box reached 0.10" in f.explanation.lower()


def test_build_output_is_valid_json_contract():
    f = ft.build("a.jpg", [box(SIGN, 0.9)], probs(none=0.9), EMB)
    again = ImageFeatures.model_validate(json.loads(f.model_dump_json()))
    assert again == f and len(again.embedding) == embed.DIM


# ---------- compare (Closure) ----------

def feat(issue, n_boxes=0, emb=EMB, version=ft.MODEL_VERSION):
    dets = [box(POTHOLE, 0.9, 0.2 + 0.3 * k, 0.5) for k in range(n_boxes)]
    f = ft.build("x.jpg", dets, probs(other=0.9) if issue else probs(none=0.9), emb)
    return f.model_copy(update={"model_version": version})


def test_compare_same_photo():
    c = ft.compare(feat("pothole", 2), feat("pothole", 2))
    assert c["similarity"] == pytest.approx(1.0) and c["issue_cleared"] is False
    assert c["detections_before"] == c["detections_after"] == 2


def test_compare_repaired_pothole_is_cleared():
    c = ft.compare(feat("pothole", 1), feat(None))
    assert c["issue_before"] == IssueType.POTHOLE and c["issue_after"] is None
    assert c["issue_cleared"] is True and c["detections_after"] == 0


def test_compare_with_nothing_before_has_nothing_to_clear():
    assert ft.compare(feat(None), feat(None))["issue_cleared"] is None


def test_compare_refuses_mixed_models_or_missing_embeddings():
    with pytest.raises(ValueError, match="model versions differ"):
        ft.compare(feat("pothole", 1), feat(None, version="yolo-v2+clip-other"))
    with pytest.raises(ValueError, match="embeddings"):
        ft.compare(feat("pothole", 1), feat(None).model_copy(update={"embedding": None}))


# ---------- embedding helpers and the embedding check ----------

def test_unit_and_cosine():
    v = embed.unit(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.linalg.norm(v, axis=1) == pytest.approx([1.0, 1.0])
    assert embed.cosine([1, 0], [2, 0]) == pytest.approx(1.0)
    assert embed.cosine([1, 0], [0, 5]) == pytest.approx(0.0)
    assert embed.cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_auc():
    assert ev.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ev.auc([0.1], [0.9]) == 0.0
    assert ev.auc([0.5], [0.5]) == 0.5


def test_dhash_similarity():
    assert ev.dhash_similarity("ff00", "ff00") == 1.0
    assert ev.dhash_similarity("0", "f") == pytest.approx(1 - 4 / 64)


def test_perturb_is_seeded_and_keeps_the_size(tmp_path):
    path = tmp_path / "a.jpg"
    pixels = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    a, b = ev.perturb(path, random.Random(1)), ev.perturb(path, random.Random(1))
    assert a == b
    assert Image.open(BytesIO(a)).size == (80, 60)
