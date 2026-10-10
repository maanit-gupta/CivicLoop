"""Step 5.2: the metrics on hand-checked cases, and the CLIP prompt scoring without the model."""

import numpy as np
import pytest

from vision import clip_zeroshot as cz
from vision import evaluate as ev
from vision.dataset import CLASSES, LABELS, OUT_DIR

POTHOLE, GARBAGE, SIGN, POLE = range(4)

# ---------- IoU and AP ----------

def test_iou():
    a = (0.25, 0.5, 0.5, 1.0)  # left half of the frame
    assert ev.iou(a, a) == pytest.approx(1.0)
    assert ev.iou(a, (0.75, 0.5, 0.5, 1.0)) == 0.0  # right half: touching, no overlap
    assert ev.iou(a, (0.5, 0.5, 0.5, 1.0)) == pytest.approx(1 / 3)  # 0.25 / (0.5 + 0.5 - 0.25)


@pytest.mark.parametrize("hits, n_true, ap", [
    ([True, True], 2, 1.0),
    ([False, False], 2, 0.0),
    ([False, True], 1, 0.5),  # the only correct box is ranked below a wrong one
    ([True, False, True], 2, 0.5 * 1 + 0.5 * 2 / 3),
    ([True], 2, 0.5),  # one true box never found: recall stops at 0.5
    ([], 3, 0.0),
])
def test_average_precision(hits, n_true, ap):
    assert ev.average_precision(hits, n_true) == pytest.approx(ap)


def test_ap_envelope_uses_best_precision_to_the_right():
    # (recall, precision) after each box: (.33, 1) (.33, .5) (.67, .67) (1, .75). The envelope
    # lifts the .67 at recall .67 to the later .75, so AP = 1/3*1 + 1/3*.75 + 1/3*.75.
    assert ev.average_precision([True, False, True, True], 3) == pytest.approx(
        (1 + 2 * 0.75) / 3)


# ---------- mAP50 ----------

TRUTH = {
    "a": [(POTHOLE, 0.3, 0.3, 0.2, 0.2), (POTHOLE, 0.7, 0.7, 0.2, 0.2)],
    "b": [(GARBAGE, 0.5, 0.5, 0.4, 0.4)],
    "c": [],  # a background photo ("none"/"other"): any box here is a false positive
}


def test_map50_perfect():
    preds = {i: [(c, 0.9, *box) for c, *box in bs] for i, bs in TRUTH.items()}
    out = ev.map50(TRUTH, preds)
    assert out["map50"] == pytest.approx(1.0)
    # classes with no true box (sign, pole) are left out of the mean, not counted as 0
    assert out["per_class"]["damaged_sign"]["n_true"] == 0


def test_map50_no_predictions_is_zero():
    assert ev.map50(TRUTH, {})["map50"] == 0.0


def test_map50_two_boxes_on_one_pothole_find_only_one():
    preds = {"a": [(POTHOLE, 0.9, 0.3, 0.3, 0.2, 0.2), (POTHOLE, 0.8, 0.31, 0.3, 0.2, 0.2)]}
    pothole = ev.map50(TRUTH, preds)["per_class"]["pothole"]
    assert pothole["recall"] == pytest.approx(0.5)
    assert pothole["ap50"] == pytest.approx(0.5)


def test_map50_wrong_class_or_poor_overlap_misses():
    preds = {"b": [(POTHOLE, 0.9, 0.5, 0.5, 0.4, 0.4),          # right place, wrong class
                   (GARBAGE, 0.9, 0.5, 0.5, 0.1, 0.1)]}          # right class, IoU 0.0625
    out = ev.map50(TRUTH, preds)
    assert out["per_class"]["garbage"]["ap50"] == 0.0
    assert out["per_class"]["pothole"]["ap50"] == 0.0


def test_map50_false_positive_on_background_lowers_ap():
    good = (GARBAGE, 0.5, 0.5, 0.5, 0.4, 0.4)
    clean = ev.map50(TRUTH, {"b": [good]})["per_class"]["garbage"]["ap50"]
    noisy = ev.map50(TRUTH, {"b": [good], "c": [(GARBAGE, 0.9, 0.5, 0.5, 0.4, 0.4)]})
    assert clean == pytest.approx(1.0)
    assert noisy["per_class"]["garbage"]["ap50"] == pytest.approx(0.5)


def test_full_image_boxes_use_mapped_issue_probability():
    probs = [{lb: 0.0 for lb in LABELS} | {"streetlight": 0.7, "pothole": 0.2}]
    dets = ev.full_image_boxes(probs, ["x"])["x"]
    assert [d[0] for d in dets] == list(range(len(CLASSES)))
    assert dets[CLASSES.index("damaged_pole")][1] == 0.7  # damaged_pole -> streetlight (5.1)
    assert dets[POTHOLE] == (POTHOLE, 0.2, 0.5, 0.5, 1.0, 1.0)


# ---------- classification ----------

def test_classification_report_by_hand():
    rep = ev.classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert rep["accuracy"] == 0.75
    assert rep["per_label"]["a"] == pytest.approx(
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2})
    assert rep["per_label"]["b"]["f1"] == pytest.approx(0.8)
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert rep["confusion"]["a"] == {"a": 1, "b": 1}


def test_label_never_predicted_scores_zero_f1_not_error():
    rep = ev.classification_report(["a", "b"], ["a", "a"], ["a", "b"])
    assert rep["per_label"]["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}


def test_majority_label_ties_break_alphabetically():
    assert ev.majority_label(["pothole", "none", "none", "pothole", "garbage"]) == "none"


# ---------- data loading ----------

def test_load_split_reads_boxes_and_empty_labels(tmp_path):
    (tmp_path / "labels" / "test").mkdir(parents=True)
    (tmp_path / "manifest.csv").write_text(
        "image_id,split,label,source\n"
        "p1,test,pothole,rdd2022_india\nn1,test,none,rdd2022_india\nt1,train,garbage,x\n")
    (tmp_path / "labels" / "test" / "p1.txt").write_text("0 0.5 0.4 0.2 0.1\n")
    (tmp_path / "labels" / "test" / "n1.txt").write_text("")
    rows = ev.load_split("test", tmp_path)
    assert [r["image_id"] for r in rows] == ["p1", "n1"]
    assert rows[0]["boxes"] == [(0, 0.5, 0.4, 0.2, 0.1)]
    assert rows[1]["boxes"] == []
    assert rows[0]["path"] == tmp_path / "images" / "test" / "p1.jpg"


@pytest.mark.skipif(not (OUT_DIR / "manifest.csv").exists(), reason="run python -m vision.dataset")
def test_real_test_split_has_every_label():
    rows = ev.load_split("test")
    assert {r["label"] for r in rows} == set(LABELS)
    assert all(r["path"].exists() for r in rows)
    # background labels carry no boxes; issue labels carry at least one
    assert all((not r["boxes"]) == (r["label"] in ("other", "none")) for r in rows)


# ---------- CLIP prompt scoring (no model download) ----------

def test_prompts_cover_exactly_the_labels():
    assert set(cz.PROMPTS) == set(LABELS)
    assert all(len(ps) >= 2 for ps in cz.PROMPTS.values())


def test_prompts_do_not_name_the_camera():
    # every "none" photo is dashcam footage (5.1): a prompt keyed on the camera would cheat
    assert not any("dashcam" in p.lower() for ps in cz.PROMPTS.values() for p in ps)


def test_label_probs_take_best_prompt_and_ignore_prompt_count():
    prompt_labels = ["x", "x", "x", "y"]
    sims = np.array([[0.20, 0.20, 0.20, 0.20],   # tie: 3 prompts must not beat 1
                     [0.10, 0.30, 0.10, 0.25]])  # x's best prompt (0.30) beats y (0.25)
    p = cz.label_probs(sims, prompt_labels, ["x", "y"])
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert p[0] == pytest.approx([0.5, 0.5])
    assert p[1, 0] == pytest.approx(1 / (1 + np.exp(-0.05 * cz.LOGIT_SCALE)))
