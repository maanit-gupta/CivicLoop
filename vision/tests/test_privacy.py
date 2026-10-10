"""Step 5.5: blurring faces/plates, storing without EXIF, failing closed, and the eval metrics."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from vision import evaluate as ev
from vision import privacy as pv

SAVED = ev.EVAL_DIR / "privacy_v1_test.json"


def noise(w=120, h=80, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def detail(a: np.ndarray) -> float:
    """Mean squared difference of neighbouring pixels: high for sharp detail, ~0 when blurred."""
    a = a.astype(float)
    return float(np.mean(np.diff(a, axis=0) ** 2) + np.mean(np.diff(a, axis=1) ** 2))


# ---------- blur ----------

def test_padded_grows_every_side_and_clips_to_the_photo():
    assert pv.padded((10, 10, 30, 20), (100, 100), pad=0.5) == (0, 5, 40, 25)
    assert pv.padded((90, 90, 100, 100), (100, 100), pad=0.5) == (85, 85, 100, 100)


def test_blur_removes_detail_inside_and_keeps_every_pixel_outside():
    img = noise()
    out = np.asarray(pv.blur(img, [(40, 20, 80, 60)]))
    src = np.asarray(img)
    inside = (slice(20, 60), slice(40, 80))
    assert detail(out[inside]) < 0.05 * detail(src[inside])
    keep = np.ones(src.shape[:2], bool)
    keep[inside] = False
    assert np.array_equal(out[keep], src[keep])


def test_blur_found_pads_each_kind_by_its_own_amount():
    img, box = noise(), (40, 30, 80, 50)
    out = np.asarray(pv.blur_found(img, [pv.Found("plate", 0.9, box)], {"plate": 0.5}))
    src = np.asarray(img)
    assert not np.array_equal(out[30:50, 25], src[30:50, 25])  # 0.5 x 40 px = 20 px wider
    assert np.array_equal(out[30:50, 15], src[30:50, 15])
    out = np.asarray(pv.blur_found(img, [pv.Found("face", 0.9, box)], {"face": 0.0}))
    assert np.array_equal(out[30:50, 35], src[30:50, 35])


def test_blur_with_no_boxes_is_an_unchanged_copy():
    img = noise()
    out = pv.blur(img, [])
    assert out is not img and np.array_equal(np.asarray(out), np.asarray(img))


def test_plate_size_cap():
    assert pv.plate_sized((0, 0, 10, 10), (100, 100))  # 1% of the photo
    assert not pv.plate_sized((0, 0, 50, 50), (100, 100))  # 25%: a road sign, not a plate


# ---------- load / store ----------

def jpeg_with_exif(path, orientation=6):
    img = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = orientation  # rotate 90 degrees when displayed
    exif[0x010F] = "PhoneMaker"  # camera make, stands in for any identifying metadata
    img.save(path, exif=exif)
    return path


def test_load_applies_exif_rotation(tmp_path):
    assert pv.load(jpeg_with_exif(tmp_path / "a.jpg")).size == (20, 40)


def test_mask_file_blurs_stores_by_hash_and_drops_exif(tmp_path, monkeypatch):
    src = tmp_path / "upload.jpg"
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"
    noise(200, 100).save(src, exif=exif)
    with Image.open(src) as im:
        assert len(im.getexif()) == 1  # the upload does carry metadata
    monkeypatch.setattr(pv, "detect_faces", lambda img: [pv.Found("face", 0.9, (10, 10, 50, 50))])
    monkeypatch.setattr(pv, "detect_plates", lambda img: [])
    out, found = pv.mask_file(src, tmp_path / "store")
    assert out == tmp_path / "store" / f"{hashlib.sha256(src.read_bytes()).hexdigest()[:16]}.jpg"
    assert [f.kind for f in found] == ["face"]
    with Image.open(out) as im:
        assert len(im.getexif()) == 0
        stored = np.asarray(im.convert("RGB"))
    assert detail(stored[10:50, 10:50]) < 0.2 * detail(np.asarray(Image.open(src))[10:50, 10:50])


def test_mask_file_fails_closed(tmp_path, monkeypatch):
    def missing(img):
        raise FileNotFoundError("no face model")

    monkeypatch.setattr(pv, "detect_faces", missing)
    noise().save(tmp_path / "upload.jpg")
    with pytest.raises(FileNotFoundError):
        pv.mask_file(tmp_path / "upload.jpg", tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_weights_with_the_wrong_hash_are_refused(tmp_path):
    repo, rev, name, _ = pv.FACE_WEIGHTS
    (tmp_path / name).write_bytes(b"not the model")
    with pytest.raises(ValueError, match="sha256"):
        pv._weights((repo, rev, name, "0" * 64), tmp_path)


# ---------- evaluation metrics ----------

FACE_PAD = {"face": 0.15}


def found(*boxes, score=0.9, kind="face"):
    return [pv.Found(kind, score, b) for b in boxes]


def item(*boxes, size=(100, 100)):
    """A labelled photo; boxes are (corners, px, must_mask)."""
    return {"size": size, "boxes": list(boxes)}


def test_iou_px():
    assert ev.iou_px((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert ev.iou_px((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert ev.iou_px((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_covered_uses_the_padded_blur():
    m = ev.blurred_mask(found((20, 20, 40, 40)), (100, 100), FACE_PAD)  # padded to 17..43
    assert ev.covered((20, 20, 40, 40), m) == 1.0
    assert ev.covered((40, 20, 60, 40), m) == pytest.approx(3 / 20)


def test_mask_scores_recall_precision_and_threshold():
    items = [item(((10, 10, 30, 30), 20, True), ((60, 60, 70, 70), 10, False))]
    hit_and_miss = [found((10, 10, 30, 30)) + found((80, 0, 99, 10), score=0.3)]
    s = ev.mask_scores(items, hit_and_miss, 0.2, FACE_PAD)
    assert (s["recall"], s["n_true"], s["precision"], s["n_det"]) == (1.0, 1, 0.5, 2)
    s = ev.mask_scores(items, hit_and_miss, 0.5, FACE_PAD)  # the stray box is below threshold
    assert (s["recall"], s["precision"]) == (1.0, 1.0)


def test_mask_scores_box_not_to_be_masked_still_counts_for_precision():
    items = [item(((60, 60, 70, 70), 10, False))]  # too small to identify
    s = ev.mask_scores(items, [found((60, 60, 70, 70))], 0.5, FACE_PAD)
    assert (s["n_true"], s["precision"]) == (0, 1.0)


def test_mask_scores_partly_blurred_face_is_a_miss():
    items = [item(((10, 10, 50, 50), 40, True))]
    s = ev.mask_scores(items, [found((10, 10, 30, 50))], 0.5, FACE_PAD, buckets=[20, 40])
    assert s["recall"] == 0.0
    assert s["recall_by_size"]["40-infpx"] == {"recall": 0.0, "n": 1}


def test_issue_blurred_measures_harm_on_the_issue_box():
    rows = [{"size": (100, 100), "boxes": [(0, 0.25, 0.25, 0.5, 0.5)]},  # issue: 0..50 px
            {"size": (100, 100), "boxes": []}]
    out = ev.issue_blurred(rows, [found((0, 0, 21.7, 43.4)), []], 0.5, FACE_PAD)  # 0..25, 0..50
    assert out["issue_blurred"] == pytest.approx(0.5, abs=0.02)
    assert out["photos_blurred"] == 0.5


def test_sweep_covers_every_pad_and_threshold():
    civic = [{"size": (100, 100), "boxes": [(0, 0.5, 0.5, 1.0, 1.0)]}]
    rows = ev.sweep_mask(civic, [[]], [item()], [[]], "face", imgsz=640)
    assert len(rows) == len(ev.PADS["face"]) * len(ev.PRIVACY_GRID)
    assert all(r["imgsz"] == 640 for r in rows)


def row(recall, hidden, pad=0.15, threshold=0.5, **kw):
    return {"recall": recall, "issue_blurred": hidden, "pad": pad, "threshold": threshold, **kw}


def test_pick_mask_takes_the_best_recall_among_harmless_settings():
    harmful = row(0.95, 0.05, threshold=0.1)  # masks most, but hides 5% of the pothole
    best = row(0.90, 0.005, threshold=0.4)
    assert ev.pick_mask([harmful, best, row(0.80, 0.0, threshold=0.6)]) == best


def test_pick_mask_ties_prefer_less_blur():
    rows = [row(0.9, 0.0, pad=0.5), row(0.9, 0.0, pad=0.15, threshold=0.3),
            row(0.9, 0.0, pad=0.15, threshold=0.6), row(0.9, 0.0, pad=0.15, imgsz=1280)]
    assert ev.pick_mask(rows) == rows[2]


def test_pick_mask_falls_back_to_the_least_harmful():
    rows = [row(0.9, 0.2), row(0.5, 0.03)]
    assert ev.pick_mask(rows) == rows[1]


def test_load_plates_splits_by_video(tmp_path):
    for vid in ("vid-1", "vid-2"):
        (tmp_path / vid).mkdir()
        Image.new("RGB", (200, 100)).save(tmp_path / vid / "f1.jpg")
        (tmp_path / vid / "f1.txt").write_text("0 0.5 0.5 0.25 0.1\n0 0.1 0.1 0.1 0.1\n")
    out = ev.load_plates(tmp_path)
    assert [it["group"] for it in out["tune"]] == ["vid-1"]
    assert [it["group"] for it in out["test"]] == ["vid-2"]
    (box, px, must), (_, px2, must2) = out["tune"][0]["boxes"]
    assert box == pytest.approx((75, 45, 125, 55)) and (px, must) == (50, True)
    assert (px2, must2) == (20, False)


def test_load_wider_parses_faces_invalid_flags_and_empty_photos(tmp_path):
    split = tmp_path / "wider_face_split"
    split.mkdir()
    (split / "wider_face_val_bbx_gt.txt").write_text(
        "0--Parade/a.jpg\n2\n10 20 30 40 0 0 0 0 0 0 \n5 5 8 8 0 0 0 1 0 0 \n"
        "1--Handshaking/b.jpg\n0\n0 0 0 0 0 0 0 0 0 0 \n")
    out = ev.load_wider(tmp_path)
    items = {it["path"].name: it for it in out["tune"] + out["test"]}
    assert items["a.jpg"]["boxes"] == [((10, 20, 40, 60), 40, True), ((5, 5, 13, 13), 8, False)]
    assert items["b.jpg"]["boxes"] == []
    assert {it["group"] for it in out["tune"]}.isdisjoint({it["group"] for it in out["test"]})


# ---------- saved evaluation ----------

@pytest.mark.skipif(not SAVED.exists(), reason="5.5 eval not run")
def test_thresholds_in_code_are_the_val_picks():
    saved = json.loads(SAVED.read_text())
    assert (saved["face_threshold"], saved["plate_threshold"], saved["plate_imgsz"]) == (
        pv.FACE_THRESHOLD, pv.PLATE_THRESHOLD, pv.PLATE_IMGSZ)
    assert saved["pad"] == pv.PAD


@pytest.mark.skipif(not SAVED.exists(), reason="5.5 eval not run")
def test_masking_beats_the_trivial_baselines():
    s = json.loads(SAVED.read_text())
    u = s["utility"]
    assert s["test_face"]["recall"] > 0.5 and s["test_plate"]["recall"] > 0.5  # > mask nothing
    assert u["masked"]["macro_f1"] > u["blur whole photo"]["macro_f1"] + 0.3
    assert u["masked"]["labels_equal_no_mask"] > 0.95
