"""Step 5.1: the dataset builder on tiny fixtures, plus sanity checks on the real build."""

import csv

import pytest
from PIL import Image, ImageDraw

from vision import dataset as ds

# ---------- pure helpers ----------

@pytest.mark.parametrize("name, stem", [
    ("img_12_jpg.rf.ab12cd.jpg", "img_12"),
    ("photo_JPG.rf.ff00.jpg", "photo"),
    ("road_png.rf.1.jpg", "road"),
    ("plain_name.jpg", "plain_name"),
])
def test_original_stem(name, stem):
    assert ds.original_stem(name) == stem


def _picture(path, seed, size=(200, 160)):
    """A photo-like test image: a smooth gradient (real photos rarely have flat regions, and
    dHash compares neighbouring pixels) plus a couple of seed-dependent shapes."""
    img = Image.linear_gradient("L").rotate(seed * 37 % 360).resize(size).convert("RGB")
    d = ImageDraw.Draw(img)
    s = seed % 7
    d.rectangle([10 + s * 9, 20, 70 + s * 9, 120], fill=(250, 240, 230))
    d.ellipse([110, 20 + s * 8, 190, 80 + s * 8], fill=(20, 20, 20))
    img.save(path)
    return path


def test_dhash_survives_resize_but_separates_photos(tmp_path):
    a = _picture(tmp_path / "a.png", 1)
    Image.open(a).resize((100, 80)).save(tmp_path / "a_small.jpg", quality=90)
    b = _picture(tmp_path / "b.png", 4)
    assert ds.dhash(a) == ds.dhash(tmp_path / "a_small.jpg")
    assert ds.dhash(a) != ds.dhash(b)


VOC = """<annotation><size><width>720</width><height>360</height></size>
<object><name>D40</name><bndbox><xmin>72</xmin><ymin>36</ymin><xmax>144</xmax><ymax>180</ymax>
</bndbox></object>
<object><name>D00</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax>
</bndbox></object></annotation>"""


def test_voc_keeps_only_potholes_and_normalises():
    boxes, codes = ds.voc_boxes(VOC)
    assert codes == {"D40", "D00"}
    assert len(boxes) == 1
    cx, cy, w, h = boxes[0]
    assert (cx, cy, w, h) == pytest.approx((0.15, 0.3, 0.1, 0.4))


def test_yolo_line_box_formats():
    assert ds.yolo_line_box("3 0.5 0.5 0.2 0.4") == pytest.approx((0.5, 0.5, 0.2, 0.4))
    # polygon -> its bounding box
    assert ds.yolo_line_box("0 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6") == pytest.approx(
        (0.2, 0.4, 0.2, 0.4))
    # box spilling over the edge is clipped to the image
    cx, cy, w, h = ds.yolo_line_box("0 0.95 0.5 0.2 0.2")
    assert cx + w / 2 == pytest.approx(1.0) and w == pytest.approx(0.15)
    assert ds.yolo_line_box("0 0.5 0.5 0.0 0.3") is None  # zero width
    assert ds.yolo_line_box("0 0.5 0.5") is None  # malformed


def test_split_is_stratified_disjoint_and_deterministic():
    items = [{"key": f"k{i}", "label": "pothole" if i % 3 else "none"} for i in range(300)]
    a, b = ds.split_items(items), ds.split_items(items)
    assert {s: [it["key"] for it in v] for s, v in a.items()} == \
        {s: [it["key"] for it in v] for s, v in b.items()}
    keys = [it["key"] for v in a.values() for it in v]
    assert len(keys) == len(set(keys)) == 300
    for label, n in (("pothole", 200), ("none", 100)):
        got = {s: sum(it["label"] == label for it in v) for s, v in a.items()}
        assert got == {"train": round(0.7 * n), "val": round(0.15 * n),
                       "test": n - round(0.7 * n) - round(0.15 * n)}


# ---------- tiny raw tree, end to end ----------

def _rdd(root, name, objects):
    (root / "India/train/images").mkdir(parents=True, exist_ok=True)
    (root / "India/train/annotations/xmls").mkdir(parents=True, exist_ok=True)
    _picture(root / f"India/train/images/{name}.jpg", len(name) + len(objects) * 3)
    objs = "".join(
        f"<object><name>{c}</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>100</xmax>"
        f"<ymax>90</ymax></bndbox></object>" for c in objects)
    (root / f"India/train/annotations/xmls/{name}.xml").write_text(
        f"<annotation><size><width>200</width><height>160</height></size>{objs}</annotation>")


def _urban(root, folder, split, name, seed, label_lines):
    img_dir = root / folder / folder / split / "images"
    lbl_dir = root / folder / folder / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    _picture(img_dir / name, seed)
    (lbl_dir / f"{name.rsplit('.', 1)[0]}.txt").write_text("\n".join(label_lines))


@pytest.fixture
def raw(tmp_path):
    r = tmp_path / "raw"
    rdd = r / "rdd2022_india"
    _rdd(rdd, "India_000001", ["D40", "D00"])  # pothole (crack box ignored)
    _rdd(rdd, "India_000002", [])  # no damage -> none
    _rdd(rdd, "India_000003", ["D20"])  # cracks only -> skipped
    u = r / "urban_issues"
    pot = "Potholes and RoadCracks"
    _urban(u, pot, "train", "p1_jpg.rf.aaa.jpg", 21, ["1 0.5 0.5 0.2 0.2", "0 0.2 0.2 0.1 0.1"])
    _urban(u, pot, "test", "p1_jpg.rf.bbb.jpg", 22, ["1 0.4 0.5 0.2 0.2"])  # augmented copy
    _urban(u, pot, "train", "c1_jpg.rf.ccc.jpg", 27, ["0 0.5 0.5 0.2 0.2"])  # cracks only
    _urban(u, "Garbage", "train", "g1_jpg.rf.c.jpg", 23,
           ["5 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4", ""])  # polygon + blank line
    _urban(u, "FallenTrees", "valid", "t1_jpg.rf.d.jpg", 24, ["4 0.5 0.5 0.5 0.5"])
    _urban(u, "Graffitti", "valid", "x1_jpg.rf.x.jpg", 28,
           ["6 0.5 0.5 0.5 0.5", "1 0.2 0.2 0.1 0.1"])  # stray "pothole" box -> skipped
    _urban(u, "DamagedRoadSigns", "train", "p1_jpg.rf.eee.jpg", 25,
           ["3 0.5 0.5 0.2 0.2"])  # same photo stem as p1 under another class -> dropped
    return r


def test_build_on_fixture(raw, tmp_path):
    out = tmp_path / "yolo"
    stats = ds.build(raw, out)
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    by_key = {r["source_key"]: r for r in rows}
    assert set(by_key) == {"rdd/India_000001", "rdd/India_000002", "urban/p1", "urban/g1",
                           "urban/t1"}
    assert stats["rdd_skipped_other_damage"] == 1
    assert stats["urban_dropped_aug_or_crossfolder_copy"] == 2
    assert stats["urban_dropped_no_target_box"] == 1
    assert stats["urban_dropped_foreign_class_box"] == 1
    assert by_key["rdd/India_000002"]["label"] == "none"
    assert by_key["urban/t1"]["label"] == "other" and by_key["urban/t1"]["n_boxes"] == "0"

    def label_lines(key):
        r = by_key[key]
        return (out / "labels" / r["split"] / f"{r['image_id']}.txt").read_text().splitlines()

    assert [ln.split()[0] for ln in label_lines("urban/p1")] == ["0"]  # source id 1 -> pothole
    # the unaugmented test/ copy of p1 wins over its train/ copy (box at x=0.4, not 0.5)
    assert label_lines("urban/p1")[0].startswith("0 0.400000")
    assert [ln.split()[0] for ln in label_lines("urban/g1")] == ["1"]  # source id 5 -> garbage
    assert [ln.split()[0] for ln in label_lines("rdd/India_000001")] == ["0"]
    assert label_lines("rdd/India_000002") == []
    for r in rows:
        assert (out / "images" / r["split"] / f"{r['image_id']}.jpg").exists()
    yaml = (out / "data.yaml").read_text()
    assert "path:" not in yaml and "3: damaged_pole" in yaml


def test_build_is_deterministic(raw, tmp_path):
    ds.build(raw, tmp_path / "a")
    ds.build(raw, tmp_path / "b")
    assert (tmp_path / "a/manifest.csv").read_text() == (tmp_path / "b/manifest.csv").read_text()


# ---------- the real dataset (skipped until `python -m vision.dataset` has run) ----------

real = pytest.mark.skipif(not (ds.OUT_DIR / "manifest.csv").exists(),
                          reason="run `python -m vision.dataset` first")


@real
def test_real_labels_are_valid_yolo():
    rows = list(csv.DictReader(open(ds.OUT_DIR / "manifest.csv")))
    for r in rows:
        assert (ds.OUT_DIR / "images" / r["split"] / f"{r['image_id']}.jpg").exists()
        lines = (ds.OUT_DIR / "labels" / r["split"] / f"{r['image_id']}.txt").read_text()
        lines = lines.splitlines()
        assert len(lines) == int(r["n_boxes"])
        assert (len(lines) == 0) == (r["label"] in ("other", "none"))
        for ln in lines:
            c, *box = ln.split()
            assert 0 <= int(c) < len(ds.CLASSES)
            cx, cy, w, h = map(float, box)
            assert 0 < w <= 1 and 0 < h <= 1
            assert 0 <= cx - w / 2 + 1e-6 and cx + w / 2 <= 1 + 1e-6
            assert 0 <= cy - h / 2 + 1e-6 and cy + h / 2 <= 1 + 1e-6


@real
def test_real_split_has_no_leak_and_every_label_everywhere():
    rows = list(csv.DictReader(open(ds.OUT_DIR / "manifest.csv")))
    keys = [r["source_key"] for r in rows]
    assert len(keys) == len(set(keys))
    hashes = [r["dhash"] for r in rows]
    assert len(hashes) == len(set(hashes))
    for split in ds.SPLITS:
        assert {r["label"] for r in rows if r["split"] == split} == set(ds.LABELS)
