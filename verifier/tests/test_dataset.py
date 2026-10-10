import shutil
from itertools import combinations
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from config import ROOT, WORLD_DIR
from contracts.corpus_gen import build_corpus
from contracts.world import load_world
from contracts.world_gen import build_world
from verifier import dataset as ds
from verifier import images as im

FAKES = {"train": 4, "test": 2}
N_IMAGES = {"train": 50, "val": 15, "test": 25}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    build_world(d)
    shutil.copy(WORLD_DIR / "departments.yaml", d)
    return build_corpus(load_world(d), n=400)


@pytest.fixture(scope="module")
def yolo(tmp_path_factory):
    """A tiny stand-in for the Phase 5 image set: noise images, one box each (none for 'none')."""
    root, rng, rows = tmp_path_factory.mktemp("yolo"), np.random.default_rng(0), []
    for split, n in N_IMAGES.items():
        for label in (*ds.PHOTO_TYPES, "none"):
            for i in range(n):
                image_id = f"{label}_{split}_{i:03d}"
                (root / "images" / split).mkdir(parents=True, exist_ok=True)
                (root / "labels" / split).mkdir(parents=True, exist_ok=True)
                pixels = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(root / "images" / split / f"{image_id}.jpg")
                box = "" if label == "none" else "0 0.5 0.5 0.3 0.4\n"
                (root / "labels" / split / f"{image_id}.txt").write_text(box)
                rows.append({"image_id": image_id, "split": split, "label": label})
    return root, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def built(corpus, yolo, tmp_path_factory):
    root, manifest = yolo
    out = tmp_path_factory.mktemp("verifier")
    return (*ds.build(corpus, manifest, root, out, fakes_per_kind=FAKES), out)


# ---------- pairs ----------

def test_every_in_window_same_incident_pair_is_a_positive(corpus, built):
    _, pairs, _, _ = built
    real = corpus[~corpus.is_spam]
    expected = set()
    for _, grp in real.groupby("incident_id"):
        for a, b in combinations(grp.sort_values("submitted_at").itertuples(), 2):
            if (b.submitted_at - a.submitted_at).total_seconds() <= ds.WINDOW_H * 3600:
                expected.add((a.complaint_id, b.complaint_id))
    pos = pairs[pairs.is_duplicate]
    assert set(zip(pos.a_id, pos.b_id, strict=True)) == expected
    assert (pairs.kind == "same_incident").equals(pairs.is_duplicate)


def test_pairs_are_ordered_in_window_and_kinds_follow_true_geometry(corpus, built):
    _, pairs, _, _ = built
    c = corpus.set_index("complaint_id")
    a, b = c.loc[pairs.a_id], c.loc[pairs.b_id]
    gap_h = (b.submitted_at.to_numpy() - a.submitted_at.to_numpy()) / np.timedelta64(1, "h")
    assert (gap_h > 0).all() and (gap_h <= ds.WINDOW_H).all()
    dist = ds.metres(a.lat.to_numpy(), a.lon.to_numpy(), b.lat.to_numpy(), b.lon.to_numpy())
    same_type = a.issue_type.to_numpy() == b.issue_type.to_numpy()
    kind = pairs.kind.to_numpy()
    assert (same_type & (dist <= ds.NEAR_M))[kind == "near_same_type"].all()
    assert (~same_type & (dist <= ds.SAME_PLACE_M))[kind == "same_place_other_type"].all()
    rnd = kind == "random"
    assert not (same_type & (dist <= ds.NEAR_M))[rnd].any()
    assert not (~same_type & (dist <= ds.SAME_PLACE_M))[rnd].any()
    assert not c.is_spam[pairs.a_id].any() and not c.is_spam[pairs.b_id].any()


def test_no_test_report_in_a_train_pair(corpus, built):
    _, pairs, _, _ = built
    split = corpus.set_index("complaint_id").split
    train = pairs[pairs.split == "train"]
    assert (split[train.a_id].to_numpy() == "train").all()
    assert (split[train.b_id].to_numpy() == "train").all()
    assert (split[pairs.b_id].to_numpy() == pairs.split.to_numpy()).all()


def test_random_negatives_are_capped_per_positive(built):
    _, pairs, _, _ = built
    for _, grp in pairs.groupby("split"):
        assert (grp.kind == "random").sum() <= ds.RANDOM_NEG_PER_POS * grp.is_duplicate.sum()


# ---------- reports and photos ----------

def test_report_table_keeps_the_corpus_and_adds_noise(corpus, built):
    reports = built[0]
    assert list(reports.columns) == ds.REPORT_COLUMNS and reports.report_id.is_unique
    orig = reports[~reports.report_id.str.startswith("F-")].set_index("report_id")
    c = corpus.set_index("complaint_id").loc[orig.index]
    assert (orig.text == c.text).all() and (orig.is_spam == c.is_spam).all()
    assert (orig.spam_kind == np.where(c.is_spam, "text", "none")).all()
    real = ~c.is_spam
    missing = orig.lat[real].isna().mean()
    assert 0.05 < missing < 0.15
    off = ds.metres(orig.lat[real], orig.lon[real], c.lat[real], c.lon[real]).dropna()
    assert 10 < off.median() < 30  # Rayleigh median for sigma 15 m is 17.7 m
    assert orig.lat[c.is_spam].isna().all()  # spam has no location at all


def test_photos_follow_incidents(corpus, yolo, built):
    reports = built[0]
    _, manifest = yolo
    real = reports[~reports.report_id.str.startswith("F-")]
    with_photo = real[real.image_path.notna()]
    assert real.image_path.notna().equals(real.base_image.notna())
    assert with_photo.issue_type.isin(ds.PHOTO_TYPES).all() and not with_photo.is_spam.any()
    assert (with_photo.groupby("incident_id").base_image.nunique() == 1).all()
    firsts = with_photo.drop_duplicates("incident_id")
    assert firsts.base_image.is_unique  # different incidents never share a source photo
    vsplit = manifest.set_index("image_id").split[with_photo.base_image].to_numpy()
    allowed = with_photo.split.map(ds.VISION_SPLITS).to_numpy()
    assert all(v in ok for v, ok in zip(vsplit, allowed, strict=True))
    label = manifest.set_index("image_id").label[with_photo.base_image].to_numpy()
    assert (label == with_photo.issue_type.to_numpy()).all()
    for p in with_photo.image_path:
        assert (ROOT / p).exists()


def test_fakes(corpus, built):
    reports, _, stock, _ = built
    fakes = reports[reports.report_id.str.startswith("F-")]
    counts = fakes.groupby(["split", "spam_kind"]).size()
    assert all(counts[s, k] == n for s, n in FAKES.items() for k in ds.FAKE_KINDS)
    assert fakes.is_spam.all() and fakes.incident_id.is_unique
    by_id, c = reports.set_index("report_id"), corpus.set_index("complaint_id")
    assert (c.issue_type[fakes.text_from].to_numpy() == fakes.issue_type.to_numpy()).all()
    genuine_bases = set(reports.base_image[reports.spam_kind == "none"].dropna())
    for f in fakes.itertuples():
        if f.spam_kind == "reused_image":
            orig = by_id.loc[f.image_from]
            assert orig.spam_kind == "none" and orig.submitted_at < f.submitted_at
            own, text = c.loc[f.image_from], c.loc[f.text_from]
            assert ds.metres(own.lat, own.lon, text.lat, text.lon) >= ds.REUSE_MIN_M
            assert f.base_image == orig.base_image
        else:
            assert f.base_image not in genuine_bases
            in_index = f.base_image in set(stock.image_id[stock.split == f.split])
            assert in_index == (f.spam_kind == "stock_image")


def test_same_seed_same_dataset(corpus, yolo, built, tmp_path):
    reports, pairs, stock, out = built
    root, manifest = yolo
    reports2, pairs2, stock2 = ds.build(corpus, manifest, root, tmp_path, fakes_per_kind=FAKES)
    pd.testing.assert_frame_equal(pairs, pairs2)
    pd.testing.assert_frame_equal(stock, stock2)

    def local(df, d):
        return df.assign(image_path=df.image_path.str.replace(str(d), "", regex=False))

    pd.testing.assert_frame_equal(local(reports, out), local(reports2, tmp_path))
    for p in reports.image_path.dropna().head(5):
        name = Path(p).relative_to(out)
        assert (out / name).read_bytes() == (tmp_path / name).read_bytes()


# ---------- pixel transforms ----------

def _scene(seed=0) -> Image.Image:
    """A blocky test picture: dhash needs structure, not noise."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks).resize((320, 240), Image.Resampling.NEAREST)


def test_crop_keeps_the_issue_in_view():
    rng = np.random.default_rng(1)
    for _ in range(200):
        start = im._crop_start(100, 80, [(0.3, 0.6)], rng)
        assert start <= 30 and start + 80 >= 60


def test_perspective_maps_corners():
    dst, src = [(0, 0), (10, 0), (10, 10), (0, 10)], [(1, 1), (9, 0), (10, 9), (0, 10)]
    a, b, c, d, e, f, g, h = im._perspective_coeffs(dst, src)
    for (x, y), (u, v) in zip(dst, src, strict=True):
        w = g * x + h * y + 1
        assert abs((a * x + b * y + c) / w - u) < 1e-9 and abs((d * x + e * y + f) / w - v) < 1e-9


def test_reshare_is_a_near_copy_and_view_and_screenshot_are_not():
    rng, img = np.random.default_rng(2), _scene()
    h = imagehash.dhash(img)
    assert imagehash.dhash(im.reshare(img, rng)) - h <= 8
    shot = im.screenshot(img, rng)
    assert shot.size == im.SCREEN and imagehash.dhash(shot) - h > 8
    view = im.photo_view(img, rng)
    lo, hi = im.VIEW_SCALE
    assert lo * 320 - 1 <= view.width <= hi * 320 + 1
    assert lo * 240 - 1 <= view.height <= hi * 240 + 1


def test_yolo_boxes(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n3 0.05 0.9 0.2 0.4\n")
    assert np.allclose(im.yolo_boxes(p), [(0.4, 0.3, 0.6, 0.7), (0.0, 0.7, 0.15, 1.0)])
    assert im.yolo_boxes(tmp_path / "missing.txt") == []


# ---------- the generated dataset, if present ----------

REAL = ds.OUT_DIR / "reports.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="run python -m verifier.dataset first")
def test_generated_dataset():
    reports = pd.read_parquet(REAL)
    pairs = pd.read_parquet(ds.OUT_DIR / "pairs.parquet")
    split = reports.set_index("report_id").split
    assert pairs.is_duplicate.sum() == 484  # every in-window pair from the 3000-row corpus
    train = pairs[pairs.split == "train"]
    assert (split[train.a_id] == "train").all() and (split[train.b_id] == "train").all()
    assert (reports.spam_kind.value_counts()[["text", *ds.FAKE_KINDS]] == 150).all()
    assert all((ROOT / p).exists() for p in reports.image_path.dropna().sample(50, random_state=0))
