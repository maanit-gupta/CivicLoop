"""Step 5.1: build the vision image dataset (YOLO format) from two public sources.

Sources (licences and citations in data/vision/SOURCES.md):
- RDD2022 India: dashcam road photos, partly from Delhi NCR. Pothole boxes (class D40), and
  photos with no damage at all, which become the "none" images.
- Urban Issues Dataset (Kaggle, mostly Roboflow exports): potholes, garbage, broken road signs,
  damaged electric poles/wires, plus fallen trees / graffiti / illegal parking / dead animals,
  which become "other".

Every image gets one image-level label (an IssueType value, or "none") for the 5.2 accuracy, and
zero or more boxes in the 4 detection classes for the YOLO mAP. "other" and "none" images are
YOLO background images: no boxes.

Leakage guard: Roboflow exports contain several augmented copies of one photo
("<photo>_jpg.rf.<hash>.jpg"), and some photos appear in more than one class folder. We keep one
image per original photo and drop exact pixel duplicates (dHash), so no photo can sit in both
train and test.

Run: python -m vision.dataset   (reads data/vision/raw/, writes data/vision/yolo/)
"""

import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

from config import DATA_DIR, SEED
from contracts.schemas import IssueType

RAW_DIR = DATA_DIR / "vision" / "raw"
OUT_DIR = DATA_DIR / "vision" / "yolo"

# YOLO detection classes, id = list index.
CLASSES = ["pothole", "garbage", "damaged_sign", "damaged_pole"]
# What each detection class means as a complaint. damaged_pole -> streetlight: there is no public
# "streetlight not working" set; a broken pole/wire is the nearest thing Electrical handles.
CLASS_TO_ISSUE = {
    "pothole": IssueType.POTHOLE,
    "garbage": IssueType.GARBAGE,
    "damaged_sign": IssueType.DAMAGED_SIGN,
    "damaged_pole": IssueType.STREETLIGHT,
}
NONE_LABEL = "none"  # a photo with no visible civic issue (ImageFeatures.issue_type = None)
LABELS = [*(i.value for i in CLASS_TO_ISSUE.values()), IssueType.OTHER.value, NONE_LABEL]

# Urban Issues folders. Label files use the dataset's global ids (0 road cracks, 1 pothole,
# 2 illegal parking, 3 broken sign, 4 fallen tree, 5 garbage, 6 graffiti, 7 dead animal,
# 9 damaged pole/wires). Per folder: (ids allowed in this folder, ids kept as our classes).
# An image with any other id is skipped: spot checks showed those strays are mislabels (e.g.
# "pothole" boxes on a graffiti figure). Folders with nothing kept become "other" (no boxes).
# Not used: "Damaged concrete structures" (looks like road damage, blurs pothole vs other).
URBAN_FOLDERS = {
    "Potholes and RoadCracks": ({0, 1}, {1: "pothole"}),  # crack boxes dropped, crack-only skipped
    "Garbage": ({5}, {5: "garbage"}),
    "DamagedRoadSigns": ({3}, {3: "damaged_sign"}),
    "DamagedElectricalPoles": ({9}, {9: "damaged_pole"}),
    "FallenTrees": ({4}, {}),
    "Graffitti": ({6}, {}),
    "IllegalParking": ({2}, {}),
    "DeadAnimalsPollution": ({7}, {}),
}

MAX_PER_LABEL = 600  # keeps CPU evaluation (CLIP, 5.2) and the GPU fine-tune (5.3) short
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_BOX_SIDE = 0.005  # normalised; anything thinner is an annotation slip


# ---------- small, testable pieces ----------

def original_stem(filename: str) -> str:
    """The original photo name behind a Roboflow export: 'img_12_jpg.rf.ab12.jpg' -> 'img_12'."""
    stem = filename.split(".rf.")[0]
    for ext in ("_jpg", "_jpeg", "_png", "_JPG", "_JPEG", "_PNG"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return Path(stem).stem


def dhash(path: Path, size: int = 8) -> str:
    """Difference hash: survives resizing and re-encoding, so it catches re-uploaded copies."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = img.tobytes()  # one byte per pixel, row by row
    bits = (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1]
            for r in range(size) for c in range(size))
    return f"{sum(b << i for i, b in enumerate(bits)):016x}"


Box = tuple[float, float, float, float]


def _yolo_box(x1: float, y1: float, x2: float, y2: float) -> Box | None:
    """Corners in 0..1 -> (cx, cy, w, h), clipped; None if the box is degenerate."""
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def voc_boxes(xml_text: str) -> tuple[list[Box], set[str]]:
    """RDD Pascal-VOC XML -> (pothole boxes in YOLO form, set of every damage code present)."""
    root = ET.fromstring(xml_text)
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    boxes, codes = [], set()
    for obj in root.iter("object"):
        code = obj.findtext("name").strip()
        codes.add(code)
        if code != "D40":
            continue
        b = obj.find("bndbox")
        box = _yolo_box(*(float(b.findtext(k)) / d for k, d in
                          (("xmin", w), ("ymin", h), ("xmax", w), ("ymax", h))))
        if box:
            boxes.append(box)
    return boxes, codes


def yolo_line_box(line: str) -> Box | None:
    """One Roboflow label line -> YOLO box. Handles plain boxes (5 numbers) and segmentation
    polygons (class + x y pairs), which Roboflow exports for some sets."""
    vals = [float(v) for v in line.split()[1:]]
    if len(vals) == 4:
        cx, cy, w, h = vals
        return _yolo_box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        return _yolo_box(min(xs), min(ys), max(xs), max(ys))
    return None


def split_items(items: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    """Stratified split by image-level label. Items are already one-per-photo, so a plain
    per-label shuffle cannot leak a photo across splits."""
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    for label in sorted({it["label"] for it in items}):
        group = sorted((it for it in items if it["label"] == label), key=lambda it: it["key"])
        rng.shuffle(group)
        n_train = round(len(group) * SPLITS["train"])
        n_val = round(len(group) * SPLITS["val"])
        out["train"] += group[:n_train]
        out["val"] += group[n_train:n_train + n_val]
        out["test"] += group[n_train + n_val:]
    return out


# ---------- the two sources ----------

def collect_rdd(root: Path) -> tuple[list[dict], Counter]:
    """RDD2022 India train set: images with a pothole -> 'pothole'; with no damage -> 'none'.
    Images with only other damage (cracks, faded markings, manholes) are skipped."""
    stats = Counter()
    items = []
    for xml in sorted((root / "India" / "train" / "annotations" / "xmls").glob("*.xml")):
        boxes, codes = voc_boxes(xml.read_text())
        image = root / "India" / "train" / "images" / f"{xml.stem}.jpg"
        if boxes:
            label = IssueType.POTHOLE.value
        elif not codes:
            label = NONE_LABEL
        else:
            stats["rdd_skipped_other_damage"] += 1
            continue
        items.append({"source": "rdd2022_india", "path": image, "key": f"rdd/{xml.stem}",
                      "label": label, "boxes": [(0, *b) for b in boxes]})
    return items, stats


def collect_urban(root: Path) -> tuple[list[dict], Counter]:
    """Urban Issues: one image per original photo, boxes remapped to our ids. Roboflow augments
    only its train folder, so a photo's valid/test copy (unaugmented) is preferred."""
    stats = Counter()
    items, seen = [], set()
    rank = {"valid": 0, "test": 1, "train": 2}
    for folder, (allowed, keep) in URBAN_FOLDERS.items():
        images = sorted((root / folder).rglob("images/*.jpg"),
                        key=lambda p: (rank.get(p.parent.parent.name, 3), p.name))
        stats[f"urban_files/{folder}"] = len(images)
        for image in images:
            # Key on the original photo across ALL folders: the same photo filed under two
            # classes is treated as one photo and kept once.
            key = f"urban/{original_stem(image.name)}"
            if key in seen:
                stats["urban_dropped_aug_or_crossfolder_copy"] += 1
                continue
            seen.add(key)
            label_file = image.parent.parent / "labels" / f"{image.stem}.txt"
            lines = [ln for ln in (label_file.read_text().splitlines()
                                   if label_file.exists() else []) if ln.strip()]
            ids = [int(ln.split()[0]) for ln in lines]
            if not set(ids) <= allowed:
                stats["urban_dropped_foreign_class_box"] += 1
                continue
            boxes = [(CLASSES.index(keep[i]), *b) for i, ln in zip(ids, lines, strict=True)
                     if i in keep and (b := yolo_line_box(ln))]
            if keep and not boxes:
                stats["urban_dropped_no_target_box"] += 1  # e.g. a crack-only road photo
                continue
            label = CLASS_TO_ISSUE[CLASSES[boxes[0][0]]].value if keep else IssueType.OTHER.value
            items.append({"source": "urban_issues", "path": image, "key": key,
                          "label": label, "boxes": boxes})
    return items, stats


# ---------- build ----------

def _dedupe_pixels(items: list[dict], stats: Counter) -> list[dict]:
    """Drop images whose dHash matches an earlier one (same photo re-uploaded under a new name)."""
    kept, seen = [], set()
    for it in items:
        it["dhash"] = dhash(it["path"])
        if it["dhash"] in seen:
            stats["dropped_pixel_duplicate"] += 1
            continue
        seen.add(it["dhash"])
        kept.append(it)
    return kept


def _cap(items: list[dict], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for label in LABELS:
        group = sorted((it for it in items if it["label"] == label), key=lambda it: it["key"])
        rng.shuffle(group)
        out += group[:MAX_PER_LABEL]
    return out


def _write(splits: dict[str, list[dict]], out_dir: Path, raw_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rows = []
    for split, items in splits.items():
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)
        for it in sorted(items, key=lambda it: it["key"]):
            image_id = f"{it['source']}_{len(rows):05d}"
            dst = out_dir / "images" / split / f"{image_id}.jpg"
            if it["path"].suffix.lower() in (".jpg", ".jpeg"):
                shutil.copyfile(it["path"], dst)
            else:
                Image.open(it["path"]).convert("RGB").save(dst, quality=95)
            # Background images still get an (empty) label file: explicit beats implicit.
            (out_dir / "labels" / split / f"{image_id}.txt").write_text("".join(
                f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in it["boxes"]))
            rows.append({"image_id": image_id, "split": split, "label": it["label"],
                         "n_boxes": len(it["boxes"]), "source": it["source"],
                         "source_key": it["key"], "dhash": it["dhash"],
                         "source_path": it["path"].relative_to(raw_dir).as_posix()})
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # No 'path:' key: Ultralytics then resolves splits relative to this file, so the same folder
    # works locally and on Colab/Kaggle.
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out_dir / "data.yaml").write_text(
        f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")


def build(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> Counter:
    rdd, stats = collect_rdd(raw_dir / "rdd2022_india")
    urban, ustats = collect_urban(raw_dir / "urban_issues")
    stats.update(ustats)
    items = _dedupe_pixels(rdd + urban, stats)
    stats["available_after_dedupe"] = len(items)
    splits = split_items(_cap(items))
    _write(splits, out_dir, raw_dir)
    return stats


def source_split_leakage(root: Path = RAW_DIR / "urban_issues") -> tuple[int, int]:
    """The trivial alternative to our split: reuse the source's own train/test folders.
    Returns (test images whose original photo is also in train, test images)."""
    train = {original_stem(p.name) for f in URBAN_FOLDERS
             for p in (root / f).rglob("train/images/*.jpg")}
    test = [original_stem(p.name) for f in URBAN_FOLDERS
            for p in (root / f).rglob("test/images/*.jpg")]
    return sum(s in train for s in test), len(test)


def summary(out_dir: Path = OUT_DIR) -> str:
    """Images per label x split, boxes per class, and the cross-split duplicate check."""
    rows = list(csv.DictReader(open(out_dir / "manifest.csv")))
    counts = Counter((r["label"], r["split"]) for r in rows)
    lines = [f"{'label':<14}" + "".join(f"{s:>7}" for s in SPLITS) + f"{'total':>7}"]
    for label in LABELS:
        n = [counts[label, s] for s in SPLITS]
        lines.append(f"{label:<14}" + "".join(f"{v:>7}" for v in n) + f"{sum(n):>7}")
    lines.append(f"{'all':<14}" + "".join(f"{sum(counts[lb, s] for lb in LABELS):>7}"
                                          for s in SPLITS) + f"{len(rows):>7}")
    boxes = Counter()
    for r in rows:
        txt = (out_dir / "labels" / r["split"] / f"{r['image_id']}.txt").read_text()
        boxes.update(CLASSES[int(ln.split()[0])] for ln in txt.splitlines())
    lines.append("boxes: " + ", ".join(f"{c} {boxes[c]}" for c in CLASSES))
    by_split = {s: {r["dhash"] for r in rows if r["split"] == s} for s in SPLITS}
    shared = len(by_split["train"] & (by_split["val"] | by_split["test"]))
    lines.append(f"photos shared between train and val/test (dHash): {shared}")
    return "\n".join(lines)


if __name__ == "__main__":
    stats = build()
    for k, v in sorted(stats.items()):
        print(f"{k}: {v}")
    leaked, n_test = source_split_leakage()
    print(f"baseline (source's own split): {leaked}/{n_test} test photos also in train "
          f"({leaked / n_test:.0%})")
    print(summary())
