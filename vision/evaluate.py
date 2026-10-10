"""Step 5.2: evaluation for the vision module, and the CLIP zero-shot baseline.

Two questions, two metrics, both on a held-out split of data/vision/yolo/ (5.1):
- Which issue does the photo show? Accuracy and macro-F1 over the 6 image labels
  (pothole, streetlight, garbage, damaged_sign, other, none). Macro-F1 because damaged_sign has
  fewer photos; it weights every label equally.
- Where is it? mAP50 over the 4 detection classes. A predicted box is correct when it overlaps an
  unmatched true box of the same class with IoU >= 0.5. Per class, rank all predicted boxes by
  confidence, trace precision against recall, and take the area under that curve (AP, all-point
  interpolation as in PASCAL VOC). mAP50 = mean AP over classes. Ultralytics reports a 101-point
  COCO-style variant; it differs slightly, so 5.3 scores YOLO with THIS code for a fair compare.

Everything here scores plain predictions, so any model (CLIP now, YOLO in 5.3) plugs in.

Baselines: majority label (the trivial floor); CLIP zero-shot (no training at all). CLIP gives
no boxes, so for mAP50 it predicts one full-image box per class, scored by its probability: this
only scores when the issue fills the frame, which is the floor a real detector must beat.

YOLO (5.3) gives boxes; its image label is the most confident box's issue if that box clears a
threshold, else "none". The threshold is picked on val (best macro-F1), then applied to test.

5.4 combines them (vision.features): YOLO names the issue when a box clears the threshold, else
a fallback rule asks CLIP, picked on val. It also checks the CLIP embedding: does a photo stay
closer to a re-shot copy of itself than to another photo of the same issue? dHash is the
trivial baseline there.

5.5 masks faces and number plates (vision.privacy) and asks three things. Privacy: of the
faces/plates big enough to identify, how many end up >= 80% blurred (WIDER FACE for faces,
Indian dashcam plates; each split into tune/test by scene or video). Harm: how much of the civic
issue's own box gets blurred, on our photos. Utility: do the 5.4 labels survive masking?
Trivial baselines: mask nothing (privacy 0) and blur the whole photo (privacy 1, labels lost).

Run: python -m vision.evaluate [--split test|val]   (~1 min on CPU for 502 photos)
     python -m vision.evaluate --model yolo [--weights models/vision/yolo_v1/best.pt]
     python -m vision.evaluate --model features   (5.4, ~3 min)
     python -m vision.evaluate --model privacy    (5.5, ~8 min)
"""

import argparse
import csv
import json
import random
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from config import DATA_DIR, SEED
from vision import privacy
from vision.dataset import CLASS_TO_ISSUE, CLASSES, LABELS, NONE_LABEL, OUT_DIR, dhash

EVAL_DIR = DATA_DIR / "vision" / "eval"
IOU_THRESHOLD = 0.5

Box = tuple[float, float, float, float]  # YOLO form: cx, cy, w, h, all normalised to 0..1
GroundTruth = dict[str, list[tuple[int, float, float, float, float]]]  # id -> [(cls, *box)]
Detections = dict[str, list[tuple[int, float, float, float, float, float]]]  # [(cls, conf, *box)]


# ---------- data ----------

def load_split(split: str, data_dir: Path = OUT_DIR) -> list[dict]:
    """Manifest rows of one split, each with its image path and true boxes."""
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv")) if r["split"] == split]
    for r in rows:
        r["path"] = data_dir / "images" / split / f"{r['image_id']}.jpg"
        txt = (data_dir / "labels" / split / f"{r['image_id']}.txt").read_text()
        r["boxes"] = [(int(v[0]), *map(float, v[1:5]))
                      for v in (ln.split() for ln in txt.splitlines()) if v]
    return rows


# ---------- image-level classification ----------

def classification_report(y_true: list[str], y_pred: list[str],
                          labels: list[str] = LABELS) -> dict:
    """Accuracy, macro-F1, per-label precision/recall/F1, and the confusion counts."""
    pairs = Counter(zip(y_true, y_pred, strict=True))
    per_label = {}
    for lb in labels:
        tp = pairs[lb, lb]
        n_pred = sum(n for (_, p), n in pairs.items() if p == lb)
        n_true = sum(n for (t, _), n in pairs.items() if t == lb)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_label[lb] = {"precision": prec, "recall": rec, "f1": f1, "support": n_true}
    return {
        "accuracy": sum(pairs[lb, lb] for lb in labels) / len(y_true),
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(labels),
        "per_label": per_label,
        "confusion": {t: {p: pairs[t, p] for p in labels} for t in labels},
    }


def majority_label(train_labels: list[str]) -> str:
    """The trivial baseline: always answer the most common training label (ties: alphabetical)."""
    counts = Counter(train_labels)
    return min(counts, key=lambda lb: (-counts[lb], lb))


# ---------- detection ----------

def iou(a: Box, b: Box) -> float:
    """Intersection over union of two (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def average_precision(hits: list[bool], n_true: int) -> float:
    """AP from predictions already sorted by confidence (True = matched a true box).
    All-point interpolation: precision at each recall is the best precision at that recall
    or higher, and AP is the area under that stepped curve."""
    if n_true == 0:
        return 0.0
    tp = fp = 0
    recall, precision = [0.0], [0.0]
    for hit in hits:
        tp, fp = tp + hit, fp + (not hit)
        recall.append(tp / n_true)
        precision.append(tp / (tp + fp))
    for i in range(len(precision) - 2, -1, -1):  # envelope: never lower than to the right
        precision[i] = max(precision[i], precision[i + 1])
    return sum((recall[i] - recall[i - 1]) * precision[i] for i in range(1, len(recall)))


def map50(truth: GroundTruth, preds: Detections, n_classes: int = len(CLASSES)) -> dict:
    """Per-class AP at IoU 0.5 and recall at the lowest confidence; mAP50 over classes that
    have at least one true box. Each true box can be matched once: a second box on the same
    pothole is a false positive, as in VOC/COCO."""
    per_class = {}
    for c in range(n_classes):
        true_boxes = {i: [b[1:] for b in bs if b[0] == c] for i, bs in truth.items()}
        n_true = sum(len(bs) for bs in true_boxes.values())
        used = {i: [False] * len(bs) for i, bs in true_boxes.items()}
        ranked = sorted(((d[1], i, d[2:]) for i, ds in preds.items() for d in ds if d[0] == c),
                        key=lambda x: (-x[0], x[1]))
        hits = []
        for _, image_id, box in ranked:
            cands = true_boxes.get(image_id, [])
            scores = [iou(box, t) for t in cands]
            best = max(range(len(cands)), key=scores.__getitem__, default=None)
            ok = best is not None and scores[best] >= IOU_THRESHOLD and not used[image_id][best]
            if ok:
                used[image_id][best] = True
            hits.append(ok)
        per_class[CLASSES[c]] = {"ap50": average_precision(hits, n_true),
                                 "recall": sum(hits) / n_true if n_true else 0.0,
                                 "n_true": n_true, "n_pred": len(hits)}
    scored = [v["ap50"] for v in per_class.values() if v["n_true"]]
    return {"map50": sum(scored) / len(scored) if scored else 0.0, "per_class": per_class}


def full_image_boxes(probs: list[dict[str, float]], image_ids: list[str]) -> Detections:
    """Turn an image classifier into a (weak) detector: one whole-frame box per detection
    class, with confidence = the classifier's probability for that class's issue type."""
    return {i: [(c, p[CLASS_TO_ISSUE[name].value], 0.5, 0.5, 1.0, 1.0)
                for c, name in enumerate(CLASSES)]
            for i, p in zip(image_ids, probs, strict=True)}


# ---------- detector -> image label ----------

THRESHOLD_GRID = [round(0.05 * k, 2) for k in range(1, 20)]  # 0.05 .. 0.95


def pick_threshold(dets: list[list], y_true: list[str],
                   grid: list[float] = THRESHOLD_GRID) -> tuple[float, float]:
    """The box-confidence threshold whose image labels get the best macro-F1 (ties: the lowest
    threshold). Call it on val only: picking it on test would leak test into the model."""
    from vision.yolo_detect import image_label

    scores = [(classification_report(y_true, [image_label(d, t) for d in dets])["macro_f1"], -t)
              for t in grid]
    f1, neg_t = max(scores)
    return -neg_t, f1


# ---------- reporting ----------

def format_confusion(confusion: dict[str, dict[str, int]]) -> str:
    corner = "true \\ pred"  # no backslashes inside f-string braces on Python 3.11
    head = f"{corner:<13}" + "".join(f"{p[:8]:>9}" for p in confusion)
    return "\n".join([head] + [f"{t:<13}" + "".join(f"{n:>9}" for n in row.values())
                               for t, row in confusion.items()])


def recall_by_source(rows: list[dict], preds: list[str]) -> dict[str, float]:
    """Recall per 'source/label'. Exposes the 5.1 bias: every "none" photo is a dashcam frame,
    so a model that answers "none" for all dashcam frames looks great on "none" alone."""
    out = {}
    for src, lb in sorted({(r["source"], r["label"]) for r in rows}):
        idx = [k for k, r in enumerate(rows) if (r["source"], r["label"]) == (src, lb)]
        out[f"{src}/{lb}"] = sum(preds[k] == lb for k in idx) / len(idx)
    return out


def _write_predictions(path: Path, rows: list[dict], probs: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "source", "label", "pred", *LABELS])
        for r, p in zip(rows, probs, strict=True):
            w.writerow([r["image_id"], r["source"], r["label"], max(p, key=p.get),
                        *(f"{p[lb]:.4f}" for lb in LABELS)])


def main(split: str = "test") -> dict:
    from vision import clip_zeroshot

    rows = load_split(split)
    y_true = [r["label"] for r in rows]
    train = [r["label"] for r in csv.DictReader(open(OUT_DIR / "manifest.csv"))
             if r["split"] == "train"]

    t0 = time.perf_counter()
    probs = clip_zeroshot.classify([r["path"] for r in rows])
    seconds = time.perf_counter() - t0
    clip_pred = [max(p, key=p.get) for p in probs]
    majority = majority_label(train)

    reports = {
        f"majority ({majority})": classification_report(y_true, [majority] * len(rows)),
        "clip zero-shot": classification_report(y_true, clip_pred),
    }
    truth = {r["image_id"]: r["boxes"] for r in rows}
    det = map50(truth, full_image_boxes(probs, list(truth)))

    print(f"split={split}  images={len(rows)}  CLIP {clip_zeroshot.MODEL_VERSION}: "
          f"{seconds:.0f} s on CPU ({1000 * seconds / len(rows):.0f} ms/photo)\n")
    print(f"{'baseline':<22}{'accuracy':>9}{'macro-F1':>9}")
    for name, rep in reports.items():
        print(f"{name:<22}{rep['accuracy']:>9.3f}{rep['macro_f1']:>9.3f}")

    clip = reports["clip zero-shot"]
    print(f"\nCLIP per label\n{'label':<14}{'prec':>6}{'recall':>8}{'F1':>6}{'n':>5}")
    for lb, v in clip["per_label"].items():
        print(f"{lb:<14}{v['precision']:>6.2f}{v['recall']:>8.2f}{v['f1']:>6.2f}"
              f"{v['support']:>5}")
    print("\n" + format_confusion(clip["confusion"]))

    by_source = recall_by_source(rows, clip_pred)
    n = Counter(f"{r['source']}/{r['label']}" for r in rows)
    print(f"\nCLIP recall by source\n{'source/label':<30}{'recall':>7}{'n':>5}")
    for key, rec in by_source.items():
        print(f"{key:<30}{rec:>7.2f}{n[key]:>5}")

    print(f"\nmAP50, CLIP + full-image box: {det['map50']:.3f}")
    print(f"{'class':<14}{'AP50':>6}{'max recall':>12}{'true boxes':>12}")
    for c, v in det["per_class"].items():
        print(f"{c:<14}{v['ap50']:>6.3f}{v['recall']:>12.3f}{v['n_true']:>12}")

    result = {
        "split": split, "n_images": len(rows), "model_version": clip_zeroshot.MODEL_VERSION,
        "ms_per_photo": 1000 * seconds / len(rows), "classification": reports,
        "clip_recall_by_source": by_source, "detection_clip_full_image": det,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"clip_zeroshot_{split}.json").write_text(json.dumps(result, indent=2))
    _write_predictions(EVAL_DIR / f"clip_zeroshot_{split}.csv", rows, probs)
    return result


def main_yolo(split: str = "test", weights: Path | None = None) -> dict:
    """5.3: score the fine-tuned YOLO on `split`, next to the saved 5.2 baselines."""
    from vision import yolo_detect

    weights = weights or yolo_detect.WEIGHTS
    clip_path = EVAL_DIR / f"clip_zeroshot_{split}.json"
    if not clip_path.exists():
        raise FileNotFoundError(f"{clip_path} is missing: run "
                                f"`python -m vision.evaluate --split {split}` (5.2) first")
    clip = json.loads(clip_path.read_text())
    clip_rep = clip["classification"]["clip zero-shot"]
    clip_det = clip["detection_clip_full_image"]["per_class"]

    def run(rows):
        t0 = time.perf_counter()
        dets = yolo_detect.detect([r["path"] for r in rows], weights)
        return dets, 1000 * (time.perf_counter() - t0) / len(rows)

    val = load_split("val")
    val_dets, ms = run(val)
    threshold, val_f1 = pick_threshold(val_dets, [r["label"] for r in val])
    rows, dets = val, val_dets
    if split != "val":
        rows = load_split(split)
        dets, ms = run(rows)

    y_true = [r["label"] for r in rows]
    y_pred = [yolo_detect.image_label(d, threshold) for d in dets]
    rep = classification_report(y_true, y_pred)
    det = map50({r["image_id"]: r["boxes"] for r in rows},
                {r["image_id"]: d for r, d in zip(rows, dets, strict=True)})
    by_source = recall_by_source(rows, y_pred)

    print(f"split={split}  images={len(rows)}  {yolo_detect.MODEL_VERSION}: {ms:.0f} ms/photo "
          f"on CPU; label threshold {threshold:.2f} (best val macro-F1 {val_f1:.3f})\n")
    print(f"{'model':<22}{'accuracy':>9}{'macro-F1':>9}{'mAP50':>7}{'ms/photo':>9}")
    for name, r in clip["classification"].items():
        cells = (f"{clip['detection_clip_full_image']['map50']:>7.3f}{clip['ms_per_photo']:>9.0f}"
                 if name == "clip zero-shot" else f"{'-':>7}{'-':>9}")
        print(f"{name:<22}{r['accuracy']:>9.3f}{r['macro_f1']:>9.3f}{cells}")
    print(f"{'yolo v1':<22}{rep['accuracy']:>9.3f}{rep['macro_f1']:>9.3f}"
          f"{det['map50']:>7.3f}{ms:>9.0f}")

    print(f"\nper label{'':<5}{'CLIP F1':>8}{'YOLO prec':>10}{'recall':>7}{'F1':>6}{'n':>5}")
    for lb, v in rep["per_label"].items():
        print(f"{lb:<14}{clip_rep['per_label'][lb]['f1']:>8.2f}{v['precision']:>10.2f}"
              f"{v['recall']:>7.2f}{v['f1']:>6.2f}{v['support']:>5}")

    print(f"\nper class{'':<5}{'CLIP AP50':>10}{'YOLO AP50':>10}{'max recall':>11}{'boxes':>7}")
    for c, v in det["per_class"].items():
        print(f"{c:<14}{clip_det[c]['ap50']:>10.3f}{v['ap50']:>10.3f}{v['recall']:>11.3f}"
              f"{v['n_true']:>7}")

    print("\nYOLO confusion\n" + format_confusion(rep["confusion"]))
    n = Counter(f"{r['source']}/{r['label']}" for r in rows)
    print(f"\nrecall by source{'':<14}{'CLIP':>6}{'YOLO':>6}{'n':>5}")
    for key, rec in by_source.items():
        print(f"{key:<30}{clip['clip_recall_by_source'][key]:>6.2f}{rec:>6.2f}{n[key]:>5}")

    result = {"split": split, "n_images": len(rows), "model_version": yolo_detect.MODEL_VERSION,
              "weights": str(weights), "ms_per_photo": ms, "label_threshold": threshold,
              "val_macro_f1_at_threshold": val_f1, "classification": rep,
              "recall_by_source": by_source, "detection": det}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"yolo_v1_{split}.json").write_text(json.dumps(result, indent=2))
    with open(EVAL_DIR / f"yolo_v1_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "source", "label", "pred", "top_class", "top_conf"])
        for r, d, p in zip(rows, dets, y_pred, strict=True):
            top = max(d, key=lambda x: x[1], default=None)
            w.writerow([r["image_id"], r["source"], r["label"], p,
                        CLASSES[top[0]] if top else "", f"{top[1]:.4f}" if top else ""])
    return result


# ---------- 5.4: YOLO + CLIP fallback, and the embedding ----------

def pick_fallback(outputs: list[tuple], y_true: list[str]) -> tuple[str, dict[str, float]]:
    """The fallback rule with the best macro-F1 (ties: the first listed, the simplest), and
    every rule's score. Call it on val only."""
    from vision.features import FALLBACK_RULES, hybrid_label

    scores = {rule: classification_report(
        y_true, [hybrid_label(d, p, rule) for d, p, _ in outputs])["macro_f1"]
        for rule in FALLBACK_RULES}
    return max(scores, key=scores.get), scores


def perturb(path: Path, rng: random.Random) -> bytes:
    """The same scene shot a second time, roughly: a random 85% crop scaled back up,
    brightness x0.8-1.2, saved as JPEG quality 70."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    w, h = im.size
    cw, ch = round(0.85 * w), round(0.85 * h)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x, y, x + cw, y + ch)).resize((w, h))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.2))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def auc(pos: list[float], neg: list[float]) -> float:
    """How often a positive pair outscores a negative pair (ties count half). 0.5 = chance."""
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def dhash_similarity(a: str, b: str) -> float:
    """1 - share of differing bits between two 64-bit dHashes (vision.dataset.dhash)."""
    return 1 - bin(int(a, 16) ^ int(b, 16)).count("1") / 64


def embedding_check(rows: list[dict], emb: np.ndarray, seed: int = SEED) -> dict:
    """Three pairs per photo: 'same scene' (the photo vs a perturbed copy), 'same label' (vs
    another photo of the same issue: the hard negative), 'other label'. Mean similarity per
    pair type, and AUC of same scene vs each negative, for CLIP cosine and the dHash baseline."""
    from vision.embed import embed_images

    rng = random.Random(seed)
    copies = [perturb(r["path"], rng) for r in rows]
    emb_copy = embed_images([BytesIO(b) for b in copies])
    hashes = [dhash(r["path"]) for r in rows]
    hash_copy = [dhash(BytesIO(b)) for b in copies]

    partner = {"same label": [], "other label": []}
    for i, r in enumerate(rows):
        same = [k for k, o in enumerate(rows) if o["label"] == r["label"] and k != i]
        other = [k for k, o in enumerate(rows) if o["label"] != r["label"]]
        partner["same label"].append(rng.choice(same))
        partner["other label"].append(rng.choice(other))

    n = range(len(rows))
    sims = {
        "clip cosine": {"same scene": [float(emb[i] @ emb_copy[i]) for i in n],
                        **{k: [float(emb[i] @ emb[j]) for i, j in zip(n, js, strict=True)]
                           for k, js in partner.items()}},
        "dhash (baseline)": {"same scene": [dhash_similarity(hashes[i], hash_copy[i]) for i in n],
                             **{k: [dhash_similarity(hashes[i], hashes[j])
                                    for i, j in zip(n, js, strict=True)]
                                for k, js in partner.items()}},
    }
    return {method: {"mean": {k: float(np.mean(v)) for k, v in s.items()},
                     "p05_same_scene": float(np.percentile(s["same scene"], 5)),
                     "p95_same_label": float(np.percentile(s["same label"], 95)),
                     "auc_vs_same_label": auc(s["same scene"], s["same label"]),
                     "auc_vs_other_label": auc(s["same scene"], s["other label"])}
            for method, s in sims.items()}


def main_features(split: str = "test") -> dict:
    """5.4: the fallback rule (picked on val), ImageFeatures, and the embedding check."""
    from vision import features
    from vision.yolo_detect import LABEL_THRESHOLD, image_label

    def run(rows):
        t0 = time.perf_counter()
        out = features.run_models([r["path"] for r in rows])
        return out, 1000 * (time.perf_counter() - t0) / len(rows)

    val = load_split("val")
    val_out, ms = run(val)
    rule, val_scores = pick_fallback(val_out, [r["label"] for r in val])
    rows, out = val, val_out
    if split != "val":
        rows = load_split(split)
        out, ms = run(rows)
    y_true = [r["label"] for r in rows]

    # Regression: 5.4 moved CLIP into vision.embed; YOLO and CLIP must answer exactly as before.
    def saved(name):
        return {r["image_id"]: r for r in csv.DictReader(open(EVAL_DIR / f"{name}_{split}.csv"))}

    saved_clip, saved_yolo = saved("clip_zeroshot"), saved("yolo_v1")
    clip_diff = max(abs(p[lb] - float(saved_clip[r["image_id"]][lb]))
                    for r, (_, p, _) in zip(rows, out, strict=True) for lb in LABELS)
    yolo_same = sum(image_label(d, LABEL_THRESHOLD) == saved_yolo[r["image_id"]]["pred"]
                    for r, (d, _, _) in zip(rows, out, strict=True)) / len(rows)

    reports = {name: classification_report(y_true, [features.hybrid_label(d, p, name)
                                                    for d, p, _ in out])
               for name in features.FALLBACK_RULES}
    clip_only = classification_report(y_true, [max(p, key=p.get) for _, p, _ in out])
    feats = [features.build(str(r["path"]), d, p, e, rule=rule)
             for r, (d, p, e) in zip(rows, out, strict=True)]
    built = [f.issue_type.value if f.issue_type else NONE_LABEL for f in feats]
    assert built == [features.hybrid_label(d, p, rule) for d, p, _ in out]

    by_label = {lb: [f for f, t in zip(feats, y_true, strict=True) if t == lb] for lb in LABELS}
    derived = {lb: {"detection_count": float(np.mean([f.detection_count for f in fs])),
                    "box_area_ratio": float(np.mean([f.box_area_ratio for f in fs])),
                    "detector_confidence": float(np.mean([f.detector_confidence for f in fs])),
                    "n": len(fs)} for lb, fs in by_label.items()}
    emb_check = embedding_check(rows, np.array([e for _, _, e in out], dtype=np.float32))

    print(f"split={split}  images={len(rows)}  {features.MODEL_VERSION}: {ms:.0f} ms/photo "
          f"on CPU (YOLO + CLIP)\nregression: CLIP probs max |diff| vs 5.2 = {clip_diff:.4f}, "
          f"YOLO labels equal to 5.3 = {yolo_same:.3f}\n")
    print(f"{f'fallback when no box >= {LABEL_THRESHOLD}':<32}"
          f"{'val F1':>7}{f'{split} acc':>9}{f'{split} F1':>8}")
    print(f"{'(clip alone, 5.2)':<32}{'':>7}{clip_only['accuracy']:>9.3f}"
          f"{clip_only['macro_f1']:>8.3f}")
    for name, rep in reports.items():
        mark = "  <- picked on val" if name == rule else ""
        print(f"{name:<32}{val_scores[name]:>7.3f}{rep['accuracy']:>9.3f}"
              f"{rep['macro_f1']:>8.3f}{mark}")

    picked = reports[rule]
    print(f"\nper label{'':<5}{'YOLO F1':>8}{'picked prec':>12}{'recall':>7}{'F1':>6}{'n':>5}")
    for lb, v in picked["per_label"].items():
        print(f"{lb:<14}{reports['yolo only']['per_label'][lb]['f1']:>8.2f}"
              f"{v['precision']:>12.2f}{v['recall']:>7.2f}{v['f1']:>6.2f}{v['support']:>5}")
    print("\nconfusion (picked)\n" + format_confusion(picked["confusion"]))

    print(f"\n{'derived features, mean by true label':<38}"
          f"{'boxes':>6}{'area':>6}{'conf':>6}{'n':>5}")
    for lb, v in derived.items():
        print(f"{lb:<38}{v['detection_count']:>6.2f}{v['box_area_ratio']:>6.2f}"
              f"{v['detector_confidence']:>6.2f}{v['n']:>5}")

    print(f"\nembedding: same scene vs ...{'':<6}{'same scene':>11}{'same label':>11}"
          f"{'other lbl':>10}{'AUC same lbl':>13}{'AUC other':>10}")
    for method, v in emb_check.items():
        m = v["mean"]
        print(f"{method:<34}{m['same scene']:>11.3f}{m['same label']:>11.3f}"
              f"{m['other label']:>10.3f}{v['auc_vs_same_label']:>13.3f}"
              f"{v['auc_vs_other_label']:>10.3f}")

    result = {"split": split, "n_images": len(rows), "model_version": features.MODEL_VERSION,
              "ms_per_photo": ms, "label_threshold": LABEL_THRESHOLD, "fallback": rule,
              "val_macro_f1_by_rule": val_scores, "clip_probs_max_diff_vs_5_2": clip_diff,
              "yolo_labels_equal_5_3": yolo_same, "classification_clip_only": clip_only,
              "classification": reports, "derived_by_label": derived, "embedding": emb_check}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"features_v1_{split}.json").write_text(json.dumps(result, indent=2))
    return result


# ---------- privacy masking (5.5) ----------

PRIVACY_DIR = DATA_DIR / "vision" / "raw" / "privacy"
MIN_FACE_PX = 20  # face box height; smaller faces cannot be recognised
MIN_PLATE_PX = 40  # plate box width; ~10 characters, each needs a few pixels to be read
COVER = 0.8  # a face/plate counts as masked when >= 80% of its box is blurred
MATCH_IOU = 0.3  # a detection is right if it overlaps a labelled face/plate this much
PRIVACY_GRID = [0.05, *(round(0.1 * k, 1) for k in range(1, 10))]  # 0.05 .. 0.9
PADS = {"face": [0.15, 0.3, 0.5], "plate": [0.15, 0.3, 0.5, 1.0]}  # see privacy.PAD
PLATE_SIZES = [640, 1280]  # plate model input size: 1280 sees small plates, ~30% slower
MAX_ISSUE_BLURRED = 0.01  # harm limit: blur may hide <= 1% of the issue-box area (civic val)
SIZE_BUCKETS = {"face": [20, 40, 80], "plate": [40, 80]}  # recall by box size (px), test

PxBox = tuple[float, float, float, float]  # pixel corners x1, y1, x2, y2


def load_wider(root: Path = PRIVACY_DIR / "wider_face", seed: int = SEED) -> dict[str, list]:
    """WIDER FACE val, split by event (Parade, Handshaking, ...) into tune/test halves, so a
    threshold is never scored on the scenes it was picked on. Each box: (corners, height px,
    must_mask). WIDER's 'invalid' faces still count for precision but never need masking."""
    lines = (root / "wider_face_split" / "wider_face_val_bbx_gt.txt").read_text().splitlines()
    items, i = [], 0
    while i < len(lines) and lines[i].strip():
        name, n = lines[i].strip(), int(lines[i + 1])
        boxes = []
        for ln in lines[i + 2:i + 2 + max(n, 1)]:  # an image with no face has one line of 0s
            x, y, w, h, *_, invalid, _, _ = map(int, ln.split()[:10])
            if n and w > 0 and h > 0:
                boxes.append(((x, y, x + w, y + h), h, not invalid and h >= MIN_FACE_PX))
        items.append({"path": root / "WIDER_val" / "images" / name,
                      "group": name.split("/")[0], "boxes": boxes})
        i += 2 + max(n, 1)
    groups = sorted({it["group"] for it in items})
    random.Random(seed).shuffle(groups)
    tune = set(groups[:len(groups) // 2])
    return {"tune": [it for it in items if it["group"] in tune],
            "test": [it for it in items if it["group"] not in tune]}


def load_plates(root: Path = PRIVACY_DIR / "plates_india") -> dict[str, list]:
    """Indian number plates from 3 dashcam videos (YOLO labels). Neighbouring frames are near
    copies, so split by video: tune = vid-1, test = vid-2 and vid-3. Box: (corners, width px,
    must_mask)."""
    out = {"tune": [], "test": []}
    for vid in sorted(root.glob("vid-*")):
        for p in sorted(vid.glob("*.jpg")):
            w, h = Image.open(p).size
            boxes = []
            for ln in p.with_suffix(".txt").read_text().splitlines():
                _, cx, cy, bw, bh = map(float, ln.split())
                box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                boxes.append((box, bw * w, bw * w >= MIN_PLATE_PX))
            out["tune" if vid.name == "vid-1" else "test"].append(
                {"path": p, "group": vid.name, "boxes": boxes})
    return out


def iou_px(a: PxBox, b: PxBox) -> float:
    """Intersection over union of two corner boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def blurred_mask(found: list, size: tuple[int, int], pad: dict[str, float]) -> np.ndarray:
    """Which pixels privacy.blur_found would blur, as an h x w bool array."""
    m = np.zeros((size[1], size[0]), dtype=bool)
    for f in found:
        x1, y1, x2, y2 = privacy.padded(f.box, size, pad[f.kind])
        m[y1:y2, x1:x2] = True
    return m


def covered(box: PxBox, m: np.ndarray) -> float:
    """Fraction of a box's pixels that are blurred."""
    h, w = m.shape
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    return float(m[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0


def mask_scores(items: list[dict], found: list[list], threshold: float,
                pad: dict[str, float] | None = None, buckets: list[int] | None = None) -> dict:
    """Recall: share of the faces/plates that must be masked with >= COVER of their box
    blurred (the privacy number). Precision: share of blurred boxes that overlap a labelled
    one (how much is blurred for nothing). Recall by box size if `buckets` are given."""
    pad = pad or privacy.PAD
    need, hit, n_det, n_right = [], [], 0, 0
    for it, fs in zip(items, found, strict=True):
        kept = [f for f in fs if f.score >= threshold]
        m = blurred_mask(kept, it["size"], pad)
        for box, px, must in it["boxes"]:
            if must:
                need.append(px)
                hit.append(covered(box, m) >= COVER)
        n_det += len(kept)
        n_right += sum(any(iou_px(f.box, t[0]) >= MATCH_IOU for t in it["boxes"]) for f in kept)
    out = {"recall": sum(hit) / len(hit) if hit else 0.0, "n_true": len(hit),
           "precision": n_right / n_det if n_det else 1.0, "n_det": n_det}
    if buckets:
        edges = [*buckets, float("inf")]
        out["recall_by_size"] = {}
        for lo, hi in zip(edges, edges[1:], strict=False):
            sel = [h for h, px in zip(hit, need, strict=True) if lo <= px < hi]
            out["recall_by_size"][f"{lo}-{hi}px"] = {
                "recall": float(np.mean(sel)) if sel else 0.0, "n": len(sel)}
    return out


def issue_blurred(rows: list[dict], found: list[list], threshold: float,
                  pad: dict[str, float] | None = None) -> dict:
    """Over-blurring where it hurts, on our own civic photos (5.1): the share of the issue's
    box area that the blur hides, the share of photos with any blur, and the mean blurred
    share of a photo. Real faces/plates in these photos are unlabelled, so no recall here."""
    pad = pad or privacy.PAD
    n_blurred = area = issue_px = issue_hidden = 0.0
    for r, fs in zip(rows, found, strict=True):
        kept = [f for f in fs if f.score >= threshold]
        w, h = r["size"]
        m = blurred_mask(kept, r["size"], pad)
        n_blurred += bool(kept)
        area += m.mean()
        for _, cx, cy, bw, bh in r["boxes"]:
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            issue_px += bw * bh * w * h
            issue_hidden += covered(box, m) * bw * bh * w * h
    return {"issue_blurred": issue_hidden / issue_px if issue_px else 0.0,
            "photos_blurred": n_blurred / len(rows), "area_blurred": area / len(rows)}


def run_detector(items: list[dict], detect) -> tuple[list[list], float]:
    """`detect(img)` on every item (records its size); also the mean ms per photo."""
    found, t = [], 0.0
    for it in items:
        img = privacy.load(it["path"])
        it["size"] = img.size
        t0 = time.perf_counter()
        found.append(detect(img))
        t += time.perf_counter() - t0
    return found, 1000 * t / max(1, len(items))


def sweep_mask(civic: list[dict], civic_found: list[list], labelled: list[dict],
               found: list[list], kind: str, **tag) -> list[dict]:
    """Every (pad, threshold): recall/precision on the labelled tune half, harm on civic val."""
    return [{**tag, "pad": pad, "threshold": t,
             **mask_scores(labelled, found, t, {kind: pad}),
             **issue_blurred(civic, civic_found, t, {kind: pad})}
            for pad in PADS[kind] for t in PRIVACY_GRID]


def pick_mask(rows: list[dict]) -> dict:
    """The setting that masks the most (recall) while hiding <= MAX_ISSUE_BLURRED of the civic
    issues; ties: less padding, smaller input, higher threshold (fewer stray blurs). If nothing
    is that harmless, the least harmful."""
    ok = [r for r in rows if r["issue_blurred"] <= MAX_ISSUE_BLURRED]
    if not ok:
        return min(rows, key=lambda r: r["issue_blurred"])
    return max(ok, key=lambda r: (r["recall"], -r["pad"], -r.get("imgsz", 0), r["threshold"]))


def _print_sweep(title: str, rows: list[dict], picked: dict) -> None:
    print(f"\n{title}\n{'threshold':>9}{'recall':>8}{'precision':>10}{'| civic val:':>13}"
          f"{'issue hidden':>13}{'photos':>8}{'area':>7}")
    for r in rows:
        mark = "  <- picked" if r == picked else ""
        print(f"{r['threshold']:>9.2f}{r['recall']:>8.3f}{r['precision']:>10.3f}{'':>13}"
              f"{r['issue_blurred']:>13.4f}{r['photos_blurred']:>8.3f}{r['area_blurred']:>7.4f}"
              f"{mark}")


def _print_settings(title: str, rows: list[dict], picked: dict) -> None:
    """One line per (input size, pad): the best harmless threshold, as pick_mask sees it."""
    print(f"\n{title}\n{'imgsz':>6}{'pad':>6}{'threshold':>10}{'recall':>8}{'precision':>10}"
          f"{'issue hidden':>13}")
    for key in dict.fromkeys((r.get("imgsz"), r["pad"]) for r in rows):
        r = pick_mask([r for r in rows if (r.get("imgsz"), r["pad"]) == key])
        mark = "  <- picked" if r == picked else ""
        print(f"{r.get('imgsz') or '':>6}{r['pad']:>6.2f}{r['threshold']:>10.2f}"
              f"{r['recall']:>8.3f}{r['precision']:>10.3f}{r['issue_blurred']:>13.4f}{mark}")


def main_privacy(split: str = "test") -> dict:
    """5.5: pick each detector's threshold and padding (and the plate input size) on the tune
    halves + civic val, then score on the test halves and on the civic `split`: privacy recall,
    over-blur, and what masking does to the 5.4 labels. Baselines: mask nothing (recall 0) and
    blur the whole photo (recall 1)."""
    import tempfile

    from vision import features

    floor = PRIVACY_GRID[0]
    wider, plates = load_wider(), load_plates()
    civic = {"val": load_split("val"), split: load_split(split)}

    def faces(img):
        return privacy.detect_faces(img, floor)

    def plates_at(sz):  # no size cap here: it is applied below, so its effect can be shown
        return lambda img: privacy.detect_plates(img, floor, imgsz=sz, max_area=1.0)

    def capped(rows, found):
        return [[f for f in fs if privacy.plate_sized(f.box, r["size"])]
                for r, fs in zip(rows, found, strict=True)]

    # det[detector, dataset, part]: detector "face" or a plate imgsz; dataset "labelled"
    # (WIDER for faces, Indian plates for plates) or "civic". ms: per civic photo, CPU.
    det, ms = {}, {}
    for part in ("tune", "test"):
        det["face", "labelled", part], _ = run_detector(wider[part], faces)
    for name, rows in civic.items():
        det["face", "civic", name], ms["face"] = run_detector(rows, faces)
    for sz in PLATE_SIZES:
        for part in ("tune", "test"):
            n_big = sum(not privacy.plate_sized(b[0], Image.open(it["path"]).size)
                        for it in plates[part] for b in it["boxes"])
            assert n_big == 0, f"{n_big} real plates are bigger than MAX_PLATE_AREA"
            raw, _ = run_detector(plates[part], plates_at(sz))
            det[sz, "labelled", part] = capped(plates[part], raw)
        for name, rows in civic.items():
            raw, ms[sz] = run_detector(rows, plates_at(sz))
            det[sz, "uncapped", name], det[sz, "civic", name] = raw, capped(rows, raw)

    face_rows = sweep_mask(civic["val"], det["face", "civic", "val"], wider["tune"],
                           det["face", "labelled", "tune"], "face")
    plate_rows = [r for sz in PLATE_SIZES
                  for r in sweep_mask(civic["val"], det[sz, "civic", "val"], plates["tune"],
                                      det[sz, "labelled", "tune"], "plate", imgsz=sz)]
    face, plate = pick_mask(face_rows), pick_mask(plate_rows)
    pad = {"face": face["pad"], "plate": plate["pad"]}
    plate_sz, face_t, plate_t = plate["imgsz"], face["threshold"], plate["threshold"]

    n_faces = sum(len(it["boxes"]) for it in wider["tune"] + wider["test"])
    print(f"faces: WIDER FACE val {len(wider['tune'])}+{len(wider['test'])} photos "
          f"(tune+test, split by event), {n_faces} faces; must mask = valid and >= "
          f"{MIN_FACE_PX} px tall. plates: {len(plates['tune'])}+{len(plates['test'])} Indian "
          f"dashcam frames (split by video), must mask = >= {MIN_PLATE_PX} px wide. masked = "
          f">= {COVER:.0%} of the box blurred. pick: best tune recall with civic val issue "
          f"hidden <= {MAX_ISSUE_BLURRED:.0%}.")
    _print_settings("FACES (YuNet) on WIDER tune", face_rows, face)
    _print_settings("PLATES (YOLO11n) on Indian plates tune", plate_rows, plate)
    _print_sweep(f"FACES, pad {face['pad']}", [r for r in face_rows if r["pad"] == face["pad"]],
                 face)
    _print_sweep(f"PLATES, imgsz {plate_sz}, pad {plate['pad']}",
                 [r for r in plate_rows if (r["imgsz"], r["pad"]) == (plate_sz, plate["pad"])],
                 plate)
    no_cap = issue_blurred(civic["val"], det[plate_sz, "uncapped", "val"], plate_t, pad)
    print(f"\nplate size cap (<= {privacy.MAX_PLATE_AREA:.0%} of the photo): civic val issue "
          f"hidden {plate['issue_blurred']:.4f} with it, {no_cap['issue_blurred']:.4f} without")

    # ---- test ----
    face_test = mask_scores(wider["test"], det["face", "labelled", "test"], face_t, pad,
                            SIZE_BUCKETS["face"])
    plate_test = mask_scores(plates["test"], det[plate_sz, "labelled", "test"], plate_t, pad,
                             SIZE_BUCKETS["plate"])
    rows = civic[split]
    both = [[f for f in fs if f.score >= face_t] + [p for p in ps if p.score >= plate_t]
            for fs, ps in zip(det["face", "civic", split], det[plate_sz, "civic", split],
                              strict=True)]
    civic_test = issue_blurred(rows, both, 0.0, pad)

    y_true = [r["label"] for r in rows]
    variants = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("no mask", "masked", "blur whole photo"):
            paths = []
            for k, r in enumerate(rows):
                img = privacy.load(r["path"])  # EXIF-rotated, as api.analyze does
                if name == "masked":
                    img = privacy.blur_found(img, both[k], pad)
                elif name == "blur whole photo":
                    img = privacy.blur(img, [(0, 0, *img.size)])
                paths.append(Path(tmp) / f"{name.replace(' ', '_')}_{k}.jpg")
                img.save(paths[-1], quality=privacy.JPEG_QUALITY)
            out = features.run_models(paths)
            variants[name] = {"labels": [features.hybrid_label(d, p, features.FALLBACK)
                                         for d, p, _ in out],
                              "emb": np.array([e for _, _, e in out])}
    ref, utility = variants["no mask"], {}
    for name, var in variants.items():
        rep = classification_report(y_true, var["labels"])
        same = [a == b for a, b in zip(var["labels"], ref["labels"], strict=True)]
        utility[name] = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
                         "labels_equal_no_mask": float(np.mean(same)),
                         "cosine_to_no_mask": float(np.mean(np.sum(var["emb"] * ref["emb"], 1)))}

    print(f"\nTEST  faces: WIDER test half, plates: vid-2+3, civic: {split} ({len(rows)} photos). "
          f"picked: face >= {face_t} pad {pad['face']}, plate >= {plate_t} pad {pad['plate']} "
          f"imgsz {plate_sz}")
    print(f"{'':<18}{'face rec':>9}{'plate rec':>10}{'issue hidden':>13}{'acc':>7}"
          f"{'macro-F1':>9}{'= no mask':>10}{'cosine':>8}")
    recalls = {"no mask": (0.0, 0.0, 0.0), "blur whole photo": (1.0, 1.0, 1.0),
               "masked": (face_test["recall"], plate_test["recall"],
                          civic_test["issue_blurred"])}
    for name, u in utility.items():
        fr, pr, hid = recalls[name]
        print(f"{name:<18}{fr:>9.3f}{pr:>10.3f}{hid:>13.4f}{u['accuracy']:>7.3f}"
              f"{u['macro_f1']:>9.3f}{u['labels_equal_no_mask']:>10.3f}"
              f"{u['cosine_to_no_mask']:>8.3f}")
    print("\nrecall by size (test)  " + "  ".join(
        f"{kind} {b}: {v['recall']:.3f} (n={v['n']})"
        for kind, s in (("face", face_test), ("plate", plate_test))
        for b, v in s["recall_by_size"].items()))
    print(f"precision (test): face {face_test['precision']:.3f}, plate "
          f"{plate_test['precision']:.3f}. civic {split}: {civic_test['photos_blurred']:.1%} of "
          f"photos get a blur, {civic_test['area_blurred']:.2%} of pixels. masking costs "
          f"{ms['face']:.0f} ms (faces) + {ms[plate_sz]:.0f} ms (plates) per civic photo on CPU")

    result = {"split": split, "model_version": privacy.MODEL_VERSION,
              "face_threshold": face_t, "plate_threshold": plate_t, "plate_imgsz": plate_sz,
              "pad": pad,
              "rules": {"min_face_px": MIN_FACE_PX, "min_plate_px": MIN_PLATE_PX,
                        "cover": COVER, "match_iou": MATCH_IOU,
                        "max_plate_area": privacy.MAX_PLATE_AREA,
                        "max_issue_blurred": MAX_ISSUE_BLURRED},
              "sweep_face": face_rows, "sweep_plate": plate_rows,
              "plate_val_issue_hidden_without_cap": no_cap["issue_blurred"],
              "test_face": face_test, "test_plate": plate_test, "test_civic": civic_test,
              "utility": utility, "ms_per_photo": {"face": ms["face"], "plate": ms[plate_sz]}}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"privacy_v1_{split}.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", default="clip", choices=["clip", "yolo", "features", "privacy"])
    parser.add_argument("--weights", type=Path, default=None, help="YOLO best.pt (default: "
                        "models/vision/yolo_v1/best.pt)")
    args = parser.parse_args()
    if args.model == "yolo":
        main_yolo(args.split, args.weights)
    elif args.model == "features":
        main_features(args.split)
    elif args.model == "privacy":
        main_privacy(args.split)
    else:
        main(args.split)
