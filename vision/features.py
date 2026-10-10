"""Step 5.4: one photo -> ImageFeatures, from YOLO (5.3) and CLIP (5.2, vision.embed).

Two models, one answer:
- YOLO knows the 4 box classes and where they are. Its boxes at or above LABEL_THRESHOLD (0.10,
  user decision 5.3) become the photo's detections; weaker boxes only ever served the mAP.
- CLIP runs once per photo. Its embedding is the photo's fingerprint (duplicates, before/after)
  and, multiplied by the prompt embeddings, gives the zero-shot probabilities at no extra cost.

Who names the issue: a confident YOLO box decides. With none, YOLO can only say "none" (it has
no "other" class), so a fallback rule asks CLIP. The rule (FALLBACK) was picked on val by
`python -m vision.evaluate --model features`, out of the candidates in FALLBACK_RULES.
"""

from collections import Counter
from pathlib import Path

from contracts.api import Detection, ImageFeatures, IssueType
from vision import clip_zeroshot, embed, yolo_detect
from vision.dataset import CLASSES, NONE_LABEL
from vision.yolo_detect import LABEL_THRESHOLD, image_label

MODEL_VERSION = f"{yolo_detect.MODEL_VERSION}+{embed.MODEL_VERSION}"
OTHER = "other"


def _top(p: dict[str, float]) -> str:
    return max(p, key=p.get)


# What to answer when no YOLO box reaches the threshold. Input: CLIP's 6 label probabilities.
FALLBACK_RULES = {
    "yolo only": lambda p: NONE_LABEL,  # = 5.3, never "other"
    "clip other vs none": lambda p: OTHER if p[OTHER] > p[NONE_LABEL] else NONE_LABEL,
    "clip top if other": lambda p: OTHER if _top(p) == OTHER else NONE_LABEL,
    "clip top": _top,  # CLIP may also name an issue YOLO missed
}
FALLBACK = "clip other vs none"  # best val macro-F1 (0.900; yolo only 0.714)


def hybrid_label(dets: list, probs: dict[str, float], rule: str,
                 threshold: float = LABEL_THRESHOLD) -> str:
    """YOLO's label if a box reaches `threshold`, else the fallback rule's."""
    label = image_label(dets, threshold)
    return label if label != NONE_LABEL else FALLBACK_RULES[rule](probs)


def to_detection(det: tuple) -> Detection | None:
    """A YOLO (cls, conf, cx, cy, w, h) tuple -> a contract Detection, corners clipped to the
    photo. None if no part of the box is inside the photo."""
    c, conf, cx, cy, w, h = det
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(label=CLASSES[c], confidence=min(1.0, conf), bbox=(x1, y1, x2, y2))


def run_models(paths: list[Path]) -> list[tuple[list, dict[str, float], list[float]]]:
    """Both models on every photo: (YOLO detections, CLIP label probabilities, CLIP embedding)."""
    dets = yolo_detect.detect(paths)
    emb = embed.embed_images(paths)
    probs = clip_zeroshot.probs_from_embeddings(emb)
    return list(zip(dets, probs, emb.tolist(), strict=True))


def _explain(f: ImageFeatures, label: str, threshold: float) -> str:
    if f.detections:
        counts = Counter(d.label.replace("_", " ") for d in f.detections)
        found = ", ".join(f"{n} {lb}" for lb, n in counts.most_common())
        return (f"The detector found {found} (most confident: "
                f"{f.detections[0].label.replace('_', ' ')}, {f.detector_confidence:.2f}), "
                f"covering about {f.box_area_ratio:.0%} of the photo.")
    if label == NONE_LABEL:
        return (f"No civic issue found: no detector box reached {threshold:.2f}, and the image "
                f"model rates 'nothing wrong' at {f.confidence:.2f}.")
    if label == OTHER:
        return (f"No detector box reached {threshold:.2f}. The image model matches the photo to "
                f"'other' (fallen tree, graffiti, illegal parking, dead animal) at "
                f"{f.confidence:.2f}.")
    return (f"No detector box reached {threshold:.2f}. The image model alone reads the photo as "
            f"{label.replace('_', ' ')} at {f.confidence:.2f}.")


def build(image_path: str, dets: list, probs: dict[str, float], embedding: list[float],
          rule: str = FALLBACK, threshold: float = LABEL_THRESHOLD) -> ImageFeatures:
    """One photo's model outputs (as from run_models) -> ImageFeatures.
    confidence = the top box's confidence when YOLO decides, else CLIP's probability of the
    label the fallback rule chose."""
    kept = sorted((d for d in dets if d[1] >= threshold), key=lambda d: -d[1])
    detections = [x for x in map(to_detection, kept) if x is not None]
    label = hybrid_label(dets, probs, rule, threshold)
    conf = kept[0][1] if kept else probs[label]
    f = ImageFeatures(
        image_path=image_path, model_version=MODEL_VERSION, confidence=min(1.0, conf),
        explanation="pending", issue_type=None if label == NONE_LABEL else IssueType(label),
        detections=detections, embedding=embedding)
    return f.model_copy(update={"explanation": _explain(f, label, threshold)})


def analyze(paths: list[Path]) -> list[ImageFeatures]:
    """Photos -> ImageFeatures, both models batched. (5.5 wraps this with masking + logging.)"""
    return [build(str(p), *out) for p, out in zip(paths, run_models(paths), strict=True)]


def compare(before: ImageFeatures, after: ImageFeatures) -> dict:
    """For Closure: the same spot before and after a repair.
    - similarity: CLIP cosine of the two photos. No same-place threshold here: deciding one
      needs real before/after pairs (typical ranges: data/vision/eval/features_v1_test.json).
    - issue_cleared: the before photo's issue no longer shows; None if before showed none.
    Both must come from the same model version: embeddings and boxes from different models
    are not comparable, so re-analyze the older photo instead."""
    if before.model_version != after.model_version:
        raise ValueError(f"model versions differ: {before.model_version} vs "
                         f"{after.model_version}; re-analyze the older photo")
    if before.embedding is None or after.embedding is None:
        raise ValueError("compare needs both photos' embeddings")
    return {
        "similarity": embed.cosine(before.embedding, after.embedding),
        "issue_before": before.issue_type, "issue_after": after.issue_type,
        "issue_cleared": None if before.issue_type is None
        else after.issue_type != before.issue_type,
        "detections_before": before.detection_count, "detections_after": after.detection_count,
        "box_area_before": before.box_area_ratio, "box_area_after": after.box_area_ratio,
    }
