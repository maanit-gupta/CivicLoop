"""Step 5.3: the fine-tuned YOLO detector, run locally on CPU.

Trained on a GPU by notebooks/yolo_finetune.ipynb (Kaggle/Colab): YOLO26s pretrained on COCO,
its head swapped for our 4 classes and fine-tuned on data/vision/yolo/ train. The weights are
dropped into models/vision/yolo_v1/ (best.pt + meta.json).

YOLO gives boxes, not an image label. The image label is the issue type of the most confident
box if that box clears a threshold, else "none". YOLO has no "other" class (fallen trees,
graffiti... are background images in training), so it can never answer "other".

Detections are (class id, confidence, cx, cy, w, h), box normalised to 0..1, the same form as
the YOLO label files and vision.evaluate.
"""

from functools import lru_cache
from pathlib import Path

from config import MODELS_DIR
from vision.dataset import CLASS_TO_ISSUE, CLASSES, NONE_LABEL

MODEL_DIR = MODELS_DIR / "vision" / "yolo_v1"
WEIGHTS = MODEL_DIR / "best.pt"
MODEL_VERSION = "yolo-v1-yolo26s"
IMGSZ = 640  # same as training
MIN_CONF = 0.001  # keep near-zero boxes: mAP needs the whole precision-recall curve
# Image label threshold (user decision, 5.3): the best val macro-F1. Low on purpose: a missed
# issue costs more than a false alarm (val: pothole recall 0.92, clean roads flagged 17%).
LABEL_THRESHOLD = 0.10
MAX_DET = 300  # per image, as in Ultralytics' own validation

Detection = tuple[int, float, float, float, float, float]


def image_label(dets: list[Detection], threshold: float) -> str:
    """The issue type of the most confident box if it reaches `threshold`, else 'none'."""
    top = max(dets, key=lambda d: d[1], default=None)
    if top is None or top[1] < threshold:
        return NONE_LABEL
    return CLASS_TO_ISSUE[CLASSES[top[0]]].value


def to_detections(cls: list[float], conf: list[float],
                  xywhn: list[list[float]]) -> list[Detection]:
    """Ultralytics' per-image tensors (as lists) -> our detection tuples."""
    return [(int(c), float(p), *map(float, b)) for c, p, b in zip(cls, conf, xywhn, strict=True)]


@lru_cache(maxsize=2)
def _model(weights: str):
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"No YOLO weights at {weights}. Run notebooks/yolo_finetune.ipynb on a GPU "
            f"(Kaggle/Colab) and unzip its yolo_v1.zip into {MODEL_DIR}.")
    from ultralytics import YOLO  # heavy import, only when a photo is actually scored

    model = YOLO(weights)
    names = [model.names[i] for i in sorted(model.names)]
    if names != CLASSES:  # wrong weights would silently give wrong labels
        raise ValueError(f"{weights} predicts {names}, expected {CLASSES}")
    return model


def detect(paths: list[Path], weights: Path = WEIGHTS,
           batch_size: int = 16) -> list[list[Detection]]:
    """One list of detections per photo, on CPU."""
    model = _model(str(weights))
    out = []
    for i in range(0, len(paths), batch_size):
        results = model.predict([str(p) for p in paths[i:i + batch_size]], imgsz=IMGSZ,
                                conf=MIN_CONF, max_det=MAX_DET, device="cpu", verbose=False)
        out += [to_detections(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                              r.boxes.xywhn.tolist()) for r in results]
    return out
