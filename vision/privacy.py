"""Step 5.5: blur faces and number plates before a photo is stored or analysed.

Order matters: mask first, then everything else. The models (vision.features), the stored copy,
the verifier and the officials only ever see the masked photo, so no unmasked copy is kept
"just for the models".

Two small open detectors, both on CPU:
- faces: YuNet (OpenCV model zoo, MIT licence, 230 KB ONNX), run by cv2.FaceDetectorYN.
- number plates: YOLO11n fine-tuned on Roboflow's licence-plate set (Hugging Face
  morsetechlab/yolov11-license-plate-detection, AGPL-3.0 like Ultralytics). Its README warns its
  own test split is contaminated, so `python -m vision.evaluate --model privacy` measures it on
  Indian plates instead.

Each box is grown by PAD[kind] on every side (detectors draw tight boxes; hair, ears and the
plate's frame would leak) and blurred hard: a Gaussian whose radius scales with the box, so no
character or facial feature survives. The photo is EXIF-rotated once and saved without its metadata
(phone EXIF can hold the GPS position; the location travels separately, through intake).

Fail closed: if a detector cannot load, this raises. An unmasked photo is never stored.
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from config import DATA_DIR, MODELS_DIR

MODEL_DIR = MODELS_DIR / "vision" / "privacy"
STORE_DIR = DATA_DIR / "vision" / "stored"  # masked photos, the only copies the pipeline keeps
# (Hugging Face repo, revision, file, sha256). Downloaded once into MODEL_DIR, checked by hash.
FACE_WEIGHTS = ("opencv/face_detection_yunet", "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
                "face_detection_yunet_2023mar.onnx",
                "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")
PLATE_WEIGHTS = ("morsetechlab/yolov11-license-plate-detection",
                 "251a30d7daedca065f56e04b0af04052c907c68f", "license-plate-finetune-v1n.pt",
                 "0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2")
MODEL_VERSION = "privacy-v1-yunet2023mar+lp-yolo11n"  # bump when a model or threshold changes

# Picked by `python -m vision.evaluate --model privacy`: the most faces/plates masked (tune
# halves) while hiding <= 1% of the civic issue's own box (civic val). Test: faces 0.81,
# plates 0.78 masked; the 5.4 labels unchanged (98% equal, macro-F1 0.876 -> 0.883).
FACE_THRESHOLD = 0.6  # 0.5 already blurs 1.5% of the issues (YuNet sees faces in rubbish)
PLATE_THRESHOLD = 0.05  # the model is unsure of Indian plates: at 0.25 it masks 4 in 10
PLATE_IMGSZ = 1280  # Indian plates in dashcam frames are small; ~240 ms per photo on CPU
# The plate model reads road signs (a rectangle with letters) as plates, often at 0.8 and
# covering half the photo, which would blur the very sign being reported. A real plate is far
# smaller (Indian plate frames: at most 0.6% of the photo), so bigger "plates" are dropped.
# Val: damaged_sign area blurred 35% -> 0.2%. (Keeping only plates on a COCO vehicle: 3.6%.)
MAX_PLATE_AREA = 0.05
# Fraction of the box's width/height added on every side. The plate model's boxes are tighter
# than the plate and often off-centre: with 0.15 it masked 58% of tune plates, with 0.5 88%.
PAD = {"face": 0.15, "plate": 0.5}
BLUR = 0.25  # Gaussian radius as a fraction of the box's longer side
JPEG_QUALITY = 95

Box = tuple[float, float, float, float]  # pixel corners x1, y1, x2, y2


class Found(NamedTuple):
    kind: str  # "face" or "plate"
    score: float
    box: Box


def _weights(spec: tuple[str, str, str, str], model_dir: Path = MODEL_DIR) -> Path:
    repo, revision, filename, sha256 = spec
    path = model_dir / filename
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo, filename, revision=revision, local_dir=model_dir)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its pinned sha256; delete it to re-download")
    return path


@lru_cache(maxsize=1)
def _face_model():
    import cv2

    # Score floor 0.05 so the evaluation can sweep thresholds; detect_faces filters higher.
    # Filtering after NMS gives the same boxes as a higher floor: NMS only lets a box be
    # suppressed by a higher-scoring one.
    return cv2.FaceDetectorYN.create(str(_weights(FACE_WEIGHTS)), "", (320, 320), 0.05, 0.3, 5000)


@lru_cache(maxsize=1)
def _plate_model():
    from ultralytics import YOLO

    return YOLO(str(_weights(PLATE_WEIGHTS)))


def load(path: str | Path) -> Image.Image:
    """The photo upright (EXIF orientation applied) and in RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def detect_faces(img: Image.Image, threshold: float = FACE_THRESHOLD) -> list[Found]:
    model = _face_model()
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    model.setInputSize(img.size)
    _, faces = model.detect(bgr)
    rows = [] if faces is None else faces.tolist()
    return [Found("face", r[14], (r[0], r[1], r[0] + r[2], r[1] + r[3]))
            for r in rows if r[14] >= threshold]


def plate_sized(box: Box, size: tuple[int, int], max_area: float = MAX_PLATE_AREA) -> bool:
    """False for a box too big to be a number plate (see MAX_PLATE_AREA)."""
    return (box[2] - box[0]) * (box[3] - box[1]) <= max_area * size[0] * size[1]


def detect_plates(img: Image.Image, threshold: float = PLATE_THRESHOLD,
                  imgsz: int = PLATE_IMGSZ, max_area: float = MAX_PLATE_AREA) -> list[Found]:
    r = _plate_model().predict(img, imgsz=imgsz, conf=0.05, device="cpu", verbose=False)[0]
    return [Found("plate", s, tuple(b)) for s, b in zip(r.boxes.conf.tolist(),
                                                        r.boxes.xyxy.tolist(), strict=True)
            if s >= threshold and plate_sized(b, img.size, max_area)]


def padded(box: Box, size: tuple[int, int], pad: float = 0.0) -> tuple[int, int, int, int]:
    """Grow a box by `pad` of its width/height on every side, round outwards, clip to the photo."""
    x1, y1, x2, y2 = box
    dx, dy = pad * (x2 - x1), pad * (y2 - y1)
    w, h = size
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(w, int(np.ceil(x2 + dx))), min(h, int(np.ceil(y2 + dy))))


def blur(img: Image.Image, boxes: list[Box]) -> Image.Image:
    """A copy of the photo with every box blurred. Pixels outside the boxes are kept."""
    out = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = padded(box, img.size)
        if x2 <= x1 or y2 <= y1:
            continue
        radius = max(2.0, BLUR * max(x2 - x1, y2 - y1))
        out.paste(out.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius)), (x1, y1))
    return out


def blur_found(img: Image.Image, found: list[Found],
               pad: dict[str, float] = PAD) -> Image.Image:
    """Blur every detection, each grown by its kind's padding."""
    return blur(img, [padded(f.box, img.size, pad[f.kind]) for f in found])


def mask(img: Image.Image) -> tuple[Image.Image, list[Found]]:
    """Find faces and plates, blur them. Returns the masked photo and what was blurred."""
    found = detect_faces(img) + detect_plates(img)
    return blur_found(img, found), found


def mask_file(src: str | Path, store_dir: Path = STORE_DIR) -> tuple[Path, list[Found]]:
    """Mask one photo and store it as <sha256 of the original bytes, 16 hex>.jpg (so the same
    upload maps to the same file), without EXIF. Returns the stored path and what was blurred."""
    src = Path(src)
    masked, found = mask(load(src))  # raises before anything is written if a model is missing
    store_dir.mkdir(parents=True, exist_ok=True)
    out = store_dir / f"{hashlib.sha256(src.read_bytes()).hexdigest()[:16]}.jpg"
    masked.save(out, "JPEG", quality=JPEG_QUALITY)
    return out, found
