"""Step 5.4: CLIP image embeddings, CPU only.

An embedding is CLIP's 512-number summary of a photo. Every vector is scaled to length 1, so the
cosine similarity of two photos is their dot product: near 1 = the same scene, lower = unlike.
The verifier (duplicate photos, Phase 6) and Closure (before/after) compare these.

Same model and weights as the 5.2 zero-shot baseline, so one CLIP pass per photo gives both the
embedding and the zero-shot label probabilities (vision.clip_zeroshot.probs_from_embeddings).

Embeddings from different models live in different spaces and must never be compared, which is
why MODEL_VERSION travels with every embedding (inside ImageFeatures.model_version).
"""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"  # open weights, ~600 MB, downloaded once from Hugging Face
MODEL_VERSION = f"clip-{MODEL_NAME}-{PRETRAINED}"
DIM = 512

ImageInput = str | Path | BytesIO | Image.Image


@lru_cache(maxsize=1)
def _model():
    import open_clip  # heavy import, only when a photo is actually embedded
    import torch

    torch.manual_seed(0)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(MODEL_NAME)


def unit(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as im:
        return im.convert("RGB")


def embed_images(images: list[ImageInput], batch_size: int = 32) -> np.ndarray:
    """Unit-length embeddings, one row per image (n x DIM, float32)."""
    import torch

    model, preprocess, _ = _model()
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(_rgb(im)) for im in images[i:i + batch_size]])
        with torch.no_grad():
            out.append(model.encode_image(batch).numpy())
    return unit(np.concatenate(out)) if out else np.zeros((0, DIM), dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Unit-length text embeddings in the same space as the images (zero-shot prompts)."""
    import torch

    model, _, tokenizer = _model()
    with torch.no_grad():
        return unit(model.encode_text(tokenizer(texts)).numpy())


def cosine(a, b) -> float:
    """Cosine similarity of two vectors: 1 same direction, 0 unrelated, -1 opposite."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
