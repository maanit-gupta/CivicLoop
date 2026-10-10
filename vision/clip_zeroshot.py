"""Step 5.2 baseline: CLIP zero-shot image classification, CPU only.

CLIP embeds images and sentences into one space. Zero-shot = no training: describe each label
in a few sentences, embed them once, and give a photo the label whose sentence it is closest to.

Each label has several prompts (sub-concepts or phrasings). A label's score is its BEST prompt
(max cosine), not the mean: "other" is a union of unlike things (fallen tree, graffiti, parked
cars, dead animal) and their averaged embedding would match none of them. Max also keeps labels
with more prompts from getting a bigger share of the probability, as summing would.

The prompts were written once, before any evaluation, and were not tuned on val or test.
The model itself lives in vision.embed (5.4): the image embedding is computed once and reused.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from vision.embed import MODEL_NAME, PRETRAINED, embed_images, embed_texts

MODEL_VERSION = f"clip-zs-{MODEL_NAME}-{PRETRAINED}"
LOGIT_SCALE = 100.0  # CLIP's learned temperature: turns cosines (~0.2-0.3) into a sharp softmax

# Keys must equal vision.dataset.LABELS (a test checks it). "streetlight" photos are damaged
# poles / hanging wires (5.1 decision), so the prompts describe those, not a dark lamp.
# No prompt mentions "dashcam": every "none" photo is dashcam footage (5.1 bias) and a prompt
# that keys on the camera instead of the content would score well here and badly on citizens.
PROMPTS = {
    "pothole": [
        "a photo of a pothole in the road",
        "a damaged road with a deep pothole",
        "a pothole filled with rainwater on a street",
    ],
    "streetlight": [
        "a photo of a broken electric pole",
        "a leaning street light pole with hanging wires",
        "tangled electric cables hanging from a damaged pole",
    ],
    "garbage": [
        "a photo of garbage dumped on the street",
        "a pile of trash and plastic waste by the roadside",
        "an overflowing garbage dump",
    ],
    "damaged_sign": [
        "a photo of a damaged road sign",
        "a bent or broken traffic sign",
        "a faded, vandalised street sign",
    ],
    "other": [
        "a photo of a fallen tree blocking the road",
        "graffiti painted on a wall",
        "cars illegally parked on the footpath",
        "a dead animal lying on the road",
    ],
    "none": [
        "a photo of a clean road in good condition",
        "a normal street with traffic and no problems",
        "an ordinary city road with nothing wrong",
    ],
}


def label_probs(sims: np.ndarray, prompt_labels: list[str], labels: list[str]) -> np.ndarray:
    """Cosine similarities (images x prompts) -> probabilities (images x labels).
    Label logit = LOGIT_SCALE * max cosine over that label's prompts; then softmax."""
    logits = np.stack([sims[:, [i for i, p in enumerate(prompt_labels) if p == lb]].max(axis=1)
                       for lb in labels], axis=1) * LOGIT_SCALE
    logits -= logits.max(axis=1, keepdims=True)  # numerically safe softmax
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


@lru_cache(maxsize=1)
def _text_matrix() -> tuple[np.ndarray, list[str]]:
    """Unit-length prompt embeddings (prompts x dim), and the label of each row."""
    prompt_labels = [lb for lb, ps in PROMPTS.items() for _ in ps]
    return embed_texts([p for ps in PROMPTS.values() for p in ps]), prompt_labels


def probs_from_embeddings(img: np.ndarray) -> list[dict[str, float]]:
    """Unit-length image embeddings (n x dim) -> one {label: probability} dict per photo."""
    text, prompt_labels = _text_matrix()
    labels = list(PROMPTS)
    probs = label_probs(img @ text.T, prompt_labels, labels)
    return [dict(zip(labels, row.tolist(), strict=True)) for row in probs]


def classify(paths: list[Path], batch_size: int = 32) -> list[dict[str, float]]:
    """One {label: probability} dict per photo, in PROMPTS order."""
    return probs_from_embeddings(embed_images(paths, batch_size))
