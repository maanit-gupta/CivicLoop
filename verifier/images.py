"""Pixel transforms that turn one source photo into the photos citizens (or spammers) upload.

- photo_view: a citizen's own photo of the issue. Two views of one source photo stand in for two
  people photographing the same pothole: different crop, slight tilt, mirror, lighting.
- reshare: a photo saved from somewhere and uploaded again: a few pixels cut, smaller, and
  recompressed harder (the caller picks the JPEG quality).
- screenshot: a photo shown inside a phone screen (status bar, app bar, caption, nav bar).
Every upload is saved as JPEG, screenshots included (messaging apps convert them), so the file
format gives nothing away.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2), normalised 0..1

VIEW_SCALE = (0.70, 0.95)   # crop side as a share of the source side
VIEW_TILT = 0.06            # corners move inwards by up to this share: a slightly different angle
VIEW_LIGHT = (0.8, 1.2)     # brightness, contrast, saturation factors
RESHARE_CROP = 0.03         # up to 3% cut from each side
RESHARE_SCALE = (0.5, 1.0)
SCREEN = (540, 1170)        # a 9:19.5 phone screen at half resolution
APP_TITLES = ("Forwarded", "Photo", "Mohalla group", "Delhi news")
CAPTIONS = ("Forwarded many times", "dekho road ka haal", "Delhi ka haal dekho", "pls share")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def yolo_boxes(label_path: Path) -> list[Box]:
    """Boxes from a YOLO label file (class cx cy w h, normalised). Missing file -> no boxes."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().split("\n"):
        if line.strip():
            cx, cy, w, h = map(float, line.split()[1:5])
            boxes.append((max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                          min(1.0, cx + w / 2), min(1.0, cy + h / 2)))
    return boxes


def save_jpeg(img: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=int(quality))


def _crop_start(size: int, crop: int, spans: list[tuple[float, float]], rng) -> int:
    """Start of a `crop`-long window on an axis of length `size`. Keeps the issue's boxes
    (normalised spans) inside the window when they fit, so the view still shows the issue."""
    lo, hi = 0, size - crop
    if spans:
        first, last = min(s[0] for s in spans) * size, max(s[1] for s in spans) * size
        keep_lo, keep_hi = max(lo, int(np.ceil(last - crop))), min(hi, int(first))
        if keep_lo <= keep_hi:
            lo, hi = keep_lo, keep_hi
    return int(rng.integers(lo, hi + 1))


def _perspective_coeffs(dst, src) -> list[float]:
    """The 8 numbers PIL needs to map each output corner in `dst` to the input point in `src`."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src, strict=True):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    return np.linalg.solve(np.array(rows, float), np.array(rhs, float)).tolist()


def _tilt(img: Image.Image, rng) -> Image.Image:
    """Pull each corner inwards by a random amount: a camera at a slightly different angle.
    Corners only move inwards, so no black borders appear."""
    w, h = img.size
    dx, dy = rng.uniform(0, VIEW_TILT, 4) * w, rng.uniform(0, VIEW_TILT, 4) * h
    src = [(dx[0], dy[0]), (w - dx[1], dy[1]), (w - dx[2], h - dy[2]), (dx[3], h - dy[3])]
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    return img.transform((w, h), Image.Transform.PERSPECTIVE, _perspective_coeffs(dst, src),
                         Image.Resampling.BICUBIC)


def photo_view(img: Image.Image, rng, boxes: list[Box] = ()) -> Image.Image:
    w, h = img.size
    cw, ch = (max(1, round(side * rng.uniform(*VIEW_SCALE))) for side in (w, h))
    x0 = _crop_start(w, cw, [(b[0], b[2]) for b in boxes], rng)
    y0 = _crop_start(h, ch, [(b[1], b[3]) for b in boxes], rng)
    out = _tilt(img.crop((x0, y0, x0 + cw, y0 + ch)), rng)
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enhancer(out).enhance(rng.uniform(*VIEW_LIGHT))
    return out


def reshare(img: Image.Image, rng) -> Image.Image:
    w, h = img.size
    left, top, right, bottom = rng.uniform(0, RESHARE_CROP, 4)
    out = img.crop((int(left * w), int(top * h), w - int(right * w), h - int(bottom * h)))
    s = rng.uniform(*RESHARE_SCALE)
    return out.resize((max(1, round(out.width * s)), max(1, round(out.height * s))),
                      Image.Resampling.BILINEAR)


def screenshot(img: Image.Image, rng) -> Image.Image:
    W, H = SCREEN
    status, app_bar, footer = 28, 56, 140  # heights in px
    bg, fg = ((0, 0, 0), (235, 235, 235)) if rng.random() < 0.5 else ((255, 255, 255), (30, 30, 30))
    out = Image.new("RGB", SCREEN, bg)
    draw = ImageDraw.Draw(out)
    small, big = ImageFont.load_default(14), ImageFont.load_default(20)
    # status bar: clock, network, battery
    draw.text((16, 6), f"{rng.integers(6, 23)}:{rng.integers(0, 60):02d}", fill=fg, font=small)
    draw.text((W - 90, 6), "4G", fill=fg, font=small)
    draw.rectangle((W - 50, 8, W - 20, 20), outline=fg)
    draw.rectangle((W - 48, 10, W - 48 + int(26 * rng.uniform(0.2, 1.0)), 18), fill=fg)
    # app bar in a random colour
    bar = tuple(int(c) for c in rng.integers(0, 200, 3))
    draw.rectangle((0, status, W, status + app_bar), fill=bar)
    title = APP_TITLES[rng.integers(len(APP_TITLES))]
    draw.text((16, status + 16), f"<  {title}", fill=(255, 255, 255), font=big)
    # the photo, as large as fits between the bars
    room_top, room_h = status + app_bar, H - status - app_bar - footer
    s = min(W / img.width, room_h / img.height)
    photo = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
    out.paste(photo, ((W - photo.width) // 2, room_top + (room_h - photo.height) // 2))
    # caption, time sent, nav bar
    caption = CAPTIONS[rng.integers(len(CAPTIONS))]
    draw.text((16, H - footer + 16), caption, fill=fg, font=big)
    draw.text((W - 70, H - footer + 50), f"{rng.integers(1, 13)}:{rng.integers(0, 60):02d} pm",
              fill=fg, font=small)
    draw.rectangle((W // 2 - 60, H - 20, W // 2 + 60, H - 14), fill=fg)
    return out
