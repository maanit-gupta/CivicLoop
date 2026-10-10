"""Step 6.1: the verifier's dataset. `python -m verifier.dataset` writes to data/verifier/:

- reports.parquet: every corpus complaint as the verifier would receive it (GPS with phone noise,
  maybe a photo), plus fake-photo reports. Labels: incident_id, is_spam, spam_kind.
- pairs.parquet: (earlier a, later b) report pairs at most WINDOW_H apart, labelled is_duplicate
  (same incident), with a `kind` for slicing the evaluation.
- stock_index.parquet: the simulated "internet": photos a stock-image lookup knows about.
- images/<split>/<report_id>.jpg: the uploaded photos.

Splits follow the corpus (by incident). A pair is a test pair when its later report b is a test
report. Pairs with b in train and a in test are dropped, so a test report never appears in a
train pair. Photos: train reports draw from vision train+val, test reports from vision test.
The corpus's true coordinates decide a pair's `kind`; the model only sees the noisy GPS.
"""

import shutil
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd

from config import CORPUS_PATH, DATA_DIR, ROOT, SEED
from verifier import images as im

OUT_DIR = DATA_DIR / "verifier"
YOLO_DIR = DATA_DIR / "vision" / "yolo"  # the Phase 5 image dataset (manifest.csv, images, labels)

WINDOW_H = 72             # open-case lookback; the corpus re-reports an incident within 72 h
GPS_SIGMA_M = 15          # per-axis noise: phone GPS error plus where the reporter stands
GPS_MISSING_P = 0.10      # reports without GPS (WhatsApp text, call centre)
PHOTO_P = 0.70            # chance a report attaches a photo, for issue types vision covers
NEAR_M = 500              # hard negative: same issue type, different incident, within this
SAME_PLACE_M = 150        # hard negative: different issue type within this
RANDOM_NEG_PER_POS = 5    # plus this many random in-window negatives per positive
FAKES_PER_KIND = {"train": 120, "test": 30}
REUSE_MIN_M = 1000        # a reused photo turns up at least this far from its own incident
VIEW_QUALITY = (75, 95)   # JPEG quality of a citizen's own photo
RESHARE_QUALITY = (40, 75)
SCREENSHOT_QUALITY = (70, 90)

# Issue types with public photos (Step 5.1). streetlight photos are damaged poles (5.1 decision);
# drain_blockage and water_leak have none, so those reports never carry a photo.
PHOTO_TYPES = ("pothole", "garbage", "streetlight", "damaged_sign", "other")
VISION_SPLITS = {"train": ("train", "val"), "test": ("test",)}
FAKE_KINDS = ("reused_image", "stock_image", "screenshot")
PAIR_KINDS = ("same_incident", "near_same_type", "same_place_other_type", "random")
REPORT_COLUMNS = ["report_id", "incident_id", "split", "submitted_at", "language", "text",
                  "issue_type", "location_text", "lat", "lon", "children", "accident", "flooding",
                  "image_path", "is_spam", "spam_kind", "base_image", "text_from", "image_from"]
M_PER_DEG_LAT = 111_320


def metres(lat1, lon1, lat2, lon2):
    """Haversine distance in metres; works on numbers or numpy arrays. NaN in -> NaN out."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.subtract(lon2, lon1)) / 2) ** 2)
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def _rel(path: Path) -> str:
    """Store paths relative to the repo so the tables survive a move; `ROOT / p` reads them."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------- pairs ----------

def build_pairs(corpus: pd.DataFrame, rng) -> pd.DataFrame:
    real = corpus[~corpus.is_spam].sort_values(["submitted_at", "complaint_id"])
    real = real.reset_index(drop=True)
    hours = (real.submitted_at - real.submitted_at.min()).dt.total_seconds().to_numpy() / 3600
    ids, inc, typ = (real[c].to_numpy() for c in ("complaint_id", "incident_id", "issue_type"))
    lat, lon, split = real.lat.to_numpy(), real.lon.to_numpy(), real.split.to_numpy()
    parts = []
    for b in range(len(real)):
        a = np.arange(np.searchsorted(hours, hours[b] - WINDOW_H), b)  # earlier, in the window
        if split[b] == "train":
            a = a[split[a] == "train"]
        if not len(a):
            continue
        dist = metres(lat[a], lon[a], lat[b], lon[b])
        same_inc, same_type = inc[a] == inc[b], typ[a] == typ[b]
        kind = np.select(
            [same_inc, same_type & (dist <= NEAR_M), ~same_type & (dist <= SAME_PLACE_M)],
            PAIR_KINDS[:3], default="random")
        parts.append(pd.DataFrame({"a_id": ids[a], "b_id": ids[b], "split": split[b],
                                   "is_duplicate": same_inc, "kind": kind}))
    pairs = pd.concat(parts, ignore_index=True)
    keep = [pairs[pairs.kind != "random"]]
    for s, grp in pairs[pairs.kind == "random"].groupby("split"):
        n = min(len(grp), RANDOM_NEG_PER_POS * int(pairs.is_duplicate[pairs.split == s].sum()))
        keep.append(grp.iloc[np.sort(rng.choice(len(grp), n, replace=False))])
    return pd.concat(keep).sort_values(["b_id", "a_id"]).reset_index(drop=True)


# ---------- reports, photos, fakes ----------

def _gps(lat: np.ndarray, lon: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    n = len(lat)
    noisy_lat = lat + rng.normal(0, GPS_SIGMA_M, n) / M_PER_DEG_LAT
    noisy_lon = lon + rng.normal(0, GPS_SIGMA_M, n) / (M_PER_DEG_LAT * np.cos(np.radians(lat)))
    missing = rng.random(n) < GPS_MISSING_P
    return np.where(missing, np.nan, noisy_lat), np.where(missing, np.nan, noisy_lon)


def _pools(manifest: pd.DataFrame, rng) -> dict[tuple[str, str], list[str]]:
    """Shuffled vision image ids per (corpus split, issue type). Pop from the end to use one."""
    pools = {}
    for split, vsplits in VISION_SPLITS.items():
        for label in PHOTO_TYPES:
            mine = manifest.split.isin(vsplits) & (manifest.label == label)
            ids = sorted(manifest.image_id[mine])
            pools[split, label] = [ids[i] for i in rng.permutation(len(ids))]
    return pools


def _assign_photos(real: pd.DataFrame, pools, rng) -> dict[str, str]:
    """report_id -> vision image id. All photos of one incident come from one source photo;
    different incidents never share one. A pool that runs dry leaves reports without photos."""
    base_of_incident, photo = {}, {}
    for r in real.sort_values(["submitted_at", "complaint_id"]).itertuples():
        wants = rng.random() < PHOTO_P
        if not wants or r.issue_type not in PHOTO_TYPES:
            continue
        if r.incident_id not in base_of_incident:
            pool = pools[r.split, r.issue_type]
            if not pool:
                continue
            base_of_incident[r.incident_id] = pool.pop()
        photo[r.complaint_id] = base_of_incident[r.incident_id]
    return photo


def _fakes(reports: pd.DataFrame, corpus: pd.DataFrame, pools, n_per_kind: dict[str, int],
           paths: dict, out_dir: Path, rng) -> tuple[list[dict], list[dict]]:
    """Fake-photo reports: plausible text cloned from a real report of the same issue type, with
    a photo that is not the reporter's own. Screenshot sources are taken out of the stock index,
    so each fake kind can only be caught by its own check. Returns (rows, stock index rows)."""
    true = corpus.set_index("complaint_id")
    genuine = reports[reports.image_path.notna()]
    rows, stock = [], []
    for split, k in n_per_kind.items():
        sources = corpus[(corpus.split == split) & ~corpus.is_spam
                         & corpus.issue_type.isin(PHOTO_TYPES)]
        sources = sources.iloc[rng.permutation(len(sources))]
        free = {t: list(pools[split, t]) for t in PHOTO_TYPES}       # not used by any report
        shot_src = {t: free[t][: len(free[t]) // 2] for t in PHOTO_TYPES}
        library = {t: free[t][len(free[t]) // 2:] for t in PHOTO_TYPES}  # the stock index
        stock += [{"image_id": i, "split": split} for t in PHOTO_TYPES for i in library[t]]
        made = dict.fromkeys(FAKE_KINDS, 0)
        for src in sources.itertuples():
            kind = next((kd for kd in FAKE_KINDS if made[kd] < k), None)
            if kind is None:
                break
            t = src.issue_type
            at = src.submitted_at + pd.Timedelta(hours=float(rng.uniform(1, 48)))
            base, image_from = None, None
            if kind == "reused_image":
                cand = genuine[(genuine.split == split) & (genuine.issue_type == t)
                               & (genuine.submitted_at < at)]
                own = true.loc[cand.report_id]  # the reused photo's own incident location
                cand = cand[metres(own.lat.to_numpy(), own.lon.to_numpy(), src.lat, src.lon)
                            >= REUSE_MIN_M]
                if not len(cand):
                    continue
                orig = cand.iloc[rng.integers(len(cand))]
                image_from, base = orig.report_id, orig.base_image
                img, quality = im.reshare(im.load_rgb(ROOT / orig.image_path), rng), RESHARE_QUALITY
            elif kind == "stock_image":
                if not library[t]:
                    continue
                base = library[t].pop()  # popped from this list only: it stays in the index
                img, quality = im.reshare(im.load_rgb(paths[base]), rng), RESHARE_QUALITY
            else:
                if not shot_src[t]:
                    continue
                base = shot_src[t].pop()
                img, quality = im.screenshot(im.load_rgb(paths[base]), rng), SCREENSHOT_QUALITY
            n = len(rows) + 1
            rid = f"F-{n:05d}"
            path = out_dir / "images" / split / f"{rid}.jpg"
            im.save_jpeg(img, path, rng.integers(quality[0], quality[1] + 1))
            lat, lon = _gps(np.array([src.lat]), np.array([src.lon]), rng)
            rows.append({
                "report_id": rid, "incident_id": f"IF-{n:05d}", "split": split, "submitted_at": at,
                "language": src.language, "text": src.text, "issue_type": t,
                "location_text": src.location_text, "lat": lat[0], "lon": lon[0],
                "children": src.children, "accident": src.accident, "flooding": src.flooding,
                "image_path": _rel(path), "is_spam": True, "spam_kind": kind, "base_image": base,
                "text_from": src.complaint_id, "image_from": image_from})
            made[kind] += 1
        if min(made.values()) < k:
            raise RuntimeError(f"{split}: could only make {made} fakes, wanted {k} of each")
    return rows, stock


def build(corpus: pd.DataFrame, manifest: pd.DataFrame, yolo_dir: Path = YOLO_DIR,
          out_dir: Path = OUT_DIR, seed: int = SEED,
          fakes_per_kind: dict[str, int] = FAKES_PER_KIND
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and write reports, pairs and the stock index. Returns them."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir / "images", ignore_errors=True)  # our own output from a previous run
    paths = {r.image_id: Path(yolo_dir) / "images" / r.split / f"{r.image_id}.jpg"
             for r in manifest.itertuples()}

    pairs = build_pairs(corpus, rng)
    pools = _pools(manifest, rng)
    real = corpus[~corpus.is_spam]
    photo = _assign_photos(real, pools, rng)

    reports = corpus.rename(columns={"complaint_id": "report_id"}).copy()
    reports = reports.sort_values(["submitted_at", "report_id"]).reset_index(drop=True)
    reports["lat"], reports["lon"] = _gps(reports.lat.to_numpy(), reports.lon.to_numpy(), rng)
    reports["spam_kind"] = np.where(reports.is_spam, "text", "none")
    reports["base_image"] = reports.report_id.map(photo)
    reports["image_path"] = None
    for i, r in reports[reports.base_image.notna()].iterrows():
        src = paths[r.base_image]
        boxes = im.yolo_boxes(Path(yolo_dir) / "labels" / src.parent.name / f"{src.stem}.txt")
        path = out_dir / "images" / r.split / f"{r.report_id}.jpg"
        im.save_jpeg(im.photo_view(im.load_rgb(src), rng, boxes), path,
                     rng.integers(VIEW_QUALITY[0], VIEW_QUALITY[1] + 1))
        reports.at[i, "image_path"] = _rel(path)
    reports["text_from"] = reports["image_from"] = None

    fakes, stock_rows = _fakes(reports, corpus, pools, fakes_per_kind, paths, out_dir, rng)
    reports = pd.concat([reports, pd.DataFrame(fakes)], ignore_index=True)[REPORT_COLUMNS]
    stock = pd.DataFrame(stock_rows)
    stock["image_path"] = [_rel(paths[i]) for i in stock.image_id]

    out_dir.mkdir(parents=True, exist_ok=True)
    reports.to_parquet(out_dir / "reports.parquet", index=False)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    stock.to_parquet(out_dir / "stock_index.parquet", index=False)
    return reports, pairs, stock


# ---------- summary ----------

def summary(reports: pd.DataFrame, pairs: pd.DataFrame, stock: pd.DataFrame,
            yolo_dir: Path = YOLO_DIR) -> str:
    """Counts, plus a difficulty check: how far apart (GPS) and how alike (dhash) pairs are."""
    hashes: dict[str, imagehash.ImageHash] = {}

    def dhash(path: str):
        if path not in hashes:
            hashes[path] = imagehash.dhash(im.load_rgb(ROOT / path))
        return hashes[path]

    by_id = reports.set_index("report_id")
    lines = ["Reports (rows = split, columns = spam_kind):",
             pd.crosstab(reports.split, reports.spam_kind, margins=True).to_string(), "",
             f"Photo share (real reports): "
             f"{by_id.image_path[by_id.spam_kind == 'none'].notna().mean():.2f}; "
             f"GPS missing: {reports.lat.isna()[reports.spam_kind != 'text'].mean():.2f}; "
             f"stock index: {len(stock)} images", ""]
    rows = []
    for (split, kind), grp in pairs.groupby(["split", "kind"]):
        a, b = by_id.loc[grp.a_id], by_id.loc[grp.b_id]
        dist = metres(a.lat.to_numpy(), a.lon.to_numpy(), b.lat.to_numpy(), b.lon.to_numpy())
        both = a.image_path.notna().to_numpy() & b.image_path.notna().to_numpy()
        ham = [dhash(x) - dhash(y) for x, y in zip(a.image_path[both], b.image_path[both],
                                                    strict=True)]
        rows.append({"split": split, "kind": kind, "pairs": len(grp),
                     "duplicates": int(grp.is_duplicate.sum()),
                     "gps_m_median": np.nanmedian(dist),
                     "within_50m": np.mean(dist[~np.isnan(dist)] <= 50),
                     "both_photos": int(both.sum()),
                     "dhash_median": np.median(ham) if ham else np.nan})
    lines += ["Pairs (gps = noisy GPS the model sees; dhash = 64-bit difference hash distance):",
              pd.DataFrame(rows).to_string(index=False, float_format="%.2f"), ""]
    fake_rows = []
    for kind, grp in reports[reports.spam_kind.isin(FAKE_KINDS)].groupby("spam_kind"):
        ham = []
        for r in grp.itertuples():
            if kind == "reused_image":
                src = by_id.image_path[r.image_from]
            else:
                src = _rel(next((Path(yolo_dir) / "images").glob(f"*/{r.base_image}.jpg")))
            ham.append(dhash(r.image_path) - dhash(src))
        fake_rows.append({"fake": kind, "n": len(grp),
                          "dhash_to_source": np.median(ham) if ham else np.nan})
    lines += ["Fakes (dhash to the photo they copy; screenshot sources are not in the index):",
              pd.DataFrame(fake_rows).to_string(index=False, float_format="%.1f")]
    return "\n".join(lines)


def main() -> None:
    corpus = pd.read_parquet(CORPUS_PATH)
    manifest = pd.read_csv(YOLO_DIR / "manifest.csv")
    reports, pairs, stock = build(corpus, manifest)
    print(summary(reports, pairs, stock))
    print(f"\nWrote {OUT_DIR}/reports.parquet, pairs.parquet, stock_index.parquet, images/")


if __name__ == "__main__":
    main()
