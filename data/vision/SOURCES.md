# Vision image sources (Step 5.1)

All images are public. They stay local (`data/vision/raw/`, `data/vision/yolo/` are gitignored)
and are used only for learning/non-commercial evaluation. They are never redistributed. Street photos
can show faces and number plates. Step 5.5 masks them in anything the pipeline outputs.
`data/vision/yolo/manifest.csv` maps every built image back to its raw file (`source_path`).

| Source | Licence | What we use | Images used |
|---|---|---|---|
| RDD2022, India subset | CC BY 4.0 | potholes (D40) + no-damage photos | 254 pothole, 600 none |
| Urban Issues Dataset (Kaggle) | CC0-1.0 as declared by the uploader; underlying Roboflow sets vary (see below) | pothole, garbage, broken sign, damaged pole/wires, plus fallen tree / graffiti / dead animal / illegal parking as "other" | 2,480 |

## RDD2022 India
- Arya, D. et al. (2022). *RDD2022: A multi-national image dataset for automatic Road Damage
  Detection.* arXiv:2209.08538. Official copy: figshare 10.6084/m9.figshare.21431547 (CC BY 4.0).
- Downloaded from the Kaggle mirror `hafsaesam/rdd2022-india`, because the official figshare server
  stalled from this network. The layout matches the official India.zip: 7,706 labelled train
  images (720x720, smartphone on a moving car) and 1,959 unlabelled test images (not used).
- Part of the India data was captured in Delhi and Gurugram (press report on the RDD project).
  This makes it the most Delhi-like source we have.
- Placement: `data/vision/raw/rdd2022_india/India/train/{images,annotations/xmls}`.

## Urban Issues Dataset
- Kaggle `akinduhiman/urban-issues-dataset` (Aug 2025), YOLO format, 10 classes with global ids.
  Licence on Kaggle: CC0-1.0. The uploader says the images are "sourced mostly from roboflow", and the
  per-image provenance is not given. Only two subsets ship a Roboflow README:
  - DamagedRoadSigns: universe.roboflow.com/road-inspection/damaged-signs-hind, CC BY 4.0
  - IllegalParking: universe.roboflow.com/parking-amu50/illegal-parking, CC BY-NC-SA 4.0
    (2 images used)
- Treat the rest as "public, licence as declared by the uploader". Attribute the source, use it for
  learning only, and don't republish the images.
- Placement: `data/vision/raw/urban_issues/<Folder>/<Folder>/{train,valid,test}/{images,labels}`
  (the "Damaged concrete structures" folder is not extracted or used).

## Not covered (no suitable public set found on Kaggle / Hugging Face, Oct 2026)
- **drain_blockage**, **water_leak**: no image data. Vision cannot see these issue types yet.
  Roboflow Universe has small sets, but downloading them needs a Roboflow account/API key.
- **streetlight** (a lamp that is out): no data. The `damaged_pole` class (broken poles, hanging
  wires, some transmission towers) stands in for it.

## Known data quirks
- 2/3 of the Urban Issues images exist only in Roboflow's (augmented) train folders, so some are
  rotated/greyscale copies with black corners. The unaugmented valid/test copy is used when one exists.
- Some "Broken Road Sign" photos show intact signs (label noise in the source).
- "none" comes only from RDD dashcam frames. A model could learn "dashcam view = no issue". The
  254 RDD potholes (also dashcam) partly break that shortcut.
- RDD potholes are small and far away (dashcam). Urban Issues potholes are close-ups.

## Privacy masking (Step 5.5): models
Both download once into `models/vision/privacy/`, pinned to a Hugging Face revision and checked
by sha256 (`vision/privacy.py`).
- Faces: **YuNet** 2023mar (`opencv/face_detection_yunet`), MIT licence, 230 KB ONNX, run by
  OpenCV's `cv2.FaceDetectorYN`.
- Number plates: **YOLO11n licence-plate** (`morsetechlab/yolov11-license-plate-detection`,
  `license-plate-finetune-v1n.pt`), AGPL-3.0 (same as Ultralytics). Trained on Roboflow's
  `license-plate-recognition-rxg4e`. Its README says that set's test split is contaminated, so
  its own metrics are not trusted here; we measure it on Indian plates (below).

## Privacy masking (Step 5.5): evaluation data
Placement: `data/vision/raw/privacy/`. Used only to measure masking, never for training.
| Source | Licence | What we use |
|---|---|---|
| WIDER FACE val (Yang et al. 2016, via HF `CUHK-CSE/wider_face`: `WIDER_val.zip`, `wider_face_split.zip`) | non-commercial research (HF card: CC BY-NC-ND 4.0) | 3,226 photos, 39,708 labelled faces; split into tune/test halves by event |
| Kaggle `deepakat002/indian-vehicle-number-plate-yolo-annotation` | CC0-1.0 | 160 dashcam frames (1920x1080) from 3 videos on Indian roads, 261 hand-labelled plates; tune = vid-1, test = vid-2 + vid-3 |
- Not used: `saisirishan/indian-vehicle-dataset` (CC BY-NC-ND), `kedarsai/...` (most labels were
  made by a YOLO model, so they are not ground truth).
- Placement: `wider_face/{WIDER_val,wider_face_split}/`, `plates_india/vid-{1,2,3}/*.jpg|txt`.
