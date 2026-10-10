# Phase 5: Vision (detection, embeddings, masking)

Goal: one shared image module used by Verifier and Closure.
Needs: Phase 0 only.

## 5.1 Image dataset
- Download a public road-damage/pothole dataset (e.g. RDD2022 subset or a Roboflow pothole set)
  plus small public sets for garbage, open drains, streetlights if available.
- If downloads are blocked, ask the user to download and place them in `data/vision/raw/`.
- Convert to YOLO format, train/val/test split, record licenses in `data/vision/SOURCES.md`.

## 5.2 Evaluation + CLIP zero-shot baseline
- `vision/evaluate.py`: mAP50 for detection; accuracy for image-level issue type.
- Baseline: CLIP zero-shot ("a photo of a pothole", ...) classification on CPU.

## 5.3 YOLO fine-tune (GPU notebook)
- `notebooks/yolo_finetune.ipynb` for Colab/Kaggle: YOLO small model, short training run,
  export weights. The user runs it and drops weights in `models/vision/yolo_v1/`.
- Locally: evaluate the weights with 5.2 and compare to the CLIP baseline.

## 5.4 CLIP embedder + image features
- `vision/embed.py`: normalized CLIP image embeddings (ViT-B/32, CPU).
- `vision/features.py`: `ImageFeatures` = detections, issue_type, detection_count,
  box_area_ratio, detector_confidence, embedding. These feed TriageFeatures.
- `compare(before, after)` helper for Closure later (embedding similarity + detection change).

## 5.5 Privacy masking + Vision API
- Blur faces and number plates (an open face detector + a plate detector or OCR-region heuristic)
  before any image is stored.
- `vision/api.py: analyze(image_path) -> ImageFeatures`. DecisionLog. Demo + tests.
