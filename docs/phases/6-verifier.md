# Phase 6: Verifier (duplicates, authenticity, spam)

Goal: one physical issue becomes one case; spam and fake images are filtered.
Needs: Phase 0. Uses `vision.api` if Phase 5 is done, otherwise the vision stub.

## 6.1 Duplicate-pair + spam dataset
- From the corpus `incident_id`: positive pairs (same incident) and hard negatives
  (same issue type, nearby but different incident; same place, different issue type).
- Attach images from the Phase 5 dataset where possible (same incident = augmented crops of one
  image; different = different images).
- Spam: corpus spam rows plus fake-image cases (stock/screenshot/reused-image hashes).

## 6.2 Evaluation
- `verifier/evaluate.py`: duplicate precision/recall/F1 at a threshold, PR curve, and the cost of
  each error type (false merge vs missed merge). Spam precision/recall.
- Ask the user which error is worse; that decides the operating threshold.

## 6.3 Duplicate scorer
- Features per pair: geo distance, time gap, same issue_type, text embedding similarity,
  image embedding similarity. Start with a weighted rule, then a logistic regression on pairs.
- Candidate generation: only compare against open cases within a radius/time window (explain why:
  avoids O(n^2)).

## 6.4 Authenticity + spam checks
- Perceptual hash against previously seen images (reused photos), EXIF sanity (missing/far GPS),
  near-duplicate of known stock images, text spam classifier (TF-IDF + logistic regression).
- Output an authenticity score with reasons; low score -> human review, never silent rejection.

## 6.5 Verifier API
- `verifier/api.py: check(Report, open_cases) -> VerifierResult` (merge_into case_id or new case,
  authenticity, spam flag, reasons). DecisionLog. Demo: 3 reports of one pothole merge into one case.
