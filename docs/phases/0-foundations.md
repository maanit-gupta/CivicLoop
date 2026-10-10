# Phase 0: Foundations

Goal: the shared contracts and synthetic world every model module builds on. After this phase,
any later module can be built and tested on its own.

## 0.1 Repo skeleton
- Folders: `contracts/ triage/ router/ trainer/ intake/ vision/ verifier/ data/ models/ notebooks/ docs/`,
  each module with `__init__.py`, `api.py` (stub), `tests/`.
- `pyproject.toml` with base deps (pydantic, numpy, pandas, pytest, ruff) and extras per module:
  `triage`/`router`/`trainer` (lightgbm, scikit-learn, mlflow, pyarrow), `intake` (faster-whisper,
  transformers, langdetect or fasttext-langid, ollama), `vision` (ultralytics, open_clip_torch, pillow),
  `verifier` (imagehash, scikit-learn).
- `config.py` with `SEED`, paths, and `MODEL_VERSIONS` dict.
- `.gitignore` for data/, models/, mlruns/, .venv.
- Done when: `pip install -e .` and `pytest` (one smoke test) pass.

## 0.2 Contracts: schemas, interfaces, stubs
- `contracts/schemas.py` (Pydantic): `RawReport`, `Report`, `GeoPoint`, `Detection`, `ImageFeatures`,
  `Case`, `TriageFeatures`, `TriageDecision`, `RoutingDecision`, `WorkOrderDraft`, `Override`,
  `DecisionLog`. Every decision model has `model_version`, `confidence`, `explanation`.
- `contracts/interfaces.py`: Protocols for each module's public API
  (`IntakeAPI.process(RawReport) -> Report`, `VisionAPI.analyze(image_path) -> ImageFeatures`,
  `VerifierAPI.check(Report, open_cases) -> VerifierResult`, `TriageAPI.score(TriageFeatures) -> TriageDecision`,
  `RouterAPI.route(Case) -> RoutingDecision`).
- `contracts/stubs.py`: dumb rule-based implementations of every interface, so any module can be
  tested before the module it depends on exists.
- `contracts/fixtures/`: 5 example Cases as JSON, validated by a test.
- Explain to the user: why contracts first makes later integration a wiring job, not a rewrite.

## 0.3 Hash-chained decision log
- `contracts/decision_log.py`: append-only JSONL. Each entry: timestamp, module, model_version,
  case_id, input, output, human_override (null), prev_hash, hash = sha256(prev_hash + canonical JSON).
- `verify_chain(path)` returns the first broken index or None.
- Tests: tampering with any line breaks verification; appends are thread-safe enough (file lock).
- Explain: this one log is the audit trail AND the trainer's dataset.

## 0.4 Synthetic city
- Ask the user which real city/area to base it on (user chose Delhi).
- `data/world/`: ~12 wards as polygons (simple grid or OSM boundaries if available offline),
  POIs (schools, hospitals) with coordinates, road segments with type (main/residential/lane).
- `contracts/world.py` helpers: `ward_of(point)`, `distance_to_nearest(point, kind)`, `road_type_at(point)`.
  Pure Python (shapely) now; PostGIS later uses the same function signatures.
- `data/world/departments.yaml`: departments (Roads, Electrical, Sanitation, Water Supply,
  Stormwater Drains, Traffic & Signage) with issue types handled and SLA hours per severity.
- Tests on the geo helpers.

## 0.5 Synthetic complaint corpus
- Generator script (local LLM via Ollama if available, else templates with slot filling) producing
  ~3,000 complaints in Hindi, English, Punjabi and Hinglish (code-mixed) text, with ground-truth fields:
  issue_type, location_text, ward, cue flags (children, accident, flooding), true point.
- Include ~5% spam/irrelevant complaints and ~15% that describe the same incident as another one
  (shared `incident_id`) for the Verifier later.
- Save `data/corpus/complaints.parquet`. Print counts by language and issue type.
- This corpus is reused by Triage, Router, Intake and Verifier. Hold out 15% by `incident_id` NOW
  in a `split` column, and never change it.
