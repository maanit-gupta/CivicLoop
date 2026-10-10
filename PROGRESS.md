# Progress

Current phase: 6

## Handoff (from last step)
- 6.1 done. Files: `verifier/dataset.py` (`.venv/bin/python -m verifier.dataset`, ~95 s CPU), `verifier/images.py` (photo_view / reshare / screenshot), `verifier/tests/test_dataset.py`; verifier extra now has pillow (imagehash installed).
- Output `data/verifier/`: `reports.parquet` 3,450 rows (2,850 real, 150 text spam, 150 each reused/stock/screenshot fake photos; GPS noise sigma 15 m/axis, 10% missing; 52% of real reports carry a photo), `pairs.parquet` 4,518 (484 dup, 706 near_same_type, 908 same_place_other_type, 2,420 random; split = the later report's), `stock_index.parquet` 718 photos, `images/`.
- Labels: `incident_id` (dup), `is_spam` + `spam_kind`. `base_image`, `text_from`, `image_from` are provenance, never features. Fakes are not in pairs. `issue_type` is corpus truth (optimistic vs intake).
- Difficulty: dups median 25 m apart (94% <= 50 m) vs same_place_other_type 95 m (12-19%); dhash can't separate dup views from random (26-29 vs 31-32 bits), so dups need CLIP. Reused/stock fakes are dhash 3-3.5 from their source, screenshots 25.
- Open for 6.4 (user): should a screenshot or stock photo set `is_spam`, or only add a flag for an official? Every fake has its own `spam_kind`, so either policy can be scored.
- pytest 436/436, ruff clean. Still open from 3.6: `trainer.retrain` MLflow logging + saved weights, cumulative overrides, serving reads `config.MODEL_VERSIONS`.

## Deferred
- **Voice clips (from 4.1, user decision).** The 12 `AUDIO_IDS` clips (`intake/evaluate.py`) are still needed in `data/intake/audio/` (`<id>.m4a|wav|mp3|ogg|flac`, read verbatim from the challenge text). Without them there is no speech WER. `python -m intake.evaluate` lists the missing ones. 4.2 was ticked without its speech part (faster-whisper + WER, low ASR confidence -> needs_clarification); do it when the clips exist.

## Checklist
Build order is chosen so the learning loop (Triage -> Router -> Trainer) works early on cheap
tabular models; heavier perception models come after. Phases 4-6 only need Phase 0.

### Phase 0: Foundations (docs/phases/0-foundations.md)
- [x] 0.1 Repo skeleton
- [x] 0.2 Contracts: schemas, interfaces, stubs
- [x] 0.3 Hash-chained decision log
- [x] 0.4 Synthetic city (wards, POIs, departments, SLAs)
- [x] 0.5 Synthetic complaint corpus

### Phase 1: Triage model (docs/phases/1-triage.md)
- [x] 1.1 Severity rubric
- [x] 1.2 Triage dataset generator
- [x] 1.3 Evaluation + trivial baselines
- [x] 1.4 Train LightGBM v1
- [x] 1.5 Triage API with explanations + logging

### Phase 2: Router model (docs/phases/2-router.md)
- [x] 2.1 Routing labels + ward lookup
- [x] 2.2 Evaluation + lookup baseline
- [x] 2.3 Train router v1
- [x] 2.4 Work-order drafter + Router API

### Phase 3: Trainer (docs/phases/3-trainer.md)
- [x] 3.1 Simulated officials
- [x] 3.2 Override dataset builder
- [x] 3.3 Retrain v2 + offline comparison
- [x] 3.4 Model registry + canary
- [x] 3.5 (Stretch, GPU) LoRA SFT + DPO small LLM
- [x] 3.6 Trainer report

### Phase 4: Intake (docs/phases/4-intake.md)
- [x] 4.1 Evaluation + keyword baseline (done except the audio evaluation: see Deferred)
- [x] 4.2 Speech, language ID, translation (done except speech: see Deferred)
- [x] 4.3 LLM structured extraction (text only)
- [x] 4.4 Intake API

### Phase 5: Vision (docs/phases/5-vision.md)
- [x] 5.1 Image dataset
- [x] 5.2 Evaluation + CLIP zero-shot baseline
- [x] 5.3 YOLO fine-tune (GPU notebook)
- [x] 5.4 CLIP embedder + image features
- [x] 5.5 Privacy masking + Vision API

### Phase 6: Verifier (docs/phases/6-verifier.md)
- [x] 6.1 Duplicate-pair + spam dataset
- [ ] 6.2 Evaluation
- [ ] 6.3 Duplicate scorer
- [ ] 6.4 Authenticity + spam checks
- [ ] 6.5 Verifier API

## Decisions
(Record user decisions here: rubric weights, thresholds, city choice, etc.)
- 0.4 City: **Delhi** (centre Connaught Place 28.6139N 77.2090E). Keep all data and examples Delhi-specific. Languages follow from this: Hindi, English, Punjabi, Hinglish.
- 0.4 SLA hours (low/medium/high/critical): Roads 168/72/24/6, Electrical 72/48/24/4, Sanitation 48/24/12/4, Water Supply 72/24/8/2, Stormwater Drains 120/48/12/4, Traffic & Signage 168/72/24/6.
- 0.5 Corpus language mix: Hindi 35%, Hinglish 30%, English 25%, Punjabi 10%.
- 0.5 Cue labels (children/accident/flooding) are True only when the complaint text explicitly mentions them; proximity to a school etc. is a separate geo feature, not a cue.
- 1.1 Rubric weights approved as drafted (`triage/rubric.md`): base pothole/water_leak/drain_blockage 2, others 1; main road +1, school <200 m +1, hospital <300 m +1, 3-9 merged reports +1, 10+ +2, children +1, accident +2, flooding +2; points 0-2 low, 3-4 medium, 5-6 high, 7+ critical; accident floor = high.
- 1.1 No separate urgency output: severity (4 levels) + the 0.4 SLA table express urgency.
- 2.1 Ambiguous routing rules: (1) water on road: clean pipe water OR sewage from a manhole -> Water Supply (owns sewers, like Delhi Jal Board); rainwater not draining -> Stormwater Drains. Applied to the corpus too: its "sewer overflowing" sentence (drain_blockage template 2) -> Water Supply. (2) fallen pole: live wires/sparks/current -> Electrical; dead pole only blocking the road -> Roads. (3) garbage at a drain: inside the drain channel -> Stormwater Drains; on the bank or road beside it -> Sanitation, regardless of water flow.
- 2.3 Router confidence threshold: **0.7**. If the top probability is below 0.7, set `needs_human_routing` and send the complaint to a human.
- 3.1 Simulated official policies approved as drafted (`trainer/officials.py`): A (severity desk) never below **high** within **200 m** of a school (an absolute floor, not "+1 over the model"); B (routing desk) water_leak/drain_blockage + flooding -> **Stormwater Drains**, deliberately against rule 2.1; C covers **20%** of cases instead of A/B and changes a field at random **15%** of the time (noise).
- 3.3 Promotion authority: **officials' override rate decides**, with a guardrail: rubric-test MAE where no policy applies (school >= 200 m) may worsen by at most **0.05** (`trainer/compare.py: GUARDRAIL_MAE`).
- 3.3 Router v2 showed no measurable improvement (B 3.0% -> 2.6%): **accept it, keep router v1 as champion**; revisit with more live data. No new router feature for now.
- 3.4 Canary rule approved as drafted (`trainer/canary.py`): **10%** of cases to the challenger (sha256(case_id) split), decide only after **100** challenger cases, promote if its override rate is lower with one-sided two-proportion z-test **p < 0.05**, else roll back. Count **all** officials' overrides (C's noise included).
- 3.5 User chose to **do the GPU stretch** rather than skip to 3.6.
- 3.5 DPO run 1 collapsed to "high": user chose **balanced pairs + rerun**, with anchor pairs 1:1 against the override pairs (`ANCHORS_PER_OVERRIDE = 1.0`). An anchor is an accepted case >= 200 m from a school; chosen = accepted severity, rejected = "high".
- 3.5 Run 2 drifted to "critical": user ordered **run 3 as the final attempt**: DPO + SFT loss (1:1), lr 1e-5, 1 epoch, anchors' rejected label balanced across severities (any accepted far case can now be an anchor), and a notebook stop if one severity is > 60% of 150 eval answers. **Hard stop:** if run 3 fails, record the negative result, keep LightGBM v2, mark 3.5 done, move to 3.6. No run 4.
- 3.5 User let Claude run Kaggle through the Kaggle CLI (`.venv`). Rules: never print or commit the token, **ask before every push** (dataset version or kernel run), and check the uploaded `dpo_train.jsonl` against the local file by hash before running.
- 3.5 Closed after run 3 (no run 4). llm-dpo is not promoted: it fails the 3.3 guardrail (+0.180 rubric MAE far from schools). LightGBM v2 stays.
- 4.1 Voice clips: **the user records them** (no TTS) so the WER reflects real speech. 12 clips, 3 per language (`AUDIO_IDS` in `intake/evaluate.py`), read verbatim. The reference transcript is the challenge item's text. Kept local in `data/intake/audio/` (gitignored).
- 4.1 Voice clips **deferred** by the user: 4.1 ticked without the audio evaluation (listed under Deferred). Go straight to **4.3 (text only)**, scored on the same challenge set as the 4.1 baseline. 4.2's speech part is skipped until the clips exist.
- 4.2 Translator: user chose **IndicTrans2** (accepted the gated HF terms and logged in) over using qwen3:4b for every language.
- 5.1 Vision classes: `damaged_pole` (broken poles / hanging wires, the only public electrical images) **maps to IssueType.streetlight**. Detection classes: pothole, garbage, damaged_sign, damaged_pole. Image labels add "other" and "none".
- 5.1 drain_blockage / water_leak: **left uncovered by vision for now** (no public set without a Roboflow account). Those cases rely on intake text. Known gap, may be revisited.
- 5.3 User approved both Kaggle pushes for 5.3: private dataset `jagpreetsingh02/civicloop-vision-yolo` (yolo.zip, hash-checked by the notebook) and kernel `civicloop-yolo-v1` on a T4.
- 5.3 YOLO image-label threshold **tau = 0.10** (best val macro-F1), chosen over Ultralytics' 0.25: a missed issue costs more than a false alarm (val pothole recall 0.92 vs 0.78; clean roads flagged 17% vs 6%). `vision/yolo_detect.py: LABEL_THRESHOLD`.
- 5.3 **Keep yolo_v1** (60 epochs, val mAP50 +0.001 over its last 5 epochs). No longer v2 run; the "other" gap goes to CLIP in 5.4.
- 5.5 Masking operating point: **max privacy** (user chose it over "fewer stray blurs" and "fast"): faces YuNet >= **0.6**, pad 15%; plates YOLO11n >= **0.05**, pad **50%**, input **1280**; plate boxes over **5%** of the photo are dropped (road signs). Picked by the rule: best tune recall with <= **1%** of the civic issue area hidden on val. Masking runs **before** the models; only the masked copy is stored.
