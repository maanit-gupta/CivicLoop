# Learning notes

One entry per step, newest at the bottom. Format:

## X.Y Step name
- Built:
- Why:
- I should be able to explain:
- Key numbers:

## 0.1 Repo skeleton
- Built: 7 top-level packages (contracts, triage, router, trainer, intake, vision, verifier), each
  with a docstring-only `api.py` and `tests/`; one `pyproject.toml` with per-module extras;
  `config.py` (SEED=42, paths, MODEL_VERSIONS); `.gitignore`; Python 3.11 conda env in `.venv`.
- Why: the folder layout enforces the architecture. `api.py` is the only door into a module, so
  later integration is wiring, not refactoring. Extras keep heavy deps (torch, whisper) out of
  modules that don't need them. One SEED and one path source make every run reproducible.
- I should be able to explain: why pytest runs with `--import-mode=importlib` (seven `tests/`
  folders would otherwise clash on duplicate test-file names), and why extras beat one
  requirements file per module (one install, one place to pin, same isolation benefit).
- Key numbers: 2 smoke tests pass; ruff clean; Python 3.11.17.

## 0.2 Contracts: schemas, interfaces, stubs
- Built: Pydantic schemas for every boundary object (`RawReport` -> `Report` -> `Case`, plus
  `ImageFeatures`, `VerifierResult`, `TriageFeatures`/`TriageDecision`, `RoutingDecision`,
  `WorkOrderDraft`, `Override`, `DecisionLog`); `Protocol` interfaces for the 5 model APIs;
  keyword/rule stubs for each; 5 synthetic fixture Cases (en, ta, hi, code-mixed).
- Why: contracts first makes later integration a wiring job, not a rewrite. Each module can be
  built and tested alone against stubs/fixtures. All model outputs inherit one `Decision` base,
  so `model_version`/`confidence`/`explanation` can't be forgotten. `extra="forbid"` turns
  typos into errors at the boundary.
- I should be able to explain: structural (`Protocol`) vs nominal (ABC) typing, and why
  `isinstance(x, Protocol)` only checks method names exist (hence the signature test).
  Also why the stubs double as the trivial baselines every later model must beat.
- Key numbers: 19 tests pass; 12 schemas, 5 protocols, 5 stubs; stub intake 3/5 correct on
  fixtures (fails both non-English ones).

## 0.3 Hash-chained decision log
- Built: `contracts/decision_log.py`, an append-only JSONL log. `append` / `log_decision` write
  entries, `read_log` loads them, and `verify_chain` returns the first broken index or None.
  Each entry stores `hash = sha256(prev_hash + canonical JSON)`; the first entry's prev_hash is 64 zeros.
  `fcntl.flock` serialises concurrent appends. The default path is `config.DECISION_LOG_PATH`.
- Why: one file is both the audit trail (who/what/which model version/on what input) and the
  trainer's dataset (input, model output, human_override). Chaining makes silent edits
  detectable: re-hashing an edited line still breaks the *next* line's prev_hash.
- I should be able to explain: why the JSON must be canonical (sorted keys, fixed separators)
  before hashing, and what a hash chain does NOT protect against (rewriting the whole tail;
  anchoring the latest hash somewhere external closes that gap).
- Key numbers: 16 new tests (35 total) pass. Edits, deletes, reorders and garbage lines are
  all caught at the right index. 8 threads x 25 appends = 200 entries, chain intact. Demo:
  20 entries logged, tampering line 6 is caught at 6.

## 0.4 Synthetic city (Delhi)
- Built: `contracts/world_gen.py` generates 12 wards (a 4x3 grid of ~2 km cells with
  jittered shared corners), 58 straight roads (main every 2 km, residential every 500 m, lanes
  every 250 m) and 98 POIs (school, hospital, market, bus_stop) around Connaught Place. It writes
  them as lon/lat GeoJSON to `data/world/`. `contracts/world.py` provides `ward_of`,
  `distance_to_nearest`, `road_type_at`, `department_for` and `sla_hours`. SLAs come from the
  hand-written `data/world/departments.yaml`.
- Why: Triage and Router need geo features (ward, road type, distance to school/hospital), and
  every module needs the same answer to "which ward is this?". Generated, not OSM: deterministic,
  offline and testable. Fixtures were moved to Delhi, and Tamil/Tanglish became Punjabi/Hinglish.
- I should be able to explain: why we project lon/lat to local metres before measuring (a degree
  of longitude is ~97.7 km here, not 111 km), and why shared jittered corners still tile the city
  with no gaps or overlaps.
- Key numbers: 20 new tests (55 total) pass. A naive "divide by 2 km" ward lookup is wrong for
  7.1% of points (jitter matters). Flat projection vs haversine: max 0.14% distance error.

## 0.5 Synthetic complaint corpus
- Built: `contracts/corpus_gen.py` (logic) + `contracts/corpus_templates.py` (hand-written phrase
  bank in hi/en/pa/Hinglish). It samples *incidents* (issue, true point near a POI or on a road,
  cues), then writes 1-4 *complaints* per incident in different languages and wordings, plus 5%
  spam. Output: `data/corpus/complaints.parquet`, read via `contracts.api.load_corpus(split)`.
- Why: every later model needs labelled complaints; generating them from the world makes the
  labels exact. Templates, not an LLM: no Ollama here, and slot filling guarantees that
  `location_text` is literally in the text and cue flags match the text exactly.
- I should be able to explain: why the held-out split is by `incident_id` and is a hash of it
  (duplicates can't leak across train/test, and regenerating the corpus can never move an
  incident between splits), and why template text inflates scores (`template_id` lets us check).
- Key numbers: 3000 complaints, 2550 incidents, 5.0% spam, 15.0% duplicates, 15.8% of incidents
  held out. Issue accuracy: majority class 22.0%, English-keyword stub 41.2% (en 83.5%, Hinglish
  55.2%, hi 7.6%, pa 5.7%). Stub "children" cue: 127/250 English hits are false positives because
  the location mentions a school. 9 new tests (64 total) pass.

## 1.1 Severity rubric
- Built: `triage/rubric.py` with `rubric_score(features) -> (severity, points, reasons)`. Each
  issue type has base points, risk modifiers add points (main road, school/hospital nearby,
  merged reports, children/accident/flooding cues), and thresholds map the total to the 4
  `Severity` levels. A reported accident is never below high. Weights and 6 worked Delhi
  examples are in `triage/rubric.md`, and the examples are pinned by tests.
- Why: the triage model needs labels, and a human-approved policy is the honest source of them.
  Integer points can be audited line by line, unlike float weights, and every point comes with
  a reason, so the explanation is exact. Urgency was dropped because severity plus the SLA
  table already gives each case a deadline.
- I should be able to explain: why the model in 1.4 *imitates* this rubric at first (it is the
  policy), and how it can later beat the rubric (by learning from official overrides that the
  rubric can't express).
- Key numbers: on 2400 corpus incidents: low 56.7%, medium 30.4%, high 12.4%, critical 0.5%.
  24 new tests (88 total) pass.

## 1.2 Triage dataset generator
- Built: `triage/generate_data.py` -> `data/triage/cases.parquet` (8000 cases). Each case takes
  a non-spam corpus incident (issue type + text cues), moves it up to 150 m, and computes ward,
  road type and school/hospital distances with `contracts.world` helpers. duplicate_count is
  negative-binomial (mode 0, long tail). image_detections depends on issue type only (the
  rubric ignores it, so the model should learn to ignore it too). Label = rubric severity with
  8% shifted one level. Split 70/15/15 by 500 m grid cell (`location_cluster`).
- Why: a clean formula label lets a tree model copy the thresholds and score ~100%, which proves
  nothing and hides how it handles disagreement. Noise also sets a ceiling (~92%): beating it
  means overfitting. A random row split would leak jittered copies of the same place into test.
  25% of cases are an "enriched" stratum (risky incidents, more duplicates) because natural
  critical is ~1%. `stratum` is kept so eval can report natural-only numbers.
- I should be able to explain: why splitting by location cluster gives an honest test score,
  and why oversampling changes the class prior the model sees.
- Key numbers: train/val/test 5568/1219/1213 over 192 clusters (135/28/29). Critical 5.3/3.8/5.8%.
  Noise 7.7% (ceiling 92.3%). 10 new tests (98 total) pass.

## 1.3 Evaluation + trivial baselines
- Built: `triage/evaluate.py` with hand-written numpy metrics on level indices 0..3: MAE,
  quadratic weighted kappa (QWK), within-1, exact accuracy, critical recall, confusion matrix,
  per-issue-type MAE. Each row is reported on all cases and on the natural stratum. Predictors
  are plain functions `df -> level indices`, so v1 plugs into the same `score_table`.
- Baselines (fitted on train): `train_mean` (constant, rounds to medium) and `issue_type_mode`
  (most common train label per issue type). A literal "rubric base severity" baseline would be
  "always low" (base points 1-2 all map to low), so the data-driven lookup is used instead.
  `rubric (ceiling)` is the noise-free formula: the best any model can do.
- Why: severity is ordinal, so a low->critical miss must cost more than low->medium. QWK
  penalises squared distance and subtracts chance agreement; plain accuracy/macro-F1 don't.
- I should be able to explain: why any constant predictor has QWK exactly 0 (O equals E).
- Key numbers (test, all): train_mean MAE 0.715 / QWK 0.000; issue_type_mode MAE 0.618 /
  QWK 0.307 / exact 0.492 / crit recall 0; rubric MAE 0.076 / QWK 0.954 / exact 0.924.
  9 new tests (107 total) pass.

## 1.4 Train LightGBM v1
- Built: `triage/train.py`: a LightGBM regressor on level index 0..3, rounded and clipped to a
  level. Native categorical splits for issue_type/ward/road_type, with category lists frozen in
  `meta.json` so later predictions encode them the same way (unseen ward -> missing). Early
  stopping on val, a 6-point grid (num_leaves x min_child_samples) chosen by val QWK, test
  scored once. Saved to `models/triage/v1`, logged to MLflow (SQLite in `mlruns/`).
- Why: ordinal target -> regression keeps "low->critical is worse than low->medium" (a 4-class
  classifier would not). Trees learn threshold rules ("school < 200 m") like the rubric's.
- Learned vs rubric: all 13 one-change probes land on the rubric's level. Importance follows
  the rubric's big weights (flooding .29, accident .27, duplicates .17); ward .001 and
  image_detections .000, which the rubric ignores. Weakest spot: hospital < 300 m (raw 0.60,
  just above the 0.5 rounding line); it is rare, so the model has few examples of it.
- I should be able to explain: why 88 of the 99 test "errors" are cases where the model is
  right and the label is noise, and why that means beating 92% exact accuracy here would be a
  warning sign of overfitting, not a win.
- Key numbers (test, all): v1 MAE 0.083 / QWK 0.948 / exact 0.918 / crit recall 0.843;
  ceiling 0.076 / 0.954 / 0.924 / 0.886; baseline QWK 0.307. Natural-stratum crit recall
  0.455 vs ceiling 0.545 (only 11 natural critical cases). 7 new tests (114 total) pass.

## 1.5 Triage API with explanations + logging
- Built: `triage/api.py: score(features, version=None, log_path=...)` loads the version named in
  `config.MODEL_VERSIONS["triage"]` (now "v1"; "stub" still works), logs every call.
  `triage/explain.py` turns LightGBM `pred_contrib` (TreeSHAP) into "Severity high. Main
  reasons: ... Counting against: ...". `triage/calibrate.py` writes `calibration.json`.
  `python -m triage.demo` scores the 5 fixtures (5/5 agree with the rubric).
- Why: contributions satisfy bias + sum = raw prediction, so the sentence is the model's actual
  arithmetic, not a story added afterwards. Reasons are relative to an average complaint, which
  is why "no flooding mentioned" can appear under "Counting against".
- Confidence = val accuracy of the bin the rounding margin falls in (distance from the raw level
  to the nearest rounding boundary). Rejected: 0.5 + margin directly (ECE 0.114, overconfident).
- I should be able to explain: why calibrated confidence barely beats a constant here (Brier
  0.0737 vs 0.0750): most test errors are random label noise, which no feature of the case can
  predict. Only the margin < 0.1 bin is genuinely shaky (test accuracy 0.72 vs ~0.92).
- Key numbers (test): Brier constant 0.0750 / raw margin 0.0923 / binned 0.0737; ECE 0.009 /
  0.114 / 0.009. 10 new tests (124 total) pass.

## 2.1 Routing labels + ward lookup
- Built: `router/dataset.py` -> `data/router/{train,test}.parquet` (3300 rows). Department label
  = corpus issue_type through departments.yaml, plus 450 generated ambiguous complaints
  (`router/templates.py`: water on road, fallen pole, garbage in drain) in all 4 languages.
  `router/ward.py: ward_for()` = GPS point-in-polygon, then ward name in the text, else None.
- Why: the ward is a fact about polygons, so it is a deterministic tool: it is always right,
  free to run, and explains itself. A model could only add errors. ML goes where the answer
  depends on fuzzy evidence: the department of an ambiguous complaint.
- Ambiguous design: one shared sentence (no department info) + one cue sentence that decides.
  The observed issue_type is drawn at random from the two plausible labels, so it carries no
  signal. Tests check both departments appear for every (scenario, issue_type) pair.
- Rules are the user's (see Decisions). "Sewage -> Water Supply" also changed 101 existing
  corpus rows: the corpus's "sewer overflowing" sentence was filed under drain_blockage. A label
  rule must hold across the whole dataset, or the model sees the same words with two labels.
- I should be able to explain: why the lookup baseline is near 50% on the ambiguous subset
  (45.9%) by construction, and why that subset needs to be reported separately in 2.2: at 17%
  of rows it barely moves overall accuracy.
- Key numbers: 2791 train / 509 test rows; 551 ambiguous, 84 in test. Ward lookup GPS 100%,
  text-only 100% (synthetic text always names the ward; real complaints often won't).
  12 new tests (136 total) pass.

## 2.2 Evaluation + lookup baseline
- Built `router/evaluate.py`: accuracy, macro-F1, confusion matrix, accuracy on clear vs
  ambiguous rows, and ambiguous accuracy per scenario. Baselines: majority (always Roads) and
  lookup (issue_type -> departments.yaml). 7 tests.
- Why: the eval exists before any model, so 2.3 is judged against a fixed bar. Macro-F1 weights
  each department equally, so a model can't hide by ignoring small ones like Traffic & Signage.
- Lookup is perfect on clear rows (100%) and a coin flip on ambiguous ones (50%), but overall it
  still scores 91.7%. That's why the ambiguous subset has its own number.
- sewer_overflow lookup = 0%: the corpus labels it drain_blockage -> Stormwater, but the user
  rule says Water Supply. Lookup misses it every time, so any v1 model has to read the text.
- I should be able to explain: why the lookup's 91.7% accuracy is misleading, and which single
  number decides whether router v1 is worth shipping (ambiguous accuracy, now 50%).
- Key numbers (test, 509 rows / 84 ambiguous): majority acc 0.273, macro-F1 0.072; lookup acc
  0.917, macro-F1 0.913, ambiguous 0.500 (fallen_pole 0.714, garbage_in_drain 0.667,
  water_on_road 0.429, sewer_overflow 0.000). 143 tests pass.

## 2.3 Train router v1
- Built `router/train.py`: char 2-5 n-gram TF-IDF of the original text + one-hot issue_type and
  road_type -> multinomial logistic regression. C picked by validation log-loss (grouped by
  incident_id); sigmoid calibration kept only if it lowers validation ECE (it didn't). Added
  `ece` and `threshold_table` to `router/evaluate.py`. Saved `models/router/v1`, MLflow
  experiment `router`. 8 tests.
- Why char n-grams: no translation exists yet (intake is Phase 4) and the text is hi/pa/en/
  Hinglish. Character pieces work in any script and survive spelling variants.
- Main lesson: test ambiguous accuracy is 0.988 (lookup 0.500), but the cue sentences come from a
  small phrase bank shared by train and test. In the unseen-phrasing check (one cue sentence
  removed from training) the router scores 0.62, about the same as lookup's 0.60. No C or
  calibration setting changed that. Confidence is also overconfident there: at 0.7 only ~7% of
  those rows go to a human. Calibration only holds on data that looks like the training data.
- I should be able to explain: why a 99.8% test score can still mean "memorised", and why a
  confidence threshold can't catch mistakes on inputs unlike the training data.
- Key numbers (test 509 / 84 amb): acc 0.998, macro-F1 0.998, clear 1.000, ambiguous 0.988,
  ECE 0.003; unseen phrasing 0.619 vs lookup 0.603. C=100, uncalibrated. 151 tests pass.
- Threshold decision (user): 0.7. On test, 0.5-0.8 give the same result (1 wrong auto-route);
  0.9 sends 6 cases to a human but still lets that 1 wrong one through. A threshold only works
  as well as the confidence scores behind it.

## 2.4 Work-order drafter + Router API
- Built `router/api.py: route(case) -> RoutingDecision`: department from model v1 (threshold read
  from `meta.json`), ward from the geo lookup (case.ward -> GPS -> location text -> "unknown"),
  `needs_human_routing`, and a `WorkOrderDraft` from `router/workorder.py` (fixed template;
  optional Ollama rewrite behind `use_llm`, rejected if it drops department/ward/severity/SLA).
  Contracts gained `needs_human_routing`, `work_order` on RoutingDecision and issue_type,
  location, before_photos, reasons and `status` ("draft"|"approved") on WorkOrderDraft.
  Explanations name the complaint words whose n-grams pushed hardest toward the department.
- Why a template, not an LLM: every fact in a work order must come from a field. A template
  can't make anything up; an LLM can only reword it, and a check throws the rewrite away if
  facts go missing. The router never sets "approved": that is a human's job.
- I should be able to explain: why the work order the router drafts (`routing.work_order`, kept
  as produced) is separate from `Case.work_order` (the version an official edits and approves).
- Key numbers (test 509, end to end through the API): dept acc 0.998 (stub 0.917), ambiguous
  0.988 (stub 0.500), ward 1.000 with or without GPS, to human 0.2%, 1 wrong auto-route (stub
  42), all work orders draft with the table SLA, log chain intact, ~10 ms/case. 169 tests pass.
- Demo lesson: "dead pole, no wires, blocking the road" (should be Roads, rule 2.1) goes to
  Electrical at 0.92. That's the unseen-phrasing weakness from 2.3, now visible in a real call.
  Some cue words are noise ('the', 'raha'), which shows the model also picked up template quirks.

## 3.1 Simulated officials
- Built `trainer/live.py` (fresh live batch: corpus generator with seed SEED+1, `L-` ids, 92 days
  after the corpus window, no spam, one case per incident; 960 cases) and `trainer/officials.py`
  (3 hidden policies, `simulate()` runs triage/router v1 via their APIs and appends `Override`s
  as new hash-chained entries in `data/trainer/decision_log.jsonl`). Overrides are SIMULATED.
- Policies: A (severity desk) never below HIGH within 200 m of a school; B (routing desk) water
  leak/drain blockage with flooding -> Stormwater Drains, against rule 2.1; C covers 20% of cases
  instead of A/B and changes a field at random 15% of the time (noise).
- Why absolute policies: "one level above whatever the model says" has no fixed point. A model
  that learned it would get bumped again, so the target keeps moving.
- I should be able to explain: why a depth-4 tree recovering A and B (F1 1.00) but not C (0.22
  vs 0.30 guessing) means only A and B are worth learning. That is the 3.3 goal in one table.
- Key numbers: 294 overrides. A 198/752 cases (26%), B 30/752 (4%), C 66 fields on 208 cases.
  32.7% of live cases are within 200 m of a school. Live texts identical to train texts: 0.2%.
- Decision (user): the default policies were approved as drafted. B contradicting rule 2.1 is
  deliberate. 3.3 has to choose between the written rule and what officials actually do.

## 3.2 Override dataset builder
- Built `trainer/dataset.py`. It reads ONLY the hash-chained decision log (and refuses a broken
  chain), joins each case's triage/router model entries with override entries by (case_id,
  field), and gives every live case a final human label (no override = accepted). Outputs:
  retrain sets (original train rows weight 1 + live rows; overrides weight 3), a 30% held-out
  override set split by case, and DPO preference pairs (chosen = human, rejected = model).
- Why weight overrides instead of training on them alone: about 200 rows that all say "go
  higher" would teach "always bump" and erase the rubric. Mixing keeps the old knowledge.
- Why hold out by case before 3.3: an override the trainer has seen can't test whether it
  generalised. Rows from one case never sit on both sides.
- I should be able to explain: why "share of policy-matching labels where the policy applies"
  predicts how far v2 can move. Policy A: 37.6% (original) -> 49.8% (retrain set). The 5568
  rubric rows outnumber the 145 overrides, so even weighted ×3 the data is split roughly 50/50
  on "near a school -> high".
- Key numbers: triage set 5568 original + 491 accepted + 165 overrides; router 2791 + 610 + 46;
  211 pairs; holdout 304 cases / 83 overrides. Per 100 live cases: A 20.6, B 3.1, C 6.9.
  Water leaks have the highest department override rate (21%). 193 tests pass.
- Logging gap: acceptances aren't logged, so the log can't give per-official override rates
  (only counts). A real system should log "accepted" too.

## 3.3 Retrain v2 + offline comparison
- Built `triage/retrain.py` and `router/retrain.py`. Both are exposed as `retrain` /
  `predict_test` in each module's `api.py`, since the trainer may only import public APIs. They
  refit v1's exact recipe (params, rounds, C) on new rows with per-row weights. On the original
  rows alone they reproduce v1's test predictions exactly, so any v1 -> v2 change comes from the
  data. `trainer/retrain.py` picks the override weight (1/3/10/30) on a 20% validation slice of
  the live training cases (criterion: disagreement with the human label), then fits v2.
  `trainer/compare.py` shows v1 vs v2 on three views.
- Why three views: the old test sets are labelled by the written rubric, which policy A
  contradicts, so learning the officials looks worse there. "Holdout vs human label" is biased
  toward v1, because officials only reacted to v1's answer. Re-running officials on v2's outputs
  is the fairest test.
- I should be able to explain: why triage got WORSE on its test set (MAE 0.083 -> 0.178) and
  that is the intended result. Also why the router didn't move: 21 B overrides against 116
  contradicting rubric rows, and about 10 B cases in the validation slice, so the sweep is blind.
- Key numbers (holdout, 304 cases): A override rate 26.3% -> 5.6%, B 3.0% -> 2.6%, A+B
  overrides 89 -> 25. Severity QWK vs human 0.664 -> 0.897. Rubric MAE where no policy applies
  +0.031. Chosen weights: triage 10, router 1. Predicted >= high at 0-100 m from a school:
  47% -> 67%; beyond 200 m unchanged. 201 tests pass.
- Decisions (user): officials' override rate decides promotion, with a guardrail (rubric MAE
  where no policy applies may worsen by at most 0.05). Router: accept "no measurable
  improvement" and keep v1 as champion for now.

## 3.4 Model registry + canary
- Built `trainer/registry.py`, which holds the MLflow registry models `civicloop-triage` and
  `civicloop-router` in the same sqlite store as the training runs. It uses aliases
  `@champion`/`@challenger` instead of the deprecated stages. Promotion and rollback only move
  an alias, and nothing is deleted. `trainer/canary.py` splits cases with
  sha256(case_id) % 100 < 10 and reads override rates from the DecisionLog only. Its rule:
  wait while the challenger has under 100 cases; promote if the challenger's rate is lower with
  a one-sided two-proportion z-test p < 0.05; otherwise roll back. `trainer/run_cycle.py` runs
  collect -> build -> train -> offline gate (3.3 metric + guardrail) -> register -> canary on
  a new batch (`K` prefix, 1600 cases) in about 35 s.
- Why a hash rather than a coin flip: the same case always gets the same model, a run can be
  repeated exactly, and an audit can say which model a citizen got.
- I should be able to explain: why "promote if the rate is lower" is a bad rule. With one model
  on both sides (A/A), it promotes about 50% of the time; the z-test promotes about 5%. Also why
  the offline gate exists: in a second cycle, v3 forgot policy A (5.6% -> 23.4%) because, once
  v2 obeyed it, the policy's evidence showed up only as "accepted" rows at weight 1. The gate
  blocked v3.
- Key numbers (cycle 1): triage canary v2 2.8% vs v1 23.3% override rate, p = 8e-9 -> promoted.
  Router v2 9.9% vs 8.0%, p = 0.79 -> rolled back (matches the 3.3 decision). A/A false
  promotion: z-test 6.5% / 4.5%, naive 54.5% / 50.5%. 214 tests pass.
- Limitation: 141 challenger cases means the case mix can differ by chance. Here the challenger
  got 17.7% water-on-road cases vs 10.3% (a 1-in-125 split), which inflates the router
  challenger's override rate.
- Decision (user): canary defaults approved: 10% split, 100 minimum challenger cases, p < 0.05, all officials' overrides count.

## 3.5 (Stretch) LoRA SFT + DPO small LLM: part 1, built and waiting on the GPU run
- Built `trainer/llm_data.py`, which exports chat-format JSONL to `data/trainer/llm/`: 5568 SFT
  rows (the rubric-labelled train rows that v1 learned from), 165 severity DPO pairs (136 from
  A, 29 from C's noise, the same pairs v2 saw) and 1517 eval prompts (1213 test, 304 holdout).
  Also built `notebooks/triage_llm.ipynb` (Qwen2.5-1.5B-Instruct: LoRA SFT for 1 epoch, then
  DPO for 3 epochs with beta 0.1, then greedy answers) and `trainer/llm_eval.py` (majority,
  LightGBM v1/v2, llm-sft, llm-dpo).
- Output design: JSON `{"explanation", "severity"}`, with the explanation first. The
  explanation restates the rubric's facts and never mentions severity, so a chosen/rejected
  pair differs only in the severity word, and DPO's gradient lands on the decision. It also
  gives a faithfulness test: the explanation must equal `explain(row)`. A model that ignores
  its input fails that test even when its severity happens to be right.
- DPO's reference is the SFT model: TRL copies the SFT adapter as a frozen "ref" adapter. Step 1
  loss = ln 2 = 0.693 shows policy == reference at the start.
- Plain LoRA on an fp16 base, not 4-bit QLoRA. 1.5B in fp16 is about 3 GB on a 15 GB T4, and
  TRL's QLoRA path casts adapters to bf16, which the T4 lacks. Pinned TRL 1.15, transformers
  5.19, peft 0.21.2. A CPU smoke run with a tiny model passed every cell.
- I should be able to explain: SFT imitates one answer, while DPO only learns "this answer over
  that one" relative to a frozen reference. beta controls how far it may drift from the
  reference. With 136 near-school pairs, watch whether DPO also raises severity far from
  schools (MAE school >= 200 m).
- Baselines (rubric-test MAE / holdout A-override rate): majority 0.965 / 32.6%, LightGBM v1
  0.083 / 26.3%, v2 0.178 / 5.6%. LLM numbers come after the notebook run.

## 3.5 part 2: GPU run 1 scored; DPO collapsed, fixed with anchor pairs (waiting on run 2)
- Run 1 (rubric MAE / QWK / holdout A-override rate): llm-sft 0.101 / 0.938 / 26.3% (close to
  LightGBM v1: 0.083 / 0.948 / 26.3%). llm-dpo 1.151 / 0.000 / 0.0%: it said "high" for all 1517
  cases, worse than the majority guess (0.965). Run 1 is archived in `models/trainer/triage_llm_run1/`.
- Diagnosis: 144 of the 165 pairs were "low/medium -> high", and only 3 rejected "high". DPO
  only learns contrasts, so "never say low/medium" fits every pair and the school distance is
  never needed. Reward accuracy 0.99 meant it fit the pairs perfectly while learning the wrong
  rule. LightGBM v2 avoided this because it also trained on every accepted case.
- Fix (user chose it): `anchor_cases` adds 165 accepted live cases >= 200 m from a school
  (seeded sample). In those pairs chosen = the accepted severity and rejected = "high". Now "high"
  is rejected 168 times and chosen 144 times, so only the school distance can tell the pairs apart.
- I should be able to explain: preference data from overrides is one-sided (people only
  correct mistakes). DPO needs "keep this" pairs as well as "change this" pairs, otherwise it
  learns the direction of the corrections, not when to apply them.

## 3.5 part 3: run 2 drifted to "critical"; run 3 prepared (final attempt)
- Run 2 (anchors, rejected always "high"): llm-sft MAE 0.092 / QWK 0.944 (steady). llm-dpo MAE
  1.819 / QWK -0.041, with 75% "critical" and 22% "high". `rewards/chosen` ended at -0.49, so even
  the officials' answers lost probability. Archived in `models/trainer/triage_llm_run2/`.
- Diagnosis: DPO only trains the chosen-minus-rejected gap, and it can widen that gap by lowering
  both. The freed probability goes to an answer no pair rejects: "critical" was rejected 0 times
  out of 330 pairs. Run 1 had the same mechanism with "high" as the sink.
- Run 3 (user's spec): (1) loss = DPO + SFT on the chosen answer, 1:1 (the RPO idea), (2) lr 1e-5
  and 1 epoch, (3) the anchors' rejected label is the least-rejected other severity (ties
  random), giving rejected counts low 78 / medium 95 / high 79 / critical 78, (4) the notebook
  checks 150 eval answers right after DPO and stops if one severity is over 60%.
- I should be able to explain: a preference loss constrains only the answers that appear in
  pairs. Every answer the model can give needs to be rejected somewhere, or an anchor (SFT term)
  must hold the chosen answers up.

## 3.5 part 4: run 3 on Kaggle (CLI) and the verdict. Step done
- Ran through the Kaggle CLI: new dataset version (uploaded `dpo_train.jsonl` hash = local),
  kernel `civicloop-2` on a T4, about 27 min. Collapse check on 150 prompts: high 60 / low 48 /
  medium 28 / critical 14 (top 40%, under the 60% stop).
- Run 3 (DPO + SFT loss 1:1, lr 1e-5, 1 epoch, balanced rejected labels). `rewards/chosen` stayed
  positive (+0.17), unlike run 2 (-0.49), so the SFT term did its job.

  | | LightGBM v1 | LightGBM v2 | llm-sft | llm-dpo |
  |---|---|---|---|---|
  | rubric MAE / QWK | 0.083 / 0.948 | 0.178 / 0.883 | 0.096 / 0.941 | 0.430 / 0.726 |
  | MAE, school >=200 m | 0.087 | 0.118 | 0.096 | 0.267 |
  | holdout MAE vs human | 0.299 | 0.138 | 0.306 | 0.214 |
  | A override rate | 26.3% | 5.6% | 26.3% | 0.0% |

- Verdict: llm-dpo learned A's school policy (0% A overrides) but also leaks escalation onto
  far cases. 21% of them land above the rubric (94 medium -> high, 64 low -> medium). Its MAE
  where no policy applies is +0.180 vs v1, against the 3.3 guardrail of +0.05, so it is **not
  promoted. LightGBM v2 stays.** The guardrail did exactly what it was made for.
- Across the 3 runs: SFT was easy and stable. DPO from 165 one-sided overrides needed anchors,
  balanced rejections and an SFT term just to avoid collapse, and it still generalised the
  correction too broadly. With few overrides, supervised retraining on the full data (v2) wins.
- I should be able to explain: why plain DPO collapsed twice (it only trains the gap; anything
  never rejected soaks up probability), what the SFT term fixes (it holds chosen answers up),
  and why a policy-free guardrail metric catches a model that "wins" on override rate.

## 3.6 Trainer report
- Built `docs/trainer_report.md` (data sources, label generation, rubric, officials, training
  runs, before/after metrics, limitations) and `trainer/report.py`. Sections 5-6 sit between auto
  markers and are rewritten by `python -m trainer.report` from the saved models and the MLflow
  store. The rest is prose. A test checks that every run id the report cites exists in MLflow
  and that every registry version links to one.
- Gap found: only the v1 fits had MLflow runs. The v2 retrains (3.3) and the three LLM runs (3.5)
  were never tracked, and registry versions had no run id. Each now has a backfilled run (tag
  `backfilled=true`, source folder, folder file time as start). Params come from meta.json /
  train_log.json and metrics are recomputed. Registry versions got a `run_id` tag.
- Why backfill, not re-run with logging: a re-run rewrites the v2 folders the registry and canary
  point at, and the LLM runs can't be repeated (no GPU, hard stop after run 3). Backfilling records
  what happened and says it was recorded later.
- I should be able to explain: why report numbers should be regenerated by code, not copied
  (copies drift, and nothing checks them), and why "it's in MLflow" isn't the same as "it was
  tracked". A backfilled run is only as trustworthy as the artifact it was rebuilt from.
- Key numbers: the regenerated numbers match 3.3-3.5 exactly (A override 26.3% -> 5.6%,
  guardrail +0.031, llm-dpo +0.180). The v2 override weights (triage 10, router 1) were never
  saved; refitting with them reproduces both v2 models exactly, and weight 3 does not. 7 new
  tests (238 total) pass.

## 4.1 Evaluation + keyword baseline (voice clips pending)
- Built `intake/evaluate.py`, which scores any `text -> raw JSON string` extractor on JSON
  validity, language accuracy, issue accuracy + macro-F1, location exact match + token-F1 and
  F1 per cue, split by language. It also gives pooled WER/CER for voice clips. With it:
  `intake/extraction.py` (the one parser: invalid output scores zero on every field) and
  `intake/keywords.py` (language by script, issue by keyword counts, location as the span around
  gazetteer hits).
- Two test sets. The corpus test split uses the same templates as train, so it can't tell
  understanding from memorisation. `intake/fixtures/challenge.jsonl` has 48 hand-written items,
  12 per language: real Delhi places, wording outside the phrase bank, a negation ("nobody hurt
  yet"), confusers (बिजली का बिल, hanging wires) and items with no location. It was frozen before
  the rules were written, and the rules were tuned on train only.
- Key numbers (keywords): corpus test 1.00 on every metric. Challenge: issue accuracy 0.65
  (majority baseline 0.15), macro-F1 0.69, location exact 0.21 / token-F1 0.58, cue F1 children
  0.89 / accident 0.00 / flooding 0.29, language 1.00 from the script alone.
- All 17 issue errors are "-> other". The rules never mix up two real issues; they just don't
  know the word (खड्डा, गंदगी, gutter, "dark"). Location finds the ward but drops the actual
  landmark (Hanuman Mandir, Chuna Mandi). Caveat: 48 items means one item is ~2 points, and cue
  F1 rests on 4-6 positives each.
- I should be able to explain: why 100% on the template test split means nothing here, why the
  extractor returns a raw string (the 4.3 LLM's broken JSON gets scored by the same code), and
  why CER is kinder than WER to Hinglish spelling ("nahi" vs "nhi" is 1 character but a whole
  word).

## 4.3 LLM structured extraction (text only)
- Built `intake/llm.py`. qwen3:4b, run locally through Ollama, fills in issue_type, location_text
  and cues. Each text gets one call constrained by the JSON schema of `Fields` (`strict_schema`
  makes every field required), then Pydantic validation, one retry, and the keyword baseline as
  a last resort. Language stays with the script rule. location_text must be a span of the text,
  or the keyword span replaces it. If the server is down the extractor raises instead of falling
  back. Replies are capped at 256 tokens.
- The prompt is in `intake/prompts/`: label definitions from the corpus taxonomy and the 0.5 cue
  decision, train-vocabulary words for each issue, and 7 few-shot examples from the train
  split. It was iterated on two 60-row train samples only, and the challenge set was run once.
  A test checks that no challenge text or location appears in the prompt.
- Model: qwen2.5:3b (the phase file's example) is under the non-commercial Qwen Research
  licence, which breaks the open-source rule. qwen3:4b (Apache-2.0) tied it on train-dev issue
  accuracy (0.95), had better locations (0.87 vs 0.77 exact) and better cues, and took twice as
  long (2.6 vs 1.2 s per item).
- Key numbers (challenge, keywords -> LLM): issue accuracy 0.65 -> 0.96, location exact 0.21 ->
  0.85, location F1 0.58 -> 0.94, cue F1 children 0.89 -> 1.00, accident 0.00 -> 0.75,
  flooding 0.29 -> 1.00. Hindi is 1.00 on every field. Still wrong: 2 Punjabi potholes -> other
  (ਖੱਡਾ, ਉੱਖੜੀ ਸੜਕ), spans one word too long ("Gutter near Chuna Mandi..."), a whole sentence
  returned as the location when there was none, and "cycle se girte hain" not read as an
  accident.
- On the corpus test split the LLM (issue 0.98, location exact 0.92) scores *below* the
  keyword rules (1.00). The template set rewards memorisation and the challenge set rewards
  understanding: picking a model on the corpus numbers would pick the wrong one.
- Constrained decoding is what makes a 4B model usable. Without the schema, 0 of 48 replies were
  valid JSON: the model described its reasoning ("We are given a complaint...") despite
  think=False and the "JSON only" instruction. Uncapped, one such reply ran for hours and
  stalled the first evaluation. Fixed with num_predict 256, results printed after each run,
  and caffeinate. With the schema, the retry and the fallback never fired.
- The span check replaced 3 locations out of 524. One was a Gurmukhi span with a stray Urdu
  letter (ਨੇੜے). The rule caught it correctly, but the keyword replacement ("ਗਲੀ") was worse
  and cost 0.02 location F1. Trade-off accepted: never return place text that isn't in the
  complaint.
- I should be able to explain: how a schema constraint differs from asking for JSON in the
  prompt (a grammar masks invalid tokens at every step; a request can be ignored), why it still
  needs a token cap, and why the model was chosen on train dev, not on the challenge set.

## 4.2 Language ID + translation (speech deferred)
- Built `intake/translate.py`. Translation adds `english_text` and the original text is always
  kept. Hindi and Punjabi go to IndicTrans2 (distilled 200M, MIT, CPU). Hinglish goes to
  qwen3:4b through `llm.translate`, constrained to `{"english": ...}`. English passes through
  unchanged. Language comes from the existing script rule. Evaluation: `python -m
  intake.evaluate --translate --llm`, against 36 reference translations
  (`intake/fixtures/challenge_en.jsonl`, written by Claude before any model ran).
- Language ID needed no model. The script rule scores 1.00 on both sets; its only "misses" are
  2 corpus spam items labelled mixed. langdetect, a learned n-gram identifier, scores 0.75 /
  0.72 because it has no Hinglish label (0% on mixed). The majority class scores 0.25 / 0.35.
- Two metrics. chrF is a character n-gram F-score against the reference, so spelling variants
  cost little. The keep rate is the share of names and numbers that survive, with accepted
  spellings separated by "|". A fluent translation that turns "N Block" into "Block 'A'"
  breaks routing, and chrF barely notices.
- Key numbers (chrF hi / pa / mixed, keep rate):
  - Copying the input: 0.1 / 0.0 / 35.7, keep 0.39.
  - IndicTrans2: 65.4 / 66.6 / 33.9, keep 0.96.
  - qwen3:4b: 59.1 / 52.4 / 59.8, keep 0.79 (Hindi 0.63).
  - **Served route: 65.4 / 66.6 / 59.8, chrF 63.9 overall, keep 1.00**, about 2 s per item.
- Why the split:
  - IndicTrans2 was trained on native scripts, so it mostly copies romanised Hindi back,
    scoring below plain copying.
  - qwen understands Hinglish but changes names in Devanagari and Gurmukhi: "C-Heptagon",
    "Darigaon", "Kanata Place", and "Block 'A'" for N Block.
- Preprocessing is part of the model. IndicProcessor rewrites Gurmukhi into Devanagari (script
  unification) because IndicTrans2 learned Hindi and Punjabi in one script. Skipping it would
  feed the model characters it has hardly seen.
- Confidence: exp(beam score), the geometric-mean probability per output token. Spearman with
  sentence chrF is 0.56 on Hindi and Punjabi but -0.02 overall, because the model is confident
  while copying Hinglish. A confidence score only means something on inputs like its training
  data.
- Friction:
  - The model is gated on Hugging Face; the user accepted the terms and logged in.
  - IndicTransToolkit 1.1.1 breaks on transformers 5, so transformers is pinned below 5.
  - The 2024 remote code crashes with transformers 4.5x's KV-cache object, so
    `use_cache=False`. That gives the same output, more slowly.
  - Unconstrained, qwen narrates its reasoning (chrF 18.8). The 4.3 lesson again.
- Still wrong in the served route: chrF misses meaning errors.
  - ch-mx-07 (qwen) came out as "People are taking goods from the hands of bandar logon", the
    reverse of monkeys snatching from people.
  - ch-hi-01 (IndicTrans2) says the auto "skidded" where the source says it almost overturned.
- I should be able to explain: why language ID by script beats a learned identifier here, what
  chrF measures and what it misses, and why one translator per language beats one for all.

## 4.4 Intake API
- Built `intake/api.py: process(RawReport) -> Report`. It is a thin orchestrator over 4.1-4.3
  with one path: pick the text (typed text, else the voice transcript, else the photos' EXIF
  captions), detect the language with the script rule, extract fields with qwen3:4b (keyword
  rules if its reply stays invalid), translate (IndicTrans2 or qwen3:4b), and take the location
  from the report's GPS, else the photo's EXIF GPS. Each report gets one DecisionLog entry,
  keyed by report_id. `intake/photo.py` reads EXIF only: what a photo *shows* is vision's job,
  and intake may not import vision.
- The models are keyword parameters (`extract`, `translate`, `transcribe`), so the 17 tests run
  without Ollama or IndicTrans2. Some tests wrap the real `llm.extract` around a fake chat
  function, so the real retry, fallback and server-down behaviour is tested too.
- Confidence uses the weakest-link rule: `Report.confidence` is the minimum over the stages.
  - Extraction is worth the issue accuracy of the path that ran, measured on the challenge set
    in 4.3: LLM 0.96, keywords 0.65.
  - Translation and speech bring their own per-item scores. The LLM translator has none, so it
    adds nothing.
  - The explanation names the weakest stage.
  - Rejected: multiplying the stage scores. That is P(all right) only if the stages are
    independent and calibrated, and these are neither.
- Voice notes fail clearly (`NotImplementedError`) until the deferred speech step exists. A
  fake transcriber would have made audio reports look supported.
- Key numbers (challenge set end to end, stub -> v1): language 0.25 -> 1.00, issue 0.33 ->
  0.96, location F1 0.12 -> 0.94, cue F1 children/accident/flooding 0.57/0.00/0.00 ->
  1.00/0.75/1.00, english_text chrF 13.6 -> 63.9. These are exactly the 4.2/4.3 numbers, so
  the wiring loses nothing. About 2.6 s per report.
- An honest weak spot: mean confidence is en 0.96, mixed 0.96, hi 0.64, pa 0.66. Hindi
  extraction is perfect, but its confidence is pulled down by the translation branch, which
  doesn't feed extraction (extraction reads the original text). The 2 errors (Punjabi potholes
  -> other) scored 0.68 and 0.70, above the Punjabi mean. So this confidence flags weak
  translations, not wrong issue types.
- I should be able to explain: why the API's evaluation must reproduce the component numbers
  (it tests the wiring, not the models), what the min rule assumes, and why a stage without a
  score is left out rather than counted as 1 or 0.

## 5.1 Image dataset
- Built `vision/dataset.py`: `python -m vision.dataset` turns two public sources into one YOLO
  dataset in `data/vision/yolo/` (`images/`, `labels/`, `data.yaml`, `manifest.csv`, about 37 s).
  - RDD2022 India: dashcam frames, partly from Delhi/Gurugram.
  - Urban Issues Dataset (Kaggle, mostly Roboflow exports).
  - Licences and quirks are in `data/vision/SOURCES.md`.
- Every image has two kinds of label:
  - an image-level label (pothole, garbage, damaged_sign, streetlight, other, none), used for the
    5.2 accuracy;
  - zero or more boxes in 4 detection classes (pothole, garbage, damaged_sign, damaged_pole), used
    for the YOLO mAP.
  - "other" (fallen tree, graffiti, dead animal, illegal parking) and "none" (RDD frames with no
    damage) are YOLO background images. They teach the detector what not to box.
- Leakage is the main lesson.
  - Roboflow exports hold several augmented copies of one photo (`<photo>_jpg.rf.<hash>.jpg`),
    and some photos are filed under two classes.
  - We keep one copy per original photo, preferring the unaugmented valid/test copy. We also drop
    exact dHash matches (a 64-bit fingerprint that survives resizing and re-encoding).
  - Only then do we make a stratified 70/15/15 split.
  - Trivial baseline: reuse the source's own folders. 71% of its test photos (2,087/2,938) also
    sit in train. Ours: 0.
- Never trust labels without looking at them. Global class ids put stray boxes in the wrong
  folders, e.g. "pothole" boxes drawn on a graffiti figure and on a bird's nest on a pylon.
  - Each folder now has an allow-list of ids. Images with any other id are skipped (469).
  - Crack boxes are dropped, and crack-only photos are skipped. Cracks are not one of our issue
    types and would blur the pothole/other boundary.
- Key numbers:
  - Raw input: ~35k Urban Issues files and 7,706 RDD frames.
  - Dropped: 21,302 augmented/cross-folder copies, 80 pixel duplicates, 2,255 RDD frames with
    only non-pothole damage.
  - Output: 3,345 images (train 2,341 / val 502 / test 502). 600 per label, except
    damaged_sign with only 345 unique photos.
  - Boxes: pothole 1,631, garbage 1,221, damaged_pole 732, damaged_sign 347.
- Gaps (user decisions): no public drain/water-leak images, so vision can't see those types.
  damaged_pole stands in for streetlight.
- Known biases:
  - "none" is all dashcam footage, so a model could learn "dashcam view = no issue".
  - Some sign photos show intact signs.
  - 2/3 of the Urban Issues photos only exist as augmented train copies.
- I should be able to explain: why splitting by file leaks when a dataset contains augmented
  copies, what a dHash is and why an exact match catches re-encoded copies, and why
  background-only images matter for a detector.

## 5.2 Evaluation + CLIP zero-shot baseline
- Built `vision/evaluate.py` (`python -m vision.evaluate [--split val|test]`) and
  `vision/clip_zeroshot.py`. Results go to `data/vision/eval/clip_zeroshot_<split>.{json,csv}`.
- Two metrics, because there are two questions:
  - "What is in the photo?" Accuracy and macro-F1 over the 6 image labels. Macro-F1 because
    damaged_sign has fewer photos.
  - "Where is it?" mAP50 over the 4 box classes. Rank every predicted box by confidence. A box
    is a hit if it overlaps an unused true box of its class with IoU >= 0.5; a second box on the
    same pothole is a miss. AP = area under the precision-recall curve. mAP50 = mean over classes.
  - Our mAP is the VOC all-point version, written by hand and tested on hand-worked cases.
    Ultralytics prints a 101-point variant, so 5.3 scores YOLO with our code for a fair compare.
- CLIP zero-shot means no training. Write 3-4 sentences per label, embed them once, and give
  each photo the label of its closest sentence. A label's score is its best sentence (max), not
  the average: "other" is fallen trees + graffiti + parking + dead animals, and the mean of those
  embeddings matches none of them. The prompts were written once and never tuned.
  - Rejected: a linear probe on CLIP embeddings. It is trained, so it is no longer a
    zero-training floor. It belongs in 5.4.
- Key numbers (test, 502 photos; ViT-B-32 laion2b; ~18 ms/photo on CPU once loaded):

  | | accuracy | macro-F1 |
  |---|---|---|
  | majority label | 0.179 | 0.051 |
  | CLIP zero-shot | 0.715 (val 0.709) | 0.704 |

  - Strong on close-ups: damaged_sign recall 0.98, garbage 0.80, other 0.77, poles 0.70.
  - Fails on potholes: recall 0.17. 74/90 are called "none", and dashcam potholes score 0/36.
    The median pothole covers ~1.5% of the frame. A whole-image embedding sees "a road".
  - "none" recall 0.99 is not a success. CLIP calls every dashcam frame "none", which is
    exactly the 5.1 bias. That is why the report splits recall by source.
  - mAP50 with one full-frame box per class: 0.176. damaged_sign 0.60 (signs fill the frame);
    pothole 0.005. This is the floor YOLO must beat in 5.3.
- I should be able to explain: why accuracy alone can't judge a detector; what IoU >= 0.5 and
  the precision-recall area mean; why a duplicate box counts as a false positive; and why CLIP
  misses small objects in wide scenes (one vector for the whole image).

## 5.3 YOLO fine-tune (GPU notebook)
- Built: `notebooks/yolo_finetune.ipynb` (Kaggle/Colab), run on a Kaggle T4 through the CLI;
  `vision/yolo_detect.py` (CPU inference, boxes -> image label); `python -m vision.evaluate
  --model yolo` scores it with the same 5.2 code that scored CLIP. Weights: `models/vision/yolo_v1/`.
- Fine-tuning = transfer learning. YOLO26s already learned edges, textures and objects on COCO.
  We swap its 80-class head for our 4 classes and train on 2,341 photos for 60 epochs (45 min on
  a T4). "other" and "none" photos have no boxes: they are background images that teach it to
  stay quiet. YOLO is one-stage: one forward pass predicts boxes + class + confidence per grid
  cell, which keeps CPU serving at ~55 ms/photo.
  - Rejected: Faster R-CNN (two-stage). A bit stronger on small objects, but 5-10x slower on CPU
    and needs a hand-written training loop. YOLO comes with augmentation and early stopping.
- Boxes -> image label: the most confident box's issue if its confidence >= tau, else "none".
  tau was picked on **val** (best macro-F1 = 0.10), never on test. User chose 0.10 over 0.25: a
  missed issue costs more than a false alarm.
- The notebook refuses to train on the wrong data: it checks the manifest SHA-256 and the image
  counts against the local build. A CPU smoke run (1 epoch, 3% of train) tested the code path
  before any GPU time was spent. A local run can't see Kaggle's folder layout, though: the real
  mount path (`/kaggle/input/datasets/<owner>/<slug>/yolo`, 4 levels deep) showed up when listing
  the uploaded dataset, and the data search was widened before the kernel was pushed.
- Key numbers (test, 502 photos):

  | | accuracy | macro-F1 | mAP50 | ms/photo (CPU) |
  |---|---|---|---|---|
  | majority label | 0.179 | 0.051 | - | - |
  | CLIP zero-shot | 0.715 | 0.704 | 0.176 | 17 |
  | YOLO v1 (tau 0.10) | 0.741 | 0.705 | 0.599 | 54 |

  - Potholes: F1 0.84 (CLIP 0.28). Dashcam potholes recall 0.75 (CLIP 0.00). Boxes see small
    things that one whole-image vector misses.
  - "other": F1 0.00. YOLO has no "other" class, so 73/90 go to "none". That is why macro-F1 is
    flat even though every box class improved. CLIP's "other" F1 is 0.73: the two are
    complementary (5.4).
  - "none" precision 0.44: 16/90 clean dashcam roads get a pothole box. That is the price of tau 0.10.
  - Our VOC mAP50 on val 0.650 vs Ultralytics' 101-point 0.661: the two metrics agree closely.
    Test (0.599) is lower than val, because val picked best.pt and tau. Quote test.
- I should be able to explain: what fine-tuning reuses and what it replaces; why background
  images matter for a detector; why tau is tuned on val, not test; why a detector beats CLIP on
  potholes but cannot say "other".

## 5.4 CLIP embedder + image features
- Built: `vision/embed.py` (unit-length CLIP ViT-B/32 embeddings, CPU; `cosine`),
  `vision/features.py` (`run_models`, `build`, `analyze`, `compare`), 3 derived properties on
  `contracts.ImageFeatures`, `python -m vision.evaluate --model features`. CLIP's loader moved from
  `clip_zeroshot.py` into `embed.py`, so one CLIP pass per photo gives both the embedding and the
  zero-shot probabilities (`probs_from_embeddings`).
- An **embedding** is a 512-number fingerprint. Scaled to length 1, cosine similarity = dot
  product. The verifier (duplicates) and Closure (before/after) compare fingerprints.
- **Two models, one answer.** YOLO names the issue when a box reaches tau 0.10. Otherwise a
  fallback rule asks CLIP. 4 candidate rules, picked on **val** (macro-F1): "clip other vs none"
  0.900, yolo only 0.714. CLIP decides only between "other" and "none". Its view on potholes is
  ignored, because YOLO is far better there.
- **Derived, not stored.** `detection_count`, `box_area_ratio` (union of boxes, overlaps counted
  once), `detector_confidence` are `@property` on `ImageFeatures`. A stored copy could disagree
  with `detections`. They are not in the JSON, so round-trips with `extra="forbid"` still work.
  `detections` keeps only boxes >= tau; the near-zero ones only ever served mAP.
- `compare(before, after)`: cosine + issue_cleared + box counts. It refuses mismatched
  model_versions (embeddings from different models live in different spaces). No same-place
  threshold yet: deciding one needs real before/after pairs.
- Rejected: a classifier head trained on CLIP embeddings. It would learn the 5.1 bias (every
  "none" is a dashcam frame) and fail on citizen photos.
- Key numbers (test, 502 photos, 68 ms/photo CPU for YOLO + CLIP):

  | | accuracy | macro-F1 | "other" F1 | "none" F1 |
  |---|---|---|---|---|
  | majority label | 0.179 | 0.051 | 0.00 | 0.00 |
  | CLIP zero-shot | 0.715 | 0.704 | 0.73 | 0.66 |
  | YOLO v1 | 0.741 | 0.705 | 0.00 | 0.56 |
  | **YOLO + CLIP other/none** | **0.873** | **0.878** | **0.81** | **0.79** |

  - Box classes are unchanged (the fallback only fires when YOLO is silent): pothole 0.84,
    garbage 0.92, sign 0.97, streetlight 0.94. No clean road was turned into "other".
  - Remaining errors: 16 clean dashcam roads -> pothole (the price of tau 0.10), 11 potholes ->
    none, "other" photos with a stray box (7 -> garbage, 6 -> streetlight).
  - Regression after the refactor: CLIP probabilities within 0.0001 of 5.2, YOLO labels 100% equal.
  - Embedding check (each photo vs a re-shot copy: 85% crop, brightness, JPEG 70):

    | | same scene | other photo, same label | other label | AUC vs same label |
    |---|---|---|---|---|
    | CLIP cosine | 0.922 (p5 0.86) | 0.559 (p95 0.82) | 0.404 | 0.993 |
    | dHash (baseline) | 0.719 | 0.516 | 0.509 | 0.959 |

    This is a synthetic upper bound. A second citizen shoots from another angle and distance.
    The real duplicate evaluation is 6.2.
- I should be able to explain: why one embedding can serve both search and zero-shot; why derived
  features are properties and not fields; why the fallback rule is picked on val; why compare()
  must refuse embeddings from different model versions.

## 5.5 Privacy masking + Vision API
- Built: `vision/privacy.py` (YuNet faces + YOLO11n licence plates, padded Gaussian blur, EXIF
  rotation applied and metadata dropped, stored as `<sha256[:16]>.jpg`, fails closed),
  `vision/api.py: analyze(image_path, case_id=...)` (mask -> store -> YOLO+CLIP on the masked copy
  -> DecisionLog), `python -m vision.demo`, `python -m vision.evaluate --model privacy`.
- **Mask first, then everything else.** The models, the stored copy and the officials only ever
  see the masked photo, and `ImageFeatures.image_path` points to it. The masker is part of
  `model_version` because it changes the pixels the embedding is computed from.
- Rejected: run the models on the original and mask only the stored copy. Then an unmasked file
  must exist for every re-analysis, and the embedding describes pixels nobody can audit.
- **Measure on your own data, not the model card.** The plate model's README says its test split
  is contaminated. On Indian dashcam plates, at its usual 0.25 it masked only **4 in 10** plates.
  It is under-confident on Indian plates, and its boxes are tighter than the plate frame. Fix:
  threshold 0.05, **50% padding**, input 1280 (tune 0.58 -> 0.88).
- **The plate model reads road signs as plates** (a rectangle with letters, score 0.8, half the
  photo). That would blur the very sign being reported (val: 35% of sign area hidden). Real plates
  cover at most 0.6% of a frame, so boxes over 5% are dropped: 0.2% hidden. Keeping only plates on
  a COCO vehicle was worse (3.6%) and needs a third model.
- **Two-sided metric.** Privacy = faces/plates big enough to identify (face >= 20 px tall, plate >=
  40 px wide) with >= 80% of the box blurred. Harm = share of the civic issue's own box that gets
  blurred. Pick (user approved): best tune recall with harm <= 1% on civic val. Tune/test splits are
  by WIDER event and by video, because neighbouring frames are near copies.
- Key numbers (test; civic = 502 photos):

  | | faces masked | plates masked | issue hidden | macro-F1 | labels = no mask |
  |---|---|---|---|---|---|
  | mask nothing | 0.00 | 0.00 | 0.000 | 0.876 | 1.000 |
  | blur whole photo | 1.00 | 1.00 | 1.000 | 0.051 | 0.183 |
  | **YuNet 0.6 + plates 0.05/pad 0.5/1280** | **0.81** | **0.78** | **0.014** | **0.883** | **0.982** |

  - Faces by size: 0.75 (20-40 px), 0.85 (40-80), 0.91 (80+). Plates: 0.71 (40-80 px), 0.85 (80+).
  - Precision: faces 0.92, plates 0.57. 44% of civic photos get some blur, 0.95% of their pixels.
    Embedding cosine to the unmasked photo: 0.993.
  - YuNet sees faces in rubbish: at 0.1 it would hide 57% of the issue area, at 0.5 1.5%.
  - CPU per photo: faces 19 ms + plates 236 ms (1280 input) + YOLO/CLIP 68 ms.
- I should be able to explain: why masking comes before the models and why the masker is in
  `model_version`; why a threshold alone could not stop the sign-as-plate error; why padding
  raised plate recall more than a bigger model did (v1m: 0.46 vs v1n 0.40 at 0.1); why the
  tune/test split is by video.

## 6.1 Duplicate-pair + spam dataset
- Built: `verifier/dataset.py` (`python -m verifier.dataset` -> `data/verifier/reports.parquet`,
  `pairs.parquet`, `stock_index.parquet`, `images/`) and `verifier/images.py` (three pixel
  transforms: a citizen's own `photo_view`, a `reshare`d copy, a phone `screenshot`).
- **Pairs, not just reports.** A duplicate is a relation between two reports, so the unit is an
  (earlier a, later b) pair at most 72 h apart (the corpus's re-report horizon). Label = same
  `incident_id`. Every in-window positive is kept (484); negatives are mostly *hard*:
  same issue type within 500 m (`near_same_type`), other type within 150 m
  (`same_place_other_type`), plus 5 random negatives per positive. `kind` lets 6.2 report each slice.
- **The corpus was too easy.** Re-reports of an incident had identical coordinates, so
  "distance == 0" would have scored perfectly. Each report now gets phone GPS noise (15 m per axis)
  and 10% have no GPS. Dups end up 25 m apart (median), same-place negatives 95 m.
- **No test report in a train pair.** Test pair = the later report b is a test report (a can be
  any older open case, as in production). Pairs with a test a and a train b are dropped.
- **Three fake-photo kinds, each caught by a different check.** Reused (another incident's photo
  at least 1 km away, recompressed) and stock (from a simulated "internet" index) stay within
  dhash 3-3.5 bits of the source, so a hash lookup catches them. A screenshot is 25 bits away
  and its source is kept out of the index, so it needs its own detector. All uploads are JPEG,
  so the file format gives nothing away.
- Rejected: two different public photos as "same incident" (no dataset photographs one pothole
  twice) and exact copies (would make image matching trivial). The phase plan's augmented views
  of one photo are the compromise: different crop, tilt, mirror and light.
- Key numbers:

  | kind (train / test) | pairs | median GPS m | <= 50 m | median dhash bits |
  |---|---|---|---|---|
  | same_incident | 407 / 77 | 25 / 27 | 0.95 / 0.93 | 29 / 26 |
  | near_same_type | 584 / 122 | 301 / 264 | 0.04 / 0.04 | 32 / 32 |
  | same_place_other_type | 744 / 164 | 98 / 94 | 0.19 / 0.12 | 32 / 31 |
  | random | 2035 / 385 | 3582 / 3361 | 0.00 / 0.00 | 32 / 31 |

  Reports: 2,850 real (52% with a photo), 150 text spam, 150 each reused / stock / screenshot.
- I should be able to explain: why hard negatives matter more than random ones; why the GPS had
  to be made noisy; why dhash finds reused photos but cannot find a second photo of the same
  pothole (CLIP's job in 6.3); why the split follows the later report.
