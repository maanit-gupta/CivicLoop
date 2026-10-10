# Phase 1: Triage model (severity + urgency)

Goal: a trained, explainable model that turns `TriageFeatures` into a `TriageDecision`.
Needs: Phase 0. Uses stub vision/intake outputs; real ones plug in later via the same features.

## 1.1 Severity rubric
- `triage/rubric.md`: severity 1-5 and urgency low/medium/high for: pothole, broken_streetlight,
  garbage_overflow, water_leak, open_drain, faded_sign.
- Base severity per issue type; modifiers for distance to school/hospital, merged report_count,
  road type, and text cues (children, accident, flooding).
- 6 worked examples with final score and a one-sentence plain-language explanation.
- Implement `triage/rubric.py: rubric_score(features) -> (severity, urgency, reasons[])`.
- USER DECISION: the user must approve or edit the weights before this step is done.

## 1.2 Triage dataset generator
- `triage/generate_data.py`: ~8,000 cases. Features from realistic distributions
  (report_count skewed to 1, right-skewed distances, box_area_ratio depends on issue type),
  with locations sampled from the Phase 0 world so geo features come from real helper calls.
- Cue flags come from the Phase 0 corpus where possible.
- Label = rubric_score + noise: shift ~8% of labels by ±1, clip 1-5.
- Group split 70/15/15 by `location_cluster` so near-identical places don't straddle splits.
- Save parquet to `data/triage/`. Print label distribution per split; warn if a class < 3%.
- Explain: why a clean formula label would make the model look perfect for a meaningless reason.

## 1.3 Evaluation + trivial baselines
- `triage/evaluate.py`: MAE, quadratic weighted kappa, within-1 accuracy, confusion matrix,
  per-issue-type MAE.
- Baselines: predict train mean; predict base severity by issue type.
- Show a table. These numbers are the bar v1 must clear.

## 1.4 Train LightGBM v1
- `triage/train.py`: LightGBM regressor, early stopping on val, round + clip to 1-5.
  Urgency: separate small classifier (or derive from severity + cues; ask the user).
- Tune on val only; touch test once at the end.
- Log params, metrics, model to MLflow (`./mlruns`); save `models/triage/v1`.
- Show feature importances, 5 good and 5 bad test predictions, and how the learned behaviour
  compares to the rubric.

## 1.5 Triage API with explanations + logging
- `triage/api.py: score(TriageFeatures) -> TriageDecision`, loading the version from config.
- Explanation: top contributing features (SHAP or LightGBM pred_contrib) rendered as a plain
  sentence an official can read, e.g. "Severity 5: open drain 80 m from a school, 6 reports."
- Confidence: from prediction distance to rounding boundary or a simple calibrated proxy.
- Every call writes a DecisionLog entry.
- `python -m triage.demo` scores the Phase 0 fixtures. Tests: schema validity, log written.
