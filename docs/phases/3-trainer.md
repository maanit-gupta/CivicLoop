# Phase 3: Trainer (learning loop)

Goal: show at least one retraining cycle with a measurable improvement, from official overrides.
Needs: Phases 1 and 2 (their v1 models and decision logs).

## 3.1 Simulated officials
- `trainer/officials.py`: 3 simulated officials, each with a consistent hidden policy, e.g.
  A bumps severity +1 within 200 m of a school; B always routes "water on road" to Stormwater;
  C sometimes overrides randomly (noise). The user picks/edits the policies.
- Run them over v1 predictions on a fresh "live" batch (from the corpus pool not in train/test)
  and write overrides into the DecisionLog (`human_override` field via a new log entry).
- Explain honestly: overrides are simulated; the hidden policy is what the model must discover.

## 3.2 Override dataset builder
- `trainer/dataset.py`: read the DecisionLog, join model output with human final value, build
  the retraining set (original labels + override-corrected rows, overrides weighted higher).
- Also export preference pairs (chosen = human value, rejected = model value) for 3.5.
- Report: override rate per official, per issue type.

## 3.3 Retrain v2 + offline comparison
- Retrain triage and router on the new dataset. Evaluate v1 vs v2 on the SAME held-out test
  plus a held-out override set the trainer never saw.
- Metrics: MAE / kappa / macro-F1, and simulated override rate (run officials on v2 outputs).
- Show a v1 vs v2 table. If no improvement, diagnose instead of tuning until it looks good.

## 3.4 Model registry + canary
- Register v1 and v2 in MLflow model registry with stages.
- `trainer/canary.py`: deterministic hash(case_id) split sends N% (default 10) to the candidate;
  promotion rule = candidate beats champion on override rate over a minimum sample, else rollback.
- `trainer/run_cycle.py`: one command runs collect -> build -> train -> eval -> register -> canary.
  This is the command the live demo calls.

## 3.5 (Stretch, GPU) LoRA SFT + DPO small LLM
- `notebooks/triage_llm.ipynb` for Colab/Kaggle T4: Qwen2.5 1.5B-3B, QLoRA SFT on teacher-labelled
  cases (JSON output: severity, urgency, explanation), then DPO on the 3.2 preference pairs.
- Local eval script compares LLM v1/v2 against LightGBM on the same test set.
- The user runs the notebook; Claude prepares data export and eval. Skip if time is short.

## 3.6 Trainer report
- `docs/trainer_report.md`: data sources (synthetic + how), label generation, rubric, simulated
  officials, training runs (MLflow run ids), before/after metrics table, limitations.
