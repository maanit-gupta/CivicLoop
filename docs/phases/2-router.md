# Phase 2: Router model (department + ward + work order)

Goal: assign department and ward, and draft a work order an official approves.
Needs: Phase 0 (world, departments, corpus). Independent of Phase 1 except reading severity
from a Case when present (use stub severity otherwise).

## 2.1 Routing labels + ward lookup
- Ward is a geo lookup (`ward_of(point)`), not ML. Explain why: deterministic facts belong in tools.
- Department: ground truth from the corpus issue_type mapped through departments.yaml, plus
  deliberate ambiguous cases (water on road: Water Supply leak or Stormwater drain? streetlight
  pole fallen on road: Electrical or Roads?) where the label depends on text cues.
- Save `data/router/` splits, reusing the corpus `split` column.

## 2.2 Evaluation + lookup baseline
- `router/evaluate.py`: accuracy, macro-F1, confusion matrix, accuracy on the ambiguous subset.
- Baseline: issue_type -> department lookup table. Show where it fails (the ambiguous cases).

## 2.3 Train router v1
- Features: issue_type, text cues / TF-IDF of English text, detection classes, road type.
- Model: logistic regression or LightGBM classifier with calibrated probabilities.
- Low confidence (top prob < threshold, user picks) -> `needs_human_routing = True`.
- MLflow logging; save `models/router/v1`.

## 2.4 Work-order drafter + Router API
- `router/workorder.py`: template-based draft (issue, location, ward, severity, SLA from
  departments.yaml, before-photo ref, reasons). Optional local-LLM polish behind a flag; template
  must work alone.
- `router/api.py: route(Case) -> RoutingDecision` with `WorkOrderDraft`, status `draft`
  (never `approved`: only a human approves).
- DecisionLog on every call. `python -m router.demo`. Tests.
