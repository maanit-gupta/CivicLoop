# CivicLoop trainer report (Phase 3)

How triage (severity) and router (department) learn from officials' overrides, and what one
learning cycle changed. **Everything here is synthetic**: the city, the complaints, the labels
and the officials. The numbers show that the loop works on a known hidden policy. They say
nothing yet about real Delhi complaints.

Sections 1-4 and 7 are hand-written. Sections 5-6 are generated from the saved models and the
MLflow store by `python -m trainer.report`. Regenerate them, don't edit them by hand.

## Summary
- **Triage v2 is the champion.** It learned official A's hidden rule ("never below high within
  200 m of a school"). A's simulated override rate on the unseen holdout fell from 26.3% to 5.6%,
  and the canary confirmed it on new cases (23.3% -> 2.8% override rate, p = 8e-9). It gets
  *worse* on the rubric test set, and that is intended: A contradicts the written rubric. Where
  no policy applies (school >= 200 m), the rubric MAE moved +0.031, inside the +0.05 guardrail.
- **Router v1 stays.** 21 training overrides from official B were too few to move it
  (3.0% -> 2.6% offline), and the canary rolled v2 back (p = 0.79).
- **The LLM stretch (Qwen2.5-1.5B, LoRA SFT + DPO) is not promoted.** SFT matched LightGBM v1.
  DPO from 165 one-sided overrides collapsed twice. The third run learned A's rule (0% A
  overrides) but also raised far-from-school cases, so it fails the guardrail.

## 1. Data sources
All generated in this repo from `config.SEED = 42`. There is no real personal data.

| Data | Made by | Size | Notes |
|---|---|---|---|
| Synthetic city | `contracts/world_gen.py` | 12 wards, 58 roads, 98 POIs | Around Connaught Place; wards are jittered grid cells; SLAs from `data/world/departments.yaml` |
| Complaint corpus | `contracts/corpus_gen.py` + `corpus_templates.py` | 3000 complaints, 2550 incidents | Hindi 35%, Hinglish 30%, English 25%, Punjabi 10%; 5% spam, 15% duplicates; held-out split = hash of `incident_id` (15.8%) |
| Triage cases | `triage/generate_data.py` | 8000 (5568 / 1219 / 1213) | Corpus incident moved up to 150 m, geo features from `contracts.world`; split by 500 m cell; 25% "enriched" risky stratum |
| Router set | `router/dataset.py` | 3300 (2791 / 509) | Corpus + 450 generated ambiguous complaints (water on road, fallen pole, garbage in drain); 551 ambiguous rows |
| Live batch | `trainer/live.py` | 960 cases (`L-`) | Same generator, seed SEED+1, 92 days later, no spam, one case per incident; 0.2% of texts identical to a training text |
| Overrides | `trainer/officials.py` | 294 overrides | Written to the hash-chained `data/trainer/decision_log.jsonl` |
| Retrain sets | `trainer/dataset.py` | triage 5568 + 491 accepted + 165 overrides; router 2791 + 610 + 46 | Built only from the decision log (refuses a broken chain); holdout = 304 live cases (83 overrides), split by case |
| Canary batch | `trainer/run_cycle.py` | 1600 cases (`K-`) | Fresh cases for the canary only |
| LLM data | `trainer/llm_data.py` | 5568 SFT rows, 165 override pairs + 165 anchors, 1517 eval prompts | Eval = the 1213 triage test cases + the 304 holdout cases |

## 2. Label generation
- **Severity** = `triage.rubric.rubric_score(features)` (section 3). Then 8% of cases are moved one
  level (7.7% in the end) as label noise. A tree can copy a clean formula perfectly, which
  proves nothing. With noise, the best possible exact accuracy is 92.3%. A model above that is
  overfitting.
- **Department** = issue type through `departments.yaml`, plus the 2.1 ambiguity rules (sewage on
  the road -> Water Supply, live wires -> Electrical, garbage inside a drain -> Stormwater Drains).
  The rules hold across the whole dataset, so 101 corpus "sewer overflowing" rows were relabelled.
- **Human labels on live cases**: the official's value if they overrode, else the model's value
  (accepted). Acceptances are implicit: they are not logged.
- The labels are exact by construction, because they are generated from the world. So the test
  scores measure how well a model copies a known policy, not real-world truth.

## 3. Severity rubric (`triage/rubric.md`, approved in 1.1)

| Component | Points |
|---|---|
| Base: pothole, water leak, drain blockage | 2 |
| Base: other issue types | 1 |
| Main road | +1 |
| School within 200 m / hospital within 300 m | +1 each |
| 3-9 merged reports / 10+ | +1 / +2 |
| Children / accident / flooding mentioned | +1 / +2 / +2 |

0-2 low, 3-4 medium, 5-6 high, 7+ critical. A reported accident is never below high. There is
no separate urgency output: severity plus the SLA table gives each case its deadline. On 2400
corpus incidents: low 56.7%, medium 30.4%, high 12.4%, critical 0.5%.

## 4. Simulated officials (`trainer/officials.py`, approved in 3.1)
The overrides are **simulated**. Each official has a fixed hidden policy, and the trainer's job
is to discover it from the overrides alone.

| Official | Covers | Hidden policy | Overrides on the live batch |
|---|---|---|---|
| A, severity desk | 752 cases (80%), same cases as B | Never below **high** within 200 m of a school | 198 of 752 cases (26%) |
| B, routing desk | 752 cases (80%), same cases as A | Water leak / drain blockage + flooding -> **Stormwater Drains** (deliberately against rule 2.1) | 30 of 752 (4%) |
| C, noise | the other 208 cases (20%) | Changes a random field 15% of the time | 66 fields on 208 cases |

The policies are absolute ("never below high"), not relative ("one level above the model").
A relative rule has no fixed point: a model that learned it would get bumped again. A depth-4
tree on the overrides recovers A and B (F1 1.00) but not C (0.22, versus 0.30 by guessing), so
only A and B are worth learning.

**How v2 was trained (3.3).** v2 uses the v1 recipe unchanged (LightGBM regressor on the
severity level; char n-gram TF-IDF + logistic regression for the router), refit on the original
rows plus the live rows. Override rows get extra weight. The weight was picked from {1, 3, 10, 30}
on a 20% validation slice of the live training cases: **triage 10, router 1**. `trainer.retrain`
did not save these weights. In 3.6, refitting with them reproduced both v2 models exactly.
Promotion is decided by an offline gate (A/B override rate plus the guardrail), then a 10%
hash-split canary (≥ 100 challenger cases, one-sided z-test p < 0.05).

## 5. Training runs and 6. Before/after metrics

<!-- auto:begin (written by python -m trainer.report; edit the code, not this block) -->

### 5.1 Training runs (MLflow)

Tracking store `mlruns/mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`). *live* = logged by the training code while it ran; *backfilled* = logged in 3.6 from the saved model folder (started = folder file time). Repeated `-v1` runs are reruns of the same deterministic fit; the registry links the newest one.

| experiment | run | run id | logged | started | registry | headline metrics |
|---|---|---|---|---|---|---|
| triage | triage-v1 | 773fc28033da4194a2f0b13c70a53784 | live | 2026-10-09 15:17 | - | test_mae 0.083, test_qwk 0.948 |
| triage | triage-v1 | c567f0284aa9476f88e5affa45e0217d | live | 2026-10-09 15:18 | civicloop-triage v1 | test_mae 0.083, test_qwk 0.948 |
| triage | triage-v2 | 371cf07bc088496b9ff3100ac3c0d587 | backfilled | 2026-10-09 18:34 | civicloop-triage v2 | test_mae 0.178, test_qwk 0.883, holdout_a_override_rate 0.056 |
| router | router-v1 | 24998468128048fb923e985c7139164b | live | 2026-10-09 15:39 | - | test_accuracy 0.996, test_macro_f1 0.996 |
| router | router-v1 | 4c3d1e64eb8f455a83b92975f17bb3ad | live | 2026-10-09 15:40 | - | test_accuracy 0.998, test_macro_f1 0.998 |
| router | router-v1 | 79e08dc976544c16a2f3647bc17c3e6b | live | 2026-10-09 15:43 | - | test_accuracy 0.998, test_macro_f1 0.998 |
| router | router-v1 | 0f551acb3d614db3955d6e3dfab27b60 | live | 2026-10-09 15:46 | civicloop-router v1 | test_accuracy 0.998, test_macro_f1 0.998 |
| router | router-v2 | b8cf47ccab16495ba398ca1c943baf79 | backfilled | 2026-10-09 18:34 | civicloop-router v2 | test_accuracy 0.996, test_macro_f1 0.996, holdout_b_override_rate 0.026 |
| triage-llm | llm-run1 | 64e717f0e9774ae89f5f7e5f481a4ab9 | backfilled | 2026-10-09 15:04 | - | sft_test_mae 0.101, dpo_test_mae 1.151, dpo_holdout_a_override_rate 0.000 |
| triage-llm | llm-run2 | faed3526b21342b9896d510e7982ca55 | backfilled | 2026-10-09 16:14 | - | sft_test_mae 0.092, dpo_test_mae 1.819, dpo_holdout_a_override_rate 0.000 |
| triage-llm | llm-run3 | 25d0d7a89ccc4ad78e018cf06bb06a75 | backfilled | 2026-10-09 17:09 | - | sft_test_mae 0.096, dpo_test_mae 0.430, dpo_holdout_a_override_rate 0.000 |

### 5.2 Model registry and canary (3.4)

`@champion` is what serving should load. Canary rates are officials' override rates on the canary batch; p is the one-sided two-proportion z-test.

| model | # | version | alias | status | run id | canary | canary_n | champion rate | challenger rate | p |
|---|---|---|---|---|---|---|---|---|---|---|
| civicloop-triage | 1 | v1 | - | retired | c567f0284aa9476f88e5affa45e0217d | - | - | - | - | - |
| civicloop-triage | 2 | v2 | champion | champion | 371cf07bc088496b9ff3100ac3c0d587 | promote | 141 | 0.2330 | 0.0284 | 8.069e-09 |
| civicloop-router | 1 | v1 | champion | champion | 0f551acb3d614db3955d6e3dfab27b60 | - | - | - | - | - |
| civicloop-router | 2 | v2 | - | rolled_back | b8cf47ccab16495ba398ca1c943baf79 | rollback | 141 | 0.0795 | 0.0993 | 0.7942 |

### 6.1 Triage and router, v1 -> v2 (offline, 3.3)

Rubric/rule test sets: 1213 triage and 509 router cases. Holdout: 304 live cases whose overrides the trainer never saw. *Officials re-review* runs policies A and B on each model's own answers.

| view | metric | v1 | v2 | change |
|---|---|---|---|---|
| triage test (rubric) | MAE | 0.083 | 0.178 | 0.095 |
| triage test (rubric) | QWK | 0.948 | 0.883 | -0.065 |
| triage test (rubric) | MAE, school <200 m | 0.072 | 0.346 | 0.274 |
| triage test (rubric) | MAE, school >=200 m | 0.087 | 0.118 | 0.031 |
| router test (2.1 rules) | accuracy | 0.998 | 0.996 | -0.002 |
| router test (2.1 rules) | macro-F1 | 0.998 | 0.996 | -0.002 |
| router test (2.1 rules) | accuracy, water on road | 1.000 | 0.974 | -0.026 |
| holdout vs human | severity MAE | 0.299 | 0.138 | -0.161 |
| holdout vs human | severity QWK | 0.664 | 0.897 | 0.233 |
| holdout vs human | department accuracy | 0.938 | 0.938 | 0.000 |
| holdout vs human | department macro-F1 | 0.931 | 0.931 | 0.000 |
| holdout, officials re-review | A override rate | 0.263 | 0.056 | -0.207 |
| holdout, officials re-review | B override rate | 0.030 | 0.026 | -0.003 |
| holdout, officials re-review | A+B overrides | 89 | 25 | -64 |

### 6.2 LLM runs (3.5)

| run | dpo pairs | dpo loss | sft MAE / QWK | dpo MAE / QWK | dpo MAE school >=200 m | dpo A override rate | rewards/chosen (end) | verdict |
|---|---|---|---|---|---|---|---|---|
| llm-run1 | 165 | sigmoid | 0.101 / 0.938 | 1.151 / 0.000 | 1.249 | 0.000 | 0.189 | collapsed: answered high for every case |
| llm-run2 | 330 | sigmoid | 0.092 / 0.944 | 1.819 / -0.041 | 2.180 | 0.000 | -0.488 | drifted: 75% of answers critical |
| llm-run3 | 330 | sigmoid + sft 1:1 | 0.096 / 0.941 | 0.430 / 0.726 | 0.267 | 0.000 | 0.172 | not promoted: fails the 3.3 guardrail |

### 6.3 Final LLM run vs LightGBM (same test set and holdout)

| view | metric | majority | LightGBM v1 | LightGBM v2 | llm-sft | llm-dpo |
|---|---|---|---|---|---|---|
| rubric test | MAE | 0.965 | 0.083 | 0.178 | 0.096 | 0.430 |
| rubric test | QWK | 0.000 | 0.948 | 0.883 | 0.941 | 0.726 |
| rubric test | MAE, school <200 m | 1.384 | 0.072 | 0.346 | 0.094 | 0.887 |
| rubric test | MAE, school >=200 m | 0.816 | 0.087 | 0.118 | 0.096 | 0.267 |
| holdout | MAE vs human | 0.868 | 0.299 | 0.138 | 0.306 | 0.214 |
| holdout | A override rate | 0.326 | 0.263 | 0.056 | 0.263 | 0.000 |
| LLM only | invalid answers | - | - | - | 0.000 | 0.000 |
| LLM only | faithful explanations | - | - | - | 0.999 | 0.997 |

<!-- auto:end -->

**Reading the tables.**
- *Triage test (rubric)* is labelled by the written rubric, which policy A contradicts. A model
  that learns A has to score worse there. The guardrail row is *MAE, school >= 200 m*, where no
  policy applies.
- *Holdout vs human* favours v1, because officials only reacted to v1's answers. *Officials
  re-review* (A and B judge each model's own answers) is the fairest offline view.
- The LLM's *rewards/chosen* going negative (run 2) means DPO pushed down even the officials'
  answers. The freed probability went to "critical", which no pair rejected. Run 3 added an
  SFT term to hold the chosen answers up.

## 7. Limitations
- **Synthetic everything.** Complaint texts come from a small phrase bank shared by train and
  test. The router scores 0.998 on its test set but only 0.62 on unseen phrasings (lookup
  baseline: 0.60). Real complaints will look more like the unseen case.
- **Simulated officials are consistent.** Real officials disagree with each other, change their
  minds over time, and leave many errors uncorrected. Policy A was learnable because it is a
  clean threshold on one feature.
- **Learning A moves the model away from the written rubric.** Whether officials or the rubric
  are right is a governance question. If A is legitimate, the rubric should be updated so the
  rubric test set stops penalising it.
- **Acceptances are not logged.** The log has override counts but not per-official override
  rates. A real system should log "accepted" too.
- **One cycle, small samples.** The holdout has 83 overrides and the canary had 141 challenger
  cases. The canary case mix differed by chance (water-on-road 17.7% vs 10.3%), which inflates
  router v2's override rate.
- **A second cycle forgets.** In a test second cycle, v3 forgot policy A (A override rate 5.6% ->
  23.4%). Once v2 obeyed A, its evidence came back only as "accepted" rows at weight 1. The
  offline gate blocked v3. Retrain data should accumulate override rows across cycles (open).
- **Serving does not read the registry yet.** The APIs still load `config.MODEL_VERSIONS`
  (triage v1), not `@champion` (triage v2). This is integration work.
- **Tracking gaps.** The v2 retrains and the LLM runs were not logged while they ran. They were
  backfilled in 3.6 (tagged `backfilled=true`). `trainer.retrain` should log its run and save
  its chosen weights. The LLM runs 1-2 notebook configs were not kept: their DPO learning rate
  is not recorded, and only what changed is.
- **LLM stretch.** One 1.5B model, 165 override pairs, three runs and no hyperparameter search.
  GPU training on Kaggle is not bit-for-bit reproducible.

## Reproduce
```bash
.venv/bin/python -m trainer.run_cycle      # collect -> build -> train -> gate -> registry -> canary
.venv/bin/python -m trainer.compare        # v1 vs v2 table (6.1)
.venv/bin/python -m trainer.llm_eval       # LLM vs LightGBM (6.3), needs models/trainer/triage_llm
.venv/bin/python -m trainer.report         # backfill MLflow runs, rewrite sections 5-6
```
