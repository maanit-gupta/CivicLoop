# CivicLoop

Multi-agent system that takes a civic complaint (voice/photo/text, any language) to a verified fix.
This is a LEARNING project: the user wants to understand every component, not just have it built.
**City: Delhi** (user decision, Step 0.4). The synthetic world is centred on Connaught Place
(`config.CITY`, `data/world/`). Complaint languages: Hindi, English, Punjabi, Hinglish (code-mixed).
Keep all data, examples, landmarks and language choices Delhi-specific.
Current stage: build each ML model as an independent, testable module. Integration (LangGraph,
Temporal, auth, Kafka, K8s) comes later and is OUT OF SCOPE until PROGRESS.md says otherwise.

## Session protocol (token budget matters)
- One session = one step. The user runs `/clear` then `/next` between steps.
- At session start read ONLY `PROGRESS.md` and the current phase file in `docs/phases/`.
  Do not read other phase files, do not scan or grep the whole repo, do not re-read unchanged files.
- Read code only in the module you are working on, plus `contracts/`.
- Keep explanations tight. Prefer showing a small result table over long prose.
- All state lives in files. Assume you remember nothing from earlier sessions.

## Teaching style
- Before coding a step: explain the concept and design choice in under ~200 words, and name one
  alternative you rejected and why.
- Write the evaluation code before tuning anything. Always compare against a trivial baseline.
- If a decision is genuinely the user's (rubric weights, thresholds, label definitions), propose a
  default and ask. Do not mark the step done until they answer.
- Keep code small and readable. Plain functions over clever abstractions.

## Modularity rules
Modules: `contracts`, `triage`, `router`, `trainer`, `intake`, `vision`, `verifier`.
- Modules talk ONLY through the schemas in `contracts/`. Allowed imports:
  - every module -> `contracts`
  - `verifier` -> `vision.api` (public API only)
  - `trainer` -> `triage.api`, `router.api`, and the decision log
  - nothing else crosses module boundaries.
- Each module exposes its public functions in `<module>/api.py`, matching `contracts/interfaces.py`.
  Everything else in the module is private.
- Each module has `<module>/tests/`, an `evaluate.py`, and a `python -m <module>.demo` that runs
  standalone on sample data.
- Every model call writes a `DecisionLog` entry through `contracts.decision_log`.
- Every model output carries `model_version`, `confidence`, and a plain-language `explanation`.

## Conventions
- Python 3.11. One `pyproject.toml` with optional extras per module: `pip install -e ".[triage]"`.
- One `SEED` in `config.py`; everything reproducible.
- Data in `data/<module>/`, models in `models/<module>/`, MLflow local store in `./mlruns`.
  These folders are gitignored.
- `pytest` for tests, `ruff` for lint. A step is not done if tests or eval fail.
- Open-source models and tools only; no paid APIs. Local LLM via Ollama.
- Synthetic or public data only. No real personal data. Faces and number plates get masked.
- Hardware is CPU-first. GPU work (YOLO fine-tune, LoRA) goes in `notebooks/` for Colab/Kaggle;
  the user runs it and drops outputs into `models/<module>/`.

## End of every step (mandatory)
1. Tick the step in `PROGRESS.md` and replace its Handoff section (max 6 lines: files touched,
   how to run, results, open questions, anything awaiting the user).
2. Append to `docs/LEARNING_NOTES.md`: what was built, why, one thing the user should be able to
   explain, key numbers.
3. Record any decision the user made under "Decisions" in `PROGRESS.md`.
4. Finish with exactly: `Step X.Y done. Run /clear, then /next.`
