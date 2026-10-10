---
description: Run the next unchecked CivicLoop step from PROGRESS.md
argument-hint: "[optional: step id like 1.3, or a note for this step]"
---

You are continuing CivicLoop. The previous context was cleared on purpose; rebuild state from files only.

1. Read `PROGRESS.md`. Find the first unchecked step, unless `$ARGUMENTS` names a step id, in which
   case use that one. If `$ARGUMENTS` contains anything else, treat it as the user's note or answer
   for this step.
2. If the Handoff section lists something awaiting the user (a decision, a notebook to run, a file
   to drop in), check whether it is resolved (via `$ARGUMENTS` or the expected file existing). If not,
   ask for it in one short question and stop.
3. Read only the matching phase file in `docs/phases/` and only the section for this step. Read code
   only in that module and in `contracts/`. Follow `CLAUDE.md`.
4. Say in one line which step you are on, then explain the concept (under ~200 words, one rejected
   alternative), then build it.
5. Run the step's tests and evaluation. Show results as a small table.
6. Do the end-of-step bookkeeping from `CLAUDE.md` (PROGRESS.md tick + Handoff, LEARNING_NOTES.md,
   Decisions).
7. Do ONE step only. Stop with exactly: `Step X.Y done. Run /clear, then /next.`
   If the step is blocked waiting on the user, stop with: `Step X.Y waiting on you: <what>.`
