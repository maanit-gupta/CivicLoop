"""Export data for the triage LLM notebook (Step 3.5): python -m trainer.llm_data

The LLM is trained on the same data as LightGBM v1 and v2:
  SFT  = the original triage training rows (rubric labels, the "teacher"); compare with v1.
  DPO  = the severity preference pairs from 3.2 (chosen = official, rejected = model; C's
         noise included, as v2 saw it) plus as many "anchor" pairs (see anchor_cases and
         anchor_rejected); compare with v2.
The prompt carries only the TriageFeatures (no complaint text), like LightGBM.

The answer is JSON {"explanation", "severity"}. The explanation restates the facts that the
rubric looks at and never depends on the severity, so a DPO pair differs only in the
severity word. Generating the explanation first also lets the evaluation test whether the
model actually read its input: its explanation must equal explain(row) exactly.
No urgency field: user decision 1.1 (severity + SLA table express urgency).

Outputs (data/trainer/llm/, upload these to the notebook):
  sft_train.jsonl  {"prompt": [system, user], "completion": [assistant]}
  dpo_train.jsonl  {"kind": "override" | "anchor", "prompt": [system, user],
                    "chosen": [assistant], "rejected": [assistant]}
  eval_prompts.jsonl  {"case_id", "set": "test" | "holdout", "prompt": [system, user]}
"""

import json
import random
import re
from collections import Counter

import pandas as pd

import config
from contracts.api import Severity
from trainer.dataset import HOLDOUT_OUT, PAIRS_OUT, TRIAGE_COLS, TRIAGE_ORIGINAL, TRIAGE_OUT
from trainer.officials import SCHOOL_RADIUS_M, TRAINER_DIR

LLM_DIR = TRAINER_DIR / "llm"
SFT_OUT = LLM_DIR / "sft_train.jsonl"
DPO_OUT = LLM_DIR / "dpo_train.jsonl"
EVAL_OUT = LLM_DIR / "eval_prompts.jsonl"
LEVELS = [s.value for s in Severity]
ANCHORS_PER_OVERRIDE = 1.0  # user decision 3.5: balance the one-sided override pairs 1:1

SYSTEM = ("You triage civic complaints for the Delhi municipal helpdesk. Read the case facts "
          "and reply with JSON only: {\"explanation\": \"<the facts that matter>\", "
          "\"severity\": \"low\" | \"medium\" | \"high\" | \"critical\"}.")


def _yes(flag) -> str:
    return "yes" if flag else "no"


def facts(row) -> str:
    """The user message. Distances are floored to whole metres (199.6 m -> 199 m, still < 200)."""
    return (f"Issue: {row.issue_type}\nRoad: {row.road_type}\nWard: {row.ward}\n"
            f"School distance: {int(row.dist_to_school_m)} m\n"
            f"Hospital distance: {int(row.dist_to_hospital_m)} m\n"
            f"Children mentioned: {_yes(row.children)}\nAccident mentioned: {_yes(row.accident)}\n"
            f"Flooding mentioned: {_yes(row.flooding)}\nOther reports: {row.duplicate_count}\n"
            f"Image detections: {row.image_detections}")


def explain(row) -> str:
    """The reference explanation: the rubric's factors, never the severity itself."""
    school, hospital = int(row.dist_to_school_m), int(row.dist_to_hospital_m)
    parts = [f"{row.issue_type} on a {row.road_type} road",
             f"school {school} m away" + (" (under 200 m)" if school < 200 else ""),
             f"hospital {hospital} m away" + (" (under 300 m)" if hospital < 300 else "")]
    parts += [f"{cue} mentioned" for cue in ("children", "accident", "flooding")
              if getattr(row, cue)]
    if row.duplicate_count:
        parts.append(f"{row.duplicate_count} other reports")
    return "; ".join(parts) + "."


def prompt(row) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": facts(row)}]


def answer(row, severity: str) -> list[dict]:
    text = json.dumps({"explanation": explain(row), "severity": severity}, ensure_ascii=False)
    return [{"role": "assistant", "content": text}]


def parse(output: str) -> dict | None:
    """{"explanation", "severity"} from a generated answer, or None if it is not valid.
    Tolerates a ```json fence around the object; nothing else."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip())
    try:
        got = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(got, dict) or got.get("severity") not in LEVELS:
        return None
    return {"explanation": str(got.get("explanation", "")), "severity": got["severity"]}


def sft_rows() -> list[dict]:
    orig = pd.read_parquet(TRIAGE_ORIGINAL).query("split == 'train'")
    return [{"prompt": prompt(r), "completion": answer(r, r.severity)}
            for r in orig.itertuples(index=False)]


def anchor_cases(retrain: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """n accepted live cases at least 200 m from a school. In the pair, chosen = the accepted
    severity, rejected = another severity (anchor_rejected).

    Why: every override pair says "raise it to high", so without anchors DPO's cheapest fit is
    "always high" (run 1: 100% high, QWK 0). Anchors say "keep the accepted answer where no
    school is near", so only the school distance can explain the overrides."""
    pool = retrain[(retrain.source == "live_accepted")
                   & (retrain.dist_to_school_m >= SCHOOL_RADIUS_M)]
    if len(pool) < n:
        raise ValueError(f"need {n} anchor cases, only {len(pool)} accepted far-from-school")
    return pool.sample(n=n, random_state=seed)


def anchor_rejected(chosen: list[str], already: Counter, seed: int = config.SEED) -> list[str]:
    """A rejected severity for each anchor, never its chosen one: the label rejected least so
    far (`already` holds the override pairs' counts), ties broken at random.

    Why: a severity that is never rejected is free probability mass. Run 2 always rejected
    "high", "critical" was never rejected, and DPO drifted to "critical" (75%). A plain uniform
    draw would leave "critical" rejected about half as often as "medium", so balance instead."""
    rng, counts, out = random.Random(seed), Counter(already), []
    for c in chosen:
        options = [lvl for lvl in LEVELS if lvl != c]
        least = min(counts[lvl] for lvl in options)
        pick = rng.choice([lvl for lvl in options if counts[lvl] == least])
        counts[pick] += 1
        out.append(pick)
    return out


def dpo_rows() -> list[dict]:
    pairs = pd.read_json(PAIRS_OUT, lines=True).query("field == 'severity'")
    retrain = pd.read_parquet(TRIAGE_OUT)
    feats = retrain[TRIAGE_COLS].drop_duplicates("case_id")
    pairs = pairs.merge(feats, on="case_id", how="left", validate="one_to_one")
    if pairs.issue_type.isna().any():
        raise ValueError("a preference pair has no features in triage_retrain.parquet")
    rows = [{"case_id": r.case_id, "kind": "override", "official_id": r.official_id,
             "prompt": prompt(r), "chosen": answer(r, r.chosen),
             "rejected": answer(r, r.rejected)} for r in pairs.itertuples(index=False)]
    anchors = anchor_cases(retrain, n=round(len(rows) * ANCHORS_PER_OVERRIDE))
    rejected = anchor_rejected(list(anchors.severity), Counter(pairs.rejected))
    rows += [{"case_id": r.case_id, "kind": "anchor", "official_id": None, "prompt": prompt(r),
              "chosen": answer(r, r.severity), "rejected": answer(r, rej)}
             for r, rej in zip(anchors.itertuples(index=False), rejected, strict=True)]
    return rows


def pair_counts(rows: list[dict]) -> pd.DataFrame:
    """How often each severity is chosen and rejected across the DPO pairs."""
    def sev(msgs):
        return json.loads(msgs[0]["content"])["severity"]
    chosen = Counter(sev(r["chosen"]) for r in rows)
    rejected = Counter(sev(r["rejected"]) for r in rows)
    return pd.DataFrame({"chosen": [chosen[lvl] for lvl in LEVELS],
                         "rejected": [rejected[lvl] for lvl in LEVELS]}, index=LEVELS)


def eval_rows() -> list[dict]:
    test = pd.read_parquet(TRIAGE_ORIGINAL).query("split == 'test'")
    holdout = pd.read_parquet(HOLDOUT_OUT)
    return [{"case_id": r.case_id, "set": name, "prompt": prompt(r)}
            for name, df in (("test", test), ("holdout", holdout))
            for r in df.itertuples(index=False)]


def _write(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    out = {SFT_OUT: sft_rows(), DPO_OUT: dpo_rows(), EVAL_OUT: eval_rows()}
    for path, rows in out.items():
        _write(path, rows)
        print(f"{path.relative_to(TRAINER_DIR.parent.parent)}: {len(rows)} rows")
    print("\nDPO pairs per severity:\n" + pair_counts(out[DPO_OUT]).to_string())
    ex = out[DPO_OUT][0]
    print("\nExample DPO pair:\n" + ex["prompt"][1]["content"])
    print("chosen:  ", ex["chosen"][0]["content"])
    print("rejected:", ex["rejected"][0]["content"])


if __name__ == "__main__":
    main()
