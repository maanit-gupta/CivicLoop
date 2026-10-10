"""python -m trainer.demo: officials review v1 on 40 live cases (temporary log), show overrides,
then rebuild human labels and preference pairs from that log."""

import json
import tempfile
from pathlib import Path

from contracts.api import read_log, verify_chain
from trainer.dataset import live_decisions, preference_pairs
from trainer.live import load_live
from trainer.officials import simulate


def main() -> None:
    live = load_live().head(40)
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "log.jsonl"
        key = simulate(live, log_path=log_path)
        log = read_log(log_path)
        print(f"{len(live)} live cases -> {len(log)} log entries, chain intact: "
              f"{verify_chain(log_path) is None}\n")
        facts = key.set_index("case_id")
        for e in log:
            if not e.human_override:
                continue
            o, f = e.human_override, facts.loc[e.case_id]
            print(f"{e.case_id}  official {o.official_id}  {o.field}: {o.model_value} -> "
                  f"{o.human_value}  | {f.issue_type}, school {f.dist_to_school_m:.0f} m, "
                  f"flooding {f.flooding}  | reason: {o.reason!r}")

        decisions = live_decisions(log_path)  # Step 3.2: rebuilt from the log alone
        pairs = preference_pairs(decisions)
        print(f"\nFrom the log: {len(decisions)} cases with human labels, "
              f"{len(pairs)} preference pairs. First pair:")
        if pairs:
            print(json.dumps(pairs[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
