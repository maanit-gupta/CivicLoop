"""One learning cycle (Step 3.4): python -m trainer.run_cycle [--reset]

  collect   officials review the registry's champions on the live batch -> decision log
  build     override dataset from that log (trainer.dataset)
  train     candidate v<n> for both modules (trainer.retrain)
  eval      offline gate on the override holdout (trainer.compare): the officials' simulated
            override rate must drop; triage must also pass the 3.3 rubric-MAE guardrail
  register  candidate -> registry; gate passed -> @challenger, else status rejected_offline
  canary    a later batch, CANARY_PCT% of cases to the challengers (trainer.canary);
            promote or roll back each module on its own

--reset forgets the registry first, so the cycle starts again from config.MODEL_VERSIONS.
The canary batch is the same every cycle: real traffic would be new each time.
"""

import argparse

import pandas as pd

from trainer import canary, dataset, registry, retrain
from trainer.compare import GUARDRAIL_MAE, compare
from trainer.live import load_canary, load_live
from trainer.officials import ANSWER_KEY_PATH, LOG_PATH, TRAINER_DIR, simulate

CANARY_LOG = TRAINER_DIR / "canary_log.jsonl"
OFFLINE_RATE = {"triage": "A override rate", "router": "B override rate"}  # whose desk


def offline_gate(table: pd.DataFrame, module: str) -> tuple[bool, str]:
    """Does the candidate (2nd column) beat the champion (1st) on the override holdout?"""
    old, new = table.loc[("holdout, officials re-review", OFFLINE_RATE[module])].iloc[:2]
    ok, why = bool(new < old), f"simulated {OFFLINE_RATE[module]} {old:.3f} -> {new:.3f}"
    if module == "triage":
        far = table.loc[("triage test (rubric)", "MAE, school >=200 m"), "change"]
        ok = ok and far <= GUARDRAIL_MAE
        why += f"; rubric MAE (school >=200 m) {far:+.3f}, limit +{GUARDRAIL_MAE}"
    return ok, why


def canary_versions(champions: dict, challengers: dict):
    def versions(case_id: str) -> dict:
        if canary.arm(case_id) == "challenger":
            return {m: challengers.get(m, v) for m, v in champions.items()}
        return dict(champions)
    return versions


def run_cycle(reset: bool = False, uri: str = registry.URI) -> pd.DataFrame:
    if reset:
        registry.reset(uri)
    champions = registry.bootstrap(uri)
    version = max((registry.next_version(m, uri) for m in registry.MODULES),
                  key=lambda v: int(v[1:]))  # one number per cycle, both modules
    print(f"champions {champions}; candidate {version}")

    LOG_PATH.unlink(missing_ok=True)
    simulate(load_live(), versions=lambda _: champions).to_parquet(ANSWER_KEY_PATH, index=False)
    print("collect: officials reviewed the champions on the live batch")
    dataset.save(dataset.build())
    print("build: override dataset written")
    weights, _ = retrain.run(version)
    print(f"train: {version} fitted, override weights {weights}")
    table = compare(champions, version)

    rows, challengers = [], {}
    for m in registry.MODULES:
        ok, why = offline_gate(table, m)
        if ok:
            registry.make_challenger(m, version, {"offline_gate": why}, uri)
            challengers[m] = version
        else:
            registry.register(m, version, {"status": "rejected_offline", "offline_gate": why},
                              uri)
        rows.append({"module": m, "champion": champions[m], "candidate": version,
                     "offline_gate": "pass" if ok else "fail", "offline": why})
    print(f"eval + register: challengers {challengers or 'none'}")

    if challengers:
        CANARY_LOG.unlink(missing_ok=True)
        simulate(load_canary(), log_path=CANARY_LOG,
                 versions=canary_versions(champions, challengers))
        print(f"canary: {canary.CANARY_PCT}% of the canary batch served by the challengers")
    for row in rows:
        m = row["module"]
        if m not in challengers:
            row |= {k: "-" for k in ("canary_n", "champion_rate", "challenger_rate",
                                     "p_value", "action")} | {"champion_after": champions[m]}
            continue
        d = canary.evaluate_module(CANARY_LOG, m, champions[m], version)
        registry.register(m, version, {
            "canary_n": d.challenger_n, "canary_rate": f"{d.challenger_rate:.4f}",
            "canary_champion_rate": f"{d.champion_rate:.4f}", "canary_p": f"{d.p_value:.4g}",
            "canary_action": d.action}, uri)
        if d.action == "promote":
            registry.promote(m, uri)
        elif d.action == "rollback":
            registry.rollback(m, uri)
        row |= {"canary_n": d.challenger_n, "champion_rate": round(d.champion_rate, 3),
                "challenger_rate": round(d.challenger_rate, 3), "p_value": f"{d.p_value:.3g}",
                "action": d.action, "champion_after": registry.alias(m, "champion", uri)}
    return pd.DataFrame(rows).set_index("module").T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reset", action="store_true", help="start from config.MODEL_VERSIONS")
    summary = run_cycle(reset=p.parse_args().reset)
    print()
    for m, why in summary.loc["offline"].items():
        print(f"{m} offline gate: {why}")
    print("\n" + summary.drop("offline").to_string())
    print("\nregistry:\n" + pd.DataFrame(registry.table()).to_string(index=False))


if __name__ == "__main__":
    main()
