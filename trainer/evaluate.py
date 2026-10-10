"""Evaluate the trainer: python -m trainer.evaluate

Step 3.1, simulated officials:
1. Do the logged overrides match the simulator's answer key, and is the log's chain intact?
2. Is each official's policy discoverable from the case's features? A depth-4 tree predicts
   "this official overrides this case" (5-fold CV F1). Baseline: guessing at the official's
   override rate, whose expected F1 equals that rate. A and B should be near 1, C near baseline.

Step 3.2, override dataset (built from the log only, checked here against the answer key):
3. Are the human labels, model values and preference pairs exactly the simulator's? Is the
   holdout disjoint from everything the trainer will see?
4. How much of each policy reaches the training data? Weighted share of policy-matching labels
   where the policy applies. Baseline: the original training data alone.

Step 3.3, v1 vs v2 (trainer.compare): old test sets, override holdout vs human labels, and
the officials' simulated override rate on v2's outputs. Baseline: v1.

Step 3.4, canary (reads data/trainer/canary_log.jsonl from trainer.run_cycle):
5. Is the hash split fair? Arm sizes and case mix (near a school, water on road) per arm.
6. A/A test: re-split only the champion arm (one model on both sides) with AA_SPLITS salts.
   Any "promote" is a false promotion. Baseline: the naive rule "promote if the rate is lower".
7. Resampling: draw both real arms with replacement RESAMPLES times. How often does each rule
   promote? For a real improvement that is the rule's power.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

import config
from contracts.api import read_log, verify_chain
from trainer.canary import arm, decide, naive, outcomes
from trainer.dataset import build
from trainer.live import load_canary, load_live, to_features
from trainer.officials import ANSWER_KEY_PATH, LEVELS, LOG_PATH

TREE_DEPTH = 4
AA_SPLITS = 200
RESAMPLES = 200


def overrides_per_official(key: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for who in ("A", "B", "C"):
        sev = key[key.severity_reviewer == who]
        dep = key[key.department_reviewer == who]
        n_sev = int((sev.model_severity != sev.human_severity).sum())
        n_dep = int((dep.model_department != dep.human_department).sum())
        mine = key[(key.severity_reviewer == who) | (key.department_reviewer == who)]
        rate = _changed(mine, who).mean()
        rows.append({"official": who, "cases": len(mine), "severity_ovr": n_sev,
                     "department_ovr": n_dep, "override_rate": rate,
                     "tree_f1": _discoverability(mine, who), "baseline_f1": rate})
    return pd.DataFrame(rows)


def _changed(df: pd.DataFrame, who: str) -> pd.Series:
    """Did `who` change a field they reviewed? (A and B share cases but not fields.)"""
    sev = (df.severity_reviewer == who) & (df.model_severity != df.human_severity)
    dep = (df.department_reviewer == who) & (df.model_department != df.human_department)
    return sev | dep


def _discoverability(df: pd.DataFrame, who: str) -> float:
    y = _changed(df, who).to_numpy()
    if y.sum() < 5:
        return float("nan")
    x = pd.concat([
        df[["dist_to_school_m"]], df.flooding.astype(int),
        df.model_severity.map(LEVELS.index).rename("model_severity_rank"),
        pd.get_dummies(df.issue_type, prefix="issue", dtype=int),
        pd.get_dummies(df.model_department, prefix="dept", dtype=int)], axis=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=TREE_DEPTH,
                                                    random_state=config.SEED), x, y, cv=cv)
    tp = int((pred & y).sum())
    return 2 * tp / (pred.sum() + y.sum())


def log_check(key: pd.DataFrame) -> dict:
    log = read_log(LOG_PATH)
    logged = {(e.case_id, e.human_override.field, e.human_override.human_value)
              for e in log if e.human_override}
    expected = ({(c, "severity", v) for c, m, v in
                 key[["case_id", "model_severity", "human_severity"]].itertuples(index=False)
                 if m != v}
                | {(c, "department", v) for c, m, v in
                   key[["case_id", "model_department", "human_department"]]
                   .itertuples(index=False) if m != v})
    models = {m: sum(e.module == m and not e.human_override for e in log)
              for m in ("triage", "router")}
    return {"chain_intact": verify_chain(LOG_PATH) is None, "overrides_match": logged == expected,
            "triage_entries": models["triage"], "router_entries": models["router"],
            "override_entries": len(logged)}


def text_overlap(live: pd.DataFrame) -> float:
    """Share of live complaint texts that also appear word for word in the training corpus."""
    from contracts.api import load_corpus
    return float(live.text.isin(set(load_corpus("train").text)).mean())


def dataset_checks(key: pd.DataFrame, out: dict) -> dict:
    live = out["live"].merge(key, on="case_id", suffixes=("", "_key"))
    same = all((live[f"{w}_{f}"] == live[f"{w}_{f}_key"]).all()
               for w in ("model", "human") for f in ("severity", "department"))
    held = set(out["holdout"].case_id)
    seen = set(out["triage"].case_id) | set(out["router"].case_id) | set(out["pairs"].case_id)
    train_live = out["live"][~out["live"].case_id.isin(held)]
    n_over = int(train_live.severity_official.notna().sum()
                 + train_live.department_official.notna().sum())
    return {"all_cases": len(out["live"]) == len(key), "labels_match_key": same,
            "holdout_disjoint": not held & seen,
            "pairs_match_overrides": len(out["pairs"]) == n_over,
            "holdout_overrides": int(
                out["holdout"].severity_official.notna().sum()
                + out["holdout"].department_official.notna().sum())}


def policy_signal(out: dict) -> pd.DataFrame:
    """Where policy A / B applies, the weighted share of rows whose label agrees with it."""
    tri, rout = out["triage"], out["router"]
    rows = []
    for policy, df, applies, agrees in (
            ("A: school <200 m -> >= high", tri, tri.dist_to_school_m < 200,
             tri.severity.isin(["high", "critical"])),
            ("B: water on road -> Stormwater", rout,
             rout.issue_type.isin(["water_leak", "drain_blockage"]) & rout.flooding,
             rout.department == "Stormwater Drains")):
        for name, mask in (("original (baseline)", df.source == "original"),
                           ("retrain set", df.source.notna())):
            m = applies & mask
            w = df.weight[m]
            rows.append({"policy": policy, "data": name, "rows": int(m.sum()),
                         "override_rows": int((m & (df.source == "live_override")).sum()),
                         "agree_share": float((w * agrees[m]).sum() / w.sum())})
    return pd.DataFrame(rows)


def arm_balance(log_path) -> pd.DataFrame:
    batch = load_canary()
    batch = batch[batch.case_id.isin(set(outcomes(log_path, "triage").case_id))]
    school = np.array([to_features(r).dist_to_school_m for r in batch.itertuples(index=False)])
    out = pd.DataFrame({
        "arm": batch.case_id.map(arm).to_numpy(), "near_school": school < 200,
        "water_on_road": (batch.issue_type.isin(["water_leak", "drain_blockage"])
                          & batch.flooding).to_numpy()})
    out = out.groupby("arm").agg(cases=("arm", "size"), near_school=("near_school", "mean"),
                                 water_on_road=("water_on_road", "mean"))
    out["share"] = out.cases / out.cases.sum()
    return out


def _counts(old: pd.Series, new: pd.Series) -> tuple[int, int, int, int]:
    """(k_new, n_new, k_old, n_old) from two boolean 'overridden' series."""
    return int(new.sum()), len(new), int(old.sum()), len(old)


def rule_checks(log_path) -> pd.DataFrame:
    """Per module and rule: A/A false promotion rate, resampled promotion rate, real call."""
    rng, rows = np.random.default_rng([config.SEED, 34]), []
    rules = {"z-test (ours)": lambda *k: decide(*k).action, "naive (baseline)": naive}
    for module in ("triage", "router"):
        out = outcomes(log_path, module)
        out = out.assign(arm=out.case_id.map(arm))
        old, new = out[out.arm == "champion"], out[out.arm == "challenger"]
        aa = [old.case_id.map(lambda c, s=f"aa{i}": arm(c, salt=s)) == "challenger"
              for i in range(AA_SPLITS)]
        resampled = [_counts(pd.Series(rng.choice(old.overridden, len(old))),
                             pd.Series(rng.choice(new.overridden, len(new))))
                     for _ in range(RESAMPLES)]
        for name, rule in rules.items():
            rows.append({
                "module": module,
                "versions": f"{old.version.iloc[0]} vs {new.version.iloc[0]}",
                "rule": name,
                "A/A false promotion": np.mean([
                    rule(*_counts(old.overridden[~s], old.overridden[s])) == "promote"
                    for s in aa]),
                "resampled promotion": np.mean([rule(*k) == "promote" for k in resampled]),
                "real decision": rule(*_counts(old.overridden, new.overridden))})
    return pd.DataFrame(rows)


def main() -> None:
    if not ANSWER_KEY_PATH.exists():
        raise SystemExit("No simulation yet. Run: python -m trainer.officials")
    key, live = pd.read_parquet(ANSWER_KEY_PATH), load_live()
    pd.set_option("display.width", 120)
    print("== 3.1 simulated officials ==")
    print(overrides_per_official(key).round(3).to_string(index=False))
    print()
    near = key.dist_to_school_m < 200
    print(f"near a school (<200 m): {near.mean():.1%} of cases; model severity there: "
          + ", ".join(f"{k} {v}" for k, v in key[near].model_severity.value_counts().items()))
    water = key.issue_type.isin(["water_leak", "drain_blockage"]) & key.flooding
    print(f"water on road: {water.mean():.1%} of cases; model dept there: "
          + ", ".join(f"{k} {v}" for k, v in key[water].model_department.value_counts().items()))
    print(f"live texts seen word for word in the train corpus: {text_overlap(live):.1%}")
    print("log:", log_check(key))
    np.testing.assert_equal(log_check(key)["overrides_match"], True)

    print("\n== 3.2 override dataset ==")
    out = build()
    checks = dataset_checks(key, out)
    print("checks:", checks)
    print(policy_signal(out).round(3).to_string(index=False))
    assert all(v for k, v in checks.items() if k != "holdout_overrides"), checks

    print("\n== 3.3 v1 vs v2 ==")
    if not (config.MODELS_DIR / "triage" / "v2").exists():
        raise SystemExit("No v2 yet. Run: python -m trainer.retrain")
    from trainer.compare import GUARDRAIL_MAE, compare
    table = compare()
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    sim = table.loc["holdout, officials re-review"]
    assert sim.loc["A+B overrides", "v2"] < sim.loc["A+B overrides", "v1"], "v2 not better"
    far = table.loc[("triage test (rubric)", "MAE, school >=200 m"), "change"]
    print(f"guardrail: rubric MAE where no policy applies +{far:.3f} (limit {GUARDRAIL_MAE})")
    assert far <= GUARDRAIL_MAE, "v2 breaks the guardrail"

    print("\n== 3.4 canary ==")
    from trainer.run_cycle import CANARY_LOG
    if not CANARY_LOG.exists():
        raise SystemExit("No canary yet. Run: python -m trainer.run_cycle --reset")
    balance = arm_balance(CANARY_LOG)
    print(balance.round(3).to_string(), "\n")
    rules = rule_checks(CANARY_LOG)
    print(rules.round(3).to_string(index=False))
    assert abs(balance.share["challenger"] - 0.10) < 0.03, "canary arm is not ~10%"
    ours = rules[rules.rule == "z-test (ours)"].set_index("module")
    assert (ours["A/A false promotion"] <= 0.10).all(), "rule promotes too often with no change"
    assert ours.loc["triage", "real decision"] == "promote", "triage v2 not promoted"


if __name__ == "__main__":
    main()
