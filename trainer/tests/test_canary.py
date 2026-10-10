from datetime import UTC, datetime

import pandas as pd

from contracts.api import Override, append
from trainer.canary import arm, bucket, decide, evaluate_module, naive, outcomes, p_lower
from trainer.run_cycle import canary_versions, offline_gate

AT = datetime(2026, 10, 1, tzinfo=UTC)
IDS = [f"C-K{i:05d}" for i in range(5000)]


def test_split_is_deterministic_and_about_pct():
    assert [arm(c) for c in IDS[:50]] == [arm(c) for c in IDS[:50]]
    share = sum(arm(c) == "challenger" for c in IDS) / len(IDS)
    assert abs(share - 0.10) < 0.015
    assert all(0 <= bucket(c) < 100 for c in IDS[:100])


def test_split_edges_and_salt():
    assert all(arm(c, pct=0) == "champion" for c in IDS[:200])
    assert all(arm(c, pct=100) == "challenger" for c in IDS[:200])
    assert [arm(c, salt="x") for c in IDS[:200]] != [arm(c) for c in IDS[:200]]


def test_p_lower():
    assert abs(p_lower(10, 100, 10, 100) - 0.5) < 1e-9          # same rate: coin flip
    assert p_lower(3, 100, 25, 100) < 0.001                      # clearly lower
    assert p_lower(25, 100, 3, 100) > 0.999                      # clearly higher
    assert p_lower(0, 100, 0, 100) == 1.0 and p_lower(0, 0, 5, 10) == 1.0


def test_decide_rules():
    assert decide(3, 50, 25, 100).action == "wait"               # below MIN_SAMPLE
    assert decide(3, 140, 300, 1300).action == "promote"
    assert decide(10, 140, 100, 1300).action == "rollback"       # lower but could be luck
    assert decide(30, 140, 100, 1300).action == "rollback"       # worse
    assert naive(10, 140, 100, 1300) == "promote"                # the baseline would ship it


def _log(path, case_id, version, overridden):
    append("triage", version, case_id, {}, {"severity": "low"}, path=path)
    if overridden:
        o = Override(case_id=case_id, module="triage", field="severity", model_value="low",
                     human_value="high", official_id="A", reason="r", at=AT)
        append("triage", version, case_id, {"severity": "low"}, {"severity": "high"},
               human_override=o, path=path)


def test_outcomes_and_evaluate_from_log(tmp_path):
    log = tmp_path / "log.jsonl"
    for i in range(120):
        _log(log, f"N{i}", "v2", overridden=i < 3)
    for i in range(400):
        _log(log, f"O{i}", "v1", overridden=i < 100)
    out = outcomes(log, "triage")
    assert len(out) == 520 and out.overridden.sum() == 103
    d = evaluate_module(log, "triage", champion="v1", challenger="v2")
    assert d.action == "promote" and d.challenger_n == 120
    assert abs(d.champion_rate - 0.25) < 1e-9 and abs(d.challenger_rate - 0.025) < 1e-9


def test_canary_versions_only_swaps_modules_with_a_challenger():
    pick = canary_versions({"triage": "v1", "router": "v1"}, {"triage": "v2"})
    chall = next(c for c in IDS if arm(c) == "challenger")
    champ = next(c for c in IDS if arm(c) == "champion")
    assert pick(chall) == {"triage": "v2", "router": "v1"}
    assert pick(champ) == {"triage": "v1", "router": "v1"}


def _table(a_old, a_new, b_old, b_new, mae_change):
    idx = pd.MultiIndex.from_tuples([
        ("holdout, officials re-review", "A override rate"),
        ("holdout, officials re-review", "B override rate"),
        ("triage test (rubric)", "MAE, school >=200 m")])
    return pd.DataFrame({"v1": [a_old, b_old, 0.1], "v2": [a_new, b_new, 0.1 + mae_change],
                         "change": [a_new - a_old, b_new - b_old, mae_change]}, index=idx)


def test_offline_gate():
    assert offline_gate(_table(0.26, 0.06, 0.03, 0.02, 0.03), "triage")[0]
    assert not offline_gate(_table(0.26, 0.06, 0.03, 0.02, 0.08), "triage")[0]  # guardrail
    assert not offline_gate(_table(0.06, 0.23, 0.03, 0.02, -0.03), "triage")[0]  # worse
    assert offline_gate(_table(0.26, 0.06, 0.03, 0.02, 0.0), "router")[0]
    assert not offline_gate(_table(0.26, 0.06, 0.03, 0.03, 0.0), "router")[0]   # not lower
