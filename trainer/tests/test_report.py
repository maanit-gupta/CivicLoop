import re
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

from trainer import registry, report


@pytest.fixture
def client(tmp_path):
    return registry.client(f"sqlite:///{tmp_path / 'mlflow.db'}")


def test_splice_replaces_only_the_generated_block():
    text = f"# Title\nprose\n{report.BEGIN}\nold numbers\n{report.END}\nlimitations\n"
    out = report.splice(text, "new numbers")
    assert "old numbers" not in out and "new numbers" in out
    assert out.startswith("# Title\nprose\n") and out.endswith(f"{report.END}\nlimitations\n")
    assert report.splice(out, "new numbers") == out  # stable on re-run


def test_splice_refuses_a_report_without_markers():
    with pytest.raises(ValueError):
        report.splice("# hand-written only", "numbers")


def test_md_table_formats_rates_counts_and_missing():
    table = report.md_table([{"metric": "A rate", "v": 0.2634}, {"metric": "count", "v": 89.0},
                             {"metric": "llm only", "v": float("nan")}])
    assert table.splitlines() == ["| metric | v |", "|---|---|", "| A rate | 0.263 |",
                                  "| count | 89 |", "| llm only | - |"]


def test_log_run_is_idempotent_and_tagged_backfilled(client):
    args = ("triage", "triage-v2", {"override_weight": 10.0}, {"test_mae": 0.2}, {"step": "3.3"},
            1_700_000_000_000)
    first = report.log_run(client, *args)
    second = report.log_run(client, *args[:3], {"test_mae": 0.18}, *args[4:])
    assert first == second
    runs = client.search_runs([client.get_experiment_by_name("triage").experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["backfilled"] == "true" and run.data.tags["step"] == "3.3"
    assert run.data.metrics["test_mae"] == 0.18  # latest re-log wins
    assert run.info.start_time == 1_700_000_000_000 and run.info.status == "FINISHED"


def test_log_run_fails_if_a_param_changes(client):
    report.log_run(client, "triage", "triage-v2", {"override_weight": 10.0}, {}, {}, 0)
    with pytest.raises(MlflowException):
        report.log_run(client, "triage", "triage-v2", {"override_weight": 3.0}, {}, {}, 0)


def test_train_log_metrics_reads_the_summary_entry():
    log = {"sft": [{"loss": 0.6, "step": 10},
                   {"train_runtime": 800.0, "train_loss": 0.03, "epoch": 1.0, "step": 348}],
           "dpo": [{"train_runtime": 113.0, "train_loss": 0.57, "epoch": 1.0, "step": 42,
                    "rewards/chosen": 0.17, "rewards/margins": 0.66, "rewards/accuracies": 0.94}]}
    m = report._train_log_metrics(log)
    assert m["sft_steps"] == 348 and m["dpo_rewards_chosen"] == 0.17
    assert m["dpo_epochs"] == 1.0 and "sft_rewards_chosen" not in m


def test_report_cites_only_runs_that_exist_in_the_store():
    db = registry.URI.removeprefix("sqlite:///")
    if not report.REPORT.exists() or not Path(db).exists():
        pytest.skip("run python -m trainer.report first")
    cited = set(re.findall(r"\b[0-9a-f]{32}\b", report.REPORT.read_text()))
    assert cited, "the report cites no MLflow run ids"
    c = registry.client()
    for run_id in cited:
        c.get_run(run_id)  # raises if the id is not in the store
    for m in registry.MODULES:  # every registry version is linked to a cited run
        for mv in c.search_model_versions(f"name = '{registry.name(m)}'"):
            assert mv.tags.get("run_id") in cited
