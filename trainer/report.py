"""Trainer report (Step 3.6): python -m trainer.report

docs/trainer_report.md is hand-written prose plus one generated block (between the auto
markers) that this script rewrites: the training runs with their MLflow run ids and the
before/after metrics. The numbers are recomputed from the saved models, not copied by hand.

Backfill: only the v1 fits logged MLflow runs while they trained. The v2 retrains (3.3) and the
three GPU LLM runs (3.5) did not, so each one gets a run here, tagged backfilled=true and with
the folder it describes. Its start time is that folder's file time. Params come from the saved
meta.json / train_log.json, and metrics from trainer.compare and trainer.llm_eval. A re-run
updates the same runs (a param that changed is an error). Each registry version also gets a
run_id tag.
"""

import json
import time

import pandas as pd
from mlflow.entities import Metric, Param, RunTag

import config
from trainer import llm_eval, registry
from trainer.compare import compare

REPORT = config.ROOT / "docs" / "trainer_report.md"
BEGIN = "<!-- auto:begin (written by python -m trainer.report; edit the code, not this block) -->"
END = "<!-- auto:end -->"
LLM_EXPERIMENT = "triage-llm"

# trainer.retrain chose these in 3.3 but did not save them. Checked in 3.6: refitting with
# them reproduces models/{triage,router}/v2 exactly (triage weight 3 and router weight 3 don't).
V2_OVERRIDE_WEIGHT = {"triage": 10.0, "router": 1.0}

# (compare view, metric) -> MLflow metric name. Same names as the v1 runs where they overlap.
COMPARE_METRICS = {
    "triage": {
        ("triage test (rubric)", "MAE"): "test_mae",
        ("triage test (rubric)", "QWK"): "test_qwk",
        ("triage test (rubric)", "MAE, school <200 m"): "test_mae_school_lt200m",
        ("triage test (rubric)", "MAE, school >=200 m"): "test_mae_school_ge200m",
        ("holdout vs human", "severity MAE"): "holdout_mae",
        ("holdout vs human", "severity QWK"): "holdout_qwk",
        ("holdout, officials re-review", "A override rate"): "holdout_a_override_rate"},
    "router": {
        ("router test (2.1 rules)", "accuracy"): "test_accuracy",
        ("router test (2.1 rules)", "macro-F1"): "test_macro_f1",
        ("router test (2.1 rules)", "accuracy, water on road"): "test_accuracy_water_on_road",
        ("holdout vs human", "department accuracy"): "holdout_accuracy",
        ("holdout vs human", "department macro-F1"): "holdout_macro_f1",
        ("holdout, officials re-review", "B override rate"): "holdout_b_override_rate"},
}
# (llm_eval view, metric) -> MLflow metric name, prefixed sft_ / dpo_.
LLM_METRICS = {
    ("rubric test", "MAE"): "test_mae",
    ("rubric test", "QWK"): "test_qwk",
    ("rubric test", "MAE, school <200 m"): "test_mae_school_lt200m",
    ("rubric test", "MAE, school >=200 m"): "test_mae_school_ge200m",
    ("holdout", "MAE vs human"): "holdout_mae",
    ("holdout", "A override rate"): "holdout_a_override_rate",
    ("LLM only", "invalid answers"): "invalid_answers",
    ("LLM only", "faithful explanations"): "faithful_explanations",
}
# Shared by all three runs (notebooks/triage_llm.ipynb). The notebook was edited for each run
# and only run 3's version is kept, so runs 1-2 record only what changed (from the 3.5 notes).
LLM_COMMON = {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "lora": "r16 alpha32 all-linear fp16",
              "sft_rows": 5568, "sft_lr": 2e-4, "sft_epochs": 1, "dpo_beta": 0.1,
              "eval_prompts": 1517}
LLM_RUNS = {  # run name -> (folder in models/trainer, params, verdict)
    "llm-run1": ("triage_llm_run1",
                 {"dpo_pairs": 165, "anchors": "none", "dpo_loss": "sigmoid"},
                 "collapsed: answered high for every case"),
    "llm-run2": ("triage_llm_run2",
                 {"dpo_pairs": 330, "anchors": "165, rejected always high", "dpo_loss": "sigmoid"},
                 "drifted: 75% of answers critical"),
    "llm-run3": ("triage_llm",
                 {"dpo_pairs": 330, "anchors": "165, rejected label balanced",
                  "dpo_loss": "sigmoid + sft 1:1", "dpo_lr": 1e-5},
                 "not promoted: fails the 3.3 guardrail"),
}
HEADLINE = {"triage": ("test_mae", "test_qwk", "holdout_a_override_rate"),
            "router": ("test_accuracy", "test_macro_f1", "holdout_b_override_rate"),
            LLM_EXPERIMENT: ("sft_test_mae", "dpo_test_mae", "dpo_holdout_a_override_rate")}


def _ms(path) -> int:
    return int(path.stat().st_mtime * 1000)


def log_run(c, experiment: str, name: str, params: dict, metrics: dict, tags: dict,
            start_ms: int) -> str:
    """Create (or update) the backfilled run `name` in `experiment`; returns its run id."""
    exp = c.get_experiment_by_name(experiment)
    exp_id = exp.experiment_id if exp else c.create_experiment(experiment)
    found = c.search_runs([exp_id], f"attributes.run_name = '{name}' and tags.backfilled = 'true'")
    if found:
        run_id = found[0].info.run_id
    else:
        run_id = c.create_run(exp_id, start_time=start_ms, run_name=name,
                              tags={"backfilled": "true"}).info.run_id
    now = int(time.time() * 1000)  # metric timestamp: the latest re-log wins
    c.log_batch(run_id, metrics=[Metric(k, float(v), now, 0) for k, v in metrics.items()],
                params=[Param(k, str(v)) for k, v in params.items()],
                tags=[RunTag(k, str(v)) for k, v in tags.items()])
    c.set_terminated(run_id, end_time=start_ms)
    return run_id


def _train_log_metrics(log: dict) -> dict:
    """Summary numbers from the trainer state that the notebook saved (last entry per stage)."""
    out = {}
    for stage in ("sft", "dpo"):
        s = next(e for e in log[stage] if "train_runtime" in e)
        out |= {f"{stage}_train_loss": s["train_loss"], f"{stage}_runtime_s": s["train_runtime"],
                f"{stage}_epochs": s["epoch"], f"{stage}_steps": s["step"]}
        if stage == "dpo":
            out |= {f"dpo_{k.replace('/', '_')}": s[k]
                    for k in ("rewards/chosen", "rewards/margins", "rewards/accuracies")}
    return out


def _latest_live_run(c, experiment: str, name: str) -> str:
    """v1 was trained (and logged) more than once; the newest run wrote the model folder."""
    exp = c.get_experiment_by_name(experiment)
    runs = c.search_runs([exp.experiment_id], f"attributes.run_name = '{name}'",
                         order_by=["attributes.start_time DESC"])
    return next(r.info.run_id for r in runs if r.data.tags.get("backfilled") != "true")


def backfill(cmp: pd.DataFrame, llm: dict[str, pd.DataFrame], uri: str = registry.URI) -> dict:
    """Log the untracked runs; tag registry versions with their run. Returns {run name: id}."""
    c, ids = registry.client(uri), {}
    for m in registry.MODULES:
        folder = config.MODELS_DIR / m / "v2"
        meta = json.loads((folder / "meta.json").read_text())
        params = {"base": meta["base"], "n_train": meta["n_train"],
                  "override_weight": V2_OVERRIDE_WEIGHT[m]}
        if m == "triage":
            params |= meta["params"] | {"num_boost_round": meta["best_iteration"]}
        else:
            params |= {k: meta[k] for k in ("C", "calibrated", "threshold")}
        metrics = {n: cmp.loc[key, "v2"] for key, n in COMPARE_METRICS[m].items()}
        tags = {"source": f"models/{m}/v2", "step": "3.3"}
        ids[f"{m}-v2"] = log_run(c, m, f"{m}-v2", params, metrics, tags, _ms(folder / "meta.json"))
    for name, (folder_name, params, verdict) in LLM_RUNS.items():
        folder = config.MODELS_DIR / "trainer" / folder_name
        log = json.loads((folder / "train_log.json").read_text())
        metrics = {f"{s}_{n}": llm[name].loc[key, f"llm-{s}"]
                   for s in ("sft", "dpo") for key, n in LLM_METRICS.items()}
        metrics |= _train_log_metrics(log)
        tags = {"source": f"models/trainer/{folder_name}", "step": "3.5", "verdict": verdict}
        if name == "llm-run3":
            tags["kaggle_kernel"] = "jagpreetsingh02/civicloop-2"
        ids[name] = log_run(c, LLM_EXPERIMENT, name, LLM_COMMON | params, metrics, tags,
                            _ms(folder / "train_log.json"))
    for m in registry.MODULES:
        registry.register(m, "v1", {"run_id": _latest_live_run(c, m, f"{m}-v1")}, uri)
        registry.register(m, "v2", {"run_id": ids[f"{m}-v2"]}, uri)
    return ids


# ---------- rendering ----------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.0f}" if v.is_integer() and abs(v) >= 2 else f"{v:.3f}"
    return str(v)


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def _frame_rows(df: pd.DataFrame) -> list[dict]:
    return [{"view": v, "metric": m, **row.to_dict()} for (v, m), row in df.iterrows()]


def _registry_rows(c, uri: str) -> list[dict]:
    aliases = {(r["model"], r["version"]): r["aliases"] for r in registry.table(uri)}
    rows = []
    for m in registry.MODULES:
        for mv in sorted(c.search_model_versions(f"name = '{registry.name(m)}'"),
                         key=lambda v: int(v.version)):
            t, version = mv.tags, mv.tags.get("civicloop_version")
            rows.append({"model": registry.name(m), "#": mv.version, "version": version,
                         "alias": aliases.get((registry.name(m), version)) or "-",
                         "status": t.get("status", ""), "run id": t.get("run_id", "-"),
                         "canary": t.get("canary_action", "-"), "canary_n": t.get("canary_n", "-"),
                         "champion rate": t.get("canary_champion_rate", "-"),
                         "challenger rate": t.get("canary_rate", "-"),
                         "p": t.get("canary_p", "-")})
    return rows


def _run_rows(c, linked: dict) -> list[dict]:
    rows = []
    for exp_name, keys in HEADLINE.items():
        exp = c.get_experiment_by_name(exp_name)
        for r in c.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"]):
            m = r.data.metrics
            rows.append({"experiment": exp_name, "run": r.info.run_name, "run id": r.info.run_id,
                         "logged": ("backfilled" if r.data.tags.get("backfilled") == "true"
                                    else "live"),
                         "started": time.strftime("%Y-%m-%d %H:%M",
                                                 time.localtime(r.info.start_time / 1000)),
                         "registry": linked.get(r.info.run_id, "-"),
                         "headline metrics": ", ".join(f"{k} {_fmt(m[k])}" for k in keys
                                                       if k in m)})
    return rows


def render(cmp: pd.DataFrame, llm_final: pd.DataFrame, uri: str = registry.URI) -> str:
    c = registry.client(uri)
    reg = _registry_rows(c, uri)
    linked = {r["run id"]: f"{r['model']} {r['version']}" for r in reg}
    llm_rows = []
    for name, (_, params, verdict) in LLM_RUNS.items():
        run = c.search_runs([c.get_experiment_by_name(LLM_EXPERIMENT).experiment_id],
                            f"attributes.run_name = '{name}'")[0]
        m = run.data.metrics
        llm_rows.append({"run": name, "dpo pairs": params["dpo_pairs"],
                         "dpo loss": params["dpo_loss"],
                         "sft MAE / QWK": f"{_fmt(m['sft_test_mae'])} / {_fmt(m['sft_test_qwk'])}",
                         "dpo MAE / QWK": f"{_fmt(m['dpo_test_mae'])} / {_fmt(m['dpo_test_qwk'])}",
                         "dpo MAE school >=200 m": m["dpo_test_mae_school_ge200m"],
                         "dpo A override rate": m["dpo_holdout_a_override_rate"],
                         "rewards/chosen (end)": m["dpo_rewards_chosen"], "verdict": verdict})
    return "\n\n".join([
        "### 5.1 Training runs (MLflow)",
        "Tracking store `mlruns/mlflow.db` (`mlflow ui --backend-store-uri "
        "sqlite:///mlruns/mlflow.db`). *live* = logged by the training code while it ran; "
        "*backfilled* = logged in 3.6 from the saved model folder (started = folder file time). "
        "Repeated `-v1` runs are reruns of the same deterministic fit; the registry links the "
        "newest one.",
        md_table(_run_rows(c, linked)),
        "### 5.2 Model registry and canary (3.4)",
        "`@champion` is what serving should load. Canary rates are officials' override rates "
        "on the canary batch; p is the one-sided two-proportion z-test.",
        md_table(reg),
        "### 6.1 Triage and router, v1 -> v2 (offline, 3.3)",
        "Rubric/rule test sets: 1213 triage and 509 router cases. Holdout: 304 live cases whose "
        "overrides the trainer never saw. *Officials re-review* runs policies A and B on each "
        "model's own answers.",
        md_table(_frame_rows(cmp)),
        "### 6.2 LLM runs (3.5)",
        md_table(llm_rows),
        "### 6.3 Final LLM run vs LightGBM (same test set and holdout)",
        md_table(_frame_rows(llm_final)),
    ])


def splice(text: str, block: str) -> str:
    """Replace what sits between BEGIN and END; everything else stays as written."""
    head, found_begin, rest = text.partition(BEGIN)
    _, found_end, tail = rest.partition(END)
    if not (found_begin and found_end):
        raise ValueError(f"{REPORT.name}: auto markers missing")
    return f"{head}{BEGIN}\n\n{block}\n\n{END}{tail}"


def main() -> None:
    cmp = compare()
    llm = {name: llm_eval.report(pd.read_json(
               config.MODELS_DIR / "trainer" / folder / "predictions.jsonl", lines=True))
           for name, (folder, _, _) in LLM_RUNS.items()}
    ids = backfill(cmp, llm)
    REPORT.write_text(splice(REPORT.read_text(), render(cmp, llm["llm-run3"])))
    print("Backfilled MLflow runs:")
    for name, run_id in ids.items():
        print(f"  {name:10s} {run_id}")
    print(f"Wrote {REPORT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
