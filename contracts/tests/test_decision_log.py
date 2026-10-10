"""The decision log chains correctly, detects any tampering, and survives concurrent appends."""

import json
import threading

import pytest

from contracts.api import (
    GENESIS_HASH,
    StubTriage,
    TriageFeatures,
    append,
    log_decision,
    read_log,
    verify_chain,
)
from contracts.decision_log import compute_hash

N = 5


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "log.jsonl"
    for i in range(N):
        append("triage", "stub", f"C-{i}", {"x": i}, {"severity": "low"}, path=path)
    return path


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_intact_chain_verifies(log):
    entries = read_log(log)
    assert len(entries) == N
    assert entries[0].prev_hash == GENESIS_HASH
    for prev, cur in zip(entries, entries[1:], strict=False):
        assert cur.prev_hash == prev.hash
    assert verify_chain(log) is None


def test_missing_or_empty_log_is_valid(tmp_path):
    assert verify_chain(tmp_path / "nope.jsonl") is None
    assert read_log(tmp_path / "nope.jsonl") == []


@pytest.mark.parametrize("i", range(N))
def test_editing_any_line_is_detected(log, i):
    lines = _lines(log)
    entry = json.loads(lines[i])
    entry["output"]["severity"] = "critical"
    lines[i] = json.dumps(entry)
    _write(log, lines)
    assert verify_chain(log) == i


@pytest.mark.parametrize("i", range(N - 1))
def test_rehashing_an_edit_breaks_the_next_line(log, i):
    lines = _lines(log)
    entry = json.loads(lines[i])
    entry["output"]["severity"] = "critical"
    entry["hash"] = compute_hash(entry)  # attacker fixes up this line's hash...
    lines[i] = json.dumps(entry)
    _write(log, lines)
    assert verify_chain(log) == i + 1  # ...but the next line still points at the old hash


def test_deleted_line_is_detected(log):
    lines = _lines(log)
    del lines[2]
    _write(log, lines)
    assert verify_chain(log) == 2


def test_reordered_lines_are_detected(log):
    lines = _lines(log)
    lines[1], lines[3] = lines[3], lines[1]
    _write(log, lines)
    assert verify_chain(log) == 1


def test_garbage_line_is_detected(log):
    lines = _lines(log)
    lines[3] = "not json"
    _write(log, lines)
    assert verify_chain(log) == 3


def test_concurrent_appends_keep_chain_intact(tmp_path):
    path = tmp_path / "log.jsonl"
    n_threads, per_thread = 8, 25

    def worker(t):
        for k in range(per_thread):
            append("router", "stub", f"T{t}-{k}", {"k": k}, {"ok": True}, path=path)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(read_log(path)) == n_threads * per_thread
    assert verify_chain(path) is None


def test_log_decision_records_a_model_call(tmp_path):
    path = tmp_path / "log.jsonl"
    feats = TriageFeatures(case_id="C-9", issue_type="pothole", ward="W1", road_type="main",
                           dist_to_school_m=100, dist_to_hospital_m=900, accident=True)
    decision = StubTriage().score(feats)
    entry = log_decision("triage", feats.case_id, feats, decision, path=path)
    assert entry.model_version == "stub"
    assert entry.input["road_type"] == "main"
    assert entry.output["severity"] == decision.severity.value
    assert entry.human_override is None
    assert read_log(path) == [entry]
