"""Append-only, hash-chained decision log (JSONL).

Each line is a DecisionLog. hash = sha256(prev_hash + canonical JSON of the entry without `hash`).
Editing any line breaks its own hash; re-hashing it breaks the next line's prev_hash. So the
log is both the audit trail and a trustworthy dataset for the trainer.
"""

import fcntl
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from config import DECISION_LOG_PATH
from contracts.schemas import Decision, DecisionLog, ModuleName, Override

GENESIS_HASH = "0" * 64  # prev_hash of the very first entry


def _canonical(entry: dict[str, Any]) -> str:
    """Same data -> same bytes: sorted keys, no whitespace, `hash` field excluded."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(entry: dict[str, Any]) -> str:
    return hashlib.sha256((entry["prev_hash"] + _canonical(entry)).encode("utf-8")).hexdigest()


def _last_hash(f) -> str:
    """Hash of the last line, read backwards from the end so appends stay O(1) in file size."""
    f.seek(0, 2)
    pos = f.tell()
    tail = b""
    while pos > 0:
        step = min(4096, pos)
        pos -= step
        f.seek(pos)
        tail = f.read(step) + tail
        lines = tail.rstrip(b"\n").split(b"\n")
        if len(lines) > 1 or pos == 0:
            return json.loads(lines[-1])["hash"]
    return GENESIS_HASH


def append(
    module: ModuleName,
    model_version: str,
    case_id: str,
    input: dict[str, Any],
    output: dict[str, Any],
    human_override: Override | None = None,
    path: Path = DECISION_LOG_PATH,
) -> DecisionLog:
    """Append one entry. An exclusive file lock makes concurrent appends safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            entry = DecisionLog(
                timestamp=datetime.now(UTC), module=module, model_version=model_version,
                case_id=case_id, input=input, output=output, human_override=human_override,
                prev_hash=_last_hash(f), hash="pending",
            ).model_dump(mode="json")
            entry["hash"] = compute_hash(entry)
            f.seek(0, 2)
            f.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return DecisionLog.model_validate(entry)


def log_decision(
    module: ModuleName,
    case_id: str,
    input: BaseModel | dict[str, Any],
    decision: Decision,
    path: Path = DECISION_LOG_PATH,
) -> DecisionLog:
    """What a module's api.py calls after every model call."""
    if isinstance(input, BaseModel):
        input = input.model_dump(mode="json")
    return append(module, decision.model_version, case_id, input,
                  decision.model_dump(mode="json"), path=path)


def read_log(path: Path = DECISION_LOG_PATH) -> list[DecisionLog]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [DecisionLog.model_validate_json(line) for line in f if line.strip()]


def verify_chain(path: Path = DECISION_LOG_PATH) -> int | None:
    """Index of the first broken entry, or None if the whole chain is intact."""
    path = Path(path)
    if not path.exists():
        return None
    expected_prev = GENESIS_HASH
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                ok = entry["prev_hash"] == expected_prev and entry["hash"] == compute_hash(entry)
            except (json.JSONDecodeError, KeyError, TypeError):
                ok = False
            if not ok:
                return i
            expected_prev = entry["hash"]
    return None
