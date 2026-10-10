"""Public API of contracts: schemas, interfaces, stubs, fixtures. Every module may import it."""

import json
from pathlib import Path

import pandas as pd

from config import CORPUS_PATH
from contracts.decision_log import GENESIS_HASH, append, log_decision, read_log, verify_chain
from contracts.interfaces import IntakeAPI, RouterAPI, TriageAPI, VerifierAPI, VisionAPI
from contracts.schemas import (
    Case,
    CaseStatus,
    Cues,
    Decision,
    DecisionLog,
    Department,
    Detection,
    GeoPoint,
    ImageFeatures,
    IssueType,
    Override,
    RawReport,
    Report,
    RoadType,
    RoutingDecision,
    Severity,
    TriageDecision,
    TriageFeatures,
    VerifierResult,
    WorkOrderDraft,
    WorkOrderStatus,
)
from contracts.stubs import StubIntake, StubRouter, StubTriage, StubVerifier, StubVision

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture_cases() -> list[Case]:
    """The 5 example Cases in contracts/fixtures/, validated."""
    return [
        Case.model_validate(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(FIXTURES_DIR.glob("*.json"))
    ]


def load_corpus(split: str | None = None, path: Path = CORPUS_PATH) -> pd.DataFrame:
    """The shared complaint corpus (Step 0.5). split='train' or 'test' filters by incident."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No corpus at {path}. Run: python -m contracts.corpus_gen")
    df = pd.read_parquet(path)
    return df if split is None else df[df.split == split].reset_index(drop=True)


__all__ = [
    "Case", "CaseStatus", "Cues", "Decision", "DecisionLog", "Department", "Detection",
    "GeoPoint", "ImageFeatures", "IssueType", "Override", "RawReport", "Report", "RoadType",
    "RoutingDecision", "Severity", "TriageDecision", "TriageFeatures", "VerifierResult",
    "WorkOrderDraft", "WorkOrderStatus",
    "IntakeAPI", "VisionAPI", "VerifierAPI", "TriageAPI", "RouterAPI",
    "StubIntake", "StubVision", "StubVerifier", "StubTriage", "StubRouter",
    "FIXTURES_DIR", "load_fixture_cases", "load_corpus",
    "GENESIS_HASH", "append", "log_decision", "read_log", "verify_chain",
]
