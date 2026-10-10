"""Protocols for each module's public API.

Structural typing: anything with matching methods conforms, no inheritance needed. A module's
`api.py` (plain functions) or a stub class (methods) both satisfy these.
"""

from typing import Protocol, runtime_checkable

from contracts.schemas import (
    Case,
    ImageFeatures,
    RawReport,
    Report,
    RoutingDecision,
    TriageDecision,
    TriageFeatures,
    VerifierResult,
)


@runtime_checkable
class IntakeAPI(Protocol):
    def process(self, raw: RawReport) -> Report: ...


@runtime_checkable
class VisionAPI(Protocol):
    def analyze(self, image_path: str) -> ImageFeatures: ...


@runtime_checkable
class VerifierAPI(Protocol):
    def check(self, report: Report, open_cases: list[Case]) -> VerifierResult: ...


@runtime_checkable
class TriageAPI(Protocol):
    def score(self, features: TriageFeatures) -> TriageDecision: ...


@runtime_checkable
class RouterAPI(Protocol):
    def route(self, case: Case) -> RoutingDecision: ...
