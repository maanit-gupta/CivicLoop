"""Pydantic schemas: the only data shapes that cross module boundaries.

Flow: RawReport -> (intake) Report -> Case <- (vision) ImageFeatures, (verifier) VerifierResult
      Case -> TriageFeatures -> (triage) TriageDecision ; Case -> (router) RoutingDecision
      -> WorkOrderDraft. Officials' corrections are Overrides; every model call is a DecisionLog.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    # extra="forbid": a typo'd field name is an error, not silently dropped data.
    model_config = ConfigDict(extra="forbid")


class Decision(_Strict):
    """Base for every model output. Subclassing it is how the 'every output carries
    model_version, confidence, explanation' rule is enforced."""

    model_version: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, description="Plain language, for officials.")


# ---------- shared vocabularies ----------

class IssueType(StrEnum):
    POTHOLE = "pothole"
    STREETLIGHT = "streetlight"
    GARBAGE = "garbage"
    WATER_LEAK = "water_leak"
    DRAIN_BLOCKAGE = "drain_blockage"
    DAMAGED_SIGN = "damaged_sign"
    OTHER = "other"


class Department(StrEnum):
    ROADS = "Roads"
    ELECTRICAL = "Electrical"
    SANITATION = "Sanitation"
    WATER_SUPPLY = "Water Supply"
    STORMWATER_DRAINS = "Stormwater Drains"
    TRAFFIC_SIGNAGE = "Traffic & Signage"


class Severity(StrEnum):
    # Levels may be revised by the severity rubric (Step 1.1).
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


RoadType = Literal["main", "residential", "lane"]
CaseStatus = Literal["open", "assigned", "resolved", "verified", "rejected"]
WorkOrderStatus = Literal["draft", "approved"]
ModuleName = Literal["intake", "vision", "verifier", "triage", "router", "trainer"]


class GeoPoint(_Strict):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Cues(_Strict):
    """Risk cues mentioned in or inferred from a complaint."""

    children: bool = False
    accident: bool = False
    flooding: bool = False


# ---------- intake ----------

class RawReport(_Strict):
    """What a citizen submits. Any mix of text, voice and photos, any language."""

    report_id: str = Field(min_length=1)
    submitted_at: datetime
    text: str | None = None
    audio_path: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    gps: GeoPoint | None = None
    channel: str = "app"

    @model_validator(mode="after")
    def _has_content(self):
        if not (self.text or self.audio_path or self.image_paths):
            raise ValueError("RawReport needs at least one of text, audio_path, image_paths")
        return self


class Report(Decision):
    """Intake output: a RawReport normalised into structured fields."""

    report_id: str
    submitted_at: datetime
    original_text: str = ""
    language: str = Field(description="ISO 639-1 code, 'mixed' if code-mixed, 'und' if unknown.")
    english_text: str = ""
    issue_type: IssueType
    location_text: str = ""
    location: GeoPoint | None = None
    cues: Cues = Field(default_factory=Cues)
    image_paths: list[str] = Field(default_factory=list)


# ---------- vision ----------

class Detection(_Strict):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] = Field(
        description="(x1, y1, x2, y2), normalised to 0..1 of image width/height."
    )

    @model_validator(mode="after")
    def _valid_box(self):
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"bbox must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1, got {self.bbox}")
        return self


def _union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Area covered by (x1, y1, x2, y2) boxes, overlaps counted once. Cut the plane into vertical
    strips at every box edge; in each strip, merge the y-intervals of the boxes spanning it."""
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:], strict=False):  # consecutive edges
        covered = top = 0.0
        for y1, y2 in sorted((b[1], b[3]) for b in boxes if b[0] <= left and b[2] >= right):
            covered += max(0.0, y2 - max(y1, top))
            top = max(top, y2)
        area += (right - left) * covered
    return area


class ImageFeatures(Decision):
    """What vision saw in one photo. `detections` holds only the boxes the detector is reasonably
    sure of. The three properties summarise them for triage; they are derived, not stored, so
    they can never disagree with the boxes (and are not part of the JSON)."""

    image_path: str
    issue_type: IssueType | None = None
    detections: list[Detection] = Field(default_factory=list)
    embedding: list[float] | None = None
    privacy_masked: bool = False

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def detector_confidence(self) -> float:
        """The most confident box's confidence; 0 without boxes."""
        return max((d.confidence for d in self.detections), default=0.0)

    @property
    def box_area_ratio(self) -> float:
        """Fraction of the photo covered by boxes (0..1), overlaps counted once."""
        return _union_area([d.bbox for d in self.detections])


# ---------- verifier ----------

class VerifierResult(Decision):
    report_id: str
    is_duplicate: bool
    duplicate_of: str | None = Field(default=None, description="case_id of the matched case.")
    is_spam: bool
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _duplicate_has_target(self):
        if self.is_duplicate != (self.duplicate_of is not None):
            raise ValueError("duplicate_of must be set exactly when is_duplicate is True")
        return self


# ---------- triage ----------

class TriageFeatures(_Strict):
    """Flat, tabular input for the triage model (LightGBM-friendly)."""

    case_id: str
    issue_type: IssueType
    ward: str
    road_type: RoadType
    dist_to_school_m: float = Field(ge=0)
    dist_to_hospital_m: float = Field(ge=0)
    children: bool = False
    accident: bool = False
    flooding: bool = False
    duplicate_count: int = Field(default=0, ge=0)
    image_detections: int = Field(default=0, ge=0)


class TriageDecision(Decision):
    case_id: str
    severity: Severity


# ---------- router ----------

class WorkOrderDraft(Decision):
    """A work order as the router drafts it. Only a human sets status='approved'."""

    case_id: str
    department: Department
    ward: str
    severity: Severity
    issue_type: IssueType
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    sla_hours: int = Field(gt=0)
    location_text: str = ""
    location: GeoPoint | None = None
    before_photos: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list, description="Why this dept/ward/severity.")
    status: WorkOrderStatus = "draft"


class RoutingDecision(Decision):
    case_id: str
    department: Department
    ward: str = Field(description="Ward id, or 'unknown' if neither GPS nor text gave one.")
    needs_human_routing: bool = Field(
        default=False, description="Confidence below threshold or ward unknown: a human routes.")
    # The router's draft, kept as it was produced. Case.work_order holds the current version,
    # which an official may edit and approve.
    work_order: WorkOrderDraft | None = None


# ---------- the case: what flows through the pipeline ----------

class Case(_Strict):
    case_id: str = Field(min_length=1)
    created_at: datetime
    status: CaseStatus = "open"
    report: Report
    ward: str | None = None
    image_features: list[ImageFeatures] = Field(default_factory=list)
    verification: VerifierResult | None = None
    triage: TriageDecision | None = None
    routing: RoutingDecision | None = None
    work_order: WorkOrderDraft | None = None


# ---------- humans and the audit trail ----------

class Override(_Strict):
    """An official correcting a model decision. The trainer learns from these."""

    case_id: str
    module: ModuleName
    field: str = Field(description="Which output field was changed, e.g. 'severity'.")
    model_value: Any
    human_value: Any
    official_id: str
    reason: str = ""
    at: datetime


class DecisionLog(_Strict):
    """One line of the append-only, hash-chained log. Written via contracts.decision_log."""

    timestamp: datetime
    module: ModuleName
    model_version: str
    case_id: str
    input: dict[str, Any]
    output: dict[str, Any]
    human_override: Override | None = None
    prev_hash: str
    hash: str
