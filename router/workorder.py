"""Work-order drafter: a fixed template filled from the case and the routing decision.

The template is the whole drafter: every fact in it comes from a field, so it can't invent
anything. `polish=True` asks a local LLM (Ollama) to reword the description for a field crew.
That's optional and checked: if Ollama is unreachable, or the rewrite drops a fact the crew
needs (department, ward, severity, SLA hours), the template text is kept.
"""

import json
import urllib.request

from contracts.api import Department, IssueType, Report, Severity, WorkOrderDraft
from contracts.world import load_world, sla_hours

TEMPLATE_VERSION = "template-v1"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
ISSUE_LABELS = {
    IssueType.POTHOLE: "Pothole", IssueType.STREETLIGHT: "Streetlight fault",
    IssueType.GARBAGE: "Garbage dump", IssueType.WATER_LEAK: "Water leak",
    IssueType.DRAIN_BLOCKAGE: "Blocked drain", IssueType.DAMAGED_SIGN: "Damaged sign",
    IssueType.OTHER: "Civic issue",
}


def ward_name(ward: str) -> str:
    names = {w.ward_id: w.name for w in load_world().wards}
    return names.get(ward, "ward not identified")


def _description(report: Report, department: Department, ward: str, severity: Severity,
                 sla: int, reasons: list[str]) -> str:
    loc = report.location_text or "no location text"
    if report.location is not None:
        loc += f" ({report.location.lat:.5f}, {report.location.lon:.5f})"
    lines = [
        f"Department: {department.value}. Ward: {ward} ({ward_name(ward)}).",
        f"Severity: {severity.value}. Fix within {sla} hours (SLA).",
        f"Issue reported: {report.issue_type.value}.",
        f"Location: {loc}.",
        f'Citizen complaint ({report.language}): "{report.original_text}"',
    ]
    if report.english_text and report.english_text != report.original_text:
        lines.append(f'English: "{report.english_text}"')
    photos = ", ".join(report.image_paths) or "none attached"
    lines.append(f"Before photos: {photos}.")
    lines += [f"- {r}" for r in reasons]
    return "\n".join(lines)


def _ollama(prompt: str, timeout: float = 30.0) -> str:
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


def polish(description: str, must_keep: list[str]) -> str | None:
    """LLM rewrite of the description, or None if unavailable or it lost a required fact."""
    prompt = ("Rewrite this municipal work order for a Delhi field crew: short, clear English. "
              "Keep every fact, number and name exactly. Add nothing.\n\n" + description)
    try:
        text = _ollama(prompt)
    except Exception:  # noqa: BLE001  any failure -> template wins
        return None
    return text if text and all(k in text for k in must_keep) else None


def draft(case_id: str, report: Report, department: Department, ward: str, severity: Severity,
          confidence: float, reasons: list[str], use_llm: bool = False) -> WorkOrderDraft:
    sla = sla_hours(department, severity)
    place = report.location_text or ward_name(ward)
    description = _description(report, department, ward, severity, sla, reasons)
    version = TEMPLATE_VERSION
    if use_llm:
        polished = polish(description, [department.value, ward, severity.value, str(sla)])
        if polished:
            description, version = polished, f"{TEMPLATE_VERSION}+{OLLAMA_MODEL}"
    return WorkOrderDraft(
        case_id=case_id, department=department, ward=ward, severity=severity,
        issue_type=report.issue_type, title=f"{ISSUE_LABELS[report.issue_type]}: {place}",
        description=description, sla_hours=sla, location_text=report.location_text,
        location=report.location, before_photos=list(report.image_paths), reasons=reasons,
        status="draft", model_version=version, confidence=confidence,
        explanation=(f"Template draft: {department.value}, ward {ward}, {severity.value} "
                     f"severity, {sla} h SLA. A human must approve it."),
    )
