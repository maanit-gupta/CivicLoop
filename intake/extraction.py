"""What a text extractor must output, and how raw output is parsed.

Every extractor (keyword rules, the LLM) returns a raw JSON *string*. Parsing lives here, once,
so a model that writes broken JSON is scored by the same code as one that doesn't.
"""

import json

from pydantic import BaseModel, ConfigDict, ValidationError

from contracts.api import Cues, IssueType


class Fields(BaseModel):
    """What a model is asked to fill in (Step 4.3). Every field is required."""

    model_config = ConfigDict(extra="forbid")

    issue_type: IssueType
    location_text: str
    cues: Cues


class Extraction(Fields):
    """A full extraction: the fields plus the language (from the script rule, not a model)."""

    language: str


def parse(raw: str, model: type[BaseModel] = Extraction) -> BaseModel | None:
    """The validated object, or None if `raw` is not valid JSON in `model`'s shape."""
    try:
        return model.model_validate(json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValidationError):
        return None


def strict_schema(model: type[BaseModel] = Fields) -> dict:
    """JSON schema with every property required, nested ones too. Pydantic leaves fields with
    defaults optional (Cues has them), and a constrained decoder would then let the model skip
    them."""
    schema = model.model_json_schema()
    for obj in [schema, *schema.get("$defs", {}).values()]:
        if "properties" in obj:
            obj["required"] = list(obj["properties"])
    return schema
