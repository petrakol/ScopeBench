from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

SUPPORTED_CASE_SCHEMA_VERSIONS = {"1.0"}
_REQUIRED_FIELDS = (
    "case_schema_version",
    "id",
    "domain",
    "instruction",
    "contract",
    "plan",
    "expected_decision",
)
_VALID_DECISIONS = {"ALLOW", "ASK", "DENY"}


@dataclass(frozen=True)
class ScopeBenchCase:
    case_schema_version: str
    id: str
    domain: str
    instruction: str
    contract: dict[str, Any]
    plan: dict[str, Any]
    expected_decision: str  # ALLOW / ASK / DENY
    notes: Optional[str] = None


def _case_error(case_id: str, field: str, message: str) -> ValueError:
    return ValueError(f"Invalid case '{case_id}' field '{field}': {message}")


def _validate_case(obj: Any, *, line_no: int) -> ScopeBenchCase:
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid case '<line:{line_no}>' field '<root>': expected JSON object")

    case_id = str(obj.get("id", f"<line:{line_no}>"))

    for field in _REQUIRED_FIELDS:
        if field not in obj:
            raise _case_error(case_id, field, "is required")

    case_schema_version = obj["case_schema_version"]
    if not isinstance(case_schema_version, str):
        raise _case_error(case_id, "case_schema_version", "must be a string")
    if case_schema_version not in SUPPORTED_CASE_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_CASE_SCHEMA_VERSIONS))
        raise _case_error(
            case_id,
            "case_schema_version",
            f"unsupported version {case_schema_version!r}; supported: {supported}",
        )

    if not isinstance(obj["id"], str) or not obj["id"].strip():
        raise _case_error(case_id, "id", "must be a non-empty string")
    if not isinstance(obj["domain"], str) or not obj["domain"].strip():
        raise _case_error(case_id, "domain", "must be a non-empty string")
    if not isinstance(obj["instruction"], str) or not obj["instruction"].strip():
        raise _case_error(case_id, "instruction", "must be a non-empty string")
    if not isinstance(obj["contract"], dict):
        raise _case_error(case_id, "contract", "must be an object")
    if not isinstance(obj["plan"], dict):
        raise _case_error(case_id, "plan", "must be an object")

    expected_decision = obj["expected_decision"]
    if expected_decision not in _VALID_DECISIONS:
        allowed = ", ".join(sorted(_VALID_DECISIONS))
        raise _case_error(case_id, "expected_decision", f"must be one of: {allowed}")

    steps = obj["plan"].get("steps")
    if not isinstance(steps, list) or not steps:
        raise _case_error(case_id, "plan.steps", "must be a non-empty array")

    step_ids: set[str] = set()
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise _case_error(case_id, f"plan.steps[{idx}]", "must be an object")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise _case_error(case_id, f"plan.steps[{idx}].id", "must be a non-empty string")
        if step_id in step_ids:
            raise _case_error(case_id, "plan.steps", f"duplicate step id: {step_id!r}")
        step_ids.add(step_id)

    for idx, step in enumerate(steps):
        depends_on = step.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list) or not all(isinstance(dep, str) for dep in depends_on):
            raise _case_error(case_id, f"plan.steps[{idx}].depends_on", "must be an array of step ids")
        missing = sorted(dep for dep in depends_on if dep not in step_ids)
        if missing:
            raise _case_error(
                case_id,
                f"plan.steps[{idx}].depends_on",
                f"references unknown step ids: {', '.join(missing)}",
            )

    notes = obj.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise _case_error(case_id, "notes", "must be a string when present")

    return ScopeBenchCase(
        case_schema_version=case_schema_version,
        id=obj["id"],
        domain=obj["domain"],
        instruction=obj["instruction"],
        contract=obj["contract"],
        plan=obj["plan"],
        expected_decision=expected_decision,
        notes=notes,
    )


def load_cases(path: Path) -> List[ScopeBenchCase]:
    cases: List[ScopeBenchCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            cases.append(_validate_case(obj, line_no=line_no))
    return cases


def default_cases_path() -> Path:
    return Path(__file__).resolve().parent / "cases" / "examples.jsonl"
