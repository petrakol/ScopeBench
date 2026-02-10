from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scopebench.bench.dataset import ScopeBenchCase, load_cases, validate_case_object
from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate
from scopebench.scoring.axes import SCOPE_AXES


def validate_cases_file(cases_path: Path) -> list[ScopeBenchCase]:
    return load_cases(cases_path)


def _read_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an object in {path}")
    return raw


def suggest_case(
    *,
    case_id: str,
    domain: str,
    instruction: str,
    contract_path: Path,
    plan_path: Path,
    expected_decision: str,
    expected_rationale: str,
    case_schema_version: str = "1.0",
    notes: str | None = None,
    policy_backend: str = "python",
) -> dict[str, Any]:
    contract_dict = _read_mapping(contract_path)
    plan_dict = _read_mapping(plan_path)

    contract = TaskContract.model_validate(contract_dict)
    plan = PlanDAG.model_validate(plan_dict)
    result = evaluate(contract, plan, policy_backend=policy_backend)

    vectors: list[dict[str, Any]] = []
    for vector in result.vectors:
        row: dict[str, Any] = {"step_id": vector.step_id}
        for axis in SCOPE_AXES:
            row[axis] = getattr(vector, axis).value
        vectors.append(row)

    case: dict[str, Any] = {
        "case_schema_version": case_schema_version,
        "id": case_id,
        "domain": domain,
        "instruction": instruction,
        "contract": contract_dict,
        "plan": plan_dict,
        "expected_decision": expected_decision,
        "expected_rationale": expected_rationale,
        "expected_step_vectors": vectors,
    }
    if notes:
        case["notes"] = notes

    # Ensure generated case is compliant before writing.
    validate_case_object(case)
    return case


def append_case(cases_path: Path, case: dict[str, Any]) -> None:
    with cases_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(case) + "\n")


def submit_pull_request(title: str, body: str) -> str:
    cmd = ["gh", "pr", "create", "--fill", "--title", title, "--body", body]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' is required to auto-submit PRs.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "Failed to create pull request with gh.") from exc
    return completed.stdout.strip()
