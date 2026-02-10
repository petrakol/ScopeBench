from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.bench.dataset import default_cases_path, load_cases  # noqa: E402
from scopebench.contracts import TaskContract  # noqa: E402
from scopebench.plan import PlanDAG  # noqa: E402
from scopebench.runtime.guard import evaluate  # noqa: E402


CASES_PATH = default_cases_path()


def test_dataset_size_and_domain_distribution() -> None:
    cases = load_cases(CASES_PATH)
    assert len(cases) >= 500

    counts = Counter(case.domain for case in cases)
    required_domains = {"swe", "ops", "finance", "health", "marketing"}
    assert required_domains.issubset(counts)

    # Logged so CI output captures distribution for visibility.
    print("domain_distribution:", json.dumps(counts, sort_keys=True))


def test_dataset_golden_decisions_exact_match() -> None:
    cases = load_cases(CASES_PATH)
    mismatches: list[str] = []

    for case in cases:
        contract = TaskContract.model_validate(case.contract)
        plan = PlanDAG.model_validate(case.plan)
        predicted = evaluate(contract, plan).policy.decision.value
        if predicted != case.expected_decision:
            mismatches.append(
                f"{case.id}: expected {case.expected_decision}, got {predicted}"
            )

    assert not mismatches, "\n".join(mismatches)


def test_loader_rejects_invalid_case_with_actionable_error(tmp_path: Path) -> None:
    bad_case = {
        "case_schema_version": "1.0",
        "id": "broken_case_001",
        "domain": "swe",
        "instruction": "Broken",
        "contract": {"goal": "Broken", "preset": "team"},
        "plan": {
            "task": "Broken",
            "steps": [{"id": "1", "description": "Oops", "tool": "analysis", "depends_on": ["9"]}],
        },
        "expected_decision": "ALLOW",
    }

    path = tmp_path / "bad_cases.jsonl"
    path.write_text(json.dumps(bad_case) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)

    message = str(exc_info.value)
    assert "broken_case_001" in message
    assert "depends_on" in message
