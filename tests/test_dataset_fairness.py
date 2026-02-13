from __future__ import annotations

import json

from typer.testing import CliRunner

from scopebench.bench.fairness import evaluate_dataset_fairness
from scopebench.bench.dataset import validate_case_object
from scopebench.cli import app


def _case(case_id: str, domain: str, decision: str, axis: str) -> dict:
    vector = {
        "step_id": "s1",
        "spatial": 0.1,
        "temporal": 0.1,
        "depth": 0.1,
        "irreversibility": 0.1,
        "resource_intensity": 0.1,
        "legal_exposure": 0.1,
        "dependency_creation": 0.1,
        "stakeholder_radius": 0.1,
        "power_concentration": 0.1,
        "uncertainty": 0.1,
    }
    vector[axis] = 0.9
    return {
        "case_schema_version": "1.0",
        "id": case_id,
        "domain": domain,
        "instruction": "Test instruction",
        "contract": {"goal": "Goal", "preset": "team"},
        "plan": {"task": "Task", "steps": [{"id": "s1", "description": "d", "tool": "analysis"}]},
        "expected_decision": decision,
        "expected_rationale": "rationale",
        "expected_step_vectors": [vector],
    }


def test_fairness_flags_underrepresented_categories() -> None:
    cases = [
        _case("c1", "swe", "ALLOW", "depth"),
        _case("c2", "swe", "ALLOW", "depth"),
        _case("c3", "ops", "DENY", "spatial"),
    ]
    report = evaluate_dataset_fairness([validate_case_object(c) for c in cases], min_share=0.4)

    assert report.total_cases == 3
    assert any(item["type"] == "decision" and item["category"] == "ASK" for item in report.underrepresented)
    assert any(item["type"] == "axis" and item["category"] == "temporal" for item in report.underrepresented)
    assert report.contribution_suggestions


def test_dataset_fairness_cli_json(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    rows = [
        _case("c1", "swe", "ALLOW", "depth"),
        _case("c2", "swe", "ALLOW", "depth"),
        _case("c3", "ops", "DENY", "spatial"),
    ]
    cases_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["dataset-fairness", str(cases_path), "--min-share", "0.4", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_cases"] == 3
    assert payload["underrepresented"]
    assert "contribution_suggestions" in payload
