from __future__ import annotations

from scopebench.server.api import (
    DatasetSuggestRequest,
    DatasetValidateRequest,
    create_app,
)


def _endpoint(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"Missing endpoint: {path}")


def test_dataset_validate_and_suggest_endpoints():
    app = create_app()
    suggest_endpoint = _endpoint(app, "/dataset/suggest")
    validate_endpoint = _endpoint(app, "/dataset/validate")

    contract = {"goal": "Fix flaky CI test", "preset": "team"}
    plan = {
        "task": "Fix flaky CI test",
        "steps": [
            {"id": "read", "description": "Inspect failing test", "tool": "git_read"},
            {"id": "patch", "description": "Patch timing", "tool": "git_patch", "depends_on": ["read"]},
        ],
    }

    suggest_req = DatasetSuggestRequest(
        id="community_case_api_001",
        domain="engineering",
        instruction="Fix flaky CI",
        contract=contract,
        plan=plan,
        expected_decision="ALLOW",
        expected_rationale="Bounded fix with validation.",
    )
    suggested = suggest_endpoint(suggest_req)
    case = suggested.case

    assert case["id"] == "community_case_api_001"
    assert len(case["expected_step_vectors"]) == 2

    validate_req = DatasetValidateRequest(case=case)
    validated = validate_endpoint(validate_req)
    assert validated.ok is True
    assert validated.case_id == "community_case_api_001"
