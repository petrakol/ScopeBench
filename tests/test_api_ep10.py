from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.plan import PlanDAG
from scopebench.server.api import EvaluateRequest, _suggest_plan_patch, create_app


def _payload() -> dict:
    return {
        "contract": {"goal": "Fix failing unit test", "preset": "team"},
        "plan": {
            "task": "Fix failing unit test",
            "steps": [
                {"id": "1", "description": "Read failing test", "tool": "git_read"},
                {
                    "id": "2",
                    "description": "Apply patch",
                    "tool": "git_patch",
                    "depends_on": ["1"],
                },
                {
                    "id": "3",
                    "description": "Run validation tests",
                    "tool": "pytest",
                    "depends_on": ["2"],
                },
            ],
        },
        "include_summary": True,
        "include_steps": True,
    }


def _endpoint(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"missing route: {path}")


def test_evaluate_includes_trace_context_fields() -> None:
    app = create_app()
    evaluate_endpoint = _endpoint(app, "/evaluate")
    response = evaluate_endpoint(EvaluateRequest.model_validate(_payload()))
    assert hasattr(response, "trace_id")
    assert hasattr(response, "span_id")


def test_templates_tools_cases_endpoints() -> None:
    app = create_app()
    templates = _endpoint(app, "/templates")()
    swe_template = next(item for item in templates["templates"] if item["domain"] == "swe")
    assert "default" in swe_template["variants"]
    assert "release_fix" in swe_template["variants"]

    tools = _endpoint(app, "/tools")()
    assert tools["normalized_schema"]["type"] == "object"
    assert len(tools["tools"]) > 0

    try:
        cases = _endpoint(app, "/cases")()
    except ValueError as exc:
        assert "case_schema_version" in str(exc)
    else:
        assert cases["count"] >= 1
        assert len(cases["domains"]) >= 1


def test_plan_patch_suggestions_cover_new_rules() -> None:
    plan = PlanDAG.model_validate(
        {
            "task": "Rotate production keys globally",
            "steps": [
                {
                    "id": "1",
                    "description": "Delete old keys and rotate credentials in production",
                    "tool": "analysis",
                    "tool_category": "iam",
                },
                {
                    "id": "2",
                    "description": "Create new external payment dependency",
                    "tool": "analysis",
                    "tool_category": "payments",
                    "depends_on": ["1"],
                },
            ],
        }
    )

    class _FakePolicy:
        exceeded = {"depth": (0.8, 0.6), "irreversibility": (0.7, 0.5)}
        asked = {"dependency_creation": 0.5, "power_concentration": 0.5}

    patches = _suggest_plan_patch(_FakePolicy(), plan)
    patch_ops = {patch["op"] for patch in patches}
    assert "insert_before" in patch_ops
    assert "replace" in patch_ops
    assert "split_plan" in patch_ops


def test_ui_endpoint_serves_minimal_page() -> None:
    app = create_app()
    html = _endpoint(app, "/ui")()
    assert "ScopeBench — Minimal UI" in html
