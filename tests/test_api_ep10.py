from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.plan import PlanDAG
from scopebench.server.api import (
    CalibrationAdjustmentRequest,
    EvaluateRequest,
    EvaluateStreamRequest,
    _suggest_plan_patch,
    create_app,
)


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

    cases = _endpoint(app, "/cases")()
    assert "count" in cases
    if cases["count"] > 0:
        assert len(cases["domains"]) >= 1
    else:
        assert "error" in cases


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
        policy_input = None

    patches = _suggest_plan_patch(_FakePolicy(), plan)
    patch_ops = {patch["op"] for patch in patches}
    assert "insert_before" in patch_ops
    assert "replace" in patch_ops
    assert "split_plan" in patch_ops



def test_evaluate_stream_endpoint_reacts_to_threshold_crossings_and_step_updates() -> None:
    app = create_app()
    evaluate_stream = _endpoint(app, "/evaluate_stream")
    request = EvaluateStreamRequest.model_validate(
        {
            "contract": {"goal": "Maintain training pipeline", "preset": "team"},
            "plan": {
                "task": "Maintain training pipeline",
                "steps": [
                    {"id": "1", "description": "Inspect current jobs", "tool": "git_read"},
                    {"id": "2", "description": "Run a small eval", "tool": "analysis", "depends_on": ["1"]},
                ],
            },
            "events": [
                {
                    "event_id": "evt-1",
                    "operation": "update_step",
                    "step_id": "2",
                    "step": {
                        "description": "Launch long-running multi-region retraining with new serving dependency",
                        "tool": "analysis",
                        "tool_category": "prod",
                    },
                },
                {
                    "event_id": "evt-2",
                    "operation": "add_step",
                    "step_id": "3",
                    "step": {
                        "id": "3",
                        "description": "Rollback check and safety validation",
                        "tool": "pytest",
                        "depends_on": ["2"],
                    },
                },
            ],
            "judge": "llm",
            "include_steps": False,
        }
    )
    response = evaluate_stream(request)

    assert response.initial.event_id == "initial"
    assert len(response.updates) == 2

    first = response.updates[0]
    assert first.event_id == "evt-1"
    trigger_kinds = {trigger.kind for trigger in first.triggers}
    assert "judge_output_changed" in trigger_kinds or "threshold_crossed" in trigger_kinds
    assert isinstance(first.judge_output_deltas, list)

    second = response.updates[1]
    assert second.event_id == "evt-2"
    assert second.operation == "add_step"


def test_evaluate_stream_remove_step_updates_dependencies() -> None:
    app = create_app()
    evaluate_stream = _endpoint(app, "/evaluate_stream")
    response = evaluate_stream(
        EvaluateStreamRequest.model_validate(
            {
                "contract": {"goal": "Operate safely", "preset": "team"},
                "plan": {
                    "task": "Operate safely",
                    "steps": [
                        {"id": "1", "description": "Read", "tool": "git_read"},
                        {"id": "2", "description": "Patch", "tool": "git_patch", "depends_on": ["1"]},
                        {"id": "3", "description": "Validate", "tool": "pytest", "depends_on": ["2"]},
                    ],
                },
                "events": [
                    {"event_id": "drop-2", "operation": "remove_step", "step_id": "2"},
                ],
            }
        )
    )

    assert len(response.updates) == 1
    update = response.updates[0]
    assert update.event_id == "drop-2"
    assert update.operation == "remove_step"

def test_ui_endpoint_serves_interactive_page() -> None:
    app = create_app()
    html = _endpoint(app, "/ui")()
    assert "ScopeBench Interactive Workbench" in html
    assert "Replay telemetry" in html
    assert "Calibration Dashboard" in html
    assert "What-if Lab" in html
    assert "Explainability: Aggregate Risk Contributions" in html


def test_telemetry_replay_endpoint_without_configuration() -> None:
    app = create_app(telemetry_jsonl_path=None)
    replay = _endpoint(app, "/telemetry/replay")()
    assert replay["enabled"] is False


def test_calibration_dashboard_and_adjustment_endpoints(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    rows = [
        {
            "telemetry": {"task_type": "bug_fix"},
            "asked": {"depth": 0.8, "uncertainty": 0.7},
            "outcome": "tests_fail",
        },
        {
            "telemetry": {"task_type": "bug_fix"},
            "asked": {"depth": 0.6},
            "outcome": "tests_pass",
        },
        {
            "telemetry": {"task_type": "general_coding"},
            "asked": {"uncertainty": 0.9},
            "outcome": "manual_override",
        },
    ]
    telemetry_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    app = create_app(telemetry_jsonl_path=str(telemetry_path))
    dashboard = _endpoint(app, "/calibration/dashboard")()
    assert dashboard.enabled is True
    domains = {entry.domain for entry in dashboard.domains}
    assert "bug_fix" in domains

    adjust_endpoint = _endpoint(app, "/calibration/adjust")
    adjusted = adjust_endpoint(
        CalibrationAdjustmentRequest.model_validate(
            {
                "domain": "bug_fix",
                "axis_threshold_factor_delta": {"depth": -0.1},
                "axis_scale_delta": {"depth": 0.05},
                "abstain_uncertainty_threshold_delta": -0.02,
            }
        )
    )
    bug_fix_entry = next(entry for entry in adjusted.domains if entry.domain == "bug_fix")
    assert bug_fix_entry.calibration["axis_threshold_factor"]["depth"] != 1.0
