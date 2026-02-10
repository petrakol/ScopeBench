from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.integrations import (
    apply_realtime_estimates,
    default_cost_connectors,
    from_autogen_messages,
    from_langchain_plan,
    guard,
)
from scopebench.plan import PlanDAG


def test_guard_entrypoint_returns_decision_and_patch() -> None:
    contract = {"goal": "Fix failing unit test", "preset": "team"}
    plan = {
        "task": "Fix failing unit test",
        "steps": [
            {"id": "1", "description": "Apply patch without reading", "tool": "git_patch"},
        ],
    }

    result = guard(plan=plan, contract=contract)

    assert result.decision in {"ASK", "DENY", "ALLOW"}
    assert isinstance(result.recommended_patch, list)
    assert isinstance(result.aggregate, dict)
    assert result.trace_id is None


def test_framework_adapters_produce_scopebench_plan() -> None:
    lc_plan = from_langchain_plan(
        {
            "goal": "ship feature",
            "steps": [
                {"id": "A", "action": "read repo", "tool": "git_read"},
                {"action": "write patch", "name": "git_patch", "depends_on": ["A"]},
            ],
        }
    )
    assert lc_plan["task"] == "ship feature"
    assert lc_plan["steps"][0]["id"] == "A"

    autogen_plan = from_autogen_messages(
        [
            {"content": "inspect logs", "tool": "analysis"},
            {"content": "apply fix", "tool": "git_patch"},
        ],
        task="incident mitigation",
    )
    assert autogen_plan["task"] == "incident mitigation"
    assert len(autogen_plan["steps"]) == 2
    assert autogen_plan["steps"][1]["depends_on"] == ["1"]


def test_default_cost_connectors_apply_realtime_estimates() -> None:
    plan = PlanDAG.model_validate(
        {
            "task": "Update billing pipeline",
            "steps": [
                {"id": "1", "description": "Process invoices for customers", "tool": "billing_sync"},
                {"id": "2", "description": "Deploy infra update", "tool": "infra_deploy", "depends_on": ["1"]},
            ],
        }
    )

    enriched = apply_realtime_estimates(plan, default_cost_connectors())
    assert len(enriched.steps[0].realtime_estimates) >= 1
    assert len(enriched.steps[1].realtime_estimates) >= 1
    assert enriched.steps[1].resolved_cost_usd() is not None
