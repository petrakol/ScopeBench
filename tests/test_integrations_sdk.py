from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.integrations import from_autogen_messages, from_langchain_plan, guard


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
