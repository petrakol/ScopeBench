from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.integrations import guard


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
