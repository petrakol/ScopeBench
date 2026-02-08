from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.plan import PlanStep  # noqa: E402
from scopebench.scoring.rules import ToolRegistry, score_step  # noqa: E402


def test_causal_rules_raise_minimums_for_lab():
    tool_registry = ToolRegistry.load_default()
    step = PlanStep(
        id="1",
        description="Run a controlled lab assay.",
        tool="lab_experiment",
    )
    vector = score_step(step, tool_registry)
    assert vector.legal_exposure.value >= 0.45
    assert vector.stakeholder_radius.value >= 0.35
    assert "lab" in vector.legal_exposure.rationale
