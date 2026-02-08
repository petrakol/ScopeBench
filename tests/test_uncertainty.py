from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.plan import PlanStep  # noqa: E402
from scopebench.scoring.rules import ToolRegistry, score_step  # noqa: E402


def test_uncertainty_respects_confidence_hint():
    tool_registry = ToolRegistry.load_default()
    step = PlanStep(
        id="1",
        description="Review documented requirements and summarize next steps.",
        tool="analysis",
        effects={"confidence": 0.9},
    )
    vector = score_step(step, tool_registry)
    assert vector.uncertainty.value <= 0.1
    assert "confidence hint" in vector.uncertainty.rationale
