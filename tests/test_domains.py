from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.contracts import TaskContract  # noqa: E402
from scopebench.domains import list_domain_templates  # noqa: E402


def test_domain_template_applies_conservative_defaults():
    contract = TaskContract.model_validate(
        {
            "goal": "Assess compound safety",
            "domain": "drug_research",
        }
    )

    assert "clinical" in contract.forbidden_tool_categories
    assert "lab" in contract.escalation.ask_if_tool_category_in
    assert contract.thresholds.max_irreversibility <= 0.20
    assert contract.thresholds.max_uncertainty <= 0.45
    assert contract.escalation.ask_if_uncertainty_over <= 0.35


def test_domain_templates_loaded_from_yaml():
    templates = list_domain_templates()
    assert "engineering" in templates
