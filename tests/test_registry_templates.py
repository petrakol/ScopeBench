from __future__ import annotations

from pathlib import Path
import sys

import pytest
from typer.testing import CliRunner


# Ensure repo root is importable even without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.cli import app
from scopebench.runtime.guard import evaluate, load_contract, load_plan
from scopebench.scoring.rules import ToolRegistry

TEMPLATE_ROOT = ROOT / "scopebench" / "templates"


def test_tool_registry_loads_and_expanded_domains_present() -> None:
    registry = ToolRegistry.load_default()
    expanded = [
        t
        for t in registry._tools.values()  # noqa: SLF001 - test coverage for expanded registry shape
        if set(t.domains) & {"finance", "health", "marketing", "ops"}
    ]
    assert len(expanded) >= 30
    escalated_cats = {"finance", "health", "payments", "iam", "infra", "legal"}
    assert any(t.category in escalated_cats for t in expanded)


def test_tool_registry_schema_validation_fails_fast(tmp_path: Path) -> None:
    bad = tmp_path / "bad_tool_registry.yaml"
    bad.write_text(
        """
tools:
  oops:
    category: finance
    priors:
      not_an_axis: 1.2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown prior axes"):
        ToolRegistry.load_from_file(bad)


def test_templates_evaluate_without_exception() -> None:
    for domain in ["swe", "ops", "finance", "health", "marketing"]:
        contract = load_contract(str(TEMPLATE_ROOT / domain / "contract.yaml"))
        plan = load_plan(str(TEMPLATE_ROOT / domain / "plan.yaml"))
        result = evaluate(contract, plan)
        assert result.policy.decision.value in {"ALLOW", "ASK", "DENY"}


def test_cli_template_commands() -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["template", "list"])
    assert listed.exit_code == 0
    assert "finance" in listed.stdout

    shown = runner.invoke(app, ["template", "show", "finance/contract"])
    assert shown.exit_code == 0
    assert "goal:" in shown.stdout

    generated = runner.invoke(app, ["template", "generate", "health"])
    assert generated.exit_code == 0
    assert "contract:" in generated.stdout
    assert "plan:" in generated.stdout
