from __future__ import annotations

from pathlib import Path
import json
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
    for domain in [
        "swe",
        "ops",
        "finance",
        "health",
        "marketing",
        "robotics",
        "biotech",
        "supply-chain",
        "fintech-risk",
        "public-sector",
    ]:
        contract = load_contract(str(TEMPLATE_ROOT / domain / "contract.yaml"))
        plan = load_plan(str(TEMPLATE_ROOT / domain / "plan.yaml"))
        result = evaluate(contract, plan)
        assert result.policy.decision.value in {"ALLOW", "ASK", "DENY"}


def test_cli_template_commands() -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["template", "list"])
    assert listed.exit_code == 0
    assert "finance" in listed.stdout
    assert "payments" in listed.stdout

    shown = runner.invoke(app, ["template", "show", "finance/contract"])
    assert shown.exit_code == 0
    assert "goal:" in shown.stdout

    variant_plan = runner.invoke(app, ["template", "show", "finance/payments/plan"])
    assert variant_plan.exit_code == 0
    assert "refund_issue" in variant_plan.stdout

    generated = runner.invoke(app, ["template", "generate", "health/medical_data"])
    assert generated.exit_code == 0
    assert "contract:" in generated.stdout
    assert "plan:" in generated.stdout


def test_cli_template_wizard_non_interactive() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "template",
            "wizard",
            "--domain",
            "health",
            "--variant",
            "medical_data",
            "--goal",
            "Generate baseline de-identification workflow",
            "--preset",
            "regulated",
            "--no-edit-steps",
        ],
    )
    assert result.exit_code == 0
    assert "Generate baseline de-identification workflow" in result.stdout
    assert "domain: health" in result.stdout
    assert "preset: regulated" in result.stdout


def test_cli_plugin_generate_and_lint(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "robotics-plugin.yaml"

    generated = runner.invoke(
        app,
        [
            "plugin-generate",
            "--non-interactive",
            "--out",
            str(out),
            "--domain",
            "robotics",
            "--publisher",
            "community",
            "--name",
            "robotics-starter",
            "--tools",
            "move_arm,scan_area",
            "--tool-definitions",
            '[{"tool":"scan_area","risk_class":"high","priors":{"uncertainty":0.4}}]',
            "--effect-mappings",
            "1",
            "--policy-rules",
            "1",
            "--secret",
            "secret",
        ],
    )
    assert generated.exit_code == 0
    assert out.exists()
    payload = json.loads(generated.stdout)
    assert payload["harness"]["passed"] is True
    assert "plugin-lint" in payload["commands"]["lint"]
    assert "plugin-install" in payload["commands"]["install"]

    linted = runner.invoke(app, ["plugin-lint", str(out)])
    assert linted.exit_code == 0
    assert "passed" in linted.stdout.lower()


def test_cli_plugin_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    source = tmp_path / "plugin.yaml"
    source.write_text(
        """
name: robotics-pack
version: 0.1.0
publisher: community
tools:
  scan_area:
    category: robotics_operations
    domains: [robotics]
    risk_class: moderate
    priors:
      uncertainty: 0.2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    target_dir = tmp_path / "plugins"
    monkeypatch.delenv("SCOPEBENCH_PLUGIN_DIRS", raising=False)

    installed = runner.invoke(app, ["plugin-install", str(source), "--plugin-dir", str(target_dir), "--json"])
    assert installed.exit_code == 0
    payload = json.loads(installed.stdout)
    assert payload["ok"] is True
    assert (target_dir / source.name).exists()
    assert str(target_dir.resolve()) in payload["configured_plugin_dirs"]
