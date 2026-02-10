from __future__ import annotations

from pathlib import Path
import sys

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.cli import app  # noqa: E402
from scopebench.plan import plan_from_dict  # noqa: E402
from scopebench.scoring.effects_annotator import suggest_effects_for_plan  # noqa: E402


def test_suggest_effects_for_plan_enriches_steps() -> None:
    plan = plan_from_dict(
        {
            "task": "Assess rollout",
            "steps": [
                {
                    "id": "1",
                    "description": "Run global cross-border API call for customer data export.",
                    "tool": "api_request",
                },
                {
                    "id": "2",
                    "description": "Local analysis only.",
                    "tool": "analysis",
                    "effects": {
                        "version": "effects_v1",
                        "legal": {"magnitude": "high", "regimes": ["gdpr"]},
                    },
                },
            ],
        }
    )
    suggestions = suggest_effects_for_plan(plan)

    assert len(suggestions) == 2

    first = suggestions[0].effects.model_dump(exclude_none=True)
    assert first["version"] == "effects_v1"
    assert first["legal"]["magnitude"] in {"medium", "high"}
    concepts = {m["concept"] for m in first.get("macro_consequences", [])}
    assert "cross_border_transfer_risk" in concepts

    second = suggestions[1].effects.model_dump(exclude_none=True)
    assert second["legal"]["magnitude"] == "high"


def test_cli_suggest_effects_json_and_in_place(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo.plan.yaml"
    plan_path.write_text(
        """
task: Demo
steps:
  - id: "1"
    description: "Process customer requests worldwide."
    tool: web_search
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    rendered = runner.invoke(app, ["suggest-effects", str(plan_path), "--json"])
    assert rendered.exit_code == 0
    assert '"version": "effects_v1"' in rendered.stdout

    rewritten = runner.invoke(app, ["suggest-effects", str(plan_path), "--in-place"])
    assert rewritten.exit_code == 0
    assert "Updated" in rewritten.stdout
    updated_text = plan_path.read_text(encoding="utf-8")
    assert "effects:" in updated_text
    assert "version: effects_v1" in updated_text
