from __future__ import annotations

import json
from pathlib import Path

import yaml

from scopebench.bench.continuous_learning import (
    analyze_continuous_learning,
    apply_learning_to_registry,
    render_learning_report_markdown,
)


def _telemetry_row(tools: list[str], decision: str, aggregate: dict[str, float]) -> dict:
    return {
        "decision": decision,
        "aggregate": aggregate,
        "asked": {},
        "exceeded": {},
        "feedback": {"outcome": "manual_override" if decision == "DENY" else "tests_pass"},
        "policy_input": {
            "plan": {
                "steps": [{"id": str(i), "tool": tool, "description": "step"} for i, tool in enumerate(tools)]
            }
        },
    }


def test_continuous_learning_recommends_priors_and_effects(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    rows = [
        _telemetry_row(
            ["shell", "api_request"],
            "DENY",
            {"legal_exposure": 0.9, "resource_intensity": 0.8, "depth": 0.4},
        ),
        _telemetry_row(
            ["shell", "api_request"],
            "ASK",
            {"legal_exposure": 0.75, "resource_intensity": 0.7, "depth": 0.35},
        ),
    ]
    telemetry_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    benchmark_path = tmp_path / "bench.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "plan": {
                    "steps": [{"tool": "shell"}, {"tool": "api_request"}],
                    "metadata": {"adversarial_tactics": ["tool_hopping"]},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    registry_copy = tmp_path / "tool_registry.yaml"
    root_registry = Path(__file__).resolve().parents[1] / "scopebench" / "tool_registry.yaml"
    registry_copy.write_text(root_registry.read_text(encoding="utf-8"), encoding="utf-8")

    report = analyze_continuous_learning(
        telemetry_path=telemetry_path,
        benchmark_path=benchmark_path,
        registry_path=registry_copy,
        min_pair_support=1,
    )

    assert report.telemetry_rows == 2
    assert report.pattern_counts["benchmark:tool_hopping"] == 1
    assert report.risky_tool_pairs
    assert "shell" in report.prior_recommendations
    assert report.default_effect_recommendations["shell"]["legal"]["magnitude"] in {
        "medium",
        "high",
        "extreme",
    }

    markdown = render_learning_report_markdown(report)
    assert "Continuous learning report" in markdown
    assert "`shell` + `api_request`" in markdown or "`api_request` + `shell`" in markdown


def test_apply_learning_updates_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "tool_registry.yaml"
    registry = {
        "tools": {
            "shell": {
                "category": "exec",
                "priors": {"legal_exposure": 0.1, "resource_intensity": 0.1},
                "default_effects": {"legal": {"magnitude": "low"}},
            }
        }
    }
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    class _Report:
        prior_recommendations = {
            "shell": {
                "legal_exposure": 0.5,
                "resource_intensity": 0.4,
            }
        }
        default_effect_recommendations = {
            "shell": {"legal": {"magnitude": "high", "rationale": "telemetry-derived"}}
        }

    updated = apply_learning_to_registry(_Report(), registry_path=registry_path)
    assert updated["tools"]["shell"]["priors"]["legal_exposure"] == 0.5
    assert updated["tools"]["shell"]["default_effects"]["legal"]["magnitude"] == "high"
