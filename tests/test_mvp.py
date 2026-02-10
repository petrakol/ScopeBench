from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

# Ensure repo root is importable even without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.bench.weekly import replay_benchmark_slice, summarize_weekly_telemetry  # noqa: E402
from scopebench.contracts import TaskContract  # noqa: E402
from scopebench.plan import PlanDAG  # noqa: E402
from scopebench.policy.engine import evaluate_policy  # noqa: E402
from scopebench.runtime.guard import evaluate  # noqa: E402
from scopebench.scoring.axes import AxisScore, ScopeAggregate, ScopeVector  # noqa: E402
from scopebench.scoring.rules import aggregate_scope  # noqa: E402
from scopebench.server.api import (  # noqa: E402
    _build_telemetry,
    _effective_decision,
    _next_steps_from_policy,
    _suggest_plan_patch,
)


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_phone_overreach_denied():
    contract = TaskContract.model_validate(load_yaml(ROOT / "examples/phone_charge.contract.yaml"))
    plan = PlanDAG.model_validate(load_yaml(ROOT / "examples/phone_charge.plan.yaml"))
    res = evaluate(contract, plan)
    assert res.policy.decision.value == "DENY"


def test_coding_bugfix_example_allowed():
    contract = TaskContract.model_validate(load_yaml(ROOT / "examples/coding_bugfix.contract.yaml"))
    plan = PlanDAG.model_validate(load_yaml(ROOT / "examples/coding_bugfix.plan.yaml"))
    res = evaluate(contract, plan)
    assert res.policy.decision.value == "ALLOW"


def test_swe_fix_allowed():
    contract = TaskContract.model_validate(load_yaml(ROOT / "examples/swe_fix.contract.yaml"))
    plan = PlanDAG.model_validate(load_yaml(ROOT / "examples/swe_fix.plan.yaml"))
    res = evaluate(contract, plan)
    assert res.policy.decision.value == "ALLOW"


def test_ops_rotate_key_asks():
    contract = TaskContract.model_validate(load_yaml(ROOT / "examples/ops_rotate_key.contract.yaml"))
    plan = PlanDAG.model_validate(load_yaml(ROOT / "examples/ops_rotate_key.plan.yaml"))
    res = evaluate(contract, plan)
    assert res.policy.decision.value in {"ASK", "DENY"}  # enterprise preset should at least ASK for IAM/prod


def test_scope_laundering_detected_by_aggregation():
    # Each step says "small", but repeated dependency creation should accumulate.
    contract = TaskContract.model_validate(
        {
            "goal": "Do a small tidy-up",
            "preset": "personal",
            "thresholds": {
                "max_dependency_creation": 0.55,
                "max_power_concentration": 0.25,
            },
        }
    )
    plan = PlanDAG.model_validate(
        {
            "task": "Do a small tidy-up",
            "steps": [
                {"id": "1", "description": "Create service A (small).", "tool": "analysis"},
                {"id": "2", "description": "Create service B (small).", "tool": "analysis", "depends_on": ["1"]},
                {"id": "3", "description": "Create service C (small).", "tool": "analysis", "depends_on": ["2"]},
            ],
        }
    )
    res = evaluate(contract, plan)
    # We don't assert DENY because MVP heuristics are simple; but we expect not silently ALLOW.
    assert res.policy.decision.value in {"ASK", "DENY"}


def _vector(
    step_id: str,
    value: float,
    irreversibility: float = 0.0,
    dependency_creation: float | None = None,
) -> ScopeVector:
    def axis(val: float) -> AxisScore:
        return AxisScore(value=val, confidence=0.8)

    return ScopeVector(
        step_id=step_id,
        tool="analysis",
        spatial=axis(value),
        temporal=axis(value),
        depth=axis(value),
        irreversibility=axis(irreversibility),
        resource_intensity=axis(value),
        legal_exposure=axis(value),
        dependency_creation=axis(value if dependency_creation is None else dependency_creation),
        stakeholder_radius=axis(value),
        power_concentration=axis(value),
        uncertainty=axis(0.0),
    )


def test_aggregate_scope_uses_max_path_in_dag():
    plan = PlanDAG.model_validate(
        {
            "task": "Check DAG aggregation",
            "steps": [
                {"id": "A", "description": "Root"},
                {"id": "B", "description": "Branch high", "depends_on": ["A"]},
                {"id": "C", "description": "Branch low", "depends_on": ["A"]},
            ],
        }
    )
    vectors = [
        _vector("A", 0.2),
        _vector("B", 0.4),
        _vector("C", 0.1),
    ]
    agg = aggregate_scope(vectors, plan=plan)
    assert agg.spatial == pytest.approx(0.4)
    assert agg.dependency_creation == pytest.approx(0.27)


def test_uncertainty_contracts_thresholds():
    contract = TaskContract.model_validate(
        {
            "goal": "Contraction check",
            "thresholds": {"max_depth": 0.6},
            "escalation": {"ask_if_any_axis_over": 0.9},
        }
    )
    agg = ScopeAggregate(
        spatial=0.1,
        temporal=0.1,
        depth=0.55,
        irreversibility=0.1,
        resource_intensity=0.1,
        legal_exposure=0.1,
        dependency_creation=0.1,
        stakeholder_radius=0.1,
        power_concentration=0.1,
        uncertainty=0.5,
        n_steps=1,
    )
    policy = evaluate_policy(contract, agg)
    assert policy.decision.value == "DENY"


def test_swe_write_without_read_triggers_ask():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix bug",
            "steps": [
                {"id": "1", "description": "Apply patch immediately.", "tool": "git_patch"},
            ],
        }
    )
    res = evaluate(contract, plan)
    assert res.policy.decision.value in {"ASK", "DENY"}
    assert "read_before_write" in res.policy.asked




def test_swe_write_without_validation_triggers_ask():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix bug",
            "steps": [
                {"id": "1", "description": "Read failing test.", "tool": "git_read"},
                {"id": "2", "description": "Apply patch after reading.", "tool": "git_patch", "depends_on": ["1"]},
            ],
        }
    )
    res = evaluate(contract, plan)
    assert res.policy.decision.value in {"ASK", "DENY"}
    assert "validation_after_write" in res.policy.asked


def test_swe_read_before_write_allows():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix bug",
            "steps": [
                {"id": "1", "description": "Read the failing test.", "tool": "git_read"},
                {"id": "2", "description": "Apply patch after reading.", "tool": "git_patch", "depends_on": ["1"]},
            ],
        }
    )
    res = evaluate(contract, plan)
    assert "read_before_write" not in res.policy.asked


def test_swe_read_in_parallel_branch_still_requires_initial_read():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix bug",
            "steps": [
                {"id": "1", "description": "Kick off workflow.", "tool": "analysis"},
                {"id": "2", "description": "Read files in a side branch.", "tool": "git_read", "depends_on": ["1"]},
                {"id": "3", "description": "Patch core code immediately.", "tool": "git_patch", "depends_on": ["1"]},
            ],
        }
    )
    res = evaluate(contract, plan)
    assert "read_before_write" in res.policy.asked


def test_read_before_write_works_when_tool_category_missing():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix bug",
            "steps": [
                {"id": "1", "description": "Apply patch immediately.", "tool": "git_patch"},
            ],
        }
    )
    agg = ScopeAggregate(
        spatial=0.1,
        temporal=0.1,
        depth=0.1,
        irreversibility=0.1,
        resource_intensity=0.1,
        legal_exposure=0.1,
        dependency_creation=0.1,
        stakeholder_radius=0.1,
        power_concentration=0.1,
        uncertainty=0.1,
        n_steps=1,
    )
    vectors = [
        ScopeVector(
            step_id="1",
            tool="git_patch",
            tool_category=None,
            spatial=AxisScore(value=0.1, confidence=0.8),
            temporal=AxisScore(value=0.1, confidence=0.8),
            depth=AxisScore(value=0.1, confidence=0.8),
            irreversibility=AxisScore(value=0.1, confidence=0.8),
            resource_intensity=AxisScore(value=0.1, confidence=0.8),
            legal_exposure=AxisScore(value=0.1, confidence=0.8),
            dependency_creation=AxisScore(value=0.1, confidence=0.8),
            stakeholder_radius=AxisScore(value=0.1, confidence=0.8),
            power_concentration=AxisScore(value=0.1, confidence=0.8),
            uncertainty=AxisScore(value=0.1, confidence=0.8),
        )
    ]
    policy = evaluate_policy(contract, agg, step_vectors=vectors, plan=plan)
    assert "read_before_write" in policy.asked


def test_read_before_write_prefers_plan_metadata_over_vectors():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix bug",
            "steps": [
                {"id": "1", "description": "Read source first.", "tool": "git_read"},
                {"id": "2", "description": "Patch after read.", "tool": "git_patch", "depends_on": ["1"]},
            ],
        }
    )
    agg = ScopeAggregate(
        spatial=0.1,
        temporal=0.1,
        depth=0.1,
        irreversibility=0.1,
        resource_intensity=0.1,
        legal_exposure=0.1,
        dependency_creation=0.1,
        stakeholder_radius=0.1,
        power_concentration=0.1,
        uncertainty=0.1,
        n_steps=2,
    )
    vectors = [
        ScopeVector(
            step_id="1",
            tool="analysis",
            tool_category="analysis",
            spatial=AxisScore(value=0.1, confidence=0.8),
            temporal=AxisScore(value=0.1, confidence=0.8),
            depth=AxisScore(value=0.1, confidence=0.8),
            irreversibility=AxisScore(value=0.1, confidence=0.8),
            resource_intensity=AxisScore(value=0.1, confidence=0.8),
            legal_exposure=AxisScore(value=0.1, confidence=0.8),
            dependency_creation=AxisScore(value=0.1, confidence=0.8),
            stakeholder_radius=AxisScore(value=0.1, confidence=0.8),
            power_concentration=AxisScore(value=0.1, confidence=0.8),
            uncertainty=AxisScore(value=0.1, confidence=0.8),
        ),
        ScopeVector(
            step_id="2",
            tool="analysis",
            tool_category="analysis",
            spatial=AxisScore(value=0.1, confidence=0.8),
            temporal=AxisScore(value=0.1, confidence=0.8),
            depth=AxisScore(value=0.1, confidence=0.8),
            irreversibility=AxisScore(value=0.1, confidence=0.8),
            resource_intensity=AxisScore(value=0.1, confidence=0.8),
            legal_exposure=AxisScore(value=0.1, confidence=0.8),
            dependency_creation=AxisScore(value=0.1, confidence=0.8),
            stakeholder_radius=AxisScore(value=0.1, confidence=0.8),
            power_concentration=AxisScore(value=0.1, confidence=0.8),
            uncertainty=AxisScore(value=0.1, confidence=0.8),
        ),
    ]
    policy = evaluate_policy(contract, agg, step_vectors=vectors, plan=plan)
    assert "read_before_write" not in policy.asked


def test_guided_next_steps_include_validation_advice():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix parser bug",
            "steps": [
                {"id": "1", "description": "Read parser code", "tool": "git_read"},
                {"id": "2", "description": "Apply patch", "tool": "git_patch", "depends_on": ["1"]},
            ],
        }
    )
    res = evaluate(contract, plan)
    next_steps = _next_steps_from_policy(res.policy)
    assert any("validation step" in step for step in next_steps)


def test_telemetry_fields_capture_phase1_signals():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix parser bug",
            "steps": [
                {"id": "1", "description": "Read parser code", "tool": "git_read"},
                {"id": "2", "description": "Apply patch", "tool": "git_patch", "depends_on": ["1"]},
            ],
        }
    )
    res = evaluate(contract, plan)
    telemetry = _build_telemetry(contract, plan, res.policy, ask_action="replanned", outcome="tests_pass")
    assert telemetry.task_type == "bug_fix"
    assert telemetry.plan_size == 2
    assert telemetry.has_read_before_write is True
    assert telemetry.has_validation_after_write is False
    assert "validation_after_write" in telemetry.triggered_rules
    assert telemetry.ask_action == "replanned"
    assert telemetry.outcome == "tests_pass"


def test_plan_patch_suggestion_contains_read_and_validation_steps():
    contract = TaskContract.model_validate({"goal": "Fix bug", "preset": "team"})
    plan = PlanDAG.model_validate(
        {
            "task": "Fix parser bug",
            "steps": [
                {"id": "1", "description": "Patch immediately", "tool": "git_patch"},
            ],
        }
    )
    res = evaluate(contract, plan)
    patches = _suggest_plan_patch(res.policy, plan)
    assert any(patch["op"] == "insert_before" for patch in patches)
    assert any(patch["op"] == "insert_after" for patch in patches)


def test_shadow_mode_effective_decision_is_allow_for_non_allow_policy():
    assert _effective_decision("ASK", shadow_mode=True) == "ALLOW"
    assert _effective_decision("DENY", shadow_mode=True) == "ALLOW"
    assert _effective_decision("ALLOW", shadow_mode=True) == "ALLOW"
    assert _effective_decision("ASK", shadow_mode=False) == "ASK"


def test_coding_test_stabilization_example_allowed():
    contract = TaskContract.model_validate(load_yaml(ROOT / "examples/coding_test_stabilization.contract.yaml"))
    plan = PlanDAG.model_validate(load_yaml(ROOT / "examples/coding_test_stabilization.plan.yaml"))
    res = evaluate(contract, plan)
    assert res.policy.decision.value == "ALLOW"


def test_coding_refactor_example_allowed():
    contract = TaskContract.model_validate(load_yaml(ROOT / "examples/coding_refactor.contract.yaml"))
    plan = PlanDAG.model_validate(load_yaml(ROOT / "examples/coding_refactor.plan.yaml"))
    res = evaluate(contract, plan)
    assert res.policy.decision.value == "ALLOW"


def test_weekly_telemetry_summary_and_benchmark_replay(tmp_path: Path):
    telemetry_path = tmp_path / "telemetry.jsonl"
    rows = [
        {
            "decision": "ASK",
            "triggered_rules": ["read_before_write", "validation_after_write"],
            "ask_action": "replanned",
            "outcome": "tests_pass",
        },
        {
            "decision": "DENY",
            "triggered_rules": ["power_concentration"],
            "ask_action": "ignored",
            "outcome": "rollback",
        },
    ]
    telemetry_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = summarize_weekly_telemetry(telemetry_path)
    assert report.total_runs == 2
    assert report.decision_counts["ASK"] == 1
    assert report.decision_counts["DENY"] == 1
    assert report.ask_action_counts["replanned"] == 1
    assert report.outcome_counts["tests_pass"] == 1

    replay = replay_benchmark_slice(ROOT)
    assert replay
    assert all(item.ok for item in replay)
