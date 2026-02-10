from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.integrations import (
    apply_realtime_estimates,
    default_cost_connectors,
    from_autogen_messages,
    from_langchain_plan,
    guard,
)
from scopebench.plan import PlanDAG


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


def test_default_cost_connectors_apply_realtime_estimates() -> None:
    plan = PlanDAG.model_validate(
        {
            "task": "Update billing pipeline",
            "steps": [
                {"id": "1", "description": "Process invoices for customers", "tool": "billing_sync"},
                {"id": "2", "description": "Deploy infra update", "tool": "infra_deploy", "depends_on": ["1"]},
            ],
        }
    )

    enriched = apply_realtime_estimates(plan, default_cost_connectors())
    assert len(enriched.steps[0].realtime_estimates) >= 1
    assert len(enriched.steps[1].realtime_estimates) >= 1
    assert enriched.steps[1].resolved_cost_usd() is not None


def test_workflow_framework_adapters_produce_scopebench_plan() -> None:
    from scopebench.integrations import from_airflow_dag, from_dagster_ops, from_prefect_tasks

    class _AirflowTask:
        def __init__(self, task_id: str, upstream_task_ids: list[str] | None = None) -> None:
            self.task_id = task_id
            self.upstream_task_ids = set(upstream_task_ids or [])
            self.task_type = "PythonOperator"

    class _AirflowDag:
        dag_id = "nightly_pipeline"
        tasks = [_AirflowTask("extract"), _AirflowTask("load", ["extract"])]

    airflow_plan = from_airflow_dag(_AirflowDag())
    assert airflow_plan["task"] == "nightly_pipeline"
    assert airflow_plan["steps"][1]["depends_on"] == ["extract"]

    class _PrefectTask:
        def __init__(self, name: str, upstream: list[str] | None = None) -> None:
            self.name = name
            self.description = f"run {name}"
            self.upstream = upstream or []

    prefect_plan = from_prefect_tasks([_PrefectTask("extract"), _PrefectTask("transform", ["extract"])], task="etl")
    assert prefect_plan["task"] == "etl"
    assert prefect_plan["steps"][1]["id"] == "transform"

    class _DagsterOp:
        def __init__(self, name: str) -> None:
            self.name = name
            self.summary = f"execute {name}"

    dagster_plan = from_dagster_ops([_DagsterOp("asset_a"), _DagsterOp("asset_b")], task="assets")
    assert dagster_plan["task"] == "assets"
    assert len(dagster_plan["steps"]) == 2


def test_workflow_guarded_task_allows_or_blocks_execution(monkeypatch) -> None:
    from scopebench.integrations.workflow_connectors import GuardResult, ScopeBenchGuardRejected, workflow_guarded_task

    allowed = GuardResult(
        trace_id=None,
        span_id=None,
        decision="ALLOW",
        effective_decision="ALLOW",
        reasons=[],
        aggregate={},
        exceeded={},
        asked={},
        recommended_patch=[],
    )

    blocked = GuardResult(
        trace_id=None,
        span_id=None,
        decision="DENY",
        effective_decision="DENY",
        reasons=["exceeded threshold"],
        aggregate={},
        exceeded={},
        asked={},
        recommended_patch=[],
    )

    calls: list[str] = []

    def _allow_guard(**kwargs):
        calls.append("guard")
        return allowed

    monkeypatch.setattr("scopebench.integrations.workflow_connectors.guard", _allow_guard)

    @workflow_guarded_task(contract={"goal": "run task", "preset": "team"}, task_id="task_a")
    def _task_a() -> str:
        calls.append("task")
        return "ok"

    assert _task_a() == "ok"
    assert calls == ["guard", "task"]

    def _block_guard(**kwargs):
        return blocked

    monkeypatch.setattr("scopebench.integrations.workflow_connectors.guard", _block_guard)

    @workflow_guarded_task(contract={"goal": "run task", "preset": "team"}, task_id="task_b")
    def _task_b() -> str:
        return "never"

    try:
        _task_b()
        assert False, "expected ScopeBenchGuardRejected"
    except ScopeBenchGuardRejected as exc:
        assert "DENY" in str(exc)
