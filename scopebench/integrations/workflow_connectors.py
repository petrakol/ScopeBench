from __future__ import annotations

from functools import wraps
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

from scopebench.integrations.sdk import GuardResult, guard

F = TypeVar("F", bound=Callable[..., Any])


class ScopeBenchGuardRejected(RuntimeError):
    """Raised when a guarded workflow task is blocked by ScopeBench policy."""



def _as_dependency_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        return [str(value)]
    deps: List[str] = []
    if isinstance(value, Iterable):
        for item in value:
            dep_id = getattr(item, "task_id", None) or getattr(item, "name", None) or item
            deps.append(str(dep_id))
    return deps



def _step_from_task_like(task: Any, fallback_id: str) -> Dict[str, Any]:
    task_id = str(
        getattr(task, "task_id", None)
        or getattr(task, "name", None)
        or getattr(task, "id", None)
        or fallback_id
    )
    description = str(
        getattr(task, "description", None)
        or getattr(task, "doc", None)
        or getattr(task, "doc_md", None)
        or getattr(task, "summary", None)
        or task_id
    )
    tool = str(
        getattr(task, "task_type", None)
        or getattr(task, "type", None)
        or getattr(task, "fn", None).__name__
        if getattr(task, "fn", None) is not None
        else getattr(task, "__class__", type(task)).__name__
    )
    depends_on = _as_dependency_ids(
        getattr(task, "upstream_task_ids", None)
        or getattr(task, "upstream_tasks", None)
        or getattr(task, "upstream", None)
    )
    return {
        "id": task_id,
        "description": description,
        "tool": tool,
        "depends_on": depends_on,
    }



def from_airflow_dag(dag: Any, task: Optional[str] = None) -> Dict[str, Any]:
    """Convert an Airflow DAG-like object into a ScopeBench PlanDAG dict."""
    dag_id = str(getattr(dag, "dag_id", None) or task or "airflow workflow")
    tasks = list(getattr(dag, "tasks", []) or [])
    steps = [_step_from_task_like(op, fallback_id=str(idx)) for idx, op in enumerate(tasks, start=1)]
    return {"task": dag_id, "steps": steps}



def from_prefect_tasks(tasks: Iterable[Any], task: str = "prefect flow") -> Dict[str, Any]:
    """Convert Prefect task-like objects into a ScopeBench PlanDAG dict."""
    steps = [_step_from_task_like(obj, fallback_id=str(idx)) for idx, obj in enumerate(tasks, start=1)]
    return {"task": task, "steps": steps}



def from_dagster_ops(ops: Iterable[Any], task: str = "dagster job") -> Dict[str, Any]:
    """Convert Dagster op/asset-like objects into a ScopeBench PlanDAG dict."""
    steps = [_step_from_task_like(obj, fallback_id=str(idx)) for idx, obj in enumerate(ops, start=1)]
    return {"task": task, "steps": steps}



def workflow_guarded_task(
    *,
    contract: Mapping[str, Any],
    task_id: Optional[str] = None,
    description: Optional[str] = None,
    tool: str = "workflow_task",
    depends_on: Optional[Sequence[str]] = None,
    shadow_mode: bool = False,
    policy_backend: str = "python",
) -> Callable[[F], F]:
    """Decorator that runs ScopeBench guard before executing a workflow task function."""

    def decorator(func: F) -> F:
        resolved_task_id = task_id or func.__name__
        resolved_description = description or (func.__doc__.strip() if func.__doc__ else func.__name__)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            plan = {
                "task": kwargs.get("scopebench_task", resolved_task_id),
                "steps": [
                    {
                        "id": str(resolved_task_id),
                        "description": resolved_description,
                        "tool": tool,
                        "depends_on": list(depends_on or []),
                    }
                ],
            }
            result: GuardResult = guard(
                plan=plan,
                contract=dict(contract),
                shadow_mode=shadow_mode,
                policy_backend=policy_backend,
            )
            if result.effective_decision != "ALLOW":
                raise ScopeBenchGuardRejected(
                    "ScopeBench rejected workflow task "
                    f"{resolved_task_id!r} with decision {result.decision}: {', '.join(result.reasons)}"
                )
            return func(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator



def airflow_task(**kwargs: Any) -> Callable[[F], F]:
    """Airflow-friendly alias for `workflow_guarded_task`."""
    return workflow_guarded_task(tool="airflow_task", **kwargs)



def prefect_task(**kwargs: Any) -> Callable[[F], F]:
    """Prefect-friendly alias for `workflow_guarded_task`."""
    return workflow_guarded_task(tool="prefect_task", **kwargs)



def dagster_op(**kwargs: Any) -> Callable[[F], F]:
    """Dagster-friendly alias for `workflow_guarded_task`."""
    return workflow_guarded_task(tool="dagster_op", **kwargs)
