from __future__ import annotations

from typing import List, Optional

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.policy.backends.base import PolicyInput, PolicyResult
from scopebench.policy.backends.factory import get_policy_backend
from scopebench.scoring.axes import ScopeAggregate, ScopeVector


def build_policy_input(
    contract: TaskContract,
    agg: ScopeAggregate,
    step_vectors: Optional[List[ScopeVector]] = None,
    plan: Optional[PlanDAG] = None,
) -> PolicyInput:
    return PolicyInput(
        policy_input_version="v1",
        contract=contract.model_dump(mode="json"),
        aggregate=agg.as_dict(),
        plan=plan.model_dump(mode="json") if plan else {},
        vectors=[v.model_dump(mode="json") for v in (step_vectors or [])],
        metadata={"n_steps": agg.n_steps},
    )


def evaluate_policy(
    contract: TaskContract,
    agg: ScopeAggregate,
    step_vectors: Optional[List[ScopeVector]] = None,
    plan: Optional[PlanDAG] = None,
    policy_backend: Optional[str] = None,
) -> PolicyResult:
    backend = get_policy_backend(policy_backend)
    result = backend.evaluate(contract, agg, step_vectors=step_vectors, plan=plan)
    result.policy_input = build_policy_input(contract, agg, step_vectors=step_vectors, plan=plan)
    return result
