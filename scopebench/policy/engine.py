from __future__ import annotations

from typing import List, Optional

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.policy.backends.base import Decision, PolicyInput, PolicyResult
from scopebench.policy.backends.factory import get_policy_backend
from scopebench.scoring.axes import ScopeAggregate, ScopeVector
from scopebench.scoring.rules import detect_knees_for_plan


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

    if plan is not None:
        knee_flags = detect_knees_for_plan(
            plan,
            min_marginal_ratio=contract.thresholds.min_marginal_ratio,
            step_vectors=step_vectors,
        )
        if len(knee_flags) > contract.thresholds.max_knee_steps:
            flagged_steps = sorted({flag.step_id for flag in knee_flags})
            result.asked.setdefault("knee_of_curve", float(len(flagged_steps)))
            result.reasons.append(
                "Knee-of-curve detected at steps "
                + ", ".join(flagged_steps)
                + f"; min_marginal_ratio={contract.thresholds.min_marginal_ratio:.2f}"
            )
            if result.decision.value == "ALLOW":
                result.decision = Decision.ASK
    abstain_threshold = contract.escalation.abstain_uncertainty_threshold
    if agg.uncertainty >= abstain_threshold:
        result.decision = Decision.ASK
        result.asked["uncertainty"] = max(result.asked.get("uncertainty", 0.0), float(agg.uncertainty))
        if "abstain_due_to_uncertainty" not in result.reasons:
            result.reasons.append("abstain_due_to_uncertainty")

    result.policy_input = build_policy_input(contract, agg, step_vectors=step_vectors, plan=plan)
    return result
