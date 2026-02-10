from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate
from scopebench.scoring.calibration import CalibratedDecisionThresholds
from scopebench.server.api import _suggest_plan_patch


@dataclass
class GuardResult:
    decision: str
    effective_decision: str
    reasons: List[str]
    aggregate: Dict[str, float]
    exceeded: Dict[str, Dict[str, float]]
    asked: Dict[str, float]
    recommended_patch: List[Dict[str, Any]]


def guard(
    plan: Dict[str, Any],
    contract: Dict[str, Any],
    *,
    calibration_scale: Optional[float] = None,
    shadow_mode: bool = False,
    policy_backend: str = "python",
) -> GuardResult:
    """Evaluate a plan and return the integration-friendly decision/patch payload."""
    contract_model = TaskContract.model_validate(contract)
    plan_model = PlanDAG.model_validate(plan)

    calibration = None
    if calibration_scale is not None:
        calibration = CalibratedDecisionThresholds(global_scale=calibration_scale)

    result = evaluate(
        contract_model,
        plan_model,
        calibration=calibration,
        policy_backend=policy_backend,
    )
    decision = result.policy.decision.value
    effective_decision = "ALLOW" if shadow_mode and decision in {"ASK", "DENY"} else decision

    reasons = list(result.policy.reasons)
    if shadow_mode and effective_decision != decision:
        reasons.append("Shadow mode bypassed blocking decision.")

    return GuardResult(
        decision=decision,
        effective_decision=effective_decision,
        reasons=reasons,
        aggregate=result.aggregate.as_dict(),
        exceeded={
            axis: {"value": float(values[0]), "threshold": float(values[1])}
            for axis, values in result.policy.exceeded.items()
        },
        asked={axis: float(threshold) for axis, threshold in result.policy.asked.items()},
        recommended_patch=_suggest_plan_patch(result.policy, plan_model),
    )
