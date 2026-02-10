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
    trace_id: Optional[str]
    span_id: Optional[str]
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
        trace_id=None,
        span_id=None,
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


def from_langchain_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a LangChain-like plan object to ScopeBench PlanDAG dict."""
    steps = []
    for idx, item in enumerate(plan.get("steps", []), start=1):
        steps.append(
            {
                "id": str(item.get("id", idx)),
                "description": item.get("description") or item.get("action") or "unspecified step",
                "tool": item.get("tool") or item.get("name"),
                "depends_on": item.get("depends_on", []),
            }
        )
    return {"task": plan.get("task") or plan.get("goal") or "agent task", "steps": steps}


def from_autogen_messages(messages: List[Dict[str, Any]], task: str = "agent task") -> Dict[str, Any]:
    """Convert AutoGen-style messages to a simple ScopeBench PlanDAG dict."""
    steps: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages, start=1):
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        steps.append(
            {
                "id": str(idx),
                "description": content,
                "tool": msg.get("tool") or msg.get("name"),
                "depends_on": [str(idx - 1)] if idx > 1 else [],
            }
        )
    return {"task": task, "steps": steps}
