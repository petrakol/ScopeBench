from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from scopebench.contracts import TaskContract
from scopebench.scoring.axes import ScopeAggregate, ScopeVector, norm_inf


class Decision(str, Enum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@dataclass
class PolicyResult:
    decision: Decision
    reasons: List[str]
    exceeded: Dict[str, Tuple[float, float]]  # axis -> (value, threshold)
    asked: Dict[str, float]  # axis -> value
    step_vectors: Optional[List[ScopeVector]] = None
    aggregate: Optional[ScopeAggregate] = None


HARD_POWER_TOOL_CATEGORIES = {"iam", "prod", "infra", "payments", "finance", "legal", "health"}


def evaluate_policy(contract: TaskContract, agg: ScopeAggregate, step_vectors: Optional[List[ScopeVector]] = None) -> PolicyResult:
    """Evaluate ALLOW/ASK/DENY from contract and aggregated scope.

    Rules (MVP):
    - DENY if any axis > max_* threshold
    - ASK if any axis > ask_if_any_axis_over or uncertainty > ask_if_uncertainty_over
    - ASK if any step uses tool category in escalation.ask_if_tool_category_in
    - DENY if tool is not allowed (if allowed_tools provided) or tool category forbidden
    """
    reasons: List[str] = []
    exceeded: Dict[str, Tuple[float, float]] = {}
    asked: Dict[str, float] = {}

    # Tool allow/deny (per-step)
    if step_vectors:
        for v in step_vectors:
            if contract.allowed_tools is not None and v.tool and v.tool not in contract.allowed_tools:
                reasons.append(f"Tool '{v.tool}' not in contract.allowed_tools")
                exceeded["allowed_tools"] = (1.0, 0.0)
            if v.tool_category and v.tool_category in contract.forbidden_tool_categories:
                reasons.append(f"Tool category '{v.tool_category}' is forbidden by contract")
                exceeded["forbidden_tool_categories"] = (1.0, 0.0)

    # Effective thresholds (uncertainty contraction)
    contraction = max(0.0, 1.0 - agg.uncertainty)

    # Axis DENY thresholds
    axis_to_thresh = {
        "spatial": contract.thresholds.max_spatial * contraction,
        "temporal": contract.thresholds.max_temporal * contraction,
        "depth": contract.thresholds.max_depth * contraction,
        "irreversibility": contract.thresholds.max_irreversibility * contraction,
        "resource_intensity": contract.thresholds.max_resource_intensity * contraction,
        "legal_exposure": contract.thresholds.max_legal_exposure * contraction,
        "dependency_creation": contract.thresholds.max_dependency_creation * contraction,
        "stakeholder_radius": contract.thresholds.max_stakeholder_radius * contraction,
        "power_concentration": contract.thresholds.max_power_concentration * contraction,
        "uncertainty": contract.thresholds.max_uncertainty,
    }
    agg_dict = agg.as_dict()
    for axis, thresh in axis_to_thresh.items():
        val = agg_dict[axis]
        if val > thresh:
            exceeded[axis] = (val, thresh)
            reasons.append(f"{axis}={val:.2f} exceeds max {thresh:.2f}")

    # ASK thresholds
    ask_any = contract.escalation.ask_if_any_axis_over * contraction
    ask_unc = contract.escalation.ask_if_uncertainty_over
    if norm_inf(agg_dict) > ask_any:
        for axis, val in agg_dict.items():
            if val > ask_any:
                asked[axis] = val
    if agg.uncertainty > ask_unc:
        asked["uncertainty"] = agg.uncertainty

    # ASK if high-risk categories appear
    if step_vectors:
        high_risk_cats = contract.escalation.ask_if_tool_category_in | HARD_POWER_TOOL_CATEGORIES
        for v in step_vectors:
            if v.tool_category and v.tool_category in high_risk_cats:
                asked.setdefault("tool_category", 0.0)
                reasons.append(f"Tool category '{v.tool_category}' triggers ASK by escalation rules")

    # Determine decision
    if exceeded:
        decision = Decision.DENY
    elif asked:
        decision = Decision.ASK
        if not reasons:
            reasons.append("Within max thresholds but triggers escalation/uncertainty thresholds")
    else:
        decision = Decision.ALLOW
        if not reasons:
            reasons.append("Within contract envelope")

    return PolicyResult(
        decision=decision,
        reasons=reasons,
        exceeded=exceeded,
        asked=asked,
        step_vectors=step_vectors,
        aggregate=agg,
    )
