from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.policy.backends.base import Decision, PolicyBackend, PolicyResult
from scopebench.scoring.axes import ScopeAggregate, ScopeVector, norm_inf

from scopebench.policy.rule_facts import (
    HARD_POWER_TOOL_CATEGORIES,
    missing_initial_read,
    missing_validation_after_write,
)


def _python_policy_hash() -> str:
    path = Path(__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class PythonPolicyBackend(PolicyBackend):
    name = "python"
    version = "python-v1"

    def evaluate(
        self,
        contract: TaskContract,
        agg: ScopeAggregate,
        step_vectors: Optional[List[ScopeVector]] = None,
        plan: Optional[PlanDAG] = None,
    ) -> PolicyResult:
        reasons: List[str] = []
        exceeded: Dict[str, Tuple[float, float]] = {}
        asked: Dict[str, float] = {}

        if step_vectors:
            for v in step_vectors:
                if (
                    contract.allowed_tools is not None
                    and v.tool
                    and v.tool not in contract.allowed_tools
                ):
                    reasons.append(f"Tool '{v.tool}' not in contract.allowed_tools")
                    exceeded["allowed_tools"] = (1.0, 0.0)
                if v.tool_category and v.tool_category in contract.forbidden_tool_categories:
                    reasons.append(f"Tool category '{v.tool_category}' is forbidden by contract")
                    exceeded["forbidden_tool_categories"] = (1.0, 0.0)

        contraction = max(0.0, 1.0 - agg.uncertainty)
        axis_to_thresh = {
            "spatial": contract.thresholds.max_spatial * contraction / contract.axis_weights.spatial,
            "temporal": contract.thresholds.max_temporal * contraction / contract.axis_weights.temporal,
            "depth": contract.thresholds.max_depth * contraction / contract.axis_weights.depth,
            "irreversibility": contract.thresholds.max_irreversibility * contraction / contract.axis_weights.irreversibility,
            "resource_intensity": contract.thresholds.max_resource_intensity * contraction / contract.axis_weights.resource_intensity,
            "legal_exposure": contract.thresholds.max_legal_exposure * contraction / contract.axis_weights.legal_exposure,
            "dependency_creation": contract.thresholds.max_dependency_creation * contraction / contract.axis_weights.dependency_creation,
            "stakeholder_radius": contract.thresholds.max_stakeholder_radius * contraction / contract.axis_weights.stakeholder_radius,
            "power_concentration": contract.thresholds.max_power_concentration * contraction / contract.axis_weights.power_concentration,
            "uncertainty": contract.thresholds.max_uncertainty / contract.axis_weights.uncertainty,
        }
        agg_dict = agg.as_dict()
        for axis, thresh in axis_to_thresh.items():
            val = agg_dict[axis]
            if val > thresh:
                exceeded[axis] = (val, thresh)
                reasons.append(f"{axis}={val:.2f} exceeds max {thresh:.2f}")

        ask_any = contract.escalation.ask_if_any_axis_over * contraction
        ask_unc = contract.escalation.ask_if_uncertainty_over
        if norm_inf(agg_dict) > ask_any:
            for axis, val in agg_dict.items():
                if val > ask_any:
                    asked[axis] = val
        if agg.uncertainty > ask_unc:
            asked["uncertainty"] = agg.uncertainty

        if step_vectors:
            high_risk_cats = contract.escalation.ask_if_tool_category_in | HARD_POWER_TOOL_CATEGORIES
            for v in step_vectors:
                if v.tool_category and v.tool_category in high_risk_cats:
                    asked.setdefault("tool_category", 0.0)
                    reasons.append(
                        f"Tool category '{v.tool_category}' triggers ASK by escalation rules"
                    )
            if plan is not None and missing_initial_read(plan, step_vectors):
                asked.setdefault("read_before_write", 1.0)
                reasons.append(
                    "SWE write step appears before any read-only step; ask for initial inspection"
                )
            if plan is not None and missing_validation_after_write(plan, step_vectors):
                asked.setdefault("validation_after_write", 1.0)
                reasons.append("SWE write step is missing a downstream validation/test step")

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
            policy_backend=self.name,
            policy_version=self.version,
            policy_hash=_python_policy_hash(),
        )
