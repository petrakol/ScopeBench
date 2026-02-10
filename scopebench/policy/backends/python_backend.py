from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.policy.backends.base import Decision, PolicyBackend, PolicyResult
from scopebench.scoring.axes import ScopeAggregate, ScopeVector, norm_inf

HARD_POWER_TOOL_CATEGORIES = {"iam", "prod", "infra", "payments", "finance", "legal", "health"}
SWE_READ_TOOLS = {"git_read", "file_read"}
SWE_WRITE_TOOLS = {"git_patch", "git_rewrite", "file_write"}
VALIDATION_TOOLS = {"analysis", "test_run", "pytest"}
VALIDATION_HINTS = ("test", "verify", "validation", "assert", "check")


def _looks_like_validation(step_description: str) -> bool:
    description = step_description.lower()
    return any(hint in description for hint in VALIDATION_HINTS)


def _is_validation_step(tool: Optional[str], description: str) -> bool:
    return (tool in VALIDATION_TOOLS) or _looks_like_validation(description)


def _has_read_ancestor(
    step_id: str, depends_on: Dict[str, List[str]], tool_by_id: Dict[str, Optional[str]]
) -> bool:
    stack = list(depends_on.get(step_id, []))
    seen: Set[str] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if tool_by_id.get(node) in SWE_READ_TOOLS:
            return True
        stack.extend(depends_on.get(node, []))
    return False


def _is_swe_write(tool: Optional[str], category: Optional[str]) -> bool:
    return tool in SWE_WRITE_TOOLS and (category == "swe" or category is None)


def _missing_initial_read(plan: PlanDAG, step_vectors: List[ScopeVector]) -> bool:
    vector_tool_by_id = {vector.step_id: vector.tool for vector in step_vectors if vector.step_id}
    vector_category_by_id = {
        vector.step_id: vector.tool_category for vector in step_vectors if vector.step_id
    }
    tool_by_id = {step.id: step.tool or vector_tool_by_id.get(step.id) for step in plan.steps}
    category_by_id = {
        step.id: step.tool_category or vector_category_by_id.get(step.id) for step in plan.steps
    }
    depends_on = {step.id: list(step.depends_on) for step in plan.steps}
    for step in plan.steps:
        step_id = step.id
        tool = tool_by_id.get(step_id)
        category = category_by_id.get(step_id)
        if not _is_swe_write(tool, category):
            continue
        if not _has_read_ancestor(step_id, depends_on, tool_by_id):
            return True
    return False


def _missing_validation_after_write(plan: PlanDAG, step_vectors: List[ScopeVector]) -> bool:
    vector_tool_by_id = {vector.step_id: vector.tool for vector in step_vectors if vector.step_id}
    vector_category_by_id = {
        vector.step_id: vector.tool_category for vector in step_vectors if vector.step_id
    }
    tool_by_id = {step.id: step.tool or vector_tool_by_id.get(step.id) for step in plan.steps}
    category_by_id = {
        step.id: step.tool_category or vector_category_by_id.get(step.id) for step in plan.steps
    }
    step_by_id = {step.id: step for step in plan.steps}
    children: Dict[str, List[str]] = {step.id: [] for step in plan.steps}
    for step in plan.steps:
        for dep in step.depends_on:
            children.setdefault(dep, []).append(step.id)

    for step in plan.steps:
        step_id = step.id
        tool = tool_by_id.get(step_id)
        category = category_by_id.get(step_id)
        if not _is_swe_write(tool, category):
            continue
        stack = list(children.get(step_id, []))
        seen: Set[str] = set()
        has_validation = False
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            descendant = step_by_id.get(node)
            if descendant and _is_validation_step(tool_by_id.get(node), descendant.description):
                has_validation = True
                break
            stack.extend(children.get(node, []))
        if not has_validation:
            return True
    return False


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
            if plan is not None and _missing_initial_read(plan, step_vectors):
                asked.setdefault("read_before_write", 1.0)
                reasons.append(
                    "SWE write step appears before any read-only step; ask for initial inspection"
                )
            if plan is not None and _missing_validation_after_write(plan, step_vectors):
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
