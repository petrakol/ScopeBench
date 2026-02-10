from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.policy.backends.base import Decision, PolicyResult
from scopebench.policy.rule_facts import (
    HARD_POWER_TOOL_CATEGORIES,
    missing_initial_read,
    missing_validation_after_write,
)
from scopebench.scoring.axes import ScopeAggregate, ScopeVector


def build_external_input(
    contract: TaskContract,
    agg: ScopeAggregate,
    step_vectors: Optional[List[ScopeVector]] = None,
    plan: Optional[PlanDAG] = None,
) -> Dict[str, Any]:
    vectors = [v.model_dump(mode="json") for v in (step_vectors or [])]
    high_risk_categories = sorted(
        contract.escalation.ask_if_tool_category_in | HARD_POWER_TOOL_CATEGORIES
    )
    present_categories = sorted(
        {
            vector["tool_category"]
            for vector in vectors
            if vector.get("tool_category") and vector["tool_category"] in high_risk_categories
        }
    )
    contract_obj = contract.model_dump(mode="json")
    contraction = max(0.0, 1.0 - float(agg.uncertainty))
    allowed_tools_violation = bool(
        contract.allowed_tools is not None
        and any(v.get("tool") and v.get("tool") not in contract.allowed_tools for v in vectors)
    )
    forbidden_tool_category_violation = any(
        v.get("tool_category") in contract.forbidden_tool_categories
        for v in vectors
        if v.get("tool_category")
    )
    return {
        "contract": contract_obj,
        "aggregate": agg.as_dict(),
        "vectors": vectors,
        "plan": plan.model_dump(mode="json") if plan else {"steps": []},
        "facts": {
            "high_risk_categories": high_risk_categories,
            "present_high_risk_categories": present_categories,
            "missing_initial_read": bool(
                plan is not None and step_vectors and missing_initial_read(plan, step_vectors)
            ),
            "missing_validation_after_write": bool(
                plan is not None and step_vectors and missing_validation_after_write(plan, step_vectors)
            ),
            "allowed_tools_active": contract_obj.get("allowed_tools") is not None,
            "allowed_tools_violation": allowed_tools_violation,
            "forbidden_tool_category_violation": forbidden_tool_category_violation,
            "contraction": contraction,
        },
    }


def normalize_policy_result(payload: Dict[str, Any], backend: str, version: str, policy_hash: str) -> PolicyResult:
    raw = payload.get("result", payload)
    if not isinstance(raw, dict):
        raise ValueError("Policy engine response missing result object")

    decision_raw = raw.get("decision")
    if decision_raw not in {Decision.ALLOW.value, Decision.ASK.value, Decision.DENY.value}:
        raise ValueError(f"Invalid decision from policy engine: {decision_raw!r}")

    reasons_raw = raw.get("reasons", [])
    reasons = [str(reason) for reason in reasons_raw] if isinstance(reasons_raw, list) else []

    exceeded: Dict[str, Tuple[float, float]] = {}
    for key, value in (raw.get("exceeded") or {}).items():
        if isinstance(value, dict):
            observed = float(value.get("value", 0.0))
            threshold = float(value.get("threshold", 0.0))
        else:
            observed, threshold = value
            observed, threshold = float(observed), float(threshold)
        exceeded[str(key)] = (observed, threshold)

    asked = {str(key): float(value) for key, value in (raw.get("asked") or {}).items()}

    return PolicyResult(
        decision=Decision(decision_raw),
        reasons=reasons,
        exceeded=exceeded,
        asked=asked,
        policy_backend=backend,
        policy_version=version,
        policy_hash=policy_hash,
    )


def parse_json_response(raw_bytes: bytes) -> Dict[str, Any]:
    return json.loads(raw_bytes.decode("utf-8"))

