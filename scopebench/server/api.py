from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG, RealtimeEstimate
from scopebench.runtime.guard import evaluate
from scopebench.scoring.axes import SCOPE_AXES, combine_aggregates
from scopebench.scoring.calibration import (
    DEFAULT_DOMAIN,
    CalibratedDecisionThresholds,
    apply_manual_adjustments,
    calibration_to_dict,
    compute_domain_calibration_from_telemetry,
)
from scopebench.scoring.rules import build_budget_ledger
from scopebench.session import MultiAgentSession
from scopebench.tracing.otel import current_trace_context, get_tracer, init_tracing

SWE_READ_TOOLS = {"git_read", "file_read"}
SWE_WRITE_TOOLS = {"git_patch", "git_rewrite", "file_write"}
VALIDATION_TOOLS = {"analysis", "test_run", "pytest"}
VALIDATION_HINTS = ("test", "verify", "validation", "assert", "check")


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "contract": {
                    "goal": "Fix flaky auth test in CI",
                    "non_goals": ["Refactor unrelated auth modules"],
                    "constraints": ["Do not change production auth behavior"],
                    "acceptance": ["Auth test passes consistently"],
                    "preset": "balanced",
                },
                "plan": {
                    "task": "Diagnose and fix flaky token refresh test",
                    "steps": [
                        {
                            "id": "read-failing-test",
                            "description": "Inspect the failing test and auth refresh implementation.",
                            "tool": "git_read",
                        },
                        {
                            "id": "patch-refresh-window",
                            "description": "Patch the refresh timing logic to avoid race conditions.",
                            "tool": "git_patch",
                        },
                        {
                            "id": "validate-auth",
                            "description": "Run targeted auth tests.",
                            "tool": "pytest",
                        },
                    ],
                },
                "include_steps": True,
                "include_summary": True,
                "include_telemetry": True,
                "shadow_mode": False,
                "policy_backend": "python",
            }
        },
    )
    contract: Dict[str, Any] = Field(..., description="TaskContract as dict")
    plan: Dict[str, Any] = Field(..., description="PlanDAG as dict")
    include_steps: bool = Field(False, description="Include step-level vectors and rationales.")
    include_summary: bool = Field(False, description="Include summary and next-step guidance.")
    include_telemetry: bool = Field(
        True, description="Include lightweight evaluation telemetry fields."
    )
    shadow_mode: bool = Field(
        False, description="If true, never block execution; return what enforcement would decide."
    )
    ask_action: Optional[str] = Field(
        None, description="Optional feedback: accepted/replanned/ignored."
    )
    outcome: Optional[str] = Field(
        None,
        description="Optional outcome feedback: tests_pass/tests_fail/rollback/manual_override.",
    )
    calibration_scale: Optional[float] = Field(
        None, ge=0.0, description="Optional scale for aggregate scores."
    )
    calibration_domain: Optional[str] = Field(
        None,
        description="Optional telemetry domain/task_type used for adaptive calibration.",
    )
    calibration_manual_adjustments: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional manual threshold adjustments (axis deltas + abstain delta).",
    )
    policy_backend: Optional[str] = Field(
        None, description="Policy backend override: python|opa|cedar."
    )


class AxisDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    rationale: str
    confidence: float


class StepDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: Optional[str]
    tool: Optional[str]
    tool_category: Optional[str]
    est_cost_usd: Optional[float] = None
    est_time_days: Optional[float] = None
    est_labor_hours: Optional[float] = None
    resolved_cost_usd: Optional[float] = None
    resolved_time_days: Optional[float] = None
    resolved_labor_hours: Optional[float] = None
    realtime_estimates: List[RealtimeEstimate] = Field(default_factory=list)
    est_benefit: Optional[float] = None
    benefit_unit: Optional[str] = None
    axes: Dict[str, AxisDetail]


class TelemetryDetail(BaseModel):
    schema_version: str = "telemetry_v1"
    model_config = ConfigDict(extra="forbid")
    preset: str
    policy_input_version: str
    task_type: str
    plan_size: int
    decision: str
    triggered_rules: List[str]
    has_read_before_write: bool
    has_validation_after_write: bool
    ask_action: Optional[str] = None
    outcome: Optional[str] = None


class EvaluateResponse(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "decision": "ALLOW",
                "policy_backend": "python",
                "policy_version": "v1",
                "policy_hash": "sha256:demo",
                "effective_decision": "ALLOW",
                "shadow_mode": False,
                "reasons": ["Aggregate risk stayed below threshold."],
                "exceeded": {},
                "asked": {},
                "aggregate": {
                    "spatial": 0.08,
                    "temporal": 0.12,
                    "depth": 0.2,
                    "irreversibility": 0.06,
                    "resource_intensity": 0.1,
                    "legal_exposure": 0.02,
                    "dependency_creation": 0.04,
                    "stakeholder_radius": 0.05,
                    "power_concentration": 0.03,
                    "uncertainty": 0.22,
                },
                "n_steps": 3,
                "steps": [
                    {
                        "step_id": "patch-refresh-window",
                        "tool": "git_patch",
                        "tool_category": "write",
                        "axes": {
                            "uncertainty": {
                                "value": 0.35,
                                "rationale": "Timing behavior is partially inferred from flaky logs.",
                                "confidence": 0.72,
                            }
                        },
                    }
                ],
                "summary": "Decision ALLOW (effective: ALLOW). Top axes: uncertainty=0.22, depth=0.20, temporal=0.12.",
                "next_steps": ["Proceed; plan appears proportionate to the contract."],
                "plan_patch_suggestion": [],
                "telemetry": {
                    "preset": "balanced",
                    "policy_input_version": "v1",
                    "task_type": "bug_fix",
                    "plan_size": 3,
                    "decision": "ALLOW",
                    "triggered_rules": [],
                    "has_read_before_write": True,
                    "has_validation_after_write": True,
                    "ask_action": None,
                    "outcome": None,
                },
                "policy_input": {
                    "task_type": "bug_fix",
                    "read_before_write": True,
                    "validation_after_write": True,
                },
            }
        },
    )
    decision: str
    policy_backend: str
    policy_version: str
    policy_hash: str
    effective_decision: str
    shadow_mode: bool
    reasons: list[str]
    exceeded: Dict[str, Dict[str, float]]
    asked: Dict[str, float]
    aggregate: Dict[str, float]
    n_steps: int
    steps: Optional[List[StepDetail]] = None
    summary: Optional[str] = None
    next_steps: Optional[List[str]] = None
    plan_patch_suggestion: Optional[List[Dict[str, Any]]] = None
    telemetry: Optional[TelemetryDetail] = None
    policy_input: Optional[Dict[str, Any]] = None


class EvaluateSessionRequest(BaseModel):
    session: Dict[str, Any] = Field(..., description="MultiAgentSession as dict")
    include_steps: bool = Field(False, description="Include step-level vectors and rationales.")
    include_telemetry: bool = Field(
        True, description="Include lightweight evaluation telemetry fields."
    )
    policy_backend: Optional[str] = Field(
        None, description="Policy backend override: python|opa|cedar."
    )


class StreamingPlanEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(..., min_length=1)
    operation: str = Field(..., pattern=r"^(add_step|update_step|remove_step|replace_plan)$")
    step_id: Optional[str] = None
    step: Optional[Dict[str, Any]] = None
    index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional insertion index used with add_step; appends when omitted.",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context payload stored in the snapshot for consumers.",
    )


class EvaluateStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract: Dict[str, Any] = Field(..., description="TaskContract as dict")
    plan: Dict[str, Any] = Field(..., description="Initial PlanDAG as dict")
    events: List[StreamingPlanEvent] = Field(
        default_factory=list,
        description="Ordered stream of plan evolution events.",
    )
    include_steps: bool = Field(False, description="Include step-level vectors for each snapshot.")
    policy_backend: Optional[str] = Field(
        None, description="Policy backend override: python|opa|cedar."
    )
    judge: str = Field(
        "heuristic",
        pattern=r"^(heuristic|llm)$",
        description="Step judge mode used for each streaming re-evaluation.",
    )


class TriggeredReevaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    axis: Optional[str] = None
    previous: Optional[float] = None
    current: Optional[float] = None
    threshold: Optional[float] = None
    details: Optional[str] = None


class StreamStepDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    changed_axes: List[str]


class StreamingEvaluationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    event_index: int
    operation: str
    decision: str
    aggregate: Dict[str, float]
    exceeded: Dict[str, Dict[str, float]]
    asked: Dict[str, float]
    triggers: List[TriggeredReevaluation]
    judge_output_deltas: List[StreamStepDelta]
    context: Dict[str, Any] = Field(default_factory=dict)
    steps: Optional[List[StepDetail]] = None


class EvaluateStreamResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial: StreamingEvaluationSnapshot
    updates: List[StreamingEvaluationSnapshot]


class SessionAggregateDetail(BaseModel):
    aggregate: Dict[str, float]
    ledger: Dict[str, Dict[str, float]]
    decision: str


class ScopeLaunderingSignal(BaseModel):
    axis: str
    global_value: float
    max_agent_value: float
    delta: float
    ask_threshold: float


class SessionDashboardEntry(BaseModel):
    aggregate: Dict[str, float]
    budget_consumption: Dict[str, float]
    budget_utilization: Dict[str, float]
    budget_projection: Dict[str, float]
    budget_projection_utilization: Dict[str, float]
    decision: str


class SessionDashboard(BaseModel):
    per_agent: Dict[str, SessionDashboardEntry]
    global_: SessionDashboardEntry = Field(..., alias="global")

    model_config = {"populate_by_name": True}


class CalibrationAdjustmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str = Field(..., description="Domain/task_type to adjust.")
    axis_threshold_factor_delta: Dict[str, float] = Field(default_factory=dict)
    axis_scale_delta: Dict[str, float] = Field(default_factory=dict)
    abstain_uncertainty_threshold_delta: Optional[float] = None


class CalibrationDashboardEntry(BaseModel):
    domain: str
    runs: int
    calibration: Dict[str, Any]
    stats: Dict[str, Dict[str, int]]


class CalibrationDashboardResponse(BaseModel):
    enabled: bool
    source: Optional[str] = None
    count: int
    domains: List[CalibrationDashboardEntry]


class EvaluateSessionResponse(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    decision: str
    per_agent: Dict[str, SessionAggregateDetail]
    global_: SessionAggregateDetail = Field(..., alias="global")
    laundering_signals: List[ScopeLaunderingSignal]
    dashboard: SessionDashboard

    model_config = {"populate_by_name": True}




def _step_detail_payload(vectors, plan: PlanDAG) -> List[StepDetail]:
    plan_steps_by_id = {step.id: step for step in plan.steps}
    details: List[StepDetail] = []
    for vec in vectors:
        axes = {
            "spatial": AxisDetail(**vec.spatial.model_dump()),
            "temporal": AxisDetail(**vec.temporal.model_dump()),
            "depth": AxisDetail(**vec.depth.model_dump()),
            "irreversibility": AxisDetail(**vec.irreversibility.model_dump()),
            "resource_intensity": AxisDetail(**vec.resource_intensity.model_dump()),
            "legal_exposure": AxisDetail(**vec.legal_exposure.model_dump()),
            "dependency_creation": AxisDetail(**vec.dependency_creation.model_dump()),
            "stakeholder_radius": AxisDetail(**vec.stakeholder_radius.model_dump()),
            "power_concentration": AxisDetail(**vec.power_concentration.model_dump()),
            "uncertainty": AxisDetail(**vec.uncertainty.model_dump()),
        }
        plan_step = plan_steps_by_id.get(vec.step_id or "")
        details.append(
            StepDetail(
                step_id=vec.step_id,
                tool=vec.tool,
                tool_category=vec.tool_category,
                est_cost_usd=plan_step.est_cost_usd if plan_step else None,
                est_time_days=plan_step.est_time_days if plan_step else None,
                est_labor_hours=plan_step.est_labor_hours if plan_step else None,
                resolved_cost_usd=plan_step.resolved_cost_usd() if plan_step else None,
                resolved_time_days=plan_step.resolved_time_days() if plan_step else None,
                resolved_labor_hours=plan_step.resolved_labor_hours() if plan_step else None,
                realtime_estimates=list(plan_step.realtime_estimates) if plan_step else [],
                est_benefit=plan_step.est_benefit if plan_step else None,
                benefit_unit=plan_step.benefit_unit if plan_step else None,
                axes=axes,
            )
        )
    return details


def _vector_axes_by_step(vectors: List[Any]) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for vector in vectors:
        if not vector.step_id:
            continue
        payload[vector.step_id] = vector.as_dict
    return payload


def _judge_output_deltas(
    previous_vectors: Optional[List[Any]],
    current_vectors: List[Any],
) -> List[StreamStepDelta]:
    if previous_vectors is None:
        return []
    previous = _vector_axes_by_step(previous_vectors)
    current = _vector_axes_by_step(current_vectors)
    deltas: List[StreamStepDelta] = []
    for step_id, current_axes in current.items():
        previous_axes = previous.get(step_id)
        if previous_axes is None:
            continue
        changed_axes = [
            axis
            for axis, value in current_axes.items()
            if abs(float(value) - float(previous_axes.get(axis, 0.0))) > 1e-6
        ]
        if changed_axes:
            deltas.append(StreamStepDelta(step_id=step_id, changed_axes=sorted(changed_axes)))
    return sorted(deltas, key=lambda item: item.step_id)


def _threshold_value(contract: TaskContract, axis: str) -> Optional[float]:
    field = f"max_{axis}"
    if hasattr(contract.thresholds, field):
        return float(getattr(contract.thresholds, field))
    return None


def _triggered_reevaluations(
    contract: TaskContract,
    previous_aggregate: Optional[Dict[str, float]],
    current_aggregate: Dict[str, float],
    judge_deltas: List[StreamStepDelta],
) -> List[TriggeredReevaluation]:
    triggers: List[TriggeredReevaluation] = []
    if previous_aggregate is not None:
        for axis in SCOPE_AXES:
            threshold = _threshold_value(contract, axis)
            if threshold is None:
                continue
            previous = float(previous_aggregate.get(axis, 0.0))
            current = float(current_aggregate.get(axis, 0.0))
            if previous < threshold <= current:
                triggers.append(
                    TriggeredReevaluation(
                        kind="threshold_crossed",
                        axis=axis,
                        previous=previous,
                        current=current,
                        threshold=threshold,
                    )
                )
    if judge_deltas:
        changed_steps = ", ".join(delta.step_id for delta in judge_deltas)
        triggers.append(
            TriggeredReevaluation(
                kind="judge_output_changed",
                details=f"LLM/heuristic judge outputs changed for step(s): {changed_steps}",
            )
        )
    return triggers


def _apply_streaming_event(plan_data: Dict[str, Any], event: StreamingPlanEvent) -> Dict[str, Any]:
    updated = dict(plan_data)
    steps = list(updated.get("steps", []))
    operation = event.operation

    if operation == "replace_plan":
        if not isinstance(event.step, dict):
            raise HTTPException(status_code=400, detail="replace_plan requires 'step' with full plan payload")
        replacement = dict(event.step)
        if "task" not in replacement or "steps" not in replacement:
            raise HTTPException(status_code=400, detail="replace_plan payload must include task and steps")
        return replacement

    if not event.step_id:
        raise HTTPException(status_code=400, detail=f"{operation} requires step_id")

    index_by_id = {str(step.get("id")): idx for idx, step in enumerate(steps)}

    if operation == "add_step":
        if not isinstance(event.step, dict):
            raise HTTPException(status_code=400, detail="add_step requires step payload")
        if str(event.step.get("id")) != event.step_id:
            raise HTTPException(status_code=400, detail="add_step step.id must match step_id")
        if event.step_id in index_by_id:
            raise HTTPException(status_code=400, detail=f"step '{event.step_id}' already exists")
        insertion = len(steps) if event.index is None else min(event.index, len(steps))
        steps.insert(insertion, event.step)
    elif operation == "update_step":
        if event.step_id not in index_by_id:
            raise HTTPException(status_code=400, detail=f"step '{event.step_id}' not found")
        if not isinstance(event.step, dict):
            raise HTTPException(status_code=400, detail="update_step requires step payload")
        existing = dict(steps[index_by_id[event.step_id]])
        existing.update(event.step)
        existing["id"] = event.step_id
        steps[index_by_id[event.step_id]] = existing
    elif operation == "remove_step":
        if event.step_id not in index_by_id:
            raise HTTPException(status_code=400, detail=f"step '{event.step_id}' not found")
        removed_idx = index_by_id[event.step_id]
        steps.pop(removed_idx)
        for idx, step in enumerate(steps):
            deps = [dep for dep in step.get("depends_on", []) if dep != event.step_id]
            if deps != step.get("depends_on", []):
                refreshed = dict(step)
                refreshed["depends_on"] = deps
                steps[idx] = refreshed
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operation '{operation}'")

    updated["steps"] = steps
    return updated


def _snapshot_from_result(
    *,
    contract: TaskContract,
    event_id: str,
    event_index: int,
    operation: str,
    context: Dict[str, Any],
    result,
    judge_deltas: List[StreamStepDelta],
    previous_aggregate: Optional[Dict[str, float]],
    include_steps: bool,
) -> StreamingEvaluationSnapshot:
    exceeded_payload = {
        key: {"value": float(values[0]), "threshold": float(values[1])}
        for key, values in result.policy.exceeded.items()
    }
    asked_payload = {key: float(value) for key, value in result.policy.asked.items()}
    aggregate_payload = result.aggregate.as_dict()
    triggers = _triggered_reevaluations(
        contract,
        previous_aggregate=previous_aggregate,
        current_aggregate=aggregate_payload,
        judge_deltas=judge_deltas,
    )
    steps_payload = _step_detail_payload(result.vectors, result.plan) if include_steps else None
    return StreamingEvaluationSnapshot(
        event_id=event_id,
        event_index=event_index,
        operation=operation,
        decision=result.policy.decision.value,
        aggregate=aggregate_payload,
        exceeded=exceeded_payload,
        asked=asked_payload,
        triggers=triggers,
        judge_output_deltas=judge_deltas,
        context=context,
        steps=steps_payload,
    )


def _budget_consumption_from_ledger(ledger: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {key: float(values.get("consumed", 0.0)) for key, values in ledger.items()}


def _budget_utilization_from_ledger(ledger: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    utilization: Dict[str, float] = {}
    for key, values in ledger.items():
        budget = float(values.get("budget", 0.0))
        consumed = float(values.get("consumed", 0.0))
        utilization[key] = 0.0 if budget <= 0.0 else consumed / budget
    return utilization




def _budget_projection_from_ledger(ledger: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    projection: Dict[str, float] = {}
    for key, values in ledger.items():
        projection[key] = float(values.get("projected", values.get("consumed", 0.0)))
    return projection


def _budget_projection_utilization_from_ledger(ledger: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    utilization: Dict[str, float] = {}
    for key, values in ledger.items():
        budget = float(values.get("budget", 0.0))
        projected = float(values.get("projected", values.get("consumed", 0.0)))
        utilization[key] = 0.0 if budget <= 0.0 else projected / budget
    return utilization

def _aggregate_session_risk(per_agent_aggregates: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    for axis in SCOPE_AXES:
        aggregated[axis] = min(
            1.0,
            sum(float(agg.get(axis, 0.0)) for agg in per_agent_aggregates.values()),
        )
    return aggregated


def _detect_cross_agent_scope_laundering(
    *,
    global_aggregate: Dict[str, float],
    per_agent_aggregates: Dict[str, Dict[str, float]],
    ask_threshold: float,
) -> List[ScopeLaunderingSignal]:
    signals: List[ScopeLaunderingSignal] = []
    for axis in SCOPE_AXES:
        global_value = float(global_aggregate.get(axis, 0.0))
        max_agent_value = max(
            (float(aggregate.get(axis, 0.0)) for aggregate in per_agent_aggregates.values()),
            default=0.0,
        )
        if global_value >= ask_threshold and max_agent_value < ask_threshold:
            signals.append(
                ScopeLaunderingSignal(
                    axis=axis,
                    global_value=global_value,
                    max_agent_value=max_agent_value,
                    delta=global_value - max_agent_value,
                    ask_threshold=ask_threshold,
                )
            )
    return signals


def _summarize_response(policy, aggregate, effective_decision: str) -> str:
    top_axes = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:3]
    axes_text = ", ".join(f"{axis}={value:.2f}" for axis, value in top_axes)
    return f"Decision {policy.decision.value} (effective: {effective_decision}). Top axes: {axes_text}."


def _next_steps_from_policy(policy) -> List[str]:
    suggestions: List[str] = []

    if "read_before_write" in policy.asked:
        suggestions.append(
            "Add an explicit read step before patching code (for example: git_read on failing files)."
        )
    if "validation_after_write" in policy.asked:
        suggestions.append(
            "Add a downstream validation step after patching (for example: run targeted tests)."
        )

    for axis, (_, threshold) in policy.exceeded.items():
        suggestions.append(
            f"Reduce {axis} below {float(threshold):.2f} or split into smaller steps."
        )

    for axis, threshold in policy.asked.items():
        if axis in {"read_before_write", "validation_after_write"}:
            continue
        suggestions.append(f"Consider approval or mitigating {axis} below {float(threshold):.2f}.")

    if any("Tool category" in reason for reason in policy.reasons):
        suggestions.append("Remove high-risk tool categories or get explicit approval.")

    knee_recommendations = []
    if getattr(policy, "policy_input", None) is not None:
        knee_recommendations = (
            policy.policy_input.metadata.get("knee_plan_patch_recommendations", [])
            if isinstance(policy.policy_input.metadata, dict)
            else []
        )
    for recommendation in knee_recommendations:
        rationale = recommendation.get("rationale") if isinstance(recommendation, dict) else None
        if rationale:
            suggestions.append(f"Knee recommendation: {rationale}")

    if not suggestions:
        suggestions.append("Proceed; plan appears proportionate to the contract.")
    return suggestions[:5]


def _suggest_plan_patch(policy, plan: PlanDAG) -> List[Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []

    knee_recommendations = []
    if getattr(policy, "policy_input", None) is not None and isinstance(policy.policy_input.metadata, dict):
        knee_recommendations = policy.policy_input.metadata.get("knee_plan_patch_recommendations", [])
    for recommendation in knee_recommendations:
        if isinstance(recommendation, dict):
            patch = recommendation.get("patch")
            if isinstance(patch, dict):
                patches.append(patch)

    triggered = set(policy.exceeded.keys()) | set(policy.asked.keys())
    if "read_before_write" in policy.asked:
        first_write = next((step for step in plan.steps if step.tool in SWE_WRITE_TOOLS), None)
        if first_write is not None:
            patches.append(
                {
                    "op": "insert_before",
                    "target_step_id": first_write.id,
                    "step": {
                        "id": "read_before_write",
                        "description": "Read failing test and impacted source files.",
                        "tool": "git_read",
                    },
                }
            )
    if "validation_after_write" in policy.asked:
        first_write = next((step for step in plan.steps if step.tool in SWE_WRITE_TOOLS), None)
        if first_write is not None:
            patches.append(
                {
                    "op": "insert_after",
                    "target_step_id": first_write.id,
                    "step": {
                        "id": "validate_after_patch",
                        "description": "Run targeted tests for modified behavior.",
                        "tool": "analysis",
                    },
                }
            )

    if "dependency_creation" in triggered:
        high_dependency = next(
            (
                step
                for step in plan.steps
                if (step.tool_category or "") in {"infra", "payments", "finance", "health"}
            ),
            None,
        )
        if high_dependency is not None:
            patches.append(
                {
                    "op": "replace",
                    "target_step_id": high_dependency.id,
                    "step": {
                        "id": f"{high_dependency.id}_reduced_tooling",
                        "description": "Use lower-risk analysis-first workflow before introducing external dependencies.",
                        "tool": "analysis",
                    },
                    "rationale": "Reduce tool category risk and defer new dependency creation.",
                }
            )

    if "depth" in triggered and len(plan.steps) >= 2:
        split_after = max(1, len(plan.steps) // 2)
        patches.append(
            {
                "op": "split_plan",
                "after_step_id": plan.steps[split_after - 1].id,
                "chunks": [
                    {
                        "name": "phase_1_safe_discovery",
                        "step_ids": [step.id for step in plan.steps[:split_after]],
                    },
                    {
                        "name": "phase_2_execution",
                        "step_ids": [step.id for step in plan.steps[split_after:]],
                    },
                ],
                "rationale": "Split high-depth execution into staged checkpoints.",
            }
        )

    if "power_concentration" in triggered:
        first_privileged = next(
            (
                step
                for step in plan.steps
                if (step.tool_category or "") in {"iam", "infra", "payments"}
            ),
            None,
        )
        if first_privileged is not None:
            patches.append(
                {
                    "op": "insert_before",
                    "target_step_id": first_privileged.id,
                    "step": {
                        "id": "approval_gate",
                        "description": "Request human approval with blast-radius summary before privileged action.",
                        "tool": "analysis",
                    },
                    "rationale": "Add approval gate before high-power operations.",
                }
            )

    if "irreversibility" in triggered:
        irreversible = next(
            (
                step
                for step in plan.steps
                if (step.tool_category or "") in {"infra", "iam", "payments", "health"}
                or any(keyword in step.description.lower() for keyword in ("delete", "destroy", "drop", "rotate"))
            ),
            None,
        )
        if irreversible is not None:
            patches.append(
                {
                    "op": "replace",
                    "target_step_id": irreversible.id,
                    "step": {
                        "id": f"{irreversible.id}_reversible_preview",
                        "description": "Run dry-run/preview and create rollback artifact before applying irreversible change.",
                        "tool": "analysis",
                    },
                    "rationale": "Reduce irreversible operations by introducing reversible preview.",
                }
            )
    return patches


def _looks_like_validation(description: str) -> bool:
    text = description.lower()
    return any(hint in text for hint in VALIDATION_HINTS)


def _has_read_before_write(plan: PlanDAG) -> bool:
    read_seen = False
    for step in plan.steps:
        if step.tool in SWE_READ_TOOLS:
            read_seen = True
        if step.tool in SWE_WRITE_TOOLS and not read_seen:
            return False
    return True


def _has_validation_after_write(plan: PlanDAG) -> bool:
    write_seen = False
    for step in plan.steps:
        if step.tool in SWE_WRITE_TOOLS:
            write_seen = True
            continue
        if write_seen and (
            (step.tool in VALIDATION_TOOLS) or _looks_like_validation(step.description)
        ):
            return True
    return not write_seen


def _infer_task_type(contract: TaskContract, plan: PlanDAG) -> str:
    text = f"{contract.goal} {plan.task}".lower()
    if "bug" in text or "fix" in text:
        return "bug_fix"
    if "test" in text:
        return "test_stabilization"
    if "refactor" in text:
        return "refactor"
    return "general_coding"


def _build_telemetry(
    contract: TaskContract, plan: PlanDAG, policy, ask_action: Optional[str], outcome: Optional[str]
) -> TelemetryDetail:
    triggered = sorted(set(policy.exceeded.keys()) | set(policy.asked.keys()))
    return TelemetryDetail(
        preset=contract.preset.value,
        policy_input_version="v1",
        task_type=_infer_task_type(contract, plan),
        plan_size=len(plan.steps),
        decision=policy.decision.value,
        triggered_rules=triggered,
        has_read_before_write=_has_read_before_write(plan),
        has_validation_after_write=_has_validation_after_write(plan),
        ask_action=ask_action,
        outcome=outcome,
    )


def _effective_decision(policy_decision: str, shadow_mode: bool) -> str:
    if shadow_mode and policy_decision in {"ASK", "DENY"}:
        return "ALLOW"
    return policy_decision


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _telemetry_row(
    telemetry: TelemetryDetail,
    policy_input: Optional[Dict[str, Any]],
    aggregate: Dict[str, float],
    asked: Dict[str, float],
    exceeded: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "schema_version": "telemetry_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy_input": policy_input or {},
        "decision": telemetry.decision,
        "aggregate": aggregate,
        "asked": asked,
        "exceeded": exceeded,
        "feedback": {
            "ask_action": telemetry.ask_action,
            "outcome": telemetry.outcome,
        },
        "telemetry": telemetry.model_dump(),
    }


def _load_telemetry_rows(path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit <= 0:
        return rows
    return rows[-limit:]



def _build_calibration_dashboard(path: Path) -> CalibrationDashboardResponse:
    domain_payload = compute_domain_calibration_from_telemetry(path)
    entries: List[CalibrationDashboardEntry] = []
    for domain, (calibration, stats) in sorted(domain_payload.items()):
        entries.append(
            CalibrationDashboardEntry(
                domain=domain,
                runs=stats.runs,
                calibration=calibration_to_dict(calibration),
                stats={
                    "triggered": stats.triggered,
                    "false_alarms": stats.false_alarms,
                    "overrides": stats.overrides,
                    "failures": stats.failures,
                },
            )
        )
    return CalibrationDashboardResponse(
        enabled=True,
        source=str(path),
        count=sum(item.runs for item in entries),
        domains=entries,
    )


def _resolve_adaptive_calibration(
    req: EvaluateRequest,
    telemetry_path: Optional[str],
) -> Optional[CalibratedDecisionThresholds]:
    calibration: Optional[CalibratedDecisionThresholds] = None
    if req.calibration_scale is not None:
        calibration = CalibratedDecisionThresholds(global_scale=req.calibration_scale)

    if telemetry_path:
        domain_payload = compute_domain_calibration_from_telemetry(Path(telemetry_path))
        domain_key = req.calibration_domain or DEFAULT_DOMAIN
        picked = domain_payload.get(domain_key)
        if picked is None and req.calibration_domain:
            picked = domain_payload.get(DEFAULT_DOMAIN)
        if picked is not None:
            domain_calibration = picked[0]
            if calibration is not None:
                domain_calibration = CalibratedDecisionThresholds(
                    global_scale=calibration.global_scale,
                    axis_scale=domain_calibration.axis_scale,
                    axis_bias=domain_calibration.axis_bias,
                    axis_threshold_factor=domain_calibration.axis_threshold_factor,
                    abstain_uncertainty_threshold=domain_calibration.abstain_uncertainty_threshold,
                )
            calibration = domain_calibration

    calibration = apply_manual_adjustments(calibration, req.calibration_manual_adjustments) if calibration else calibration
    return calibration


def create_app(default_policy_backend: str = "python", telemetry_jsonl_path: Optional[str] = None) -> FastAPI:
    init_tracing(enable_console=False)
    tracer = get_tracer("scopebench")
    app = FastAPI(title="ScopeBench", version="0.1.0")
    configured_telemetry_path = telemetry_jsonl_path or os.getenv("SCOPEBENCH_TELEMETRY_JSONL_PATH")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ui", response_class=HTMLResponse)
    def ui_index():
        from pathlib import Path

        ui_path = Path(__file__).resolve().parents[1] / "ui" / "index.html"
        return ui_path.read_text(encoding="utf-8")

    @app.get("/templates")
    def templates_endpoint():
        from pathlib import Path
        import yaml

        templates_root = Path(__file__).resolve().parents[1] / "templates"
        payload: List[Dict[str, Any]] = []
        for domain_dir in sorted(p for p in templates_root.iterdir() if p.is_dir()):
            variants: Dict[str, Dict[str, Any]] = {}

            def load_variant(variant_name: str, contract_path: Path, plan_path: Path, notes_path: Path) -> None:
                variants[variant_name] = {
                    "metadata": {
                        "has_contract": contract_path.exists(),
                        "has_plan": plan_path.exists(),
                        "has_notes": notes_path.exists(),
                    },
                    "content": {
                        "contract": yaml.safe_load(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else None,
                        "plan": yaml.safe_load(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None,
                        "notes": notes_path.read_text(encoding="utf-8") if notes_path.exists() else None,
                    },
                }

            load_variant(
                "default",
                domain_dir / "contract.yaml",
                domain_dir / "plan.yaml",
                domain_dir / "notes.md",
            )

            for contract_path in sorted(domain_dir.glob("*.contract.yaml")):
                variant = contract_path.name[: -len(".contract.yaml")]
                if not variant:
                    continue
                load_variant(
                    variant,
                    contract_path,
                    domain_dir / f"{variant}.plan.yaml",
                    domain_dir / f"{variant}.notes.md",
                )

            payload.append({"domain": domain_dir.name, "variants": variants})
        return {"templates": payload}

    @app.get("/tools")
    def tools_endpoint():
        from scopebench.scoring.rules import ToolRegistry

        registry = ToolRegistry.load_default()
        tools = []
        for tool_name, tool_info in sorted(registry._tools.items()):  # noqa: SLF001
            tools.append(
                {
                    "tool": tool_name,
                    "category": tool_info.category,
                    "domains": list(tool_info.domains),
                    "risk_class": tool_info.risk_class,
                    "priors": dict(tool_info.priors),
                    "default_effects": dict(tool_info.default_effects),
                }
            )

        schema = {
            "type": "object",
            "required": ["tool", "category", "domains", "risk_class", "priors", "default_effects"],
            "properties": {
                "tool": {"type": "string"},
                "category": {"type": "string"},
                "domains": {"type": "array", "items": {"type": "string"}},
                "risk_class": {"type": "string", "enum": ["low", "moderate", "high", "critical"]},
                "priors": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
                "default_effects": {"type": "object"},
            },
            "additionalProperties": False,
        }
        return {"tools": tools, "normalized_schema": schema}

    @app.get("/cases")
    def cases_endpoint():
        from scopebench.bench.dataset import default_cases_path, load_cases

        try:
            cases = load_cases(default_cases_path())
        except ValueError as exc:
            return {
                "datasets": [],
                "domains": [],
                "count": 0,
                "cases": [],
                "error": str(exc),
            }

        return {
            "datasets": sorted({case.id for case in cases}),
            "domains": sorted({case.domain for case in cases}),
            "count": len(cases),
            "cases": [
                {
                    "id": case.id,
                    "domain": case.domain,
                    "instruction": case.instruction,
                    "expected_decision": case.expected_decision,
                    "contract": case.contract,
                }
                for case in cases
            ],
        }

    @app.get("/telemetry/replay")
    def telemetry_replay(limit: int = 50):
        if not configured_telemetry_path:
            return {
                "enabled": False,
                "rows": [],
                "message": "Set SCOPEBENCH_TELEMETRY_JSONL_PATH to enable replay.",
            }

        rows = _load_telemetry_rows(Path(configured_telemetry_path), limit=limit)
        return {
            "enabled": True,
            "source": configured_telemetry_path,
            "count": len(rows),
            "rows": rows,
        }


    @app.get("/calibration/dashboard", response_model=CalibrationDashboardResponse)
    def calibration_dashboard_endpoint():
        if not configured_telemetry_path:
            return CalibrationDashboardResponse(
                enabled=False,
                source=None,
                count=0,
                domains=[],
            )
        path = Path(configured_telemetry_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Telemetry path not found")
        return _build_calibration_dashboard(path)

    @app.post("/calibration/adjust", response_model=CalibrationDashboardResponse)
    def calibration_adjust_endpoint(req: CalibrationAdjustmentRequest):
        if not configured_telemetry_path:
            raise HTTPException(status_code=400, detail="Telemetry path not configured")
        path = Path(configured_telemetry_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Telemetry path not found")

        domain_payload = compute_domain_calibration_from_telemetry(path)
        picked = domain_payload.get(req.domain)
        if picked is None:
            raise HTTPException(status_code=404, detail=f"Domain '{req.domain}' not found")
        calibration, _ = picked
        adjusted = apply_manual_adjustments(
            calibration,
            {
                "axis_threshold_factor_delta": req.axis_threshold_factor_delta,
                "axis_scale_delta": req.axis_scale_delta,
                "abstain_uncertainty_threshold_delta": req.abstain_uncertainty_threshold_delta,
            },
        )
        domain_payload[req.domain] = (adjusted, picked[1])

        entries: List[CalibrationDashboardEntry] = []
        for domain, (calibration, stats) in sorted(domain_payload.items()):
            entries.append(
                CalibrationDashboardEntry(
                    domain=domain,
                    runs=stats.runs,
                    calibration=calibration_to_dict(calibration),
                    stats={
                        "triggered": stats.triggered,
                        "false_alarms": stats.false_alarms,
                        "overrides": stats.overrides,
                        "failures": stats.failures,
                    },
                )
            )
        return CalibrationDashboardResponse(
            enabled=True,
            source=str(path),
            count=sum(item.runs for item in entries),
            domains=entries,
        )

    @app.post("/evaluate", response_model=EvaluateResponse)
    @app.post(
        "/evaluate",
        response_model=EvaluateResponse,
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/json": {
                        "example": EvaluateRequest.model_config["json_schema_extra"]["example"]
                    }
                }
            },
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "example": EvaluateResponse.model_config["json_schema_extra"]["example"]
                        }
                    }
                }
            },
        },
    )
    def evaluate_endpoint(req: EvaluateRequest):
        contract = TaskContract.model_validate(req.contract)
        plan = PlanDAG.model_validate(req.plan)
        calibration = _resolve_adaptive_calibration(req, configured_telemetry_path)
        backend = req.policy_backend or default_policy_backend
        res = evaluate(contract, plan, calibration=calibration, policy_backend=backend)
        pol = res.policy

        decision = pol.decision.value
        effective_decision = _effective_decision(decision, req.shadow_mode)

        steps = None
        if req.include_steps:
            steps = []
            plan_steps_by_id = {step.id: step for step in plan.steps}
            for vec in res.vectors:
                axes = {
                    "spatial": AxisDetail(**vec.spatial.model_dump()),
                    "temporal": AxisDetail(**vec.temporal.model_dump()),
                    "depth": AxisDetail(**vec.depth.model_dump()),
                    "irreversibility": AxisDetail(**vec.irreversibility.model_dump()),
                    "resource_intensity": AxisDetail(**vec.resource_intensity.model_dump()),
                    "legal_exposure": AxisDetail(**vec.legal_exposure.model_dump()),
                    "dependency_creation": AxisDetail(**vec.dependency_creation.model_dump()),
                    "stakeholder_radius": AxisDetail(**vec.stakeholder_radius.model_dump()),
                    "power_concentration": AxisDetail(**vec.power_concentration.model_dump()),
                    "uncertainty": AxisDetail(**vec.uncertainty.model_dump()),
                }
                plan_step = plan_steps_by_id.get(vec.step_id or "")
                steps.append(
                    StepDetail(
                        step_id=vec.step_id,
                        tool=vec.tool,
                        tool_category=vec.tool_category,
                        est_cost_usd=plan_step.est_cost_usd if plan_step else None,
                        est_time_days=plan_step.est_time_days if plan_step else None,
                        est_labor_hours=plan_step.est_labor_hours if plan_step else None,
                        resolved_cost_usd=plan_step.resolved_cost_usd() if plan_step else None,
                        resolved_time_days=plan_step.resolved_time_days() if plan_step else None,
                        resolved_labor_hours=plan_step.resolved_labor_hours() if plan_step else None,
                        realtime_estimates=list(plan_step.realtime_estimates) if plan_step else [],
                        est_benefit=plan_step.est_benefit if plan_step else None,
                        benefit_unit=plan_step.benefit_unit if plan_step else None,
                        axes=axes,
                    )
                )

        summary = None
        next_steps = None
        patch_suggestion = None
        if req.include_summary:
            summary = _summarize_response(pol, res.aggregate.as_dict(), effective_decision)
            next_steps = _next_steps_from_policy(pol)
            patch_suggestion = _suggest_plan_patch(pol, plan)

        telemetry = (
            _build_telemetry(contract, plan, pol, req.ask_action, req.outcome)
            if req.include_telemetry
            else None
        )

        reasons = list(pol.reasons)
        if req.shadow_mode and decision != effective_decision:
            reasons.append(
                "Shadow mode enabled: returning effective_decision=ALLOW while preserving policy decision for analysis"
            )

        with tracer.start_as_current_span("scopebench.evaluate.response"):
            trace_context = current_trace_context()
        exceeded_payload = {
            k: {"value": float(v[0]), "threshold": float(v[1])} for k, v in pol.exceeded.items()
        }
        asked_payload = {k: float(v) for k, v in pol.asked.items()}
        policy_input_payload = pol.policy_input.__dict__ if (req.include_telemetry and pol.policy_input) else None

        if req.include_telemetry and telemetry and configured_telemetry_path:
            _append_jsonl(
                Path(configured_telemetry_path),
                _telemetry_row(
                    telemetry=telemetry,
                    policy_input=policy_input_payload,
                    aggregate=res.aggregate.as_dict(),
                    asked=asked_payload,
                    exceeded=exceeded_payload,
                ),
            )

        return EvaluateResponse(
            trace_id=trace_context.get("trace_id"),
            span_id=trace_context.get("span_id"),
            decision=decision,
            policy_backend=pol.policy_backend,
            policy_version=pol.policy_version,
            policy_hash=pol.policy_hash,
            effective_decision=effective_decision,
            shadow_mode=req.shadow_mode,
            reasons=reasons,
            exceeded=exceeded_payload,
            asked=asked_payload,
            aggregate=res.aggregate.as_dict(),
            n_steps=res.aggregate.n_steps,
            steps=steps,
            summary=summary,
            next_steps=next_steps,
            plan_patch_suggestion=patch_suggestion,
            telemetry=telemetry,
            policy_input=policy_input_payload,
        )

    @app.post("/evaluate_stream", response_model=EvaluateStreamResponse)
    def evaluate_stream_endpoint(req: EvaluateStreamRequest):
        contract = TaskContract.model_validate(req.contract)
        backend = req.policy_backend or default_policy_backend

        plan_data: Dict[str, Any] = dict(req.plan)
        previous_vectors = None
        previous_aggregate = None

        initial_plan = PlanDAG.model_validate(plan_data)
        initial_result = evaluate(contract, initial_plan, policy_backend=backend, judge=req.judge)
        initial_snapshot = _snapshot_from_result(
            contract=contract,
            event_id="initial",
            event_index=0,
            operation="initial",
            context={},
            result=initial_result,
            judge_deltas=[],
            previous_aggregate=None,
            include_steps=req.include_steps,
        )
        previous_vectors = initial_result.vectors
        previous_aggregate = initial_result.aggregate.as_dict()

        updates: List[StreamingEvaluationSnapshot] = []
        for index, event in enumerate(req.events, start=1):
            plan_data = _apply_streaming_event(plan_data, event)
            plan_model = PlanDAG.model_validate(plan_data)
            result = evaluate(contract, plan_model, policy_backend=backend, judge=req.judge)
            judge_deltas = _judge_output_deltas(previous_vectors, result.vectors)
            snapshot = _snapshot_from_result(
                contract=contract,
                event_id=event.event_id,
                event_index=index,
                operation=event.operation,
                context=dict(event.context),
                result=result,
                judge_deltas=judge_deltas,
                previous_aggregate=previous_aggregate,
                include_steps=req.include_steps,
            )
            updates.append(snapshot)
            previous_vectors = result.vectors
            previous_aggregate = result.aggregate.as_dict()

        return EvaluateStreamResponse(initial=initial_snapshot, updates=updates)

    @app.post("/evaluate_session", response_model=EvaluateSessionResponse)
    def evaluate_session_endpoint(req: EvaluateSessionRequest):
        session = MultiAgentSession.model_validate(req.session)
        backend = req.policy_backend or default_policy_backend

        per_agent: Dict[str, SessionAggregateDetail] = {}
        per_agent_aggregates: Dict[str, Dict[str, float]] = {}
        per_agent_dashboard: Dict[str, SessionDashboardEntry] = {}
        agent_aggregates = []
        global_plans: List[PlanDAG] = []
        global_decision = "ALLOW"

        for agent in sorted(session.agents, key=lambda item: item.agent_id):
            agent_plans = session.plans_for(agent.agent_id)
            contract = session.contract_for(agent.agent_id)
            global_plans.extend(agent_plans)

            with tracer.start_as_current_span("scopebench.evaluate_session.agent") as span:
                span.set_attribute("scopebench.agent_id", agent.agent_id)
                span.set_attribute("scopebench.agent_plan_count", len(agent_plans))
                agent_results = [evaluate(contract, plan, policy_backend=backend) for plan in agent_plans]
            agent_aggregate = combine_aggregates([result.aggregate for result in agent_results])
            agent_decision = "ALLOW"
            for result in agent_results:
                if result.policy.decision.value == "DENY":
                    agent_decision = "DENY"
                    break
                if result.policy.decision.value == "ASK":
                    agent_decision = "ASK"

            agent_ledger = build_budget_ledger(contract, agent_plans)
            if any(entry["exceeded"] > 0 for entry in agent_ledger.values()) and agent_decision != "DENY":
                agent_decision = "ASK"

            aggregate_dict = agent_aggregate.as_dict()
            per_agent[agent.agent_id] = SessionAggregateDetail(
                aggregate=aggregate_dict,
                ledger=agent_ledger,
                decision=agent_decision,
            )
            per_agent_aggregates[agent.agent_id] = aggregate_dict
            per_agent_dashboard[agent.agent_id] = SessionDashboardEntry(
                aggregate=aggregate_dict,
                budget_consumption=_budget_consumption_from_ledger(agent_ledger),
                budget_utilization=_budget_utilization_from_ledger(agent_ledger),
                budget_projection=_budget_projection_from_ledger(agent_ledger),
                budget_projection_utilization=_budget_projection_utilization_from_ledger(agent_ledger),
                decision=agent_decision,
            )
            agent_aggregates.append(agent_aggregate)

            if agent_decision == "DENY":
                global_decision = "DENY"
            elif agent_decision == "ASK" and global_decision != "DENY":
                global_decision = "ASK"

        global_aggregate_dict = _aggregate_session_risk(per_agent_aggregates)
        global_ledger = build_budget_ledger(session.global_contract, global_plans)
        if any(entry["exceeded"] > 0 for entry in global_ledger.values()) and global_decision != "DENY":
            global_decision = "ASK"

        global_threshold = float(session.global_contract.escalation.ask_if_any_axis_over)
        laundering_signals = _detect_cross_agent_scope_laundering(
            global_aggregate=global_aggregate_dict,
            per_agent_aggregates=per_agent_aggregates,
            ask_threshold=global_threshold,
        )
        if laundering_signals and global_decision == "ALLOW":
            global_decision = "ASK"

        global_scope = SessionAggregateDetail(
            aggregate=global_aggregate_dict,
            ledger=global_ledger,
            decision=global_decision,
        )
        with tracer.start_as_current_span("scopebench.evaluate_session.response"):
            trace_context = current_trace_context()
        dashboard = SessionDashboard(
            per_agent=per_agent_dashboard,
            global_=SessionDashboardEntry(
                aggregate=global_aggregate_dict,
                budget_consumption=_budget_consumption_from_ledger(global_ledger),
                budget_utilization=_budget_utilization_from_ledger(global_ledger),
                budget_projection=_budget_projection_from_ledger(global_ledger),
                budget_projection_utilization=_budget_projection_utilization_from_ledger(global_ledger),
                decision=global_decision,
            ),
        )
        return EvaluateSessionResponse(
            trace_id=trace_context.get("trace_id"),
            span_id=trace_context.get("span_id"),
            decision=global_decision,
            per_agent=per_agent,
            global_=global_scope,
            laundering_signals=laundering_signals,
            dashboard=dashboard,
        )

    return app
