from __future__ import annotations

import json
import os
import re
import tempfile
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
import yaml

from scopebench.contracts import TaskContract
from scopebench.plan import EffectSpec, PlanDAG, RealtimeEstimate, plan_from_dict
from scopebench.contracts import Preset, TaskContract
from scopebench.plan import PlanDAG, RealtimeEstimate
from scopebench.runtime.guard import evaluate
from scopebench.scoring.axes import (
    AxisScore,
    SCOPE_AXES,
    ScopeVector,
    combine_aggregates,
)
from scopebench.scoring.calibration import (
    DEFAULT_DOMAIN,
    CalibratedDecisionThresholds,
    apply_manual_adjustments,
    calibration_to_dict,
    compute_domain_calibration_from_telemetry,
)
from scopebench.scoring.rules import aggregate_scope, build_budget_ledger
from scopebench.scoring.effects_annotator import suggest_effects_for_plan
from scopebench.scoring.rules import build_budget_ledger
from scopebench.plugins import PluginManager, lint_plugin_bundle, sign_plugin_bundle
from scopebench.bench.plugin_harness import run_plugin_test_harness
from scopebench.bench.dataset import default_cases_path
from scopebench.session import MultiAgentSession
from scopebench.bench.community import suggest_case
from scopebench.bench.dataset import validate_case_object
from scopebench.tracing.otel import current_trace_context, get_tracer, init_tracing

SWE_READ_TOOLS = {"git_read", "file_read"}
SWE_WRITE_TOOLS = {"git_patch", "git_rewrite", "file_write"}
VALIDATION_TOOLS = {"analysis", "test_run", "pytest"}
VALIDATION_HINTS = ("test", "verify", "validation", "assert", "check")
RATIONALE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


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
    include_steps: bool = Field(
        False, description="Include step-level vectors and rationales."
    )
    include_summary: bool = Field(
        False, description="Include summary and next-step guidance."
    )
    include_telemetry: bool = Field(
        True, description="Include lightweight evaluation telemetry fields."
    )
    shadow_mode: bool = Field(
        False,
        description="If true, never block execution; return what enforcement would decide.",
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
    rationale_summary: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    next_steps: Optional[List[str]] = None
    plan_patch_suggestion: Optional[List[Dict[str, Any]]] = None
    telemetry: Optional[TelemetryDetail] = None
    policy_input: Optional[Dict[str, Any]] = None


class EvaluateSessionRequest(BaseModel):
    session: Dict[str, Any] = Field(..., description="MultiAgentSession as dict")
    include_steps: bool = Field(
        False, description="Include step-level vectors and rationales."
    )
    include_telemetry: bool = Field(
        True, description="Include lightweight evaluation telemetry fields."
    )
    policy_backend: Optional[str] = Field(
        None, description="Policy backend override: python|opa|cedar."
    )


class DatasetValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: Dict[str, Any] = Field(..., description="Single benchmark case object")


class DatasetValidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    case_id: str


class DatasetSuggestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    domain: str
    instruction: str
    contract: Dict[str, Any]
    plan: Dict[str, Any]
    expected_decision: str
    expected_rationale: str
    notes: Optional[str] = None
    policy_backend: str = "python"


class DatasetSuggestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: Dict[str, Any]


class DatasetRenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: Dict[str, Any]
    format: str = Field(default="json", description="json|yaml")


class DatasetRenderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    content_type: str
    content: str


class DatasetWizardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    domain: str
    instruction: str
    contract: Dict[str, Any]
    plan: Dict[str, Any]
    expected_decision: str
    expected_rationale: str
    notes: Optional[str] = None
    policy_backend: str = "python"
    format: str = Field(default="json", description="json|yaml")


class DatasetWizardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    case_id: str
    case: Dict[str, Any]
    rendered: DatasetRenderResponse


class DatasetReviewCommentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    reviewer: str
    comment: str
    draft_case: Optional[Dict[str, Any]] = None


class DatasetReviewSuggestEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    reviewer: str
    field_path: str = Field(description="Dot-path describing the suggested edit target")
    proposed_value: str
    rationale: str
    draft_case: Optional[Dict[str, Any]] = None


class DatasetReviewVoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    reviewer: str
    vote: str = Field(description="accept|reject|abstain")
    draft_case: Optional[Dict[str, Any]] = None


class DatasetReviewStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    draft_case: Optional[Dict[str, Any]] = None
    comments: List[Dict[str, str]]
    suggested_edits: List[Dict[str, str]]
    votes: Dict[str, str]
    acceptance: Dict[str, Any]
    updated_utc: str


class PluginWizardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str
    publisher: str
    name: str
    version: str = "0.1.0"
    tools: List[str] = Field(default_factory=list)
    tool_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    effects_mappings: List[Dict[str, Any]] = Field(default_factory=list)
    policy_rule_templates: List[str] = Field(default_factory=list)
    key_id: str = "community-main"
    secret: str


class PluginWizardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    lint_errors: List[str]
    bundle: Dict[str, Any]
    harness: Dict[str, Any]
    publish_guidance: List[str]


class PolicyRuleProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    action: str = Field(description="ALLOW|ASK|DENY")
    axis: Optional[str] = None
    operator: Optional[str] = None
    value: Optional[float] = None
    reason: Optional[str] = None


class PolicyWorkbenchStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_backend: str
    thresholds: Dict[str, float]
    escalation: Dict[str, Any]
    backend_assets: Dict[str, Optional[str]]
    signed_policy_rules: List[Dict[str, Any]]
    authorization: Dict[str, Any]


class PolicyWorkbenchTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_backend: str = "python"
    contract: Dict[str, Any]
    plan: Dict[str, Any]
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)
    proposed_rules: List[PolicyRuleProposal] = Field(default_factory=list)


class PolicyWorkbenchApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_backend: str = "python"
    summary: Optional[str] = None
    contract: Dict[str, Any]
    plan: Dict[str, Any]
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)
    proposed_rules: List[PolicyRuleProposal] = Field(default_factory=list)


class PolicyWorkbenchApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    saved_to: str
    proposal_id: str
class SuggestEffectsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: Dict[str, Any]


class SuggestEffectsItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    tool: Optional[str] = None
    effects: Dict[str, Any]


class SuggestEffectsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: Dict[str, Any]
    suggestions: List[SuggestEffectsItem]


class StreamingPlanEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(..., min_length=1)
    operation: str = Field(
        ..., pattern=r"^(add_step|update_step|remove_step|replace_plan)$"
    )
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
    include_steps: bool = Field(
        False, description="Include step-level vectors for each snapshot."
    )
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
    axis_deltas: List[Dict[str, Any]] = Field(default_factory=list)


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


class NegotiationRequest(BaseModel):
    agent_id: str
    amount: float


class NegotiationTransfer(BaseModel):
    from_agent: str
    to_agent: str
    amount: float


class NegotiationReallocation(BaseModel):
    agent_id: str
    target_budget: float
    current_budget: float
    delta: float


class NegotiationConsensus(BaseModel):
    protocol: str
    quorum_ratio: float
    approvals: int
    participants: int
    status: str
    note: str


class NegotiationBudgetRecommendation(BaseModel):
    budget_key: str
    fairness_rule: str
    total_requested: float
    global_headroom: float
    allocated_from_headroom: float
    allocated_from_transfers: float
    remaining_unmet: float
    requests: List[NegotiationRequest]
    transfers: List[NegotiationTransfer]
    reallocation: List[NegotiationReallocation]
    consensus: NegotiationConsensus


class SessionNegotiation(BaseModel):
    triggered: bool
    reason_codes: List[str]
    recommendations: List[NegotiationBudgetRecommendation]


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


class DomainDecisionAnalytics(BaseModel):
    domain: str
    total_cases: int
    decision_counts: Dict[str, int]
    decision_rates: Dict[str, float]


class AxisTriggerAnalytics(BaseModel):
    axis: str
    ask_count: int
    deny_count: int


class EffectThresholdAnalytics(BaseModel):
    axis: str
    average_effect: float
    average_threshold: float
    average_margin: float
    over_threshold_cases: int
    over_threshold_rate: float


class DecisionAxisLeader(BaseModel):
    axis: str
    count: int
    rate_within_decision: float


class TopTriggerAxesByDecision(BaseModel):
    decision: str
    total_cases: int
    top_axes: List[DecisionAxisLeader]


class EffectMagnitudeProfile(BaseModel):
    effect_type: str
    magnitude: str
    count: int
    decision_counts: Dict[str, int]
    decision_rates: Dict[str, float]


class CasesAnalyticsResponse(BaseModel):
    source: str
    count: int
    decision_distribution_by_domain: List[DomainDecisionAnalytics]
    trigger_axes: List[AxisTriggerAnalytics]
    effect_magnitude_vs_threshold: List[EffectThresholdAnalytics]
    top_trigger_axes_by_decision: List[TopTriggerAxesByDecision] = Field(default_factory=list)
    effect_magnitude_profiles: List[EffectMagnitudeProfile] = Field(default_factory=list)
_AXIS_TO_THRESHOLD_FIELD = {
    "spatial": "max_spatial",
    "temporal": "max_temporal",
    "depth": "max_depth",
    "irreversibility": "max_irreversibility",
    "resource_intensity": "max_resource_intensity",
    "legal_exposure": "max_legal_exposure",
    "dependency_creation": "max_dependency_creation",
    "stakeholder_radius": "max_stakeholder_radius",
    "power_concentration": "max_power_concentration",
    "uncertainty": "max_uncertainty",
}


def _dashboard_row_domain(row: Dict[str, Any]) -> str:
    for key in ("domain", "task_type"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    telemetry = row.get("telemetry")
    if isinstance(telemetry, dict):
        task_type = telemetry.get("task_type")
        if isinstance(task_type, str) and task_type:
            return task_type
    policy_input = row.get("policy_input")
    if isinstance(policy_input, dict):
        task_type = policy_input.get("task_type")
        if isinstance(task_type, str) and task_type:
            return task_type
    return DEFAULT_DOMAIN


def _dashboard_axis_signal(row: Dict[str, Any], axis: str) -> float:
    values: List[float] = []
    for container_name in ("asked", "aggregate"):
        container = row.get(container_name)
        if isinstance(container, dict):
            value = container.get(axis)
            if isinstance(value, (int, float)):
                values.append(float(value))
    exceeded = row.get("exceeded")
    if isinstance(exceeded, dict):
        value = exceeded.get(axis)
        if isinstance(value, dict):
            raw = value.get("value")
            if isinstance(raw, (int, float)):
                values.append(float(raw))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return 0.0
    return max(0.0, min(1.0, max(values)))


def _preset_thresholds() -> Dict[str, Dict[str, float]]:
    presets: Dict[str, Dict[str, float]] = {}
    for preset in Preset:
        contract = TaskContract(goal="calibration dashboard", preset=preset)
        thresholds: Dict[str, float] = {}
        for axis, field in _AXIS_TO_THRESHOLD_FIELD.items():
            thresholds[axis] = float(getattr(contract.thresholds, field))
        presets[preset.value] = thresholds
    return presets


class EvaluateSessionResponse(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    decision: str
    per_agent: Dict[str, SessionAggregateDetail]
    global_: SessionAggregateDetail = Field(..., alias="global")
    laundering_signals: List[ScopeLaunderingSignal]
    dashboard: SessionDashboard
    negotiation: SessionNegotiation

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
                resolved_time_days=(
                    plan_step.resolved_time_days() if plan_step else None
                ),
                resolved_labor_hours=(
                    plan_step.resolved_labor_hours() if plan_step else None
                ),
                realtime_estimates=(
                    list(plan_step.realtime_estimates) if plan_step else []
                ),
                est_benefit=plan_step.est_benefit if plan_step else None,
                benefit_unit=plan_step.benefit_unit if plan_step else None,
                axes=axes,
            )
        )
    return details


def _vector_axes_by_step(vectors: List[Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    payload: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for vector in vectors:
        if not vector.step_id:
            continue
        step_axes: Dict[str, Dict[str, Any]] = {}
        for axis in SCOPE_AXES:
            detail = getattr(vector, axis, None)
            if detail is None:
                continue
            step_axes[axis] = {
                "value": float(getattr(detail, "value", 0.0)),
                "rationale": str(getattr(detail, "rationale", "") or ""),
            }
        payload[vector.step_id] = step_axes
    return payload


def _summarize_step_rationales(
    step_details: Optional[List[StepDetail]],
    aggregate: Dict[str, float],
    top_themes: int = 3,
    top_reasons_per_theme: int = 3,
) -> List[Dict[str, Any]]:
    if not step_details:
        return []

    def _theme_tokens(axis: str, rationale: str) -> List[str]:
        axis_tokens = set(axis.split("_"))
        tokens = re.findall(r"[a-z0-9]+", rationale.lower())
        return [
            token
            for token in tokens
            if token not in RATIONALE_STOPWORDS and token not in axis_tokens and len(token) > 2
        ]

    def _theme_signature(axis: str, rationale: str) -> set[str]:
        return set(_theme_tokens(axis, rationale))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for step in step_details:
        for axis, detail in (step.axes or {}).items():
            grouped.setdefault(axis, []).append(
                {
                    "step_id": step.step_id or "unknown",
                    "value": float(detail.value),
                    "confidence": float(detail.confidence),
                    "rationale": (detail.rationale or "").strip(),
                }
            )

    summaries: List[Dict[str, Any]] = []
    for axis, entries in grouped.items():
        sorted_entries = sorted(entries, key=lambda item: item["value"], reverse=True)
        rationale_index: Dict[str, Dict[str, Any]] = {}
        theme_clusters: List[Dict[str, Any]] = []
        for entry in sorted_entries:
            rationale = entry["rationale"] or "n/a"
            bucket = rationale_index.setdefault(
                rationale,
                {
                    "rationale": rationale,
                    "count": 0,
                    "peak_value": 0.0,
                    "steps": [],
                },
            )
            bucket["count"] += 1
            bucket["peak_value"] = max(bucket["peak_value"], entry["value"])
            bucket["steps"].append(entry["step_id"])

            theme_tokens = _theme_signature(axis, rationale)
            if not theme_tokens:
                theme_tokens = {"unspecified"}
            theme_bucket = next(
                (
                    cluster
                    for cluster in theme_clusters
                    if len(cluster["tokens"] & theme_tokens) > 0
                ),
                None,
            )
            if theme_bucket is None:
                theme_bucket = {
                    "tokens": set(theme_tokens),
                    "count": 0,
                    "peak_value": 0.0,
                    "total_value": 0.0,
                    "steps": [],
                    "rationale_examples": [],
                    "token_freq": {},
                }
                theme_clusters.append(theme_bucket)

            theme_bucket["count"] += 1
            theme_bucket["peak_value"] = max(theme_bucket["peak_value"], entry["value"])
            theme_bucket["total_value"] += entry["value"]
            theme_bucket["steps"].append(entry["step_id"])
            if rationale not in theme_bucket["rationale_examples"]:
                theme_bucket["rationale_examples"].append(rationale)
            theme_bucket["tokens"] |= theme_tokens
            for token in theme_tokens:
                theme_bucket["token_freq"][token] = theme_bucket["token_freq"].get(token, 0) + 1

        top_reasons = sorted(
            rationale_index.values(),
            key=lambda item: (item["count"], item["peak_value"]),
            reverse=True,
        )[:top_reasons_per_theme]

        top_theme_clusters = sorted(
            theme_clusters,
            key=lambda item: (item["count"], item["peak_value"], item["total_value"]),
            reverse=True,
        )[:top_reasons_per_theme]
        axis_contribution = float(aggregate.get(axis, 0.0))
        contribution_score = axis_contribution * sorted_entries[0]["value"] * len(entries)

        summaries.append(
            {
                "theme": axis,
                "contribution": axis_contribution,
                "contribution_score": contribution_score,
                "step_count": len(entries),
                "peak_value": sorted_entries[0]["value"],
                "top_reasons": [
                    {
                        "rationale": reason["rationale"],
                        "count": reason["count"],
                        "peak_value": reason["peak_value"],
                        "steps": sorted(set(reason["steps"])),
                    }
                    for reason in top_reasons
                ],
                "theme_clusters": [
                    {
                        "theme": " ".join(
                            sorted(
                                cluster["token_freq"].keys(),
                                key=lambda token: (
                                    cluster["token_freq"].get(token, 0),
                                    len(token),
                                    token,
                                ),
                                reverse=True,
                            )[:3]
                        ),
                        "count": cluster["count"],
                        "peak_value": cluster["peak_value"],
                        "avg_value": (
                            cluster["total_value"] / cluster["count"]
                            if cluster["count"]
                            else 0.0
                        ),
                        "steps": sorted(set(cluster["steps"])),
                        "rationale_examples": cluster["rationale_examples"][:2],
                    }
                    for cluster in top_theme_clusters
                ],
            }
        )

    return sorted(
        summaries,
        key=lambda item: (item["contribution_score"], item["contribution"], item["peak_value"]),
        reverse=True,
    )[:top_themes]


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
        axis_deltas: List[Dict[str, Any]] = []
        changed_axes = [
            axis
            for axis, detail in current_axes.items()
            if (
                abs(
                    float(detail.get("value", 0.0))
                    - float(previous_axes.get(axis, {}).get("value", 0.0))
                )
                > 1e-6
                or str(detail.get("rationale", ""))
                != str(previous_axes.get(axis, {}).get("rationale", ""))
            )
        ]
        for axis in changed_axes:
            before = previous_axes.get(axis, {})
            after = current_axes.get(axis, {})
            axis_deltas.append(
                {
                    "axis": axis,
                    "previous": float(before.get("value", 0.0)),
                    "current": float(after.get("value", 0.0)),
                    "previous_rationale": str(before.get("rationale", "") or ""),
                    "current_rationale": str(after.get("rationale", "") or ""),
                }
            )
        if changed_axes:
            deltas.append(
                StreamStepDelta(
                    step_id=step_id,
                    changed_axes=sorted(changed_axes),
                    axis_deltas=sorted(
                        axis_deltas, key=lambda item: str(item.get("axis", ""))
                    ),
                )
            )
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


def _apply_streaming_event(
    plan_data: Dict[str, Any], event: StreamingPlanEvent
) -> Dict[str, Any]:
    updated = dict(plan_data)
    steps = list(updated.get("steps", []))
    operation = event.operation

    if operation == "replace_plan":
        if not isinstance(event.step, dict):
            raise HTTPException(
                status_code=400,
                detail="replace_plan requires 'step' with full plan payload",
            )
        replacement = dict(event.step)
        if "task" not in replacement or "steps" not in replacement:
            raise HTTPException(
                status_code=400,
                detail="replace_plan payload must include task and steps",
            )
        return replacement

    if not event.step_id:
        raise HTTPException(status_code=400, detail=f"{operation} requires step_id")

    index_by_id = {str(step.get("id")): idx for idx, step in enumerate(steps)}

    if operation == "add_step":
        if not isinstance(event.step, dict):
            raise HTTPException(
                status_code=400, detail="add_step requires step payload"
            )
        if str(event.step.get("id")) != event.step_id:
            raise HTTPException(
                status_code=400, detail="add_step step.id must match step_id"
            )
        if event.step_id in index_by_id:
            raise HTTPException(
                status_code=400, detail=f"step '{event.step_id}' already exists"
            )
        insertion = len(steps) if event.index is None else min(event.index, len(steps))
        steps.insert(insertion, event.step)
    elif operation == "update_step":
        if event.step_id not in index_by_id:
            raise HTTPException(
                status_code=400, detail=f"step '{event.step_id}' not found"
            )
        if not isinstance(event.step, dict):
            raise HTTPException(
                status_code=400, detail="update_step requires step payload"
            )
        existing = dict(steps[index_by_id[event.step_id]])
        existing.update(event.step)
        existing["id"] = event.step_id
        steps[index_by_id[event.step_id]] = existing
    elif operation == "remove_step":
        if event.step_id not in index_by_id:
            raise HTTPException(
                status_code=400, detail=f"step '{event.step_id}' not found"
            )
        removed_idx = index_by_id[event.step_id]
        steps.pop(removed_idx)
        for idx, step in enumerate(steps):
            deps = [dep for dep in step.get("depends_on", []) if dep != event.step_id]
            if deps != step.get("depends_on", []):
                refreshed = dict(step)
                refreshed["depends_on"] = deps
                steps[idx] = refreshed
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported operation '{operation}'"
        )

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
    steps_payload = (
        _step_detail_payload(result.vectors, result.plan) if include_steps else None
    )
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


def _budget_consumption_from_ledger(
    ledger: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    return {key: float(values.get("consumed", 0.0)) for key, values in ledger.items()}


def _budget_utilization_from_ledger(
    ledger: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    utilization: Dict[str, float] = {}
    for key, values in ledger.items():
        budget = float(values.get("budget", 0.0))
        consumed = float(values.get("consumed", 0.0))
        utilization[key] = 0.0 if budget <= 0.0 else consumed / budget
    return utilization


def _budget_projection_from_ledger(
    ledger: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    projection: Dict[str, float] = {}
    for key, values in ledger.items():
        projection[key] = float(values.get("projected", values.get("consumed", 0.0)))
    return projection


def _budget_projection_utilization_from_ledger(
    ledger: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    utilization: Dict[str, float] = {}
    for key, values in ledger.items():
        budget = float(values.get("budget", 0.0))
        projected = float(values.get("projected", values.get("consumed", 0.0)))
        utilization[key] = 0.0 if budget <= 0.0 else projected / budget
    return utilization


def _aggregate_session_risk(
    per_agent_aggregates: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    for axis in SCOPE_AXES:
        aggregated[axis] = min(
            1.0,
            sum(float(agg.get(axis, 0.0)) for agg in per_agent_aggregates.values()),
        )
    return aggregated


def _proportional_allocations(
    requests: Dict[str, float], available: float
) -> Dict[str, float]:
    if available <= 0:
        return {agent_id: 0.0 for agent_id in requests}
    total_requested = sum(max(0.0, value) for value in requests.values())
    if total_requested <= 0:
        return {agent_id: 0.0 for agent_id in requests}
    return {
        agent_id: min(
            max(0.0, requested), available * (max(0.0, requested) / total_requested)
        )
        for agent_id, requested in requests.items()
    }


def _build_session_negotiation(
    *,
    per_agent: Dict[str, SessionAggregateDetail],
    global_ledger: Dict[str, Dict[str, float]],
) -> SessionNegotiation:
    reason_codes: List[str] = []
    recommendations: List[NegotiationBudgetRecommendation] = []

    all_budget_keys = sorted(
        {key for detail in per_agent.values() for key in detail.ledger.keys()}
        | set(global_ledger.keys())
    )

    for budget_key in all_budget_keys:
        requests = {
            agent_id: max(
                0.0, float(detail.ledger.get(budget_key, {}).get("exceeded", 0.0))
            )
            for agent_id, detail in per_agent.items()
        }
        requests = {
            agent_id: value for agent_id, value in requests.items() if value > 0.0
        }
        if not requests:
            continue

        reason_codes.append(f"agent_over_budget:{budget_key}")
        total_requested = sum(requests.values())
        global_headroom = max(
            0.0, float(global_ledger.get(budget_key, {}).get("remaining", 0.0))
        )

        headroom_allocations = _proportional_allocations(
            requests, min(global_headroom, total_requested)
        )
        unmet = {
            agent_id: max(
                0.0, requests[agent_id] - headroom_allocations.get(agent_id, 0.0)
            )
            for agent_id in requests
        }
        remaining_unmet = sum(unmet.values())

        donors = {
            agent_id: max(
                0.0, float(detail.ledger.get(budget_key, {}).get("remaining", 0.0))
            )
            for agent_id, detail in per_agent.items()
            if float(detail.ledger.get(budget_key, {}).get("remaining", 0.0)) > 0.0
        }
        transfer_capacity = min(sum(donors.values()), remaining_unmet)
        transfer_targets = _proportional_allocations(unmet, transfer_capacity)
        transfer_supply = _proportional_allocations(donors, transfer_capacity)

        transfer_records: List[NegotiationTransfer] = []
        donor_remaining = dict(transfer_supply)
        for recipient, recipient_amount in sorted(transfer_targets.items()):
            needed = recipient_amount
            if needed <= 0:
                continue
            for donor in sorted(donor_remaining):
                available = donor_remaining[donor]
                if available <= 0 or needed <= 0:
                    continue
                amount = min(available, needed)
                transfer_records.append(
                    NegotiationTransfer(
                        from_agent=donor, to_agent=recipient, amount=amount
                    )
                )
                donor_remaining[donor] -= amount
                needed -= amount

        allocated_from_headroom = sum(headroom_allocations.values())
        allocated_from_transfers = sum(record.amount for record in transfer_records)
        still_unmet = max(
            0.0, total_requested - allocated_from_headroom - allocated_from_transfers
        )
        if still_unmet > 0.0:
            reason_codes.append(f"tight_envelope:{budget_key}")

        current_budgets = {
            agent_id: float(detail.ledger.get(budget_key, {}).get("budget", 0.0))
            for agent_id, detail in per_agent.items()
        }
        total_budget = sum(current_budgets.values())
        consumption = {
            agent_id: float(detail.ledger.get(budget_key, {}).get("consumed", 0.0))
            for agent_id, detail in per_agent.items()
        }
        minimum_budget = consumption
        flex_pool = max(0.0, total_budget - sum(minimum_budget.values()))
        deficit_weights = {
            agent_id: max(
                0.0, unmet.get(agent_id, 0.0) - transfer_targets.get(agent_id, 0.0)
            )
            for agent_id in current_budgets
        }
        rebalance = _proportional_allocations(deficit_weights, flex_pool)
        target_budgets = {
            agent_id: minimum_budget[agent_id] + rebalance.get(agent_id, 0.0)
            for agent_id in current_budgets
        }
        reallocation = [
            NegotiationReallocation(
                agent_id=agent_id,
                current_budget=current_budgets[agent_id],
                target_budget=target_budgets[agent_id],
                delta=target_budgets[agent_id] - current_budgets[agent_id],
            )
            for agent_id in sorted(current_budgets)
        ]

        participant_count = len(per_agent)
        approvals = sum(
            1 for delta in (entry.delta for entry in reallocation) if delta >= 0.0
        )
        quorum_ratio = 0.67
        needed_approvals = max(1, int(participant_count * quorum_ratio + 0.999999))
        consensus_status = "reached" if approvals >= needed_approvals else "pending"

        recommendations.append(
            NegotiationBudgetRecommendation(
                budget_key=budget_key,
                fairness_rule="proportional_by_deficit",
                total_requested=total_requested,
                global_headroom=global_headroom,
                allocated_from_headroom=allocated_from_headroom,
                allocated_from_transfers=allocated_from_transfers,
                remaining_unmet=still_unmet,
                requests=[
                    NegotiationRequest(agent_id=agent_id, amount=amount)
                    for agent_id, amount in sorted(requests.items())
                ],
                transfers=transfer_records,
                reallocation=reallocation,
                consensus=NegotiationConsensus(
                    protocol="supermajority_with_non_regression",
                    quorum_ratio=quorum_ratio,
                    approvals=approvals,
                    participants=participant_count,
                    status=consensus_status,
                    note=(
                        "Consensus reached"
                        if consensus_status == "reached"
                        else "Awaiting more approvals"
                    ),
                ),
            )
        )

    return SessionNegotiation(
        triggered=bool(recommendations),
        reason_codes=sorted(set(reason_codes)),
        recommendations=recommendations,
    )


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
            (
                float(aggregate.get(axis, 0.0))
                for aggregate in per_agent_aggregates.values()
            ),
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
        suggestions.append(
            f"Consider approval or mitigating {axis} below {float(threshold):.2f}."
        )

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
        rationale = (
            recommendation.get("rationale")
            if isinstance(recommendation, dict)
            else None
        )
        if rationale:
            suggestions.append(f"Knee recommendation: {rationale}")

    if not suggestions:
        suggestions.append("Proceed; plan appears proportionate to the contract.")
    return suggestions[:5]


def _suggest_plan_patch(policy, plan: PlanDAG) -> List[Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []

    knee_recommendations = []
    if getattr(policy, "policy_input", None) is not None and isinstance(
        policy.policy_input.metadata, dict
    ):
        knee_recommendations = policy.policy_input.metadata.get(
            "knee_plan_patch_recommendations", []
        )
    for recommendation in knee_recommendations:
        if isinstance(recommendation, dict):
            patch = recommendation.get("patch")
            if isinstance(patch, dict):
                patches.append(patch)

    triggered = set(policy.exceeded.keys()) | set(policy.asked.keys())
    if "read_before_write" in policy.asked:
        first_write = next(
            (step for step in plan.steps if step.tool in SWE_WRITE_TOOLS), None
        )
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
        first_write = next(
            (step for step in plan.steps if step.tool in SWE_WRITE_TOOLS), None
        )
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
                if (step.tool_category or "")
                in {"infra", "payments", "finance", "health"}
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
                or any(
                    keyword in step.description.lower()
                    for keyword in ("delete", "destroy", "drop", "rotate")
                )
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
    contract: TaskContract,
    plan: PlanDAG,
    policy,
    ask_action: Optional[str],
    outcome: Optional[str],
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




def _policy_backend_assets(policy_backend: str) -> Dict[str, Optional[str]]:
    base = Path(__file__).resolve().parents[1] / "policy"
    if policy_backend == "opa":
        rego = base / "opa" / "policy.rego"
        return {"rego": str(rego), "cedar": None, "schema": None}
    if policy_backend == "cedar":
        cedar = base / "cedar" / "policy.cedar"
        schema = base / "cedar" / "schema.json"
        return {"rego": None, "cedar": str(cedar), "schema": str(schema)}
    return {"rego": None, "cedar": None, "schema": None}


def _rule_threshold_overrides(rules: List[PolicyRuleProposal]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for rule in rules:
        axis = (rule.axis or "").strip()
        op = (rule.operator or "").strip()
        if axis and axis in SCOPE_AXES and op in {">", ">="} and rule.value is not None:
            overrides[f"max_{axis}"] = float(rule.value)
    return overrides


def _apply_workbench_overrides(contract_payload: Dict[str, Any], threshold_overrides: Dict[str, float], rules: List[PolicyRuleProposal]) -> TaskContract:
    patched = json.loads(json.dumps(contract_payload))
    thresholds = patched.get("thresholds")
    if not isinstance(thresholds, dict):
        thresholds = {}
        patched["thresholds"] = thresholds
    merged = dict(threshold_overrides)
    merged.update(_rule_threshold_overrides(rules))
    for key, value in merged.items():
        if key.startswith("max_") and isinstance(value, (int, float)):
            thresholds[key] = float(value)
    return TaskContract.model_validate(patched)


def _authorized_policy_editor(token: Optional[str]) -> tuple[bool, str]:
    expected = os.getenv("SCOPEBENCH_POLICY_EDITOR_TOKEN", "").strip()
    if not expected:
        return False, "SCOPEBENCH_POLICY_EDITOR_TOKEN is not configured"
    if not token:
        return False, "missing X-ScopeBench-Policy-Token header"
    if token.strip() != expected:
        return False, "invalid policy editor token"
    return True, "authorized"
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


def _scope_vector_from_expected_payload(payload: Dict[str, Any]) -> ScopeVector:
    return ScopeVector(
        step_id=str(payload.get("step_id", "")) or None,
        spatial=AxisScore(value=float(payload["spatial"])),
        temporal=AxisScore(value=float(payload["temporal"])),
        depth=AxisScore(value=float(payload["depth"])),
        irreversibility=AxisScore(value=float(payload["irreversibility"])),
        resource_intensity=AxisScore(value=float(payload["resource_intensity"])),
        legal_exposure=AxisScore(value=float(payload["legal_exposure"])),
        dependency_creation=AxisScore(value=float(payload["dependency_creation"])),
        stakeholder_radius=AxisScore(value=float(payload["stakeholder_radius"])),
        power_concentration=AxisScore(value=float(payload["power_concentration"])),
        uncertainty=AxisScore(value=float(payload["uncertainty"])),
    )


def _build_cases_analytics(
    path: Path, plugin_cases: List[Any]
) -> CasesAnalyticsResponse:
    from collections import Counter, defaultdict

    from scopebench.bench.dataset import load_cases

    builtin_cases = load_cases(path)
    all_cases = list(builtin_cases) + list(plugin_cases)

    decision_by_domain: Dict[str, Counter] = defaultdict(Counter)
    axis_trigger_counts: Dict[str, Counter] = {
        axis: Counter({"ASK": 0, "DENY": 0}) for axis in SCOPE_AXES
    }
    axis_effect_total = {axis: 0.0 for axis in SCOPE_AXES}
    axis_threshold_total = {axis: 0.0 for axis in SCOPE_AXES}
    axis_margin_total = {axis: 0.0 for axis in SCOPE_AXES}
    axis_over_threshold = Counter({axis: 0 for axis in SCOPE_AXES})
    dominant_axes_by_decision: Dict[str, Counter] = {
        decision: Counter() for decision in ("ALLOW", "ASK", "DENY")
    }
    effect_magnitude_counts: Dict[tuple[str, str], Counter] = defaultdict(Counter)

    for case in all_cases:
        decision = str(case.expected_decision)
        decision_by_domain[case.domain][decision] += 1

        contract = TaskContract.model_validate(case.contract)
        plan = plan_from_dict(case.plan)
        vectors = [
            _scope_vector_from_expected_payload(row)
            for row in case.expected_step_vectors
        ]
        aggregate_dict = aggregate_scope(vectors, plan).as_dict()

        thresholds = {
            "spatial": contract.thresholds.max_spatial,
            "temporal": contract.thresholds.max_temporal,
            "depth": contract.thresholds.max_depth,
            "irreversibility": contract.thresholds.max_irreversibility,
            "resource_intensity": contract.thresholds.max_resource_intensity,
            "legal_exposure": contract.thresholds.max_legal_exposure,
            "dependency_creation": contract.thresholds.max_dependency_creation,
            "stakeholder_radius": contract.thresholds.max_stakeholder_radius,
            "power_concentration": contract.thresholds.max_power_concentration,
            "uncertainty": contract.thresholds.max_uncertainty,
        }

        top_axis = max(SCOPE_AXES, key=lambda axis: float(aggregate_dict[axis]))
        dominant_axes_by_decision.setdefault(decision, Counter())[top_axis] += 1

        for step in getattr(plan, "steps", []):
            effects = step.effects
            if effects is None:
                continue
            effects_dict = effects.model_dump(mode="python", exclude_none=True)
            for effect_type, payload in effects_dict.items():
                if effect_type == "version" or not isinstance(payload, dict):
                    continue
                magnitude = payload.get("magnitude")
                if isinstance(magnitude, str):
                    effect_magnitude_counts[(effect_type, magnitude)][decision] += 1

        ask_threshold = float(contract.escalation.ask_if_any_axis_over)
        ask_uncertainty = float(contract.escalation.ask_if_uncertainty_over)
        for axis in SCOPE_AXES:
            effect = float(aggregate_dict[axis])
            threshold = float(thresholds[axis])
            margin = effect - threshold

            axis_effect_total[axis] += effect
            axis_threshold_total[axis] += threshold
            axis_margin_total[axis] += margin
            if margin > 0:
                axis_over_threshold[axis] += 1

            if decision == "DENY" and margin > 0:
                axis_trigger_counts[axis]["DENY"] += 1
            if decision == "ASK" and (
                effect > ask_threshold
                or (axis == "uncertainty" and effect > ask_uncertainty)
            ):
                axis_trigger_counts[axis]["ASK"] += 1

    domain_entries: List[DomainDecisionAnalytics] = []
    for domain, counts in sorted(decision_by_domain.items()):
        total = sum(counts.values())
        decision_counts = {
            label: int(counts.get(label, 0)) for label in ("ALLOW", "ASK", "DENY")
        }
        decision_rates = {
            label: (decision_counts[label] / total if total else 0.0)
            for label in decision_counts
        }
        domain_entries.append(
            DomainDecisionAnalytics(
                domain=domain,
                total_cases=total,
                decision_counts=decision_counts,
                decision_rates=decision_rates,
            )
        )

    trigger_axes = [
        AxisTriggerAnalytics(
            axis=axis,
            ask_count=int(axis_trigger_counts[axis]["ASK"]),
            deny_count=int(axis_trigger_counts[axis]["DENY"]),
        )
        for axis in SCOPE_AXES
    ]

    top_trigger_axes_by_decision: List[TopTriggerAxesByDecision] = []
    for decision in ("ASK", "DENY"):
        counts = dominant_axes_by_decision.get(decision, Counter())
        total = int(sum(counts.values()))
        top_axes = [
            DecisionAxisLeader(
                axis=axis,
                count=int(count),
                rate_within_decision=(count / total if total else 0.0),
            )
            for axis, count in counts.most_common(5)
        ]
        top_trigger_axes_by_decision.append(
            TopTriggerAxesByDecision(
                decision=decision,
                total_cases=total,
                top_axes=top_axes,
            )
        )

    effect_magnitude_profiles: List[EffectMagnitudeProfile] = []
    for (effect_type, magnitude), counts in sorted(effect_magnitude_counts.items()):
        total = int(sum(counts.values()))
        decision_counts = {
            label: int(counts.get(label, 0)) for label in ("ALLOW", "ASK", "DENY")
        }
        decision_rates = {
            label: (decision_counts[label] / total if total else 0.0)
            for label in decision_counts
        }
        effect_magnitude_profiles.append(
            EffectMagnitudeProfile(
                effect_type=effect_type,
                magnitude=magnitude,
                count=total,
                decision_counts=decision_counts,
                decision_rates=decision_rates,
            )
        )

    total_cases = len(all_cases)
    effect_entries = [
        EffectThresholdAnalytics(
            axis=axis,
            average_effect=(
                axis_effect_total[axis] / total_cases if total_cases else 0.0
            ),
            average_threshold=(
                axis_threshold_total[axis] / total_cases if total_cases else 0.0
            ),
            average_margin=(
                axis_margin_total[axis] / total_cases if total_cases else 0.0
            ),
            over_threshold_cases=int(axis_over_threshold[axis]),
            over_threshold_rate=(
                axis_over_threshold[axis] / total_cases if total_cases else 0.0
            ),
        )
        for axis in SCOPE_AXES
    ]

    return CasesAnalyticsResponse(
        source=str(path),
        count=total_cases,
        decision_distribution_by_domain=domain_entries,
        trigger_axes=trigger_axes,
        effect_magnitude_vs_threshold=effect_entries,
        top_trigger_axes_by_decision=top_trigger_axes_by_decision,
        effect_magnitude_profiles=effect_magnitude_profiles,
    )


def _build_calibration_dashboard(
    path: Path,
    calibration_overrides: Optional[Dict[str, CalibratedDecisionThresholds]] = None,
) -> CalibrationDashboardResponse:
    domain_payload = compute_domain_calibration_from_telemetry(path)
    rows = _load_telemetry_rows(path, limit=0)
    rows_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_domain.setdefault(_dashboard_row_domain(row), []).append(row)

    preset_thresholds = _preset_thresholds()

    entries: List[CalibrationDashboardEntry] = []
    for domain, (base_calibration, stats) in sorted(domain_payload.items()):
        calibration = (
            calibration_overrides.get(domain, base_calibration)
            if calibration_overrides
            else base_calibration
        )
        domain_rows = rows_by_domain.get(domain, [])
        distributions: Dict[str, Dict[str, Any]] = {}
        telemetry_delta: Dict[str, Dict[str, Any]] = {}
        rates: Dict[str, Dict[str, float]] = {}
        for axis in SCOPE_AXES:
            values = [_dashboard_axis_signal(row, axis) for row in domain_rows]
            values = [value for value in values if value > 0]
            histogram = [0] * 10
            for value in values:
                bucket = min(9, int(value * 10))
                histogram[bucket] += 1
            quantiles = {"p50": 0.0, "p90": 0.0, "p95": 0.0}
            if values:
                sorted_values = sorted(values)

                def percentile(p: float) -> float:
                    index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p))))
                    return float(sorted_values[index])

                quantiles = {
                    "p50": percentile(0.5),
                    "p90": percentile(0.9),
                    "p95": percentile(0.95),
                }

            distributions[axis] = {
                "samples": len(values),
                "histogram": histogram,
                "quantiles": quantiles,
            }

            false_alarms = stats.false_alarms.get(axis, 0)
            overrides = stats.overrides.get(axis, 0)
            triggered = max(1, stats.triggered.get(axis, 0))
            rates[axis] = {
                "false_alarm_rate": float(false_alarms) / float(triggered),
                "override_rate": float(overrides) / float(triggered),
            }

            telemetry_delta[axis] = {
                "axis_scale_delta": calibration.resolved_axis_scale()[axis] - 1.0,
                "threshold_factor_delta": calibration.resolved_axis_threshold_factor()[axis] - 1.0,
                "axis_bias": calibration.resolved_axis_bias()[axis],
            }

        calibrated_thresholds: Dict[str, Dict[str, float]] = {}
        threshold_factors = calibration.resolved_axis_threshold_factor()
        for preset, thresholds in preset_thresholds.items():
            calibrated_thresholds[preset] = {
                axis: max(0.0, min(1.0, threshold * threshold_factors.get(axis, 1.0)))
                for axis, threshold in thresholds.items()
            }

        entry_calibration = calibration_to_dict(calibration)
        entry_calibration["preset_thresholds"] = {
            "base": preset_thresholds,
            "calibrated": calibrated_thresholds,
        }
        entry_calibration["axis_distributions"] = distributions
        entry_calibration["rates"] = rates
        entry_calibration["telemetry_delta"] = telemetry_delta

        entries.append(
            CalibrationDashboardEntry(
                domain=domain,
                runs=stats.runs,
                calibration=entry_calibration,
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

    calibration = (
        apply_manual_adjustments(calibration, req.calibration_manual_adjustments)
        if calibration
        else calibration
    )
    return calibration


def create_app(
    default_policy_backend: str = "python", telemetry_jsonl_path: Optional[str] = None
) -> FastAPI:
    init_tracing(enable_console=False)
    tracer = get_tracer("scopebench")
    app = FastAPI(title="ScopeBench", version="0.1.0")
    configured_telemetry_path = telemetry_jsonl_path or os.getenv(
        "SCOPEBENCH_TELEMETRY_JSONL_PATH"
    )
    runtime_plugin_dirs = [
        p.strip() for p in os.getenv("SCOPEBENCH_PLUGIN_DIRS", "").split(os.pathsep) if p.strip()
    ]
    plugin_manager = PluginManager.from_dirs(
        runtime_plugin_dirs,
        PluginManager._load_keyring(os.getenv("SCOPEBENCH_PLUGIN_KEYS_JSON", "")),
    )

    dataset_review_state: Dict[str, Dict[str, Any]] = {}

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _empty_review(case_id: str, draft_case: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "case_id": case_id,
            "draft_case": draft_case,
            "comments": [],
            "suggested_edits": [],
            "votes": {},
            "acceptance": {
                "accept": 0,
                "reject": 0,
                "abstain": 0,
                "total": 0,
                "ready": False,
                "status": "pending",
            },
            "updated_utc": _utc_now_iso(),
        }

    def _refresh_acceptance(review: Dict[str, Any]) -> None:
        votes = review.get("votes", {})
        accept = sum(1 for value in votes.values() if value == "accept")
        reject = sum(1 for value in votes.values() if value == "reject")
        abstain = sum(1 for value in votes.values() if value == "abstain")
        total = len(votes)
        ready = total >= 2 and accept > reject
        status = "accepted" if ready else ("rejected" if total >= 2 and reject >= accept else "pending")
        review["acceptance"] = {
            "accept": accept,
            "reject": reject,
            "abstain": abstain,
            "total": total,
            "ready": ready,
            "status": status,
        }
        review["updated_utc"] = _utc_now_iso()

    def _ensure_review(case_id: str, draft_case: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        review = dataset_review_state.get(case_id)
        if review is None:
            review = _empty_review(case_id, draft_case=draft_case)
            dataset_review_state[case_id] = review
        elif draft_case is not None:
            review["draft_case"] = draft_case
            review["updated_utc"] = _utc_now_iso()
        return review

    def _reload_plugins() -> None:
        nonlocal plugin_manager
        plugin_manager = PluginManager.from_dirs(
            runtime_plugin_dirs,
            PluginManager._load_keyring(os.getenv("SCOPEBENCH_PLUGIN_KEYS_JSON", "")),
        )

    def _sync_plugin_dirs_env() -> None:
        os.environ["SCOPEBENCH_PLUGIN_DIRS"] = os.pathsep.join(runtime_plugin_dirs)

    plugin_reviews: Dict[str, List[Dict[str, Any]]] = {}

    def _plugin_key(publisher: Any, bundle_name: Any) -> str:
        return f"{str(publisher or 'community').strip()}::{str(bundle_name or '').strip()}"

    def _scan_bundle_security(bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(bundle, dict):
            return {
                "status": "missing",
                "issues": ["Bundle not installed or not discoverable in runtime plugin directories."],
                "recommendations": ["Install bundle locally before running security scan."],
            }
        signed = bool(bundle.get("signed"))
        signature_valid = bool(bundle.get("signature_valid"))
        policy_rules_count = int(bundle.get("policy_rules_count") or 0)
        issues: List[str] = []
        if policy_rules_count and not signed:
            issues.append("Bundle defines policy_rules but is unsigned.")
        if policy_rules_count and signed and not signature_valid:
            issues.append("Bundle signature verification failed for policy_rules.")
        if signed and not signature_valid and bundle.get("signature_error"):
            issues.append(f"signature_error: {bundle.get('signature_error')}")
        status = "pass" if not issues else "fail"
        recommendations: List[str] = []
        if issues:
            recommendations.append("Re-sign bundle with a trusted key and verify SCOPEBENCH_PLUGIN_KEYS_JSON configuration.")
            recommendations.append("Avoid loading policy_rules from unsigned or unverifiable bundles.")
        else:
            recommendations.append("Signed policy rules verified successfully.")
        return {
            "status": status,
            "signed": signed,
            "signature_valid": signature_valid,
            "policy_rules_count": policy_rules_count,
            "issues": issues,
            "recommendations": recommendations,
        }

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ui", response_class=HTMLResponse)
    def ui_index():
        from pathlib import Path

        ui_path = Path(__file__).resolve().parents[1] / "ui" / "index.html"
        return ui_path.read_text(encoding="utf-8")

    @app.get("/plugin_marketplace")
    def plugin_marketplace_endpoint():
        from pathlib import Path
        import yaml

        marketplace_path = (
            Path(__file__).resolve().parents[2] / "docs" / "plugin_marketplace.yaml"
        )
        if not marketplace_path.exists():
            return {
                "plugins": [],
                "count": 0,
                "source": str(marketplace_path),
                "error": "marketplace file not found",
            }
        payload = yaml.safe_load(marketplace_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            return {
                "plugins": [],
                "count": 0,
                "source": str(marketplace_path),
                "error": "invalid marketplace format",
            }
        domains = (
            payload.get("domains") if isinstance(payload.get("domains"), list) else []
        )
        rows = [row for row in domains if isinstance(row, dict)]

        def _proportionality_signals(
            row_payload: Dict[str, Any],
            installed_bundle: Optional[Dict[str, Any]],
            rating_value: Optional[float],
            security_scan_payload: Dict[str, Any],
        ) -> Dict[str, Any]:
            warnings: List[str] = []
            policy_rules_count = int((installed_bundle or {}).get("policy_rules_count") or 0)
            effects_mappings_count = int((installed_bundle or {}).get("effects_mappings_count") or 0)
            risk_classes = {
                str(item).strip().lower()
                for item in (row_payload.get("risk_classes") or [])
                if str(item).strip()
            }
            if security_scan_payload.get("status") == "fail":
                warnings.append(
                    "Security scan failed: policy contributions may bypass trusted proportionality controls."
                )
            if effects_mappings_count > 0 and policy_rules_count == 0:
                warnings.append(
                    "Effects mappings are present without policy_rules safeguards; review proportionality constraints before install."
                )
            if policy_rules_count > 3 and not (installed_bundle or {}).get("signature_valid"):
                warnings.append(
                    "Multiple policy rules are present but signature validation is missing/invalid."
                )
            if risk_classes.intersection({"high", "critical"}) and policy_rules_count == 0:
                warnings.append(
                    "High-impact risk classes listed without plugin policy guardrails; require manual review."
                )
            if rating_value is not None and rating_value < 3.0:
                warnings.append(
                    "Community rating is below 3.0★; inspect policy/effects behavior for disproportionate side effects."
                )
            status = "warn" if warnings else "ok"
            return {
                "status": status,
                "warnings": warnings,
                "policy_rules_count": policy_rules_count,
                "effects_mappings_count": effects_mappings_count,
            }

        installed_by_bundle = {
            f"{item.get('publisher')}::{item.get('name')}": item for item in plugin_manager.bundles_payload()
        }
        enriched_rows = []
        trust_totals = {"reviews": 0, "ratings": 0.0}
        for row in rows:
            bundle_name = row.get("plugin_bundle")
            publisher = row.get("publisher", row.get("maintainer", "community"))
            key = _plugin_key(publisher, bundle_name)
            installed = installed_by_bundle.get(key)
            installed_tools = (
                installed.get("tools")
                if installed and isinstance(installed.get("tools"), dict)
                else {}
            )
            discovered_risk_classes = sorted(
                {
                    str(tool.get("risk_class")).strip().lower()
                    for tool in installed_tools.values()
                    if isinstance(tool, dict)
                    and isinstance(tool.get("risk_class"), str)
                    and tool.get("risk_class").strip()
                }
            )
            listed_risk_classes = row.get("risk_classes")
            risk_classes = (
                listed_risk_classes
                if isinstance(listed_risk_classes, list) and listed_risk_classes
                else discovered_risk_classes
            )
            usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
            trust_seed = row.get("trust") if isinstance(row.get("trust"), dict) else {}
            ratings_seed = (
                trust_seed.get("ratings") if isinstance(trust_seed.get("ratings"), dict) else {}
            )
            reviews = plugin_reviews.get(key, [])
            review_count = len(reviews)
            seeded_count = int(ratings_seed.get("count") or 0)
            seeded_avg = (
                round(float(ratings_seed.get("average")), 2)
                if isinstance(ratings_seed.get("average"), (int, float))
                else None
            )
            live_rating_total = sum(float(item.get("rating", 0)) for item in reviews)
            seeded_rating_total = (seeded_avg * seeded_count) if seeded_avg is not None else 0.0
            combined_count = review_count + seeded_count
            avg_rating = (
                round((live_rating_total + seeded_rating_total) / combined_count, 2)
                if combined_count
                else None
            )
            trust_totals["reviews"] += review_count
            trust_totals["ratings"] += sum(float(item.get("rating", 0)) for item in reviews)
            security_scan = _scan_bundle_security(installed)
            seeded_scan = (
                trust_seed.get("security_scan")
                if isinstance(trust_seed.get("security_scan"), dict)
                else None
            )
            if security_scan.get("status") == "unknown" and seeded_scan:
                security_scan = {
                    **security_scan,
                    **seeded_scan,
                }
            proportionality = _proportionality_signals(row, installed, avg_rating, security_scan)
            enriched_rows.append(
                {
                    **row,
                    "domain_focus": row.get("domain_focus") or row.get("title") or row.get("slug"),
                    "description": row.get("description") or row.get("summary") or "",
                    "risk_classes": risk_classes,
                    "version": row.get("version") or (installed.get("version") if installed else None),
                    "signature_status": (
                        "valid"
                        if installed and installed.get("signed") and installed.get("signature_valid")
                        else "unsigned_or_unverified"
                    ),
                    "installed": installed is not None,
                    "installed_bundle": installed,
                    "usage": {
                        "downloads_30d": int(usage.get("downloads_30d") or 0),
                        "active_installs": int(usage.get("active_installs") or 0),
                        "invocations_7d": int(usage.get("invocations_7d") or 0),
                    },
                    "trust": {
                        "average_rating": avg_rating,
                        "review_count": combined_count,
                        "live_review_count": review_count,
                        "seeded_review_count": seeded_count,
                        "recent_reviews": reviews[-5:],
                        "security_scan": security_scan,
                        "proportionality": proportionality,
                    },
                }
            )
        trust_summary = {
            "total_reviews": trust_totals["reviews"],
            "average_rating": round(trust_totals["ratings"] / trust_totals["reviews"], 2) if trust_totals["reviews"] else None,
            "scanned_plugins": len(enriched_rows),
            "failed_scans": sum(1 for item in enriched_rows if item.get("trust", {}).get("security_scan", {}).get("status") == "fail"),
        }
        return {
            "plugins": enriched_rows,
            "count": len(enriched_rows),
            "source": str(marketplace_path),
            "version": payload.get("version"),
            "updated_utc": payload.get("updated_utc"),
            "trust_summary": trust_summary,
        }

    @app.post("/plugin_marketplace/review")
    def plugin_marketplace_review_endpoint(payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return {"ok": False, "error": "payload must be an object"}
        bundle_name = str(payload.get("plugin_bundle") or "").strip()
        publisher = str(payload.get("publisher") or "community").strip()
        rating = payload.get("rating")
        if not bundle_name:
            return {"ok": False, "error": "plugin_bundle is required"}
        if not isinstance(rating, (int, float)) or not (1 <= float(rating) <= 5):
            return {"ok": False, "error": "rating must be between 1 and 5"}
        review = {
            "reviewer": str(payload.get("reviewer") or "anonymous").strip() or "anonymous",
            "rating": float(rating),
            "comment": str(payload.get("comment") or "").strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        key = _plugin_key(publisher, bundle_name)
        plugin_reviews.setdefault(key, []).append(review)
        reviews = plugin_reviews[key]
        avg_rating = round(sum(item["rating"] for item in reviews) / len(reviews), 2)
        return {
            "ok": True,
            "plugin_bundle": bundle_name,
            "publisher": publisher,
            "trust": {
                "average_rating": avg_rating,
                "review_count": len(reviews),
                "recent_reviews": reviews[-5:],
            },
        }

    @app.post("/plugins/security_scan")
    def plugins_security_scan_endpoint(payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return {"ok": False, "error": "payload must be an object"}
        bundle_name = str(payload.get("plugin_bundle") or "").strip()
        publisher = str(payload.get("publisher") or "community").strip()
        if not bundle_name:
            return {"ok": False, "error": "plugin_bundle is required"}
        installed_by_bundle = {
            f"{item.get('publisher')}::{item.get('name')}": item for item in plugin_manager.bundles_payload()
        }
        bundle = installed_by_bundle.get(_plugin_key(publisher, bundle_name))
        scan = _scan_bundle_security(bundle)
        return {
            "ok": scan.get("status") != "missing",
            "plugin_bundle": bundle_name,
            "publisher": publisher,
            "security_scan": scan,
        }

    @app.get("/plugins/schema")
    def plugins_schema_endpoint():
        return {
            "required": ["name", "version", "publisher"],
            "version_pattern": "MAJOR.MINOR.PATCH",
            "risk_classes": ["low", "moderate", "high", "critical"],
            "contribution_keys": ["tool_categories", "effects_mappings", "scoring_axes", "policy_rules"],
            "notes": [
                "policy_rules load only for valid signed bundles",
                "effects_mappings axes should use built-in scope axes in [0,1]",
            ],
        }

    @app.post("/plugins/lint")
    def plugins_lint_endpoint(payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return {"ok": False, "errors": ["payload must be an object"]}
        errors = lint_plugin_bundle(payload)
        return {"ok": not errors, "errors": errors}

    @app.post("/plugins/wizard/generate", response_model=PluginWizardResponse)
    def plugins_wizard_generate(payload: Dict[str, Any]):
        req = PluginWizardRequest.model_validate(payload)
        tool_category = f"{req.domain}_operations"
        policy_rules = [
            {
                "id": f"{req.domain}.template_{idx+1}",
                "when": {"tool_category": tool_category},
                "action": "ASK",
                "template": template,
            }
            for idx, template in enumerate(req.policy_rule_templates)
        ]
        mappings = []
        for idx, mapping in enumerate(req.effects_mappings):
            trigger = mapping.get("trigger") if isinstance(mapping, dict) else None
            axes = mapping.get("axes") if isinstance(mapping, dict) else None
            if isinstance(trigger, str) and isinstance(axes, dict):
                mappings.append({"trigger": trigger, "axes": axes})
            else:
                mappings.append({"trigger": f"tool_{idx+1}", "axes": {"uncertainty": 0.4}})

        explicit_tools: Dict[str, Dict[str, Any]] = {}
        for row in req.tool_definitions:
            if not isinstance(row, dict):
                continue
            name = row.get("tool")
            if not isinstance(name, str) or not name.strip():
                continue
            explicit_tools[name.strip()] = {
                "category": str(row.get("category") or tool_category),
                "domains": row.get("domains") if isinstance(row.get("domains"), list) else [req.domain],
                "risk_class": str(row.get("risk_class") or "moderate"),
                "priors": row.get("priors") if isinstance(row.get("priors"), dict) else {"uncertainty": 0.3},
            }

        for tool in req.tools:
            if tool not in explicit_tools:
                explicit_tools[tool] = {
                    "category": tool_category,
                    "domains": [req.domain],
                    "risk_class": "moderate",
                    "priors": {"uncertainty": 0.3, "dependency_creation": 0.2},
                }

        bundle = {
            "name": req.name,
            "version": req.version,
            "publisher": req.publisher,
            "contributions": {
                "tool_categories": {tool_category: {"description": f"Operations for {req.domain}"}},
                "effects_mappings": mappings,
                "scoring_axes": {f"{req.domain}_safety": {"description": f"Safety profile for {req.domain}"}},
                "policy_rules": policy_rules,
            },
            "tools": explicit_tools,
            "cases": [],
        }
        errors = lint_plugin_bundle(bundle)
        signed = sign_plugin_bundle(bundle, key_id=req.key_id, secret=req.secret) if not errors else bundle

        harness: Dict[str, Any] = {"passed": False, "reason": "lint_failed"}
        if not errors:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                yaml.safe_dump(signed, tmp, sort_keys=False)
                tmp_path = Path(tmp.name)
            try:
                harness = run_plugin_test_harness(
                    tmp_path,
                    keys_json=json.dumps({req.key_id: req.secret}),
                    golden_cases_path=default_cases_path(),
                    max_golden_cases=3,
                ).to_dict()
            finally:
                tmp_path.unlink(missing_ok=True)

        guidance = [
            "Run /plugins/lint and plugin-harness before publishing.",
            "Install locally via /plugins/install to validate runtime loading.",
            "Publish signed bundle in an immutable release artifact.",
            "Submit plugin listing update to docs/plugin_marketplace.yaml.",
        ]
        return PluginWizardResponse(ok=not errors, lint_errors=errors, bundle=signed, harness=harness, publish_guidance=guidance)

    @app.get("/plugins")
    def plugins_endpoint():
        bundles = plugin_manager.bundles_payload()
        return {
            "plugins": bundles,
            "count": len(bundles),
            "configured_plugin_dirs": runtime_plugin_dirs,
            "merged_environment": {
                "tools": len(plugin_manager.tools),
                "cases": len(plugin_manager.cases),
                "tool_categories": len(plugin_manager.tool_categories),
                "effects_mappings": len(plugin_manager.effects_mappings),
                "scoring_axes": len(plugin_manager.scoring_axes),
                "policy_rules": len(plugin_manager.policy_rules),
            },
        }

    @app.post("/plugins/install")
    def plugins_install_endpoint(payload: Dict[str, Any]):
        source_path = payload.get("source_path") if isinstance(payload, dict) else None
        plugin_dir = payload.get("plugin_dir") if isinstance(payload, dict) else None
        if not isinstance(source_path, str) or not source_path.strip():
            return {"ok": False, "error": "source_path is required"}
        if not isinstance(plugin_dir, str) or not plugin_dir.strip():
            plugin_dir = ".scopebench/plugins"

        src = Path(source_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            return {"ok": False, "error": f"source_path not found: {src}"}

        target_dir = Path(plugin_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        if str(target_dir) not in runtime_plugin_dirs:
            runtime_plugin_dirs.append(str(target_dir))
            _sync_plugin_dirs_env()

        _reload_plugins()
        installed = next(
            (
                item
                for item in plugin_manager.bundles_payload()
                if Path(item.get("source_path", "")).name == src.name
            ),
            None,
        )
        if installed is None:
            target.unlink(missing_ok=True)
            _reload_plugins()
            return {"ok": False, "error": "bundle failed to load"}
        return {
            "ok": True,
            "installed": installed,
            "target_path": str(target),
            "configured_plugin_dirs": runtime_plugin_dirs,
            "merged_environment": {
                "tools": len(plugin_manager.tools),
                "cases": len(plugin_manager.cases),
            },
        }

    @app.post("/plugins/uninstall")
    def plugins_uninstall_endpoint(payload: Dict[str, Any]):
        source_path = payload.get("source_path") if isinstance(payload, dict) else None
        if not isinstance(source_path, str) or not source_path.strip():
            return {"ok": False, "error": "source_path is required"}
        target = Path(source_path).expanduser().resolve()
        if not target.exists():
            return {"ok": False, "error": f"bundle not found: {target}"}
        target.unlink()
        _reload_plugins()
        return {
            "ok": True,
            "removed": str(target),
            "configured_plugin_dirs": runtime_plugin_dirs,
            "merged_environment": {
                "tools": len(plugin_manager.tools),
                "cases": len(plugin_manager.cases),
            },
        }

    @app.get("/templates")
    def templates_endpoint():
        from pathlib import Path
        import yaml

        templates_root = Path(__file__).resolve().parents[1] / "templates"
        payload: List[Dict[str, Any]] = []
        for domain_dir in sorted(p for p in templates_root.iterdir() if p.is_dir()):
            variants: Dict[str, Dict[str, Any]] = {}

            def load_variant(
                variant_name: str,
                contract_path: Path,
                plan_path: Path,
                notes_path: Path,
            ) -> None:
                variants[variant_name] = {
                    "metadata": {
                        "has_contract": contract_path.exists(),
                        "has_plan": plan_path.exists(),
                        "has_notes": notes_path.exists(),
                    },
                    "content": {
                        "contract": (
                            yaml.safe_load(contract_path.read_text(encoding="utf-8"))
                            if contract_path.exists()
                            else None
                        ),
                        "plan": (
                            yaml.safe_load(plan_path.read_text(encoding="utf-8"))
                            if plan_path.exists()
                            else None
                        ),
                        "notes": (
                            notes_path.read_text(encoding="utf-8")
                            if notes_path.exists()
                            else None
                        ),
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
        merged_tools = dict(registry._tools)  # noqa: SLF001
        merged_tools.update(plugin_manager.tools)

        tools = []
        for tool_name, tool_info in sorted(merged_tools.items()):
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
            "required": [
                "tool",
                "category",
                "domains",
                "risk_class",
                "priors",
                "default_effects",
            ],
            "properties": {
                "tool": {"type": "string"},
                "category": {"type": "string"},
                "domains": {"type": "array", "items": {"type": "string"}},
                "risk_class": {
                    "type": "string",
                    "enum": ["low", "moderate", "high", "critical"],
                },
                "priors": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "default_effects": {"type": "object"},
            },
            "additionalProperties": False,
        }
        return {
            "tools": tools,
            "count": len(tools),
            "normalized_schema": schema,
            "extensions": {
                "tool_categories": plugin_manager.tool_categories,
                "effects_mappings": plugin_manager.effects_mappings,
                "scoring_axes": plugin_manager.scoring_axes,
                "policy_rules": plugin_manager.policy_rules,
            },
            "plugins": plugin_manager.bundles_payload(),
        }

    @app.get("/policy/workbench", response_model=PolicyWorkbenchStateResponse)
    def policy_workbench_state(policy_backend: str = default_policy_backend, x_scopebench_policy_token: Optional[str] = Header(default=None)):
        authorized, message = _authorized_policy_editor(x_scopebench_policy_token)
        baseline_contract = TaskContract(goal="Policy tuning sandbox")
        assets = _policy_backend_assets(policy_backend)
        return PolicyWorkbenchStateResponse(
            policy_backend=policy_backend,
            thresholds={
                "max_spatial": baseline_contract.thresholds.max_spatial,
                "max_temporal": baseline_contract.thresholds.max_temporal,
                "max_depth": baseline_contract.thresholds.max_depth,
                "max_irreversibility": baseline_contract.thresholds.max_irreversibility,
                "max_resource_intensity": baseline_contract.thresholds.max_resource_intensity,
                "max_legal_exposure": baseline_contract.thresholds.max_legal_exposure,
                "max_dependency_creation": baseline_contract.thresholds.max_dependency_creation,
                "max_stakeholder_radius": baseline_contract.thresholds.max_stakeholder_radius,
                "max_power_concentration": baseline_contract.thresholds.max_power_concentration,
                "max_uncertainty": baseline_contract.thresholds.max_uncertainty,
            },
            escalation={
                "ask_if_any_axis_over": baseline_contract.escalation.ask_if_any_axis_over,
                "ask_if_uncertainty_over": baseline_contract.escalation.ask_if_uncertainty_over,
                "ask_if_tool_category_in": sorted(baseline_contract.escalation.ask_if_tool_category_in),
            },
            backend_assets=assets,
            signed_policy_rules=[dict(rule) for rule in plugin_manager.policy_rules],
            authorization={"authorized": authorized, "message": message},
        )

    @app.post("/policy/workbench/test")
    def policy_workbench_test(req: PolicyWorkbenchTestRequest):
        contract = _apply_workbench_overrides(req.contract, req.threshold_overrides, req.proposed_rules)
        plan = PlanDAG.model_validate(req.plan)
        result = evaluate(contract, plan, policy_backend=req.policy_backend)
        applied = dict(req.threshold_overrides)
        applied.update(_rule_threshold_overrides(req.proposed_rules))
        return {
            "decision": result.policy.decision.value,
            "policy_backend": result.policy.policy_backend,
            "policy_version": result.policy.policy_version,
            "policy_hash": result.policy.policy_hash,
            "reasons": result.policy.reasons,
            "aggregate": result.aggregate.as_dict(),
            "asked": {key: float(value) for key, value in result.policy.asked.items()},
            "exceeded": {
                key: {"value": float(values[0]), "threshold": float(values[1])}
                for key, values in result.policy.exceeded.items()
            },
            "applied_threshold_overrides": applied,
            "proposed_rules": [rule.model_dump() for rule in req.proposed_rules],
            "n_steps": len(plan.steps),
        }

    @app.post("/policy/workbench/apply", response_model=PolicyWorkbenchApplyResponse)
    def policy_workbench_apply(req: PolicyWorkbenchApplyRequest, x_scopebench_policy_token: Optional[str] = Header(default=None)):
        authorized, message = _authorized_policy_editor(x_scopebench_policy_token)
        if not authorized:
            raise HTTPException(status_code=403, detail=message)

        target_dir = Path(os.getenv("SCOPEBENCH_POLICY_PROPOSALS_DIR", ".scopebench/policy_proposals")).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        proposal_id = datetime.now(timezone.utc).strftime("policy-proposal-%Y%m%dT%H%M%SZ")
        target_path = target_dir / f"{proposal_id}.json"
        payload = {
            "proposal_id": proposal_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "summary": req.summary,
            "policy_backend": req.policy_backend,
            "contract": req.contract,
            "plan": req.plan,
            "threshold_overrides": req.threshold_overrides,
            "proposed_rules": [rule.model_dump() for rule in req.proposed_rules],
        }
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return PolicyWorkbenchApplyResponse(ok=True, saved_to=str(target_path), proposal_id=proposal_id)

    @app.get("/cases")
    def cases_endpoint():
        from scopebench.bench.dataset import default_cases_path, load_cases

        try:
            builtin_cases = load_cases(default_cases_path())
        except ValueError as exc:
            return {
                "datasets": [],
                "domains": [],
                "count": 0,
                "cases": [],
                "error": str(exc),
                "plugins": plugin_manager.bundles_payload(),
            }

        all_cases = list(builtin_cases) + list(plugin_manager.cases)
        return {
            "datasets": sorted({case.id for case in all_cases}),
            "domains": sorted({case.domain for case in all_cases}),
            "count": len(all_cases),
            "cases": [
                {
                    "case_schema_version": case.case_schema_version,
                    "id": case.id,
                    "domain": case.domain,
                    "instruction": case.instruction,
                    "expected_decision": case.expected_decision,
                    "expected_rationale": case.expected_rationale,
                    "expected_step_vectors": case.expected_step_vectors,
                    "contract": case.contract,
                    "plan": case.plan,
                    "notes": case.notes,
                }
                for case in all_cases
            ],
            "plugins": plugin_manager.bundles_payload(),
        }

    @app.get("/cases/analytics", response_model=CasesAnalyticsResponse)
    def cases_analytics_endpoint():
        from scopebench.bench.dataset import default_cases_path

        path = default_cases_path()
        try:
            return _build_cases_analytics(path, plugin_manager.cases)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
            raise HTTPException(
                status_code=404, detail=f"Domain '{req.domain}' not found"
            )
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

        return _build_calibration_dashboard(
            path,
            calibration_overrides={
                domain: calibration for domain, (calibration, _stats) in domain_payload.items()
            },
        )

    @app.post("/dataset/validate", response_model=DatasetValidateResponse)
    def dataset_validate_endpoint(req: DatasetValidateRequest):
        validated = validate_case_object(req.case)
        return DatasetValidateResponse(ok=True, case_id=validated.id)

    @app.post("/dataset/suggest", response_model=DatasetSuggestResponse)
    def dataset_suggest_endpoint(req: DatasetSuggestRequest):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            contract_path = tmp_dir / "contract.yaml"
            plan_path = tmp_dir / "plan.yaml"
            import yaml

            contract_path.write_text(
                yaml.safe_dump(req.contract, sort_keys=False), encoding="utf-8"
            )
            plan_path.write_text(
                yaml.safe_dump(req.plan, sort_keys=False), encoding="utf-8"
            )
            case = suggest_case(
                case_id=req.id,
                domain=req.domain,
                instruction=req.instruction,
                contract_path=contract_path,
                plan_path=plan_path,
                expected_decision=req.expected_decision,
                expected_rationale=req.expected_rationale,
                notes=req.notes,
                policy_backend=req.policy_backend,
            )
        return DatasetSuggestResponse(case=case)

    @app.post("/dataset/render", response_model=DatasetRenderResponse)
    def dataset_render_endpoint(req: DatasetRenderRequest):
        validated = validate_case_object(req.case)
        normalized_case = asdict(validated)
        output_format = (req.format or "json").strip().lower()
        if output_format == "yaml":
            return DatasetRenderResponse(
                filename=f"{validated.id}.yaml",
                content_type="application/x-yaml",
                content=yaml.safe_dump(normalized_case, sort_keys=False),
            )
        if output_format != "json":
            raise HTTPException(status_code=400, detail="format must be 'json' or 'yaml'")
        return DatasetRenderResponse(
            filename=f"{validated.id}.json",
            content_type="application/json",
            content=json.dumps(normalized_case, indent=2),
        )

    @app.post("/dataset/wizard", response_model=DatasetWizardResponse)
    def dataset_wizard_endpoint(req: DatasetWizardRequest):
        suggested = dataset_suggest_endpoint(
            DatasetSuggestRequest(
                id=req.id,
                domain=req.domain,
                instruction=req.instruction,
                contract=req.contract,
                plan=req.plan,
                expected_decision=req.expected_decision,
                expected_rationale=req.expected_rationale,
                notes=req.notes,
                policy_backend=req.policy_backend,
            )
        )
        validate_result = dataset_validate_endpoint(DatasetValidateRequest(case=suggested.case))
        rendered = dataset_render_endpoint(
            DatasetRenderRequest(case=suggested.case, format=req.format)
        )
        return DatasetWizardResponse(
            ok=validate_result.ok,
            case_id=validate_result.case_id,
            case=suggested.case,
            rendered=rendered,
        )

    @app.post("/dataset/review/comment", response_model=DatasetReviewStateResponse)
    def dataset_review_comment_endpoint(req: DatasetReviewCommentRequest):
        review = _ensure_review(req.case_id, draft_case=req.draft_case)
        text = req.comment.strip()
        if not text:
            raise HTTPException(status_code=400, detail="comment must not be empty")
        review["comments"].append(
            {
                "reviewer": req.reviewer.strip() or "anonymous",
                "comment": text,
                "created_utc": _utc_now_iso(),
            }
        )
        review["updated_utc"] = _utc_now_iso()
        _refresh_acceptance(review)
        return DatasetReviewStateResponse(**review)

    @app.post("/dataset/review/suggest_edit", response_model=DatasetReviewStateResponse)
    def dataset_review_suggest_edit_endpoint(req: DatasetReviewSuggestEditRequest):
        review = _ensure_review(req.case_id, draft_case=req.draft_case)
        field_path = req.field_path.strip()
        proposed = req.proposed_value.strip()
        rationale = req.rationale.strip()
        if not field_path:
            raise HTTPException(status_code=400, detail="field_path must not be empty")
        if not proposed:
            raise HTTPException(status_code=400, detail="proposed_value must not be empty")
        if not rationale:
            raise HTTPException(status_code=400, detail="rationale must not be empty")
        review["suggested_edits"].append(
            {
                "reviewer": req.reviewer.strip() or "anonymous",
                "field_path": field_path,
                "proposed_value": proposed,
                "rationale": rationale,
                "created_utc": _utc_now_iso(),
            }
        )
        review["updated_utc"] = _utc_now_iso()
        _refresh_acceptance(review)
        return DatasetReviewStateResponse(**review)

    @app.post("/dataset/review/vote", response_model=DatasetReviewStateResponse)
    def dataset_review_vote_endpoint(req: DatasetReviewVoteRequest):
        review = _ensure_review(req.case_id, draft_case=req.draft_case)
        vote = req.vote.strip().lower()
        if vote not in {"accept", "reject", "abstain"}:
            raise HTTPException(status_code=400, detail="vote must be one of: accept, reject, abstain")
        reviewer = req.reviewer.strip() or "anonymous"
        review["votes"][reviewer] = vote
        review["updated_utc"] = _utc_now_iso()
        _refresh_acceptance(review)
        return DatasetReviewStateResponse(**review)

    @app.get("/dataset/review/{case_id}", response_model=DatasetReviewStateResponse)
    def dataset_review_state_endpoint(case_id: str):
        review = _ensure_review(case_id)
        _refresh_acceptance(review)
        return DatasetReviewStateResponse(**review)

    @app.post("/suggest_effects", response_model=SuggestEffectsResponse)
    def suggest_effects_endpoint(req: SuggestEffectsRequest):
        plan = PlanDAG.model_validate(req.plan)
        suggestions: List[SuggestEffectsItem] = []

        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            plan_path = tmp_dir / "plan.yaml"
            plan_path.write_text(
                yaml.safe_dump(plan.model_dump(mode="json", exclude_none=True), sort_keys=False),
                encoding="utf-8",
            )
            cmd = [
                "python",
                "-m",
                "scopebench.cli",
                "suggest-effects",
                str(plan_path),
                "--json",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if completed.returncode == 0:
                try:
                    payload = json.loads(completed.stdout)
                    for item in payload.get("steps", []):
                        if isinstance(item, dict) and item.get("id"):
                            suggestions.append(
                                SuggestEffectsItem(
                                    step_id=str(item.get("id")),
                                    tool=item.get("tool"),
                                    effects=item.get("effects") or {},
                                )
                            )
                except json.JSONDecodeError:
                    suggestions = []

            if not suggestions:
                # Fall back to in-process annotation when CLI invocation is unavailable.
                for item in suggest_effects_for_plan(plan):
                    suggestions.append(
                        SuggestEffectsItem(
                            step_id=item.step_id,
                            tool=item.tool,
                            effects=item.effects.model_dump(mode="json", exclude_none=True),
                        )
                    )

        step_lookup = {step.id: step for step in plan.steps}
        suggestion_payload: List[SuggestEffectsItem] = []
        for suggestion in suggestions:
            effects_payload = suggestion.effects
            if suggestion.step_id in step_lookup:
                step_lookup[suggestion.step_id].effects = EffectSpec.model_validate(effects_payload)
            suggestion_payload.append(
                SuggestEffectsItem(
                    step_id=suggestion.step_id,
                    tool=suggestion.tool,
                    effects=effects_payload,
                )
            )

        return SuggestEffectsResponse(
            plan=plan.model_dump(mode="json", exclude_none=True),
            suggestions=suggestion_payload,
        )

    @app.post("/evaluate", response_model=EvaluateResponse)
    @app.post(
        "/evaluate",
        response_model=EvaluateResponse,
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/json": {
                        "example": EvaluateRequest.model_config["json_schema_extra"][
                            "example"
                        ]
                    }
                }
            },
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "example": EvaluateResponse.model_config[
                                "json_schema_extra"
                            ]["example"]
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
                    "resource_intensity": AxisDetail(
                        **vec.resource_intensity.model_dump()
                    ),
                    "legal_exposure": AxisDetail(**vec.legal_exposure.model_dump()),
                    "dependency_creation": AxisDetail(
                        **vec.dependency_creation.model_dump()
                    ),
                    "stakeholder_radius": AxisDetail(
                        **vec.stakeholder_radius.model_dump()
                    ),
                    "power_concentration": AxisDetail(
                        **vec.power_concentration.model_dump()
                    ),
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
                        est_labor_hours=(
                            plan_step.est_labor_hours if plan_step else None
                        ),
                        resolved_cost_usd=(
                            plan_step.resolved_cost_usd() if plan_step else None
                        ),
                        resolved_time_days=(
                            plan_step.resolved_time_days() if plan_step else None
                        ),
                        resolved_labor_hours=(
                            plan_step.resolved_labor_hours() if plan_step else None
                        ),
                        realtime_estimates=(
                            list(plan_step.realtime_estimates) if plan_step else []
                        ),
                        est_benefit=plan_step.est_benefit if plan_step else None,
                        benefit_unit=plan_step.benefit_unit if plan_step else None,
                        axes=axes,
                    )
                )

        summary = None
        next_steps = None
        patch_suggestion = None
        rationale_summary = None
        if req.include_summary:
            summary = _summarize_response(
                pol, res.aggregate.as_dict(), effective_decision
            )
            next_steps = _next_steps_from_policy(pol)
            patch_suggestion = _suggest_plan_patch(pol, plan)
            rationale_summary = _summarize_step_rationales(
                steps, res.aggregate.as_dict()
            )

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
            k: {"value": float(v[0]), "threshold": float(v[1])}
            for k, v in pol.exceeded.items()
        }
        asked_payload = {k: float(v) for k, v in pol.asked.items()}
        policy_input_payload = (
            pol.policy_input.__dict__
            if (req.include_telemetry and pol.policy_input)
            else None
        )

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
            rationale_summary=rationale_summary,
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
        initial_result = evaluate(
            contract, initial_plan, policy_backend=backend, judge=req.judge
        )
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
            result = evaluate(
                contract, plan_model, policy_backend=backend, judge=req.judge
            )
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

            with tracer.start_as_current_span(
                "scopebench.evaluate_session.agent"
            ) as span:
                span.set_attribute("scopebench.agent_id", agent.agent_id)
                span.set_attribute("scopebench.agent_plan_count", len(agent_plans))
                agent_results = [
                    evaluate(contract, plan, policy_backend=backend)
                    for plan in agent_plans
                ]
            agent_aggregate = combine_aggregates(
                [result.aggregate for result in agent_results]
            )
            agent_decision = "ALLOW"
            for result in agent_results:
                if result.policy.decision.value == "DENY":
                    agent_decision = "DENY"
                    break
                if result.policy.decision.value == "ASK":
                    agent_decision = "ASK"

            agent_ledger = build_budget_ledger(contract, agent_plans)
            if (
                any(entry["exceeded"] > 0 for entry in agent_ledger.values())
                and agent_decision != "DENY"
            ):
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
                budget_projection_utilization=_budget_projection_utilization_from_ledger(
                    agent_ledger
                ),
                decision=agent_decision,
            )
            agent_aggregates.append(agent_aggregate)

            if agent_decision == "DENY":
                global_decision = "DENY"
            elif agent_decision == "ASK" and global_decision != "DENY":
                global_decision = "ASK"

        global_aggregate_dict = _aggregate_session_risk(per_agent_aggregates)
        global_ledger = build_budget_ledger(session.global_contract, global_plans)
        if (
            any(entry["exceeded"] > 0 for entry in global_ledger.values())
            and global_decision != "DENY"
        ):
            global_decision = "ASK"

        global_threshold = float(
            session.global_contract.escalation.ask_if_any_axis_over
        )
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
                budget_projection_utilization=_budget_projection_utilization_from_ledger(
                    global_ledger
                ),
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
            negotiation=_build_session_negotiation(
                per_agent=per_agent,
                global_ledger=global_ledger,
            ),
        )

    return app
