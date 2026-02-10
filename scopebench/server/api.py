from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate
from scopebench.scoring.axes import combine_aggregates
from scopebench.scoring.calibration import CalibratedDecisionThresholds
from scopebench.scoring.rules import build_budget_ledger
from scopebench.session import MultiAgentSession
from scopebench.tracing.otel import get_tracer, init_tracing

SWE_READ_TOOLS = {"git_read", "file_read"}
SWE_WRITE_TOOLS = {"git_patch", "git_rewrite", "file_write"}
VALIDATION_TOOLS = {"analysis", "test_run", "pytest"}
VALIDATION_HINTS = ("test", "verify", "validation", "assert", "check")


class EvaluateRequest(BaseModel):
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
    policy_backend: Optional[str] = Field(
        None, description="Policy backend override: python|opa|cedar."
    )


class AxisDetail(BaseModel):
    value: float
    rationale: str
    confidence: float


class StepDetail(BaseModel):
    step_id: Optional[str]
    tool: Optional[str]
    tool_category: Optional[str]
    axes: Dict[str, AxisDetail]


class TelemetryDetail(BaseModel):
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


class SessionAggregateDetail(BaseModel):
    aggregate: Dict[str, float]
    ledger: Dict[str, Dict[str, float]]
    decision: str


class EvaluateSessionResponse(BaseModel):
    decision: str
    per_agent: Dict[str, SessionAggregateDetail]
    global_: SessionAggregateDetail = Field(..., alias="global")

    model_config = {"populate_by_name": True}


def _summarize_response(policy, aggregate, effective_decision: str) -> str:
    top_axes = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:3]
    axes_text = ", ".join(f"{axis}={value:.2f}" for axis, value in top_axes)
    return f"Decision {policy.decision.value} (effective: {effective_decision}). Top axes: {axes_text}."


def _next_steps_from_policy(policy) -> List[str]:
    suggestions: List[str] = []
    for axis, (_, threshold) in policy.exceeded.items():
        suggestions.append(
            f"Reduce {axis} below {float(threshold):.2f} or split into smaller steps."
        )

    if "read_before_write" in policy.asked:
        suggestions.append(
            "Add an explicit read step before patching code (for example: git_read on failing files)."
        )
    if "validation_after_write" in policy.asked:
        suggestions.append(
            "Add a downstream validation step after patching (for example: run targeted tests)."
        )

    for axis, threshold in policy.asked.items():
        if axis in {"read_before_write", "validation_after_write"}:
            continue
        suggestions.append(f"Consider approval or mitigating {axis} below {float(threshold):.2f}.")

    if any("Tool category" in reason for reason in policy.reasons):
        suggestions.append("Remove high-risk tool categories or get explicit approval.")
    if not suggestions:
        suggestions.append("Proceed; plan appears proportionate to the contract.")
    return suggestions[:5]


def _suggest_plan_patch(policy, plan: PlanDAG) -> List[Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []
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


def create_app(default_policy_backend: str = "python") -> FastAPI:
    init_tracing(enable_console=False)
    tracer = get_tracer("scopebench")
    app = FastAPI(title="ScopeBench", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/evaluate", response_model=EvaluateResponse)
    def evaluate_endpoint(req: EvaluateRequest):
        contract = TaskContract.model_validate(req.contract)
        plan = PlanDAG.model_validate(req.plan)
        calibration = None
        if req.calibration_scale is not None:
            calibration = CalibratedDecisionThresholds(global_scale=req.calibration_scale)
        backend = req.policy_backend or default_policy_backend
        res = evaluate(contract, plan, calibration=calibration, policy_backend=backend)
        pol = res.policy

        decision = pol.decision.value
        effective_decision = _effective_decision(decision, req.shadow_mode)

        steps = None
        if req.include_steps:
            steps = []
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
                steps.append(
                    StepDetail(
                        step_id=vec.step_id,
                        tool=vec.tool,
                        tool_category=vec.tool_category,
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

        return EvaluateResponse(
            decision=decision,
            policy_backend=pol.policy_backend,
            policy_version=pol.policy_version,
            policy_hash=pol.policy_hash,
            effective_decision=effective_decision,
            shadow_mode=req.shadow_mode,
            reasons=reasons,
            exceeded={
                k: {"value": float(v[0]), "threshold": float(v[1])} for k, v in pol.exceeded.items()
            },
            asked={k: float(v) for k, v in pol.asked.items()},
            aggregate=res.aggregate.as_dict(),
            n_steps=res.aggregate.n_steps,
            steps=steps,
            summary=summary,
            next_steps=next_steps,
            plan_patch_suggestion=patch_suggestion,
            telemetry=telemetry,
            policy_input=(pol.policy_input.__dict__ if (req.include_telemetry and pol.policy_input) else None),
        )

    @app.post("/evaluate_session", response_model=EvaluateSessionResponse)
    def evaluate_session_endpoint(req: EvaluateSessionRequest):
        session = MultiAgentSession.model_validate(req.session)
        backend = req.policy_backend or default_policy_backend

        per_agent: Dict[str, SessionAggregateDetail] = {}
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

            per_agent[agent.agent_id] = SessionAggregateDetail(
                aggregate=agent_aggregate.as_dict(),
                ledger=agent_ledger,
                decision=agent_decision,
            )
            agent_aggregates.append(agent_aggregate)

            if agent_decision == "DENY":
                global_decision = "DENY"
            elif agent_decision == "ASK" and global_decision != "DENY":
                global_decision = "ASK"

        global_aggregate = combine_aggregates(agent_aggregates)
        global_ledger = build_budget_ledger(session.global_contract, global_plans)
        if any(entry["exceeded"] > 0 for entry in global_ledger.values()) and global_decision != "DENY":
            global_decision = "ASK"

        global_scope = SessionAggregateDetail(
            aggregate=global_aggregate.as_dict(),
            ledger=global_ledger,
            decision=global_decision,
        )
        return EvaluateSessionResponse(
            decision=global_decision,
            per_agent=per_agent,
            global_=global_scope,
        )

    return app
