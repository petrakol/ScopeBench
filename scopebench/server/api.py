from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from scopebench.contracts import TaskContract
from scopebench.domains import list_domain_templates
from scopebench.scoring.causal import list_causal_rules
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate
from scopebench.scoring.calibration import CalibratedDecisionThresholds
from scopebench.tracing.otel import init_tracing


class EvaluateRequest(BaseModel):
    contract: Dict[str, Any] = Field(..., description="TaskContract as dict")
    plan: Dict[str, Any] = Field(..., description="PlanDAG as dict")
    include_steps: bool = Field(False, description="Include step-level vectors and rationales.")
    include_summary: bool = Field(False, description="Include summary and next-step guidance.")
    calibration_scale: Optional[float] = Field(None, ge=0.0, description="Optional scale for aggregate scores.")


class AxisDetail(BaseModel):
    value: float
    rationale: str
    confidence: float


class StepDetail(BaseModel):
    step_id: Optional[str]
    tool: Optional[str]
    tool_category: Optional[str]
    axes: Dict[str, AxisDetail]


class EvaluateResponse(BaseModel):
    decision: str
    reasons: list[str]
    exceeded: Dict[str, Dict[str, float]]
    asked: Dict[str, float]
    aggregate: Dict[str, float]
    n_steps: int
    steps: Optional[List[StepDetail]] = None
    summary: Optional[str] = None
    next_steps: Optional[List[str]] = None


class DomainTemplateResponse(BaseModel):
    name: str
    description: str
    forbidden_tool_categories: list[str]
    escalation_tool_categories: list[str]
    thresholds: Dict[str, float]
    escalation: Dict[str, float]
    budgets: Dict[str, float]
    allowed_tools: Optional[list[str]] = None
    notes: Dict[str, str]


class CausalRuleResponse(BaseModel):
    category: str
    axis_minimums: Dict[str, float]
    rationale: str


def _summarize_response(policy, aggregate) -> str:
    top_axes = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:3]
    axes_text = ", ".join(f"{axis}={value:.2f}" for axis, value in top_axes)
    return f"Decision {policy.decision.value}. Top axes: {axes_text}."


def _next_steps_from_policy(policy) -> List[str]:
    suggestions: List[str] = []
    for axis, (_, threshold) in policy.exceeded.items():
        suggestions.append(f"Reduce {axis} below {float(threshold):.2f} or split into smaller steps.")
    for axis, threshold in policy.asked.items():
        suggestions.append(f"Consider approval or mitigating {axis} below {float(threshold):.2f}.")
    if any("Tool category" in reason for reason in policy.reasons):
        suggestions.append("Remove high-risk tool categories or get explicit approval.")
    if not suggestions:
        suggestions.append("Proceed; plan appears proportionate to the contract.")
    return suggestions[:5]


def create_app() -> FastAPI:
    init_tracing(enable_console=False)
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
        res = evaluate(contract, plan, calibration=calibration)
        pol = res.policy
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
        if req.include_summary:
            summary = _summarize_response(pol, res.aggregate.as_dict())
            next_steps = _next_steps_from_policy(pol)
        return EvaluateResponse(
            decision=pol.decision.value,
            reasons=pol.reasons,
            exceeded={k: {"value": float(v[0]), "threshold": float(v[1])} for k, v in pol.exceeded.items()},
            asked={k: float(v) for k, v in pol.asked.items()},
            aggregate=res.aggregate.as_dict(),
            n_steps=res.aggregate.n_steps,
            steps=steps,
            summary=summary,
            next_steps=next_steps,
        )

    @app.get("/domains", response_model=List[DomainTemplateResponse])
    def domains_endpoint():
        templates = list_domain_templates()
        payload = []
        for template in templates.values():
            payload.append(
                DomainTemplateResponse(
                    name=template.name,
                    description=template.description,
                    forbidden_tool_categories=sorted(template.forbidden_tool_categories),
                    escalation_tool_categories=sorted(template.escalation_tool_categories),
                    thresholds=template.thresholds,
                    escalation=template.escalation,
                    budgets=template.budgets,
                    allowed_tools=sorted(template.allowed_tools) if template.allowed_tools else None,
                    notes=template.notes,
                )
            )
        return sorted(payload, key=lambda item: item.name)

    @app.get("/causal-rules", response_model=List[CausalRuleResponse])
    def causal_rules_endpoint():
        rules = list_causal_rules()
        payload = []
        for rule in rules.values():
            payload.append(
                CausalRuleResponse(
                    category=rule.category,
                    axis_minimums=rule.axis_minimums,
                    rationale=rule.rationale,
                )
            )
        return sorted(payload, key=lambda item: item.category)

    return app
