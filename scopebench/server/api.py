from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate
from scopebench.scoring.calibration import CalibratedDecisionThresholds
from scopebench.tracing.otel import init_tracing


class EvaluateRequest(BaseModel):
    contract: Dict[str, Any] = Field(..., description="TaskContract as dict")
    plan: Dict[str, Any] = Field(..., description="PlanDAG as dict")
    include_steps: bool = Field(False, description="Include step-level vectors and rationales.")
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
        return EvaluateResponse(
            decision=pol.decision.value,
            reasons=pol.reasons,
            exceeded={k: {"value": float(v[0]), "threshold": float(v[1])} for k, v in pol.exceeded.items()},
            asked={k: float(v) for k, v in pol.asked.items()},
            aggregate=res.aggregate.as_dict(),
            n_steps=res.aggregate.n_steps,
            steps=steps,
        )

    return app
