from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate
from scopebench.tracing.otel import init_tracing


class EvaluateRequest(BaseModel):
    contract: Dict[str, Any] = Field(..., description="TaskContract as dict")
    plan: Dict[str, Any] = Field(..., description="PlanDAG as dict")


class EvaluateResponse(BaseModel):
    decision: str
    reasons: list[str]
    exceeded: Dict[str, Dict[str, float]]
    asked: Dict[str, float]
    aggregate: Dict[str, float]
    n_steps: int


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
        res = evaluate(contract, plan)
        pol = res.policy
        return EvaluateResponse(
            decision=pol.decision.value,
            reasons=pol.reasons,
            exceeded={k: {"value": float(v[0]), "threshold": float(v[1])} for k, v in pol.exceeded.items()},
            asked={k: float(v) for k, v in pol.asked.items()},
            aggregate=res.aggregate.as_dict(),
            n_steps=res.aggregate.n_steps,
        )

    return app
