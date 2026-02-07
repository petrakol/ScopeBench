from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class PlanStep(BaseModel):
    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)

    tool: Optional[str] = None
    tool_category: Optional[str] = None

    # Dependencies by step id (DAG edges)
    depends_on: List[str] = Field(default_factory=list)

    # Optional structured "effects" the planner can provide.
    effects: Dict[str, Any] = Field(default_factory=dict)

    # Optional estimated cost/time (can be filled by planner, or by a predictor).
    est_cost_usd: Optional[float] = Field(default=None, ge=0)
    est_time_days: Optional[float] = Field(default=None, ge=0)


class PlanDAG(BaseModel):
    """A minimal explicit plan format.

    - steps are nodes
    - depends_on edges define a DAG (acyclicity not fully checked in MVP)
    """

    task: str = Field(..., min_length=1)
    steps: List[PlanStep] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_dag(self) -> "PlanDAG":
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan step IDs must be unique.")
        known = set(ids)
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in known:
                    raise ValueError(f"Step {s.id} depends on unknown step {dep}.")
        return self


def plan_from_dict(data: Dict[str, Any]) -> PlanDAG:
    return PlanDAG.model_validate(data)
