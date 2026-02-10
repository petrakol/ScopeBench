from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class EffectMagnitude(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class EffectCategory(BaseModel):
    magnitude: EffectMagnitude
    rationale: Optional[str] = None


class ResourceEffect(EffectCategory):
    kinds: List[str] = Field(default_factory=list)


class LegalEffect(EffectCategory):
    regimes: List[str] = Field(default_factory=list)


class StakeholderEffect(EffectCategory):
    groups: List[str] = Field(default_factory=list)


class IrreversibleActionEffect(EffectCategory):
    actions: List[str] = Field(default_factory=list)


class GeoScopeEffect(EffectCategory):
    regions: List[str] = Field(default_factory=list)


class TimeHorizonEffect(EffectCategory):
    horizons: List[str] = Field(default_factory=list)


class MacroConsequence(BaseModel):
    concept: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: Optional[str] = None


class EffectSpec(BaseModel):
    version: str = Field(default="effects_v1", pattern=r"^effects_v1$")
    resources: Optional[ResourceEffect] = None
    legal: Optional[LegalEffect] = None
    stakeholders: Optional[StakeholderEffect] = None
    irreversible_actions: Optional[IrreversibleActionEffect] = None
    geo_scope: Optional[GeoScopeEffect] = None
    time_horizon: Optional[TimeHorizonEffect] = None
    macro_consequences: List[MacroConsequence] = Field(default_factory=list)


class PlanStep(BaseModel):
    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)

    tool: Optional[str] = None
    tool_category: Optional[str] = None

    # Dependencies by step id (DAG edges)
    depends_on: List[str] = Field(default_factory=list)

    # Optional structured "effects" the planner can provide.
    effects: Optional[EffectSpec] = None

    # Optional estimated cost/time (can be filled by planner, or by a predictor).
    est_cost_usd: Optional[float] = Field(default=None, ge=0)
    est_time_days: Optional[float] = Field(default=None, ge=0)
    est_benefit: Optional[float] = Field(default=None, ge=0)
    benefit_unit: Optional[str] = None


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
        graph = {s.id: s.depends_on for s in self.steps}
        visiting: List[str] = []
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                cycle_start = visiting.index(node)
                cycle = visiting[cycle_start:] + [node]
                raise ValueError(f"Plan step dependencies contain a cycle: {' -> '.join(cycle)}.")
            visiting.append(node)
            for dep in graph.get(node, []):
                visit(dep)
            visiting.pop()
            visited.add(node)

        for node in graph:
            visit(node)
        return self


def plan_from_dict(data: Dict[str, Any]) -> PlanDAG:
    return PlanDAG.model_validate(data)
