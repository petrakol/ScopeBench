from __future__ import annotations

from typing import Dict, Optional, Tuple

from pydantic import BaseModel, Field


class AxisScore(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(default="")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ScopeVector(BaseModel):
    """Normalized scope footprint scores in [0,1].

    Higher = larger footprint / higher probability of "scale mismatch".
    """

    spatial: AxisScore
    temporal: AxisScore
    depth: AxisScore
    irreversibility: AxisScore
    resource_intensity: AxisScore
    legal_exposure: AxisScore
    dependency_creation: AxisScore
    stakeholder_radius: AxisScore
    power_concentration: AxisScore
    uncertainty: AxisScore

    step_id: Optional[str] = None
    tool: Optional[str] = None
    tool_category: Optional[str] = None

    @property
    def axes(self) -> Tuple[str, ...]:
        return (
            "spatial",
            "temporal",
            "depth",
            "irreversibility",
            "resource_intensity",
            "legal_exposure",
            "dependency_creation",
            "stakeholder_radius",
            "power_concentration",
            "uncertainty",
        )

    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            "spatial": self.spatial.value,
            "temporal": self.temporal.value,
            "depth": self.depth.value,
            "irreversibility": self.irreversibility.value,
            "resource_intensity": self.resource_intensity.value,
            "legal_exposure": self.legal_exposure.value,
            "dependency_creation": self.dependency_creation.value,
            "stakeholder_radius": self.stakeholder_radius.value,
            "power_concentration": self.power_concentration.value,
            "uncertainty": self.uncertainty.value,
        }


class ScopeAggregate(BaseModel):
    """Aggregated scope footprint across a plan/session."""

    # Aggregated axis values in [0,1]
    spatial: float = Field(..., ge=0.0, le=1.0)
    temporal: float = Field(..., ge=0.0, le=1.0)
    depth: float = Field(..., ge=0.0, le=1.0)
    irreversibility: float = Field(..., ge=0.0, le=1.0)
    resource_intensity: float = Field(..., ge=0.0, le=1.0)
    legal_exposure: float = Field(..., ge=0.0, le=1.0)
    dependency_creation: float = Field(..., ge=0.0, le=1.0)
    stakeholder_radius: float = Field(..., ge=0.0, le=1.0)
    power_concentration: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)

    # Helpful stats
    n_steps: int = Field(default=0, ge=0)

    def as_dict(self) -> Dict[str, float]:
        return {
            "spatial": self.spatial,
            "temporal": self.temporal,
            "depth": self.depth,
            "irreversibility": self.irreversibility,
            "resource_intensity": self.resource_intensity,
            "legal_exposure": self.legal_exposure,
            "dependency_creation": self.dependency_creation,
            "stakeholder_radius": self.stakeholder_radius,
            "power_concentration": self.power_concentration,
            "uncertainty": self.uncertainty,
        }


SCOPE_AXES: Tuple[str, ...] = (
    "spatial",
    "temporal",
    "depth",
    "irreversibility",
    "resource_intensity",
    "legal_exposure",
    "dependency_creation",
    "stakeholder_radius",
    "power_concentration",
    "uncertainty",
)


def norm_inf(values: Dict[str, float]) -> float:
    return max(values.values()) if values else 0.0


def norm_l1(values: Dict[str, float]) -> float:
    return sum(values.values()) if values else 0.0
