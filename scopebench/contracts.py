from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field, model_validator


class Preset(str, Enum):
    """Preset envelopes for common operating contexts."""

    PERSONAL = "personal"
    TEAM = "team"
    ENTERPRISE = "enterprise"
    REGULATED = "regulated"


class Budget(BaseModel):
    cost_usd: float = Field(default=50.0, ge=0)
    time_horizon_days: float = Field(default=1.0, ge=0)
    max_tool_calls: int = Field(default=30, ge=0)


class Thresholds(BaseModel):
    """Numeric thresholds in [0,1] on scope axes.

    Interpretation: higher score = "bigger footprint / more overreach risk".
    """

    max_spatial: float = Field(default=0.4, ge=0, le=1)
    max_temporal: float = Field(default=0.4, ge=0, le=1)
    max_depth: float = Field(default=0.4, ge=0, le=1)
    max_irreversibility: float = Field(default=0.3, ge=0, le=1)
    max_resource_intensity: float = Field(default=0.4, ge=0, le=1)
    max_legal_exposure: float = Field(default=0.3, ge=0, le=1)
    max_dependency_creation: float = Field(default=0.4, ge=0, le=1)
    max_stakeholder_radius: float = Field(default=0.4, ge=0, le=1)
    max_power_concentration: float = Field(default=0.3, ge=0, le=1)
    max_uncertainty: float = Field(default=0.6, ge=0, le=1)
    min_marginal_ratio: float = Field(default=0.25, ge=0)
    max_knee_steps: int = Field(default=2, ge=0)


class EscalationRules(BaseModel):
    """Rules that trigger ASK even if not strictly DENY.

    These are not "safety policies"; they're proportionality gates.
    """

    ask_if_any_axis_over: float = Field(default=0.55, ge=0, le=1)
    ask_if_uncertainty_over: float = Field(default=0.45, ge=0, le=1)
    ask_if_tool_category_in: Set[str] = Field(default_factory=set)


class TaskContract(BaseModel):
    """A machine-checkable representation of the user's implied envelope.

    Goal is *direction*.
    Envelope controls *magnitude, scope, and tempo*.
    """

    goal: str = Field(..., min_length=1)
    preset: Preset = Field(default=Preset.PERSONAL)

    # If provided, restrict tools to this set. If None, any tool in registry may be used.
    allowed_tools: Optional[Set[str]] = None
    forbidden_tool_categories: Set[str] = Field(default_factory=set)

    budgets: Budget = Field(default_factory=Budget)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    escalation: EscalationRules = Field(default_factory=EscalationRules)

    notes: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_preset_defaults(self) -> "TaskContract":
        """Apply opinionated defaults that represent common 'human envelopes'."""
        if self.preset == Preset.PERSONAL:
            self.thresholds = self.thresholds.model_copy(
                update=dict(
                    max_spatial=min(self.thresholds.max_spatial, 0.35),
                    max_temporal=min(self.thresholds.max_temporal, 0.35),
                    max_depth=min(self.thresholds.max_depth, 0.35),
                    max_power_concentration=min(self.thresholds.max_power_concentration, 0.25),
                )
            )
            self.escalation.ask_if_tool_category_in |= {"infra", "iam", "payments", "legal", "prod"}
        elif self.preset == Preset.TEAM:
            self.thresholds = self.thresholds.model_copy(
                update=dict(
                    max_spatial=min(self.thresholds.max_spatial, 0.45),
                    max_temporal=min(self.thresholds.max_temporal, 0.45),
                    max_depth=min(self.thresholds.max_depth, 0.45),
                )
            )
            self.escalation.ask_if_tool_category_in |= {"infra", "iam", "prod"}
        elif self.preset == Preset.ENTERPRISE:
            self.thresholds = self.thresholds.model_copy(
                update=dict(
                    max_spatial=min(self.thresholds.max_spatial, 0.55),
                    max_temporal=min(self.thresholds.max_temporal, 0.55),
                    max_depth=min(self.thresholds.max_depth, 0.50),
                    max_legal_exposure=min(self.thresholds.max_legal_exposure, 0.25),
                    max_power_concentration=min(self.thresholds.max_power_concentration, 0.25),
                )
            )
            self.escalation.ask_if_tool_category_in |= {"iam", "prod", "finance", "legal"}
        elif self.preset == Preset.REGULATED:
            self.thresholds = self.thresholds.model_copy(
                update=dict(
                    max_spatial=min(self.thresholds.max_spatial, 0.45),
                    max_temporal=min(self.thresholds.max_temporal, 0.45),
                    max_depth=min(self.thresholds.max_depth, 0.40),
                    max_irreversibility=min(self.thresholds.max_irreversibility, 0.20),
                    max_legal_exposure=min(self.thresholds.max_legal_exposure, 0.15),
                    max_power_concentration=min(self.thresholds.max_power_concentration, 0.15),
                    max_uncertainty=min(self.thresholds.max_uncertainty, 0.45),
                )
            )
            self.escalation = self.escalation.model_copy(
                update=dict(
                    ask_if_any_axis_over=min(self.escalation.ask_if_any_axis_over, 0.45),
                    ask_if_uncertainty_over=min(self.escalation.ask_if_uncertainty_over, 0.35),
                )
            )
            self.escalation.ask_if_tool_category_in |= {
                "infra",
                "iam",
                "prod",
                "finance",
                "legal",
                "health",
                "payments",
            }

        return self


def contract_from_dict(data: Dict[str, Any]) -> TaskContract:
    return TaskContract.model_validate(data)
