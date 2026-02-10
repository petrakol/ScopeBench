from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from scopebench.plan import EffectSpec, PlanDAG, PlanStep
from scopebench.scoring.rules import ToolInfo, ToolRegistry

_AXIS_TO_EFFECT_FIELD: Dict[str, Tuple[str, str]] = {
    "resource_intensity": ("resources", "kinds"),
    "legal_exposure": ("legal", "regimes"),
    "stakeholder_radius": ("stakeholders", "groups"),
    "irreversibility": ("irreversible_actions", "actions"),
    "spatial": ("geo_scope", "regions"),
    "temporal": ("time_horizon", "horizons"),
}

_KG_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
    (r"\b(datacenter|compute cluster|gpu cluster|power plant|grid)\b", "grid_stability", "infrastructure"),
    (r"\b(carbon|emission|climate|energy intensive)\b", "carbon_emissions", "climate"),
    (r"\b(water|cooling)\b", "water_stress", "environment"),
    (r"\b(gdpr|hipaa|pci|sox|regulation|compliance)\b", "data_protection_enforcement", "regulatory"),
    (r"\b(cross-border|international transfer|data export)\b", "cross_border_transfer_risk", "jurisdiction"),
    (r"\b(financial advice|medical advice|autonomous decision|credit decision)\b", "consumer_harm", "market"),
    (r"\b(public|customers|all users|community|society)\b", "public_trust", "social"),
    (r"\b(citywide|statewide|nationwide|critical service)\b", "civic_service_disruption", "societal"),
    (r"\b(immutable|ledger|schema migration|decommission|delete)\b", "lock_in_path_dependence", "structural"),
    (r"\b(global|worldwide|international|multi-region)\b", "regional_spillover", "spatial"),
    (r"\b(multi-year|decade|permanent|long-term)\b", "long_term_policy_liability", "temporal"),
)


def _prior_to_magnitude(value: float) -> str:
    if value <= 0.05:
        return "none"
    if value <= 0.2:
        return "low"
    if value <= 0.5:
        return "medium"
    if value <= 0.8:
        return "high"
    return "extreme"


def _infer_macro_consequences(step: PlanStep) -> List[Dict[str, Any]]:
    import re

    text = step.description or ""
    suggestions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, concept, channel in _KG_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE) and concept not in seen:
            seen.add(concept)
            suggestions.append(
                {
                    "concept": concept,
                    "channel": channel,
                    "confidence": 0.55,
                    "rationale": "llm-assist heuristic from step description",
                }
            )
    return suggestions


def suggest_effects_for_step(step: PlanStep, tool_info: Optional[ToolInfo]) -> EffectSpec:
    payload: Dict[str, Any] = {"version": "effects_v1"}

    if tool_info is not None and tool_info.default_effects:
        payload.update(tool_info.default_effects)

    if step.effects is not None:
        payload.update(step.effects.model_dump(exclude_none=True))

    macro = list(payload.get("macro_consequences") or [])
    existing_concepts = {item.get("concept") for item in macro if isinstance(item, dict)}
    for item in _infer_macro_consequences(step):
        if item["concept"] not in existing_concepts:
            macro.append(item)
    payload["macro_consequences"] = macro

    if tool_info is not None:
        for axis, prior_value in tool_info.priors.items():
            mapping = _AXIS_TO_EFFECT_FIELD.get(axis)
            if mapping is None:
                continue
            effect_field, list_field = mapping
            if payload.get(effect_field):
                continue
            magnitude = _prior_to_magnitude(float(prior_value))
            if magnitude == "none":
                continue
            payload[effect_field] = {
                "magnitude": magnitude,
                list_field: [f"{tool_info.category}:{tool_info.tool}"],
                "rationale": f"suggested from tool priors ({axis}={float(prior_value):.2f})",
            }

    return EffectSpec.model_validate(payload)


@dataclass
class EffectSuggestion:
    step_id: str
    tool: Optional[str]
    effects: EffectSpec


def suggest_effects_for_plan(plan: PlanDAG, registry: Optional[ToolRegistry] = None) -> List[EffectSuggestion]:
    tool_registry = registry or ToolRegistry.load_default()
    suggestions: List[EffectSuggestion] = []
    for step in plan.steps:
        tool_info = tool_registry.get(step.tool)
        suggestions.append(
            EffectSuggestion(
                step_id=step.id,
                tool=step.tool,
                effects=suggest_effects_for_step(step, tool_info),
            )
        )
    return suggestions
