from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Protocol

from pydantic import BaseModel, ValidationError

from scopebench.plan import PlanStep
from scopebench.scoring.axes import AxisScore, ScopeVector
from scopebench.scoring.cache import JudgeCache
from scopebench.scoring.providers import EnvJSONProvider, LLMProvider

PROMPT_VERSION = "v1"


class JudgeTelemetry(BaseModel):
    mode: str
    prompt_version: Optional[str] = None
    provider: Optional[str] = None
    cache_hit: bool = False
    fallback_reason: Optional[str] = None


class JudgeOutput(BaseModel):
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


class StepJudge(Protocol):
    telemetry: JudgeTelemetry

    def judge_step(
        self,
        step: PlanStep,
        *,
        tool_category: Optional[str],
        tool_priors: Dict[str, float],
    ) -> Optional[ScopeVector]:
        ...


class NoopStepJudge:
    def __init__(self, mode: str):
        self.telemetry = JudgeTelemetry(mode=mode)

    def judge_step(
        self,
        step: PlanStep,
        *,
        tool_category: Optional[str],
        tool_priors: Dict[str, float],
    ) -> Optional[ScopeVector]:
        del step, tool_category, tool_priors
        return None


@dataclass
class LLMStepJudge:
    provider: LLMProvider
    cache: JudgeCache
    telemetry: JudgeTelemetry = field(
        default_factory=lambda: JudgeTelemetry(
            mode="llm",
            prompt_version=PROMPT_VERSION,
        )
    )

    def judge_step(
        self,
        step: PlanStep,
        *,
        tool_category: Optional[str],
        tool_priors: Dict[str, float],
    ) -> Optional[ScopeVector]:
        prompt = build_prompt_v1(step, tool_category=tool_category, tool_priors=tool_priors)
        cache_payload = {
            "prompt_version": PROMPT_VERSION,
            "step_description": step.description,
            "tool": step.tool or "",
            "tool_category": tool_category or "",
            "tool_priors": tool_priors,
        }
        key = self.cache.make_key(cache_payload)

        cached = self.cache.get(key)
        if cached is not None:
            self.telemetry.cache_hit = True
            return _parse_scope_vector(cached, step=step, tool_category=tool_category)

        self.telemetry.cache_hit = False
        try:
            provider_result = self.provider.complete(prompt)
            self.telemetry.provider = provider_result.provider_name
            raw = json.loads(provider_result.raw_text)
            self.cache.set(key, raw)
            return _parse_scope_vector(raw, step=step, tool_category=tool_category)
        except (RuntimeError, json.JSONDecodeError, ValidationError) as exc:
            self.telemetry.fallback_reason = f"llm_judge_fallback:{type(exc).__name__}"
            return None


def _parse_scope_vector(
    raw_payload: dict,
    *,
    step: PlanStep,
    tool_category: Optional[str],
) -> ScopeVector:
    parsed = JudgeOutput.model_validate(raw_payload)
    return ScopeVector(
        step_id=step.id,
        tool=step.tool,
        tool_category=tool_category,
        spatial=parsed.spatial,
        temporal=parsed.temporal,
        depth=parsed.depth,
        irreversibility=parsed.irreversibility,
        resource_intensity=parsed.resource_intensity,
        legal_exposure=parsed.legal_exposure,
        dependency_creation=parsed.dependency_creation,
        stakeholder_radius=parsed.stakeholder_radius,
        power_concentration=parsed.power_concentration,
        uncertainty=parsed.uncertainty,
    )


def build_prompt_v1(
    step: PlanStep,
    *,
    tool_category: Optional[str],
    tool_priors: Dict[str, float],
) -> str:
    spec_path = Path(__file__).resolve().parent / "prompts" / "prompt_v1.md"
    spec = spec_path.read_text(encoding="utf-8").strip()
    payload = {
        "step_id": step.id,
        "step_description": step.description,
        "tool": step.tool or "",
        "tool_category": tool_category or "",
        "tool_priors": tool_priors,
    }
    context = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{spec}\n\nINPUT_JSON={context}\n"


JudgeMode = Literal["none", "heuristic", "llm"]


def build_step_judge(
    mode: JudgeMode,
    *,
    provider: Optional[LLMProvider] = None,
    cache_path: Optional[str] = None,
) -> StepJudge:
    if mode in {"none", "heuristic"}:
        return NoopStepJudge(mode=mode)
    chosen_provider = provider or EnvJSONProvider()
    cache = JudgeCache.from_path(cache_path)
    return LLMStepJudge(provider=chosen_provider, cache=cache)
