from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scopebench.plan import PlanStep
from scopebench.scoring.axes import AxisScore, ScopeVector
from scopebench.scoring.cache import JudgeCache
from scopebench.scoring.providers import EnvJSONProvider, LLMProvider, ProviderResult

PROMPT_VERSION = "v2"
OUTPUT_SCHEMA_VERSION = "scopebench.llm_judge.v1"


class JudgeTelemetry(BaseModel):
    mode: str
    prompt_version: Optional[str] = None
    provider: Optional[str] = None
    cache_hit: bool = False
    fallback_reason: Optional[str] = None


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scopebench.llm_judge.v1"]
    axes: "AxisBundle"


class AxisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class AxisBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spatial: AxisOutput
    temporal: AxisOutput
    depth: AxisOutput
    irreversibility: AxisOutput
    resource_intensity: AxisOutput
    legal_exposure: AxisOutput
    dependency_creation: AxisOutput
    stakeholder_radius: AxisOutput
    power_concentration: AxisOutput
    uncertainty: AxisOutput


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
class ProviderAgnosticClient:
    providers: tuple[LLMProvider, ...]

    def complete(self, prompt: str) -> "ProviderAttempt":
        last_error: Optional[RuntimeError] = None
        failures: list[str] = []
        for provider in self.providers:
            try:
                result = provider.complete(prompt)
                return ProviderAttempt(result=result, failures=tuple(failures))
            except RuntimeError as exc:
                last_error = exc
                provider_name = type(provider).__name__
                failures.append(f"{provider_name}:{type(exc).__name__}")

        if last_error is not None:
            raise RuntimeError("all_providers_failed") from last_error
        raise RuntimeError("no_providers_configured")


@dataclass(frozen=True)
class ProviderAttempt:
    result: ProviderResult
    failures: tuple[str, ...] = ()


@dataclass
class LLMStepJudge:
    provider_client: ProviderAgnosticClient
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
        self.telemetry.fallback_reason = None
        prompt = build_prompt_v2(step, tool_category=tool_category, tool_priors=tool_priors)
        cache_payload = {
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
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
            attempt = self.provider_client.complete(prompt)
            provider_result = attempt.result
            self.telemetry.provider = provider_result.provider_name
            if attempt.failures:
                self.telemetry.fallback_reason = (
                    "llm_judge_provider_retries:" + ",".join(attempt.failures)
                )
            raw = json.loads(provider_result.raw_text)
            vector = _parse_scope_vector(raw, step=step, tool_category=tool_category)
            self.cache.set(key, raw)
            return vector
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
    weighted_axes = _confidence_weight_axes(parsed.axes)
    return ScopeVector(
        step_id=step.id,
        tool=step.tool,
        tool_category=tool_category,
        **weighted_axes,
    )


def _confidence_weight_axes(axes: AxisBundle) -> Dict[str, AxisScore]:
    def _weight(axis: AxisOutput) -> AxisScore:
        return AxisScore(
            value=round(axis.value * axis.confidence, 6),
            rationale=axis.rationale,
            confidence=axis.confidence,
        )

    return {
        "spatial": _weight(axes.spatial),
        "temporal": _weight(axes.temporal),
        "depth": _weight(axes.depth),
        "irreversibility": _weight(axes.irreversibility),
        "resource_intensity": _weight(axes.resource_intensity),
        "legal_exposure": _weight(axes.legal_exposure),
        "dependency_creation": _weight(axes.dependency_creation),
        "stakeholder_radius": _weight(axes.stakeholder_radius),
        "power_concentration": _weight(axes.power_concentration),
        "uncertainty": _weight(axes.uncertainty),
    }


def build_prompt_v2(
    step: PlanStep,
    *,
    tool_category: Optional[str],
    tool_priors: Dict[str, float],
) -> str:
    spec_path = Path(__file__).resolve().parent / "prompts" / "prompt_v2.md"
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


def build_prompt_v1(
    step: PlanStep,
    *,
    tool_category: Optional[str],
    tool_priors: Dict[str, float],
) -> str:
    return build_prompt_v2(step, tool_category=tool_category, tool_priors=tool_priors)


JudgeMode = Literal["none", "heuristic", "llm"]


def build_step_judge(
    mode: JudgeMode,
    *,
    provider: Optional[LLMProvider | Iterable[LLMProvider]] = None,
    cache_path: Optional[str] = None,
) -> StepJudge:
    if mode in {"none", "heuristic"}:
        return NoopStepJudge(mode=mode)
    if provider is None:
        providers = (EnvJSONProvider(),)
    elif isinstance(provider, Iterable) and not isinstance(provider, (str, bytes)):
        providers = tuple(provider)
    else:
        providers = (provider,)

    if not providers:
        providers = (EnvJSONProvider(),)

    cache = JudgeCache.from_path(cache_path)
    return LLMStepJudge(provider_client=ProviderAgnosticClient(providers=providers), cache=cache)
