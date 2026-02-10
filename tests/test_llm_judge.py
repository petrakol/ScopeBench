from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.contracts import TaskContract  # noqa: E402
from scopebench.plan import PlanDAG, PlanStep  # noqa: E402
from scopebench.runtime.guard import evaluate  # noqa: E402
from scopebench.scoring.llm_judge import build_prompt_v1  # noqa: E402
from scopebench.scoring.providers.base import ProviderResult  # noqa: E402


class CountingProvider:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def complete(self, prompt: str) -> ProviderResult:
        assert "INPUT_JSON=" in prompt
        self.calls += 1
        return ProviderResult(raw_text=json.dumps(self.payload), provider_name="counting")


def _judge_payload(value: float) -> dict:
    axis = {"value": value, "rationale": "llm", "confidence": 0.9}
    return {
        "spatial": axis,
        "temporal": axis,
        "depth": axis,
        "irreversibility": axis,
        "resource_intensity": axis,
        "legal_exposure": axis,
        "dependency_creation": axis,
        "stakeholder_radius": axis,
        "power_concentration": axis,
        "uncertainty": axis,
    }


def test_prompt_builder_is_deterministic():
    step = PlanStep(id="1", description="Read code", tool="git_read")
    p1 = build_prompt_v1(step, tool_category="code", tool_priors={"depth": 0.1})
    p2 = build_prompt_v1(step, tool_category="code", tool_priors={"depth": 0.1})
    assert p1 == p2


def test_llm_judge_runs_for_each_step_and_overrides(tmp_path: Path):
    contract = TaskContract.model_validate({"goal": "x", "preset": "personal"})
    plan = PlanDAG.model_validate(
        {
            "task": "x",
            "steps": [
                {"id": "1", "description": "small edit", "tool": "analysis"},
                {"id": "2", "description": "small edit 2", "tool": "analysis"},
            ],
        }
    )

    provider = CountingProvider(_judge_payload(0.91))
    from scopebench.scoring.llm_judge import LLMStepJudge
    from scopebench.scoring.cache import JudgeCache

    # Build a judge with custom provider and monkeypatch factory for evaluate path.
    import scopebench.runtime.guard as guard

    original_builder = guard.build_step_judge

    def _factory(mode, *, provider=None, cache_path=None):
        if mode == "llm":
            return LLMStepJudge(provider=provider_obj, cache=JudgeCache.from_path(cache_path))
        return original_builder(mode, provider=provider, cache_path=cache_path)

    provider_obj = provider
    guard.build_step_judge = _factory
    try:
        res = evaluate(contract, plan, judge="llm", judge_cache_path=str(tmp_path / "cache"))
    finally:
        guard.build_step_judge = original_builder

    assert provider.calls == 2
    assert all(v.spatial.value == 0.91 for v in res.vectors)


def test_llm_cache_hit_avoids_provider_calls(tmp_path: Path):
    step = PlanStep(id="1", description="Read code", tool="git_read")
    payload = _judge_payload(0.3)
    provider = CountingProvider(payload)

    from scopebench.scoring.cache import JudgeCache
    from scopebench.scoring.llm_judge import LLMStepJudge

    cache = JudgeCache.from_path(str(tmp_path / "cache"))
    judge = LLMStepJudge(provider=provider, cache=cache)

    first = judge.judge_step(step, tool_category="code", tool_priors={"depth": 0.1})
    second = judge.judge_step(step, tool_category="code", tool_priors={"depth": 0.1})

    assert first is not None and second is not None
    assert provider.calls == 1


def test_invalid_provider_output_falls_back_none_and_increases_uncertainty(tmp_path: Path):
    contract = TaskContract.model_validate({"goal": "x", "preset": "personal"})
    plan = PlanDAG.model_validate({"task": "x", "steps": [{"id": "1", "description": "x", "tool": "analysis"}]})

    class BadProvider:
        def complete(self, prompt: str) -> ProviderResult:
            del prompt
            return ProviderResult(raw_text="{}", provider_name="bad")

    from scopebench.scoring.cache import JudgeCache
    from scopebench.scoring.llm_judge import LLMStepJudge
    import scopebench.runtime.guard as guard

    original_builder = guard.build_step_judge

    def _factory(mode, *, provider=None, cache_path=None):
        if mode == "llm":
            return LLMStepJudge(provider=BadProvider(), cache=JudgeCache.from_path(cache_path))
        return original_builder(mode, provider=provider, cache_path=cache_path)

    guard.build_step_judge = _factory
    try:
        res = evaluate(contract, plan, judge="llm", judge_cache_path=str(tmp_path / "cache"))
    finally:
        guard.build_step_judge = original_builder

    assert res.vectors[0].uncertainty.value >= 0.45
    assert "llm_judge_fallback" in res.vectors[0].uncertainty.rationale
