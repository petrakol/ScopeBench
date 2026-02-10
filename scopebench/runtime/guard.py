from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import yaml

from scopebench.contracts import TaskContract, contract_from_dict
from scopebench.plan import PlanDAG, plan_from_dict
from scopebench.policy.engine import PolicyResult, evaluate_policy
from scopebench.scoring.axes import ScopeAggregate, ScopeVector
from scopebench.scoring.calibration import CalibratedDecisionThresholds, apply_calibration
from scopebench.scoring.llm_judge import JudgeMode, build_step_judge
from scopebench.scoring.rules import ToolRegistry, aggregate_scope, score_step
from scopebench.tracing.otel import get_tracer


@dataclass
class EvaluationResult:
    contract: TaskContract
    plan: PlanDAG
    vectors: List[ScopeVector]
    aggregate: ScopeAggregate
    policy: PolicyResult


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return cast(Dict[str, Any], data)


def load_contract(path: str) -> TaskContract:
    return contract_from_dict(load_yaml(path))


def load_plan(path: str) -> PlanDAG:
    return plan_from_dict(load_yaml(path))


def evaluate(
    contract: TaskContract,
    plan: PlanDAG,
    tool_registry: Optional[ToolRegistry] = None,
    calibration: Optional[CalibratedDecisionThresholds] = None,
    policy_backend: Optional[str] = None,
    judge: JudgeMode = "heuristic",
    judge_cache_path: Optional[str] = None,
) -> EvaluationResult:
    tracer = get_tracer("scopebench")

    if tool_registry is None:
        tool_registry = ToolRegistry.load_default()

    step_judge = build_step_judge(judge, cache_path=judge_cache_path)

    with tracer.start_as_current_span("scopebench.evaluate") as span:
        span.set_attribute("scopebench.task", plan.task)
        span.set_attribute("scopebench.preset", contract.preset.value)
        span.set_attribute("scopebench.judge.mode", judge)
        if step_judge.telemetry.prompt_version:
            span.set_attribute("scopebench.judge.prompt_version", step_judge.telemetry.prompt_version)

        vectors: List[ScopeVector] = []
        for step in plan.steps:
            with tracer.start_as_current_span("scopebench.score_step") as step_span:
                step_span.set_attribute("scopebench.step_id", step.id)
                step_span.set_attribute("scopebench.tool", step.tool or "")
                step_span.set_attribute(
                    "scopebench.tool_category",
                    step.tool_category or tool_registry.category_of(step.tool) or "",
                )
                tool_info = tool_registry.get(step.tool)
                effective_tool_category = step.tool_category or (
                    tool_info.category if tool_info else None
                )
                tool_priors = dict(tool_info.priors) if tool_info else {}

                v = score_step(step, tool_registry)

                override = step_judge.judge_step(
                    step,
                    tool_category=effective_tool_category,
                    tool_priors=tool_priors,
                )
                if override is not None:
                    v = override
                elif judge == "llm":
                    v.uncertainty.value = min(1.0, v.uncertainty.value + 0.05)
                    if step_judge.telemetry.fallback_reason:
                        prefix = v.uncertainty.rationale + "; " if v.uncertainty.rationale else ""
                        v.uncertainty.rationale = (
                            f"{prefix}{step_judge.telemetry.fallback_reason}"
                        )

                step_span.set_attribute("scopebench.judge.cache_hit", step_judge.telemetry.cache_hit)
                if step_judge.telemetry.provider:
                    step_span.set_attribute("scopebench.judge.provider", step_judge.telemetry.provider)
                if step_judge.telemetry.fallback_reason:
                    step_span.set_attribute(
                        "scopebench.judge.fallback_reason", step_judge.telemetry.fallback_reason
                    )

                # Emit axis values as span attributes (OTel-friendly)
                for axis, val in v.as_dict.items():
                    step_span.set_attribute(f"scopebench.axis.{axis}", float(val))

                vectors.append(v)

        agg = aggregate_scope(vectors, plan=plan)
        if calibration is not None:
            agg = apply_calibration(agg, calibration)
            span.set_attribute(
                "scopebench.calibration.global_scale", float(calibration.global_scale)
            )
        for axis, val in agg.as_dict().items():
            span.set_attribute(f"scopebench.aggregate.{axis}", float(val))
        span.set_attribute("scopebench.n_steps", agg.n_steps)

        policy = evaluate_policy(
            contract,
            agg,
            step_vectors=vectors,
            plan=plan,
            policy_backend=policy_backend,
        )
        span.set_attribute("scopebench.decision", policy.decision.value)

        return EvaluationResult(
            contract=contract,
            plan=plan,
            vectors=vectors,
            aggregate=agg,
            policy=policy,
        )


def evaluate_from_files(
    contract_path: str,
    plan_path: str,
    calibration: Optional[CalibratedDecisionThresholds] = None,
    policy_backend: Optional[str] = None,
    judge: JudgeMode = "heuristic",
    judge_cache_path: Optional[str] = None,
) -> EvaluationResult:
    contract = load_contract(contract_path)
    plan = load_plan(plan_path)
    return evaluate(
        contract,
        plan,
        calibration=calibration,
        policy_backend=policy_backend,
        judge=judge,
        judge_cache_path=judge_cache_path,
    )
