from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from scopebench.contracts import TaskContract, contract_from_dict
from scopebench.plan import PlanDAG, plan_from_dict
from scopebench.policy.engine import PolicyResult, evaluate_policy
from scopebench.scoring.axes import ScopeAggregate, ScopeVector
from scopebench.scoring.calibration import CalibratedDecisionThresholds, apply_calibration
from scopebench.scoring.llm_judge import judge_step
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
        return yaml.safe_load(f)


def load_contract(path: str) -> TaskContract:
    return contract_from_dict(load_yaml(path))


def load_plan(path: str) -> PlanDAG:
    return plan_from_dict(load_yaml(path))


def evaluate(
    contract: TaskContract,
    plan: PlanDAG,
    tool_registry: Optional[ToolRegistry] = None,
    calibration: Optional[CalibratedDecisionThresholds] = None,
) -> EvaluationResult:
    tracer = get_tracer("scopebench")

    if tool_registry is None:
        tool_registry = ToolRegistry.load_default()

    with tracer.start_as_current_span("scopebench.evaluate") as span:
        span.set_attribute("scopebench.task", plan.task)
        span.set_attribute("scopebench.preset", contract.preset.value)

        vectors: List[ScopeVector] = []
        for step in plan.steps:
            with tracer.start_as_current_span("scopebench.score_step") as step_span:
                step_span.set_attribute("scopebench.step_id", step.id)
                step_span.set_attribute("scopebench.tool", step.tool or "")
                step_span.set_attribute("scopebench.tool_category", step.tool_category or tool_registry.category_of(step.tool) or "")

                v = score_step(step, tool_registry)

                # Optional LLM-judge override
                override = judge_step(step)
                if override is not None:
                    v = override

                # Emit axis values as span attributes (OTel-friendly)
                for axis, val in v.as_dict.items():
                    step_span.set_attribute(f"scopebench.axis.{axis}", float(val))

                vectors.append(v)

        agg = aggregate_scope(vectors)
        if calibration is not None:
            agg = apply_calibration(agg, calibration)
            span.set_attribute("scopebench.calibration.global_scale", float(calibration.global_scale))
        for axis, val in agg.as_dict().items():
            span.set_attribute(f"scopebench.aggregate.{axis}", float(val))
        span.set_attribute("scopebench.n_steps", agg.n_steps)

        policy = evaluate_policy(contract, agg, step_vectors=vectors)
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
) -> EvaluationResult:
    contract = load_contract(contract_path)
    plan = load_plan(plan_path)
    return evaluate(contract, plan, calibration=calibration)
