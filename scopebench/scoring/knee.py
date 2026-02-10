from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scopebench.plan import PlanDAG
from scopebench.scoring.axes import ScopeVector


@dataclass(frozen=True)
class KneeFlag:
    step_id: str
    path: Tuple[str, ...]
    ratio: float
    benefit: float
    cost: float
    source: str
    rationale: str


@dataclass(frozen=True)
class KneeStepLog:
    step_id: str
    path: Tuple[str, ...]
    source: str
    benefit: float
    cost: float
    ratio: float
    threshold: float
    is_knee: bool
    cumulative_knees: int


@dataclass(frozen=True)
class KneeRecommendation:
    recommendation_type: str
    target_step_id: Optional[str]
    rationale: str
    patch: Dict[str, Any]


@dataclass(frozen=True)
class KneeAnalysis:
    flags: List[KneeFlag]
    step_logs: List[KneeStepLog]
    recommendations: List[KneeRecommendation]


def _all_root_to_leaf_paths(plan: PlanDAG) -> List[List[str]]:
    by_id = {step.id: step for step in plan.steps}
    children: Dict[str, List[str]] = {step.id: [] for step in plan.steps}
    for step in plan.steps:
        for dep in step.depends_on:
            children[dep].append(step.id)
    roots = [step.id for step in plan.steps if not step.depends_on]
    leaves = {step_id for step_id, kids in children.items() if not kids}

    paths: List[List[str]] = []

    def walk(node: str, prefix: List[str]) -> None:
        cur = [*prefix, node]
        if node in leaves:
            paths.append(cur)
            return
        for nxt in children.get(node, []):
            walk(nxt, cur)

    for root in roots:
        walk(root, [])
    return [path for path in paths if all(step_id in by_id for step_id in path)]


def _risk_for_step(step_id: str, vectors_by_id: Dict[str, ScopeVector]) -> float:
    vector = vectors_by_id.get(step_id)
    if vector is None:
        return 0.0
    return vector.depth.value + vector.irreversibility.value + vector.resource_intensity.value


def _max_risk_path(paths: Sequence[List[str]], vectors: Optional[List[ScopeVector]]) -> Optional[List[str]]:
    if not paths:
        return None
    if not vectors:
        return list(paths[0])
    vectors_by_id = {v.step_id: v for v in vectors if v.step_id}
    return max(paths, key=lambda path: sum(_risk_for_step(step_id, vectors_by_id) for step_id in path))


def detect_knees(
    plan: PlanDAG,
    min_marginal_ratio: float,
    step_vectors: Optional[List[ScopeVector]] = None,
) -> List[KneeFlag]:
    return analyze_knees(
        plan=plan,
        min_marginal_ratio=min_marginal_ratio,
        max_knee_steps=2,
        step_vectors=step_vectors,
    ).flags


def _knee_recommendations(
    *,
    plan: PlanDAG,
    flags: List[KneeFlag],
    max_knee_steps: int,
    min_marginal_ratio: float,
) -> List[KneeRecommendation]:
    if not flags:
        return []

    steps_by_id = {step.id: step for step in plan.steps}
    flagged_steps = [steps_by_id[f.step_id] for f in flags if f.step_id in steps_by_id]
    recommendations: List[KneeRecommendation] = []

    if len(flags) > max_knee_steps:
        recommendations.append(
            KneeRecommendation(
                recommendation_type="halt_further_optimization",
                target_step_id=None,
                rationale=(
                    f"Detected {len(flags)} knee steps above max_knee_steps={max_knee_steps}; "
                    "prefer stopping after core objective is satisfied."
                ),
                patch={
                    "op": "truncate_after_last_non_knee",
                    "knee_step_ids": sorted({flag.step_id for flag in flags}),
                    "rationale": (
                        "Marginal benefit/cost dropped below min_marginal_ratio "
                        f"{min_marginal_ratio:.2f}."
                    ),
                },
            )
        )

    if flagged_steps:
        biggest_cost = max(flagged_steps, key=lambda step: float(step.est_cost_usd or step.est_time_days or 0.0))
        recommendations.append(
            KneeRecommendation(
                recommendation_type="split_large_step",
                target_step_id=biggest_cost.id,
                rationale=(
                    f"Step {biggest_cost.id} has high cost estimate and appears post-knee; "
                    "split into scoped checkpoints."
                ),
                patch={
                    "op": "split_step",
                    "target_step_id": biggest_cost.id,
                    "chunks": ["scoped_discovery", "implementation", "validation"],
                    "rationale": "Reduce late-stage over-optimization risk.",
                },
            )
        )

        high_risk_category = next(
            (
                step
                for step in flagged_steps
                if (step.tool_category or "") in {"infra", "iam", "payments", "finance", "prod"}
            ),
            None,
        )
        if high_risk_category is not None:
            recommendations.append(
                KneeRecommendation(
                    recommendation_type="lower_tool_category",
                    target_step_id=high_risk_category.id,
                    rationale=(
                        f"Step {high_risk_category.id} is post-knee and uses high-impact tool category "
                        f"'{high_risk_category.tool_category}'."
                    ),
                    patch={
                        "op": "replace",
                        "target_step_id": high_risk_category.id,
                        "step": {
                            "id": f"{high_risk_category.id}_analysis_first",
                            "description": "Use analysis-first fallback before high-impact action.",
                            "tool": "analysis",
                            "tool_category": "analysis",
                        },
                        "rationale": "Lower tool category while keeping optional optimization gated.",
                    },
                )
            )

    return recommendations


def analyze_knees(
    plan: PlanDAG,
    min_marginal_ratio: float,
    max_knee_steps: int,
    step_vectors: Optional[List[ScopeVector]] = None,
) -> KneeAnalysis:
    steps = {step.id: step for step in plan.steps}
    paths = _all_root_to_leaf_paths(plan)
    max_risk = _max_risk_path(paths, step_vectors)

    candidates: List[Tuple[List[str], str]] = [(path, "path") for path in paths]
    if max_risk is not None:
        candidates.append((max_risk, "max_risk_path"))

    flags: Dict[Tuple[str, str], KneeFlag] = {}
    logs: Dict[Tuple[str, str], KneeStepLog] = {}
    for path, source in candidates:
        seen_high_ratio = False
        cumulative_knees = 0
        for step_id in path:
            step = steps[step_id]
            benefit = float(step.est_benefit or 0.0)
            cost = float(step.est_cost_usd or step.est_time_days or 0.0)
            if cost <= 0.0 or benefit <= 0.0:
                continue
            ratio = benefit / cost
            is_knee = False
            if ratio >= min_marginal_ratio:
                seen_high_ratio = True
            elif seen_high_ratio:
                is_knee = True
                cumulative_knees += 1
                rationale = (
                    f"knee-of-curve: step {step_id} marginal ratio {ratio:.3f} "
                    f"({benefit:.3f}/{cost:.3f}) below threshold {min_marginal_ratio:.3f}"
                )
                key = (step_id, source)
                prev = flags.get(key)
                if prev is None or ratio < prev.ratio:
                    flags[key] = KneeFlag(
                        step_id=step_id,
                        path=tuple(path),
                        ratio=ratio,
                        benefit=benefit,
                        cost=cost,
                        source=source,
                        rationale=rationale,
                    )
            logs[(step_id, source)] = KneeStepLog(
                step_id=step_id,
                path=tuple(path),
                source=source,
                benefit=benefit,
                cost=cost,
                ratio=ratio,
                threshold=min_marginal_ratio,
                is_knee=is_knee,
                cumulative_knees=cumulative_knees,
            )

    ordered_flags = sorted(flags.values(), key=lambda item: (item.ratio, item.step_id, item.source))
    recommendations = _knee_recommendations(
        plan=plan,
        flags=ordered_flags,
        max_knee_steps=max_knee_steps,
        min_marginal_ratio=min_marginal_ratio,
    )
    ordered_logs = sorted(logs.values(), key=lambda item: (item.path, item.step_id, item.source))
    return KneeAnalysis(flags=ordered_flags, step_logs=ordered_logs, recommendations=recommendations)
