from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    steps = {step.id: step for step in plan.steps}
    paths = _all_root_to_leaf_paths(plan)
    max_risk = _max_risk_path(paths, step_vectors)

    candidates: List[Tuple[List[str], str]] = [(path, "path") for path in paths]
    if max_risk is not None:
        candidates.append((max_risk, "max_risk_path"))

    flags: Dict[Tuple[str, str], KneeFlag] = {}
    for path, source in candidates:
        seen_high_ratio = False
        for step_id in path:
            step = steps[step_id]
            benefit = float(step.est_benefit or 0.0)
            cost = float(step.est_cost_usd or step.est_time_days or 0.0)
            if cost <= 0.0 or benefit <= 0.0:
                continue
            ratio = benefit / cost
            if ratio >= min_marginal_ratio:
                seen_high_ratio = True
                continue
            if seen_high_ratio:
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
    return sorted(flags.values(), key=lambda item: (item.ratio, item.step_id, item.source))
