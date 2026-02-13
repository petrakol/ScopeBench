from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from scopebench.bench.dataset import ScopeBenchCase
from scopebench.scoring.axes import SCOPE_AXES

_VALID_DECISIONS = ("ALLOW", "ASK", "DENY")


@dataclass(frozen=True)
class FairnessBucket:
    category: str
    count: int
    share: float
    deficit: int


@dataclass(frozen=True)
class DatasetFairnessReport:
    total_cases: int
    domain_distribution: list[FairnessBucket]
    decision_distribution: list[FairnessBucket]
    effect_distribution: list[FairnessBucket]
    axis_distribution: list[FairnessBucket]
    decision_by_domain: dict[str, dict[str, int]]
    decision_by_effect: dict[str, dict[str, int]]
    underrepresented: list[dict[str, Any]]
    priority_matrix: list[dict[str, Any]]
    contribution_suggestions: list[str]


def _case_axis_profile(case: ScopeBenchCase) -> dict[str, float]:
    totals = {axis: 0.0 for axis in SCOPE_AXES}
    vectors = case.expected_step_vectors or []
    if not vectors:
        return totals

    for vector in vectors:
        for axis in SCOPE_AXES:
            value = vector.get(axis, 0.0)
            totals[axis] += float(value)

    steps = max(1, len(vectors))
    return {axis: totals[axis] / steps for axis in SCOPE_AXES}


def _dominant_axis(case: ScopeBenchCase) -> str:
    profile = _case_axis_profile(case)
    return max(SCOPE_AXES, key=lambda axis: profile[axis])


def _case_effect_categories(case: ScopeBenchCase) -> set[str]:
    categories: set[str] = set()
    steps = case.plan.get("steps", []) if isinstance(case.plan, dict) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        effects = step.get("effects")
        if not isinstance(effects, dict):
            continue
        for key in effects:
            if key != "version":
                categories.add(str(key))
    return categories or {"none"}


def _build_buckets(
    counts: Counter[str], *, total: int, all_categories: Iterable[str] | None = None, min_share: float
) -> list[FairnessBucket]:
    if all_categories is None:
        categories = sorted(counts.keys())
    else:
        categories = list(all_categories)

    min_count = max(1, int(total * min_share)) if total else 0
    buckets: list[FairnessBucket] = []
    for category in categories:
        count = int(counts.get(category, 0))
        share = (count / total) if total else 0.0
        deficit = max(0, min_count - count)
        buckets.append(FairnessBucket(category=category, count=count, share=share, deficit=deficit))

    buckets.sort(key=lambda row: (row.count, row.category))
    return buckets


def evaluate_dataset_fairness(cases: list[ScopeBenchCase], *, min_share: float = 0.1) -> DatasetFairnessReport:
    if min_share <= 0 or min_share >= 1:
        raise ValueError("min_share must be in the range (0, 1)")

    total = len(cases)
    domain_counts = Counter(case.domain for case in cases)
    decision_counts = Counter(case.expected_decision for case in cases)
    axis_counts = Counter(_dominant_axis(case) for case in cases)
    effect_counts: Counter[str] = Counter()
    effect_categories_seen: set[str] = set()

    decision_by_domain: dict[str, dict[str, int]] = {}
    decision_by_effect: dict[str, dict[str, int]] = {}
    combo_counts: Counter[tuple[str, str, str]] = Counter()

    for case in cases:
        effect_categories = _case_effect_categories(case)
        effect_categories_seen.update(effect_categories)

        domain_row = decision_by_domain.setdefault(
            case.domain, {decision: 0 for decision in _VALID_DECISIONS}
        )
        domain_row[case.expected_decision] += 1

        for effect in effect_categories:
            effect_counts[effect] += 1
            combo_counts[(case.domain, effect, case.expected_decision)] += 1

            effect_row = decision_by_effect.setdefault(
                effect, {decision: 0 for decision in _VALID_DECISIONS}
            )
            effect_row[case.expected_decision] += 1

    domain_distribution = _build_buckets(domain_counts, total=total, min_share=min_share)
    decision_distribution = _build_buckets(
        decision_counts, total=total, all_categories=_VALID_DECISIONS, min_share=min_share
    )
    effect_distribution = _build_buckets(
        effect_counts, total=total, all_categories=sorted(effect_categories_seen), min_share=min_share
    )
    axis_distribution = _build_buckets(axis_counts, total=total, all_categories=SCOPE_AXES, min_share=min_share)

    underrepresented: list[dict[str, Any]] = []
    for label, buckets in (
        ("domain", domain_distribution),
        ("decision", decision_distribution),
        ("effect", effect_distribution),
        ("axis", axis_distribution),
    ):
        for bucket in buckets:
            if bucket.deficit > 0:
                underrepresented.append(
                    {
                        "type": label,
                        "category": bucket.category,
                        "count": bucket.count,
                        "share": bucket.share,
                        "target_count": bucket.count + bucket.deficit,
                        "deficit": bucket.deficit,
                    }
                )

    underrepresented.sort(key=lambda row: (-row["deficit"], row["type"], row["category"]))

    domain_deficits = {bucket.category: bucket.deficit for bucket in domain_distribution}
    decision_deficits = {bucket.category: bucket.deficit for bucket in decision_distribution}
    effect_deficits = {bucket.category: bucket.deficit for bucket in effect_distribution}

    priority_matrix: list[dict[str, Any]] = []
    for domain, effect, decision in product(
        sorted(domain_counts.keys()), sorted(effect_categories_seen), _VALID_DECISIONS
    ):
        count = int(combo_counts.get((domain, effect, decision), 0))
        deficit_score = int(
            domain_deficits.get(domain, 0)
            + effect_deficits.get(effect, 0)
            + decision_deficits.get(decision, 0)
            + (1 if count == 0 else 0)
        )
        priority_matrix.append(
            {
                "domain": domain,
                "effect_category": effect,
                "decision": decision,
                "count": count,
                "priority_score": deficit_score,
            }
        )

    priority_matrix.sort(
        key=lambda row: (-row["priority_score"], row["count"], row["domain"], row["effect_category"], row["decision"])
    )

    suggestions = [
        (
            f"Add about {entry['deficit']} case(s) in {entry['type']}='{entry['category']}' "
            f"to reach at least {entry['target_count']} examples."
        )
        for entry in underrepresented[:10]
    ]
    for item in priority_matrix[:10]:
        suggestions.append(
            "Prioritize "
            f"domain='{item['domain']}', effect='{item['effect_category']}', decision='{item['decision']}' "
            f"(current={item['count']}, score={item['priority_score']})."
        )

    return DatasetFairnessReport(
        total_cases=total,
        domain_distribution=domain_distribution,
        decision_distribution=decision_distribution,
        effect_distribution=effect_distribution,
        axis_distribution=axis_distribution,
        decision_by_domain=decision_by_domain,
        decision_by_effect=decision_by_effect,
        underrepresented=underrepresented,
        priority_matrix=priority_matrix,
        contribution_suggestions=suggestions,
    )


def fairness_report_to_dict(report: DatasetFairnessReport) -> dict[str, Any]:
    def to_rows(items: list[FairnessBucket]) -> list[dict[str, Any]]:
        return [
            {
                "category": item.category,
                "count": item.count,
                "share": item.share,
                "deficit": item.deficit,
            }
            for item in items
        ]

    return {
        "total_cases": report.total_cases,
        "domain_distribution": to_rows(report.domain_distribution),
        "decision_distribution": to_rows(report.decision_distribution),
        "effect_distribution": to_rows(report.effect_distribution),
        "axis_distribution": to_rows(report.axis_distribution),
        "decision_by_domain": report.decision_by_domain,
        "decision_by_effect": report.decision_by_effect,
        "underrepresented": report.underrepresented,
        "priority_matrix": report.priority_matrix,
        "contribution_suggestions": report.contribution_suggestions,
    }
