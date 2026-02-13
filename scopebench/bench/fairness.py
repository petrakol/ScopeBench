from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
    axis_distribution: list[FairnessBucket]
    underrepresented: list[dict[str, Any]]
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

    domain_distribution = _build_buckets(domain_counts, total=total, min_share=min_share)
    decision_distribution = _build_buckets(
        decision_counts, total=total, all_categories=_VALID_DECISIONS, min_share=min_share
    )
    axis_distribution = _build_buckets(axis_counts, total=total, all_categories=SCOPE_AXES, min_share=min_share)

    underrepresented: list[dict[str, Any]] = []
    for label, buckets in (
        ("domain", domain_distribution),
        ("decision", decision_distribution),
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

    suggestions = [
        (
            f"Add about {entry['deficit']} case(s) in {entry['type']}='{entry['category']}' "
            f"to reach at least {entry['target_count']} examples."
        )
        for entry in underrepresented[:10]
    ]

    return DatasetFairnessReport(
        total_cases=total,
        domain_distribution=domain_distribution,
        decision_distribution=decision_distribution,
        axis_distribution=axis_distribution,
        underrepresented=underrepresented,
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
        "axis_distribution": to_rows(report.axis_distribution),
        "underrepresented": report.underrepresented,
        "contribution_suggestions": report.contribution_suggestions,
    }
