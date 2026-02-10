from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict

from scopebench.bench.dataset import load_cases
from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG


def run_judge_bench(cases_jsonl: Path, evaluate_case: Callable[[TaskContract, PlanDAG], Any]) -> Dict[str, Any]:
    cases = load_cases(cases_jsonl)
    per_domain_counts = defaultdict(lambda: {"matched": 0, "total": 0})
    error_case_ids: list[str] = []

    for case in cases:
        contract = TaskContract.model_validate(case.contract)
        plan = PlanDAG.model_validate(case.plan)
        res = evaluate_case(contract, plan)
        predicted = res.policy.decision.value
        matched = predicted == case.expected_decision

        stats = per_domain_counts[case.domain]
        stats["total"] += 1
        if matched:
            stats["matched"] += 1
        else:
            error_case_ids.append(case.id)

    per_domain = {}
    for domain, stats in per_domain_counts.items():
        total = stats["total"]
        matched = stats["matched"]
        per_domain[domain] = {
            "matched": matched,
            "total": total,
            "agreement": (matched / total) if total else 0.0,
        }

    return {
        "total_cases": len(cases),
        "per_domain": per_domain,
        "error_case_ids": error_case_ids,
    }
