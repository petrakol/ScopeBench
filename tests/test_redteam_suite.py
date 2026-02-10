from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.bench.dataset import ScopeBenchCase, load_cases  # noqa: E402
from scopebench.bench.judge import run_judge_bench  # noqa: E402
from scopebench.contracts import TaskContract  # noqa: E402
from scopebench.plan import PlanDAG  # noqa: E402
from scopebench.redteam.generate import generate_cases, write_jsonl  # noqa: E402
from scopebench.runtime.guard import evaluate  # noqa: E402

REDTEAM_PATH = ROOT / "scopebench" / "bench" / "cases" / "redteam.jsonl"


def test_redteam_generator_outputs_valid_jsonl_and_minimum_size(tmp_path: Path) -> None:
    output_path = tmp_path / "generated_redteam.jsonl"
    generated_cases = generate_cases(count=60, seed=123)
    write_jsonl(generated_cases, output_path)

    with output_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    assert len(rows) >= 50
    parsed = [
        ScopeBenchCase(
            case_schema_version=row["case_schema_version"],
            id=row["id"],
            domain=row["domain"],
            instruction=row["instruction"],
            contract=row["contract"],
            plan=row["plan"],
            expected_decision=row["expected_decision"],
            expected_rationale=row["expected_rationale"],
            expected_step_vectors=row["expected_step_vectors"],
            notes=row.get("notes"),
        )
        for row in rows
    ]
    assert len(parsed) == len(rows)


def test_redteam_dataset_golden_decisions_match_evaluator() -> None:
    assert REDTEAM_PATH.exists(), "red-team dataset must be checked in"
    cases = load_cases(REDTEAM_PATH)
    assert len(cases) >= 100

    result = run_judge_bench(
        REDTEAM_PATH, evaluate_case=lambda contract, plan: evaluate(contract, plan)
    )
    assert result["error_case_ids"] == []


def _random_dag(rng: random.Random) -> PlanDAG:
    n_steps = rng.randint(2, 15)
    steps = []
    verbs = [
        "Read local file",
        "Apply patch",
        "Run tests",
        "Create service",
        "Grant access key",
        "Deploy to production",
        "Analyze logs",
    ]
    tools = ["git_read", "git_patch", "analysis", "iam_change", "prod_deploy", "infra_provision"]

    for idx in range(1, n_steps + 1):
        depends_on = [str(dep) for dep in range(1, idx) if rng.random() < 0.25]
        step = {
            "id": str(idx),
            "description": f"{rng.choice(verbs)} #{idx}",
            "tool": rng.choice(tools),
            "depends_on": depends_on,
        }
        steps.append(step)

    return PlanDAG.model_validate({"task": "Random fuzz plan", "steps": steps})


def test_no_crash_policy_and_scoring_over_random_dags() -> None:
    rng = random.Random(2026)
    for _ in range(1000):
        contract = TaskContract.model_validate(
            {
                "goal": "Fuzz test proportionality",
                "preset": rng.choice(["personal", "team", "enterprise", "regulated"]),
            }
        )
        plan = _random_dag(rng)
        result = evaluate(contract, plan)
        assert result.policy.decision.value in {"ALLOW", "ASK", "DENY"}
