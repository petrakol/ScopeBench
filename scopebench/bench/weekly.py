from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from scopebench.runtime.guard import evaluate_from_files
from scopebench.scoring.calibration import (
    CalibratedDecisionThresholds,
    compute_axis_calibration_from_telemetry,
    write_calibration_file,
)


@dataclass
class WeeklyCalibrationReport:
    total_runs: int
    decision_counts: Dict[str, int]
    top_triggered_rules: List[tuple[str, int]]
    ask_action_counts: Dict[str, int]
    outcome_counts: Dict[str, int]


@dataclass
class BenchmarkReplayResult:
    name: str
    decision: str
    expected: set[str]
    ok: bool


def _iter_telemetry_rows(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def summarize_weekly_telemetry(path: Path) -> WeeklyCalibrationReport:
    decision_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    ask_actions: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()

    total = 0
    for row in _iter_telemetry_rows(path):
        total += 1
        decision = row.get("decision")
        if isinstance(decision, str):
            decision_counts[decision] += 1

        for rule in row.get("triggered_rules", []):
            if isinstance(rule, str):
                rule_counts[rule] += 1

        ask_action = row.get("ask_action")
        if isinstance(ask_action, str) and ask_action:
            ask_actions[ask_action] += 1

        outcome = row.get("outcome")
        if isinstance(outcome, str) and outcome:
            outcomes[outcome] += 1

    return WeeklyCalibrationReport(
        total_runs=total,
        decision_counts=dict(decision_counts),
        top_triggered_rules=rule_counts.most_common(5),
        ask_action_counts=dict(ask_actions),
        outcome_counts=dict(outcomes),
    )


def replay_benchmark_slice(
    root: Path, cases: Optional[List[dict]] = None
) -> List[BenchmarkReplayResult]:
    if cases is None:
        cases = [
            {
                "name": "coding_bugfix",
                "contract": root / "examples" / "coding_bugfix.contract.yaml",
                "plan": root / "examples" / "coding_bugfix.plan.yaml",
                "expected": {"ALLOW"},
            },
            {
                "name": "swe_fix",
                "contract": root / "examples" / "swe_fix.contract.yaml",
                "plan": root / "examples" / "swe_fix.plan.yaml",
                "expected": {"ALLOW"},
            },
            {
                "name": "phone_charge",
                "contract": root / "examples" / "phone_charge.contract.yaml",
                "plan": root / "examples" / "phone_charge.plan.yaml",
                "expected": {"DENY"},
            },
        ]

    results: List[BenchmarkReplayResult] = []
    for case in cases:
        res = evaluate_from_files(str(case["contract"]), str(case["plan"]))
        decision = res.policy.decision.value
        expected = set(case["expected"])
        results.append(
            BenchmarkReplayResult(
                name=case["name"],
                decision=decision,
                expected=expected,
                ok=decision in expected,
            )
        )
    return results


def refresh_weekly_calibration(
    telemetry_path: Path, output_path: Path
) -> tuple[CalibratedDecisionThresholds, WeeklyCalibrationReport]:
    """Compute and persist latest calibration factors from weekly telemetry."""
    calibration, _ = compute_axis_calibration_from_telemetry(telemetry_path)
    write_calibration_file(output_path, calibration)
    report = summarize_weekly_telemetry(telemetry_path)
    return calibration, report
