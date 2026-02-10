from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from scopebench.bench.dataset import ScopeBenchCase, default_cases_path, load_cases
from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.plugins import PluginBundle, PluginManager
from scopebench.runtime.guard import evaluate, load_contract, load_plan

_VALID_DECISIONS = {"ALLOW", "ASK", "DENY"}


@dataclass(frozen=True)
class HarnessCheckResult:
    name: str
    passed: bool
    details: Dict[str, Any]


@dataclass(frozen=True)
class PluginHarnessReport:
    plugin: Dict[str, Any]
    checks: List[HarnessCheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "plugin": self.plugin,
            "checks": [
                {"name": c.name, "passed": c.passed, "details": c.details}
                for c in self.checks
            ],
        }


class PluginHarnessError(ValueError):
    """Raised when a plugin harness input is invalid."""


def _evaluate_case(case: ScopeBenchCase) -> str:
    contract = TaskContract.model_validate(case.contract)
    plan = PlanDAG.model_validate(case.plan)
    return evaluate(contract, plan).policy.decision.value


def _run_plugin_case_checks(cases: List[ScopeBenchCase], max_cases: int = 5) -> HarnessCheckResult:
    evaluated = 0
    mismatches: List[str] = []

    for case in cases[:max_cases]:
        predicted = _evaluate_case(case)
        evaluated += 1
        if predicted != case.expected_decision:
            mismatches.append(
                f"{case.id}: expected {case.expected_decision}, got {predicted}"
            )

    return HarnessCheckResult(
        name="plugin_cases",
        passed=not mismatches,
        details={"evaluated": evaluated, "mismatches": mismatches},
    )


def _run_representative_examples(repo_root: Path) -> HarnessCheckResult:
    pairs = [
        ("phone_charge", "phone_charge.contract.yaml", "phone_charge.plan.yaml"),
        ("coding_bugfix", "coding_bugfix.contract.yaml", "coding_bugfix.plan.yaml"),
        ("ops_rotate_key", "ops_rotate_key.contract.yaml", "ops_rotate_key.plan.yaml"),
    ]
    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    for name, contract_file, plan_file in pairs:
        contract_path = repo_root / "examples" / contract_file
        plan_path = repo_root / "examples" / plan_file
        contract = load_contract(str(contract_path))
        plan = load_plan(str(plan_path))
        decision = evaluate(contract, plan).policy.decision.value
        results.append({"name": name, "decision": decision})
        if decision not in _VALID_DECISIONS:
            failures.append(f"{name}: invalid decision {decision}")

    return HarnessCheckResult(
        name="representative_examples",
        passed=not failures,
        details={"results": results, "failures": failures},
    )


def _run_golden_regression(golden_cases: Path, max_cases: Optional[int]) -> HarnessCheckResult:
    cases = load_cases(golden_cases)
    subset = cases[:max_cases] if max_cases is not None else cases
    mismatches: List[str] = []

    for case in subset:
        predicted = _evaluate_case(case)
        if predicted != case.expected_decision:
            mismatches.append(
                f"{case.id}: expected {case.expected_decision}, got {predicted}"
            )

    return HarnessCheckResult(
        name="golden_regression",
        passed=not mismatches,
        details={
            "dataset": str(golden_cases),
            "evaluated": len(subset),
            "mismatches": mismatches[:25],
            "total_mismatches": len(mismatches),
        },
    )


def run_plugin_test_harness(
    bundle_path: Path,
    *,
    keys_json: str = "",
    golden_cases_path: Optional[Path] = None,
    max_golden_cases: Optional[int] = None,
) -> PluginHarnessReport:
    if not bundle_path.exists():
        raise PluginHarnessError(f"plugin bundle does not exist: {bundle_path}")

    keyring = PluginManager._load_keyring(keys_json)
    bundle = PluginManager._load_bundle(bundle_path, keyring)

    plugin_info = {
        "name": bundle.name,
        "version": bundle.version,
        "publisher": bundle.publisher,
        "source_path": bundle.source_path,
        "signed": bundle.signed,
        "signature_valid": bundle.signature_valid,
        "signature_error": bundle.signature_error,
    }

    checks: List[HarnessCheckResult] = []
    checks.append(
        HarnessCheckResult(
            name="signature_validation",
            passed=(not bundle.signed) or bundle.signature_valid,
            details={
                "signed": bundle.signed,
                "signature_valid": bundle.signature_valid,
                "signature_error": bundle.signature_error,
            },
        )
    )

    policy_loaded = not bundle.policy_rules or (bundle.signed and bundle.signature_valid)
    checks.append(
        HarnessCheckResult(
            name="policy_rule_gating",
            passed=policy_loaded,
            details={
                "policy_rules_declared": len(bundle.policy_rules),
                "policy_rules_loadable": policy_loaded,
            },
        )
    )

    checks.append(_run_plugin_case_checks(bundle.cases))

    repo_root = Path(__file__).resolve().parents[2]
    checks.append(_run_representative_examples(repo_root))

    golden_path = golden_cases_path or default_cases_path()
    checks.append(_run_golden_regression(golden_path, max_golden_cases))

    return PluginHarnessReport(plugin=plugin_info, checks=checks)
