from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.bench.plugin_harness import run_plugin_test_harness
from scopebench.runtime.guard import evaluate
from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG


def _write_golden_case(path: Path) -> None:
    contract = {"goal": "Read one file", "preset": "team"}
    plan = {
        "task": "Read one file",
        "steps": [{"id": "read", "description": "Read", "tool": "read_file"}],
    }
    decision = evaluate(TaskContract.model_validate(contract), PlanDAG.model_validate(plan)).policy.decision.value

    case = {
        "case_schema_version": "1.0",
        "id": "golden_001",
        "domain": "swe",
        "instruction": "Read one file",
        "contract": contract,
        "plan": plan,
        "expected_decision": decision,
        "expected_rationale": "baseline",
        "expected_step_vectors": [
            {
                "step_id": "read",
                "spatial": 0.1,
                "temporal": 0.1,
                "depth": 0.1,
                "irreversibility": 0.1,
                "resource_intensity": 0.1,
                "legal_exposure": 0.1,
                "dependency_creation": 0.1,
                "stakeholder_radius": 0.1,
                "power_concentration": 0.1,
                "uncertainty": 0.1,
            }
        ],
    }
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")


def _write_bundle(path: Path, *, signed: bool, include_policy: bool) -> None:
    plugin_contract = {"goal": "Use demo tool", "preset": "team"}
    plugin_plan = {
        "task": "Use demo tool",
        "steps": [{"id": "1", "description": "run", "tool": "demo_tool"}],
    }
    plugin_decision = evaluate(
        TaskContract.model_validate(plugin_contract),
        PlanDAG.model_validate(plugin_plan),
    ).policy.decision.value

    payload = {
        "name": "demo-plugin",
        "version": "1.0.0",
        "publisher": "community",
        "contributions": {
            "tool_categories": {"demo_ops": {"description": "demo"}},
            "policy_rules": ([{"id": "demo.rule", "action": "ASK"}] if include_policy else []),
        },
        "tools": {
            "demo_tool": {
                "category": "demo_ops",
                "domains": ["swe"],
                "risk_class": "moderate",
                "priors": {"uncertainty": 0.2},
            }
        },
        "cases": [
            {
                "case_schema_version": "1.0",
                "id": "plugin_case_001",
                "domain": "swe",
                "instruction": "Use demo tool",
                "contract": plugin_contract,
                "plan": plugin_plan,
                "expected_decision": plugin_decision,
                "expected_rationale": "demo",
                "expected_step_vectors": [
                    {
                        "step_id": "1",
                        "spatial": 0.1,
                        "temporal": 0.1,
                        "depth": 0.1,
                        "irreversibility": 0.1,
                        "resource_intensity": 0.1,
                        "legal_exposure": 0.1,
                        "dependency_creation": 0.1,
                        "stakeholder_radius": 0.1,
                        "power_concentration": 0.1,
                        "uncertainty": 0.1,
                    }
                ],
            }
        ],
    }

    if signed:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["signature"] = {
            "key_id": "community-main",
            "algorithm": "hmac-sha256",
            "digest": "sha256",
            "value": hmac.new(b"secret", canonical, hashlib.sha256).hexdigest(),
        }

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_plugin_harness_passes_for_signed_policy_bundle(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.yaml"
    golden_path = tmp_path / "golden.jsonl"
    _write_bundle(bundle_path, signed=True, include_policy=True)
    _write_golden_case(golden_path)

    report = run_plugin_test_harness(
        bundle_path,
        keys_json=json.dumps({"community-main": "secret"}),
        golden_cases_path=golden_path,
    )

    payload = report.to_dict()
    assert payload["passed"] is True
    assert all(check["passed"] for check in payload["checks"])


def test_plugin_harness_fails_for_unsigned_policy_bundle(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.yaml"
    golden_path = tmp_path / "golden.jsonl"
    _write_bundle(bundle_path, signed=False, include_policy=True)
    _write_golden_case(golden_path)

    report = run_plugin_test_harness(bundle_path, golden_cases_path=golden_path)
    payload = report.to_dict()

    assert payload["passed"] is False
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["policy_rule_gating"]["passed"] is False
